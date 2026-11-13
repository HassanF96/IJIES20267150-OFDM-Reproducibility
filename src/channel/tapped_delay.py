"""Generic discrete tapped-delay laws, not named 3GPP profiles."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ChannelLaw:
    family_id: str
    delays: tuple[int, ...]
    powers: tuple[float, ...]
    evolution: str = 'frame_static'

    def __post_init__(self):
        if (not self.delays or tuple(sorted(set(self.delays))) != self.delays or
                any(type(d) is not int or d < 0 for d in self.delays)):
            raise ValueError('unique increasing nonnegative integer delays required')
        if len(self.delays) != len(self.powers) or any(not np.isfinite(p) or p <= 0 for p in self.powers):
            raise ValueError('positive powers required')
        if not np.isclose(sum(self.powers), 1, rtol=0, atol=1e-12):
            raise ValueError('ensemble powers must sum to one; no per-realization normalization')
        if self.evolution != 'frame_static':
            raise ValueError('E4 implements frame-static channels only')

    def draw(self, rng):
        values = (rng.standard_normal(len(self.delays))+1j*rng.standard_normal(len(self.delays)))
        values *= np.sqrt(np.asarray(self.powers)/2)
        taps = np.zeros(max(self.delays)+1, np.complex128)
        taps[list(self.delays)] = values
        return taps


def convolve(samples, taps):
    """Ordinary linear convolution of whole serialized frame; keep receive window."""
    samples, taps = np.asarray(samples), np.asarray(taps)
    if samples.ndim != 1 or taps.ndim != 1 or not len(taps):
        raise ValueError('single-frame vectors required')
    return np.convolve(samples, taps, mode='full')[:len(samples)]


def frequency_response(taps, nfft, symbols):
    if len(taps) > nfft:
        raise ValueError('tap support exceeds FFT size')
    # Ordinary FFT: sqrt(N) times the unitary FFT, as required by convolution.
    return np.repeat(np.fft.fft(taps, nfft)[:, None], symbols, axis=1)
