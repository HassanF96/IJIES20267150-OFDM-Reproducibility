"""MSB-first labels, I bits followed by Q bits; explicit Gray PAM axes."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True, slots=True)
class Constellation:
    name: str
    energy: float = 1.0

    def __post_init__(self):
        if self.name not in ('QPSK', '16QAM') or not np.isfinite(self.energy) or self.energy <= 0:
            raise ValueError('QPSK/16QAM and positive finite symbol energy required')

    @property
    def bits_per_symbol(self):
        return 2 if self.name == 'QPSK' else 4

    @property
    def labels(self):
        b = self.bits_per_symbol
        return ((np.arange(2**b)[:, None] >> np.arange(b-1, -1, -1)) & 1).astype(np.uint8)

    @property
    def points(self):
        labels = self.labels
        if self.name == 'QPSK':
            return ((1-2*labels[:, 0].astype(float)) +
                    1j*(1-2*labels[:, 1].astype(float))) * np.sqrt(self.energy/2)
        # Natural binary index -> Gray amplitude: 00:-3, 01:-1, 10:+3, 11:+1.
        axis = np.array([-3, -1, 3, 1])
        return (axis[2*labels[:, 0]+labels[:, 1]] +
                1j*axis[2*labels[:, 2]+labels[:, 3]]) * np.sqrt(self.energy/10)

    def modulate(self, bits):
        bits = np.asarray(bits)
        if bits.ndim < 1 or bits.shape[-1] % self.bits_per_symbol or not np.isin(bits, [0, 1]).all():
            raise ValueError('binary last dimension divisible by bits per symbol required')
        grouped = bits.reshape(*bits.shape[:-1], -1, self.bits_per_symbol).astype(np.int64)
        index = grouped @ (1 << np.arange(self.bits_per_symbol-1, -1, -1))
        return self.points[index]

    def bits_from_indices(self, indices):
        labels = self.labels[np.asarray(indices)]
        return labels.reshape(*labels.shape[:-2], -1)

    def demap(self, symbols):
        return self.bits_from_indices(hard_indices(symbols, np.ones_like(symbols), self))


def distances(y, h, constellation):
    y, h = np.broadcast_arrays(np.asarray(y, np.complex128), np.asarray(h, np.complex128))
    if not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('nonfinite observation or channel estimate')
    return np.abs(y[..., None]-h[..., None]*constellation.points)**2


def hard_indices(y, h, constellation):
    """Receive-domain ML; deterministic lowest constellation-index tie."""
    return distances(y, h, constellation).argmin(axis=-1)


def posterior(y, h, variance, constellation):
    v = np.asarray(variance, dtype=float)
    if not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError('strictly positive finite receiver variance required')
    d = distances(y, h, constellation)
    # Subtract minimum before division to avoid high-SNR exponent underflow issues.
    logits = -(d-d.min(axis=-1, keepdims=True))/v[..., None]
    logp = logits-logsumexp(logits, axis=-1, keepdims=True)
    p = np.exp(logp)
    mean = p @ constellation.points
    var = p @ np.abs(constellation.points)**2 - np.abs(mean)**2
    if np.any(var < -1e-10*constellation.energy):
        raise ArithmeticError('negative posterior variance beyond roundoff')
    return p, mean, np.maximum(var, 0), logp


def bit_llrs(logp, constellation):
    """E3 sign: log P(bit=1)/P(bit=0)."""
    labels = constellation.labels
    return np.stack([logsumexp(logp[..., labels[:, b] == 1], axis=-1) -
                     logsumexp(logp[..., labels[:, b] == 0], axis=-1)
                     for b in range(constellation.bits_per_symbol)], axis=-1)
