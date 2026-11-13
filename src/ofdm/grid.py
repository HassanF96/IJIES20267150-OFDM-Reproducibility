from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Grid:
    nfft: int
    symbols: int
    cp: int
    active_signed: tuple[int, ...]
    pilot_stride: int
    pilot_pattern: str = 'staggered_comb'
    pilot_energy: float = 1.0
    data_energy: float = 1.0
    sample_rate_hz: float = 1_920_000.0

    def __post_init__(self):
        for x in (self.nfft, self.symbols, self.cp, self.pilot_stride):
            if type(x) is not int:
                raise ValueError('integer grid dimensions required')
        if self.nfft < 4 or self.nfft % 2 or self.symbols < 1 or not 0 <= self.cp <= self.nfft:
            raise ValueError('invalid FFT/symbol/CP dimensions')
        a = self.active_signed
        if not a or tuple(sorted(set(a))) != a or 0 in a:
            raise ValueError('active signed bins must be unique, sorted, exclude DC')
        if any(type(k) is not int or not -self.nfft//2 <= k < self.nfft//2 for k in a):
            raise ValueError('active bin outside signed FFT range')
        if self.pilot_stride < 2 or len(a) % self.pilot_stride:
            raise ValueError('pilot fraction must divide active carrier count exactly')
        if self.pilot_pattern not in ('staggered_comb', 'comb'):
            raise ValueError('unsupported pilot pattern')
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.pilot_energy, self.data_energy, self.sample_rate_hz)):
            raise ValueError('energies/rate must be finite and positive')

    @property
    def shape(self):
        return self.nfft, self.symbols

    @property
    def active_bins(self):
        return np.asarray(self.active_signed) % self.nfft

    @property
    def active_mask(self):
        m = np.zeros(self.shape, bool)
        m[self.active_bins, :] = True
        return m

    @property
    def pilot_mask(self):
        m = np.zeros(self.shape, bool)
        for ell in range(self.symbols):
            offset = ell % self.pilot_stride if self.pilot_pattern == 'staggered_comb' else 0
            m[self.active_bins[offset::self.pilot_stride], ell] = True
        return m

    @property
    def data_mask(self):
        return self.active_mask & ~self.pilot_mask

    @property
    def null_mask(self):
        return ~self.active_mask

    @property
    def guard_bins(self):
        return np.flatnonzero(self.null_mask[:, 0] & (np.arange(self.nfft) != 0))

    @property
    def dc_bin(self):
        return 0

    @property
    def pilot_grid(self):
        p = np.zeros(self.shape, np.complex128)
        # Known constant complex pilot, no stochastic phase or hidden normalization.
        p[self.pilot_mask] = np.sqrt(self.pilot_energy) * (1+1j)/np.sqrt(2)
        return p

    @property
    def frame_samples(self):
        return self.symbols * (self.nfft+self.cp)

    @property
    def subcarrier_spacing_hz(self):
        return self.sample_rate_hz/self.nfft

    def map_data(self, data):
        data = np.asarray(data, dtype=np.complex128)
        if data.shape != (int(self.data_mask.sum()),) or not np.isfinite(data).all():
            raise ValueError('data must be a finite vector in C-order (k,ell) RE order')
        x = self.pilot_grid
        x[self.data_mask] = data
        return x

    def extract(self, grid, kind='data'):
        value = np.asarray(grid)
        if value.shape[-2:] != self.shape:
            raise ValueError('expected [...,N,L]')
        masks = {'data': self.data_mask, 'pilot': self.pilot_mask,
                 'active': self.active_mask, 'null': self.null_mask}
        return value[..., masks[kind]]
