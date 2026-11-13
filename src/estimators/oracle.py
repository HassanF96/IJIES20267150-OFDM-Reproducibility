"""Explicit ORACLE/EVALUATION namespace; never used by practical estimation."""
import numpy as np
from src.ofdm.information import require_pilots
from src.ofdm.receiver import detect
from src.channel.tapped_delay import frequency_response


def matched_lmmse(observation, delays, powers, true_variance, solve_floor=1e-12):
    require_pilots(observation)
    if true_variance < 0 or solve_floor <= 0:
        raise ValueError('invalid oracle variance/floor')
    grid = observation.grid
    # Independent observation-space covariance implementation, not practical wrapper.
    k = np.arange(grid.nfft)
    basis = np.exp(-2j*np.pi*k[:, None]*np.asarray(delays)[None, :]/grid.nfft)
    field = np.repeat(basis[:, None, :], grid.symbols, axis=1)
    a = field[observation.fit_mask]*grid.pilot_grid[observation.fit_mask, None]
    r = np.diag(powers)
    weight = np.linalg.solve(a@r@a.conj().T+(true_variance+solve_floor)*np.eye(len(a)),
                             observation.y_allowed[observation.fit_mask])
    out = field@(r@a.conj().T@weight)
    out[~grid.active_mask] = 0
    return out


def perfect_csi(observation, labels, constellation):
    # True scale remains in this explicitly named oracle-only entry point.
    return detect(observation, labels.h_true, constellation, max(labels.noise_variance_true, 1e-12))
