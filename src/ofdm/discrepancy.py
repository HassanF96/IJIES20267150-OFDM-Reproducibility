"""E3 arithmetic primitives for analytic fixtures only in E4; no gates/thresholds."""
import numpy as np
from src.ofdm.modulation import hard_indices, posterior


def normalized_energies(y, h, estimated_variance, constellation, soft=False):
    if not np.isfinite(estimated_variance) or estimated_variance <= 0:
        raise ValueError('positive estimated scale required')
    y, h = np.broadcast_arrays(y, h)
    if soft:
        _, mean, variance, _ = posterior(y, h, estimated_variance, constellation)
        return (np.abs(y-h*mean)**2+np.abs(h)**2*variance)/estimated_variance
    decision = constellation.points[hard_indices(y, h, constellation)]
    return np.abs(y-h*decision)**2/estimated_variance


def radial_huber_mean(energies, kappa=np.sqrt(np.log(20))):
    u = np.asarray(energies)
    if not u.size or not np.isfinite(u).all() or np.any(u < 0) or not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('finite nonnegative energies and positive kappa required')
    return float(np.mean(np.where(u <= kappa*kappa, u, 2*kappa*np.sqrt(u)-kappa*kappa)))
