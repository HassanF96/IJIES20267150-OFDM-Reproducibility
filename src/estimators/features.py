"""Pilot-only future residual input contract. Does not define/train a network."""
import numpy as np
from src.ofdm.information import require_pilots, pilot_scale


def pilot_features(observation, baseline, estimated_variance, scale_floor=1e-12):
    require_pilots(observation)
    g = observation.grid
    baseline = np.asarray(baseline)
    if baseline.shape != g.shape or not np.isfinite(baseline).all():
        raise ValueError('finite baseline grid required')
    if not np.isfinite(estimated_variance) or estimated_variance <= 0:
        raise ValueError('positive practical disturbance proxy required')
    scale = pilot_scale(observation, scale_floor)
    z = np.zeros(g.shape, complex)
    z[observation.fit_mask] = observation.y_allowed[observation.fit_mask]/g.pilot_grid[observation.fit_mask]/scale
    base = baseline*g.active_mask/scale
    features = np.stack([z.real, z.imag, base.real, base.imag,
                         observation.fit_mask, g.active_mask,
                         np.full(g.shape, estimated_variance/scale**2)]).astype(np.float64)
    return features[None, ...], scale  # [B=1,C=7,N,L]
