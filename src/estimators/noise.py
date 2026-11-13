import numpy as np
from src.ofdm.information import require_pilots
from src.utils.cost import count


def estimate_variance(observation, floor=1e-10, ledger=None):
    require_pilots(observation)
    if not np.isfinite(floor) or floor <= 0 or observation.null_samples.size < 1:
        raise ValueError('null samples and positive finite floor required')
    count(ledger, 'null_scale_estimates')
    raw = float(np.median(np.abs(observation.null_samples)**2)/np.log(2))
    return max(raw, floor)
