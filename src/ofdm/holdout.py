"""Receiver-seeded pilot masks only; no gate, data score or threshold search."""
import numpy as np


def pilot_holdout(grid, fraction, policy, rng):
    if fraction not in (.25, .5) or policy not in ('random', 'interleaved', 'block'):
        raise ValueError('E3 S1 infrastructure supports 1/4, 1/2 and declared mask policies')
    validation = np.zeros(grid.shape, bool)
    for ell in range(grid.symbols):
        bins = grid.active_bins[grid.pilot_mask[grid.active_bins, ell]]
        # Floor per-symbol heldout count, clamped to preserve at least one fitting pilot.
        count = min(len(bins)-1, max(1, int(np.floor(fraction*len(bins)))))
        if count < 1:
            raise ValueError('need at least two pilots per symbol for holdout')
        if policy == 'random':
            selected = rng.choice(len(bins), count, replace=False)
        elif policy == 'block':
            start = int(rng.integers(0, len(bins)-count+1))
            selected = np.arange(start, start+count)
        else:
            selected = np.floor(np.arange(count)*len(bins)/count).astype(int)
        validation[bins[selected], ell] = True
    return grid.pilot_mask & ~validation, validation
