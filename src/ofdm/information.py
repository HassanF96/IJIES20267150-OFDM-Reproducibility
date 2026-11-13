"""Restricted receiver value objects. No simulator/label object in estimator API."""
from dataclasses import dataclass
import numpy as np
from src.ofdm.grid import Grid


def readonly(value, dtype):
    value = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite receiver input')
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class PilotObservation:
    grid: Grid
    y_allowed: np.ndarray
    fit_mask: np.ndarray
    null_samples: np.ndarray

    def __post_init__(self):
        mask = readonly(self.fit_mask, bool)
        values = readonly(self.y_allowed, np.complex128)
        null = readonly(self.null_samples, np.complex128)
        if mask.shape != self.grid.shape or values.shape != mask.shape or not mask.any():
            raise ValueError('invalid pilot observation shapes')
        if np.any(mask & ~self.grid.pilot_mask) or np.any(values[~mask] != 0):
            raise ValueError('only allowed fitting pilots may be populated')
        if null.shape != (int(self.grid.null_mask.sum()),):
            raise ValueError('null observation count mismatch')
        object.__setattr__(self, 'y_allowed', values)
        object.__setattr__(self, 'fit_mask', mask)
        object.__setattr__(self, 'null_samples', null)


@dataclass(frozen=True, slots=True)
class ReceiverObservation:
    grid: Grid
    y: np.ndarray

    def __post_init__(self):
        y = readonly(self.y, np.complex128)
        if y.shape != self.grid.shape:
            raise ValueError('receiver grid shape mismatch')
        object.__setattr__(self, 'y', y)

    def pilots(self, fit_mask=None):
        mask = self.grid.pilot_mask if fit_mask is None else np.asarray(fit_mask, bool)
        if mask.shape != self.grid.shape or np.any(mask & ~self.grid.pilot_mask):
            raise ValueError('fit mask must select transmitted pilots')
        # Materialize the mask before ANY estimator statistic/normalizer operation.
        masked = np.zeros(self.grid.shape, np.complex128)
        masked[mask] = self.y[mask]
        return PilotObservation(self.grid, masked, mask, self.y[self.grid.null_mask])


def require_pilots(observation):
    if type(observation) is not PilotObservation:
        raise TypeError('practical estimator accepts PilotObservation only')


def pilot_scale(observation, floor=1e-12):
    require_pilots(observation)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('positive scale floor required')
    z = observation.y_allowed[observation.fit_mask]/observation.grid.pilot_grid[observation.fit_mask]
    return float(np.sqrt(max(np.mean(np.abs(z)**2), floor)))
