"""Offline/evaluation only, active-grid loss with no hidden energy floor."""
import numpy as np


def frame_loss(estimate, truth, active_mask):
    estimate, truth = np.asarray(estimate, np.complex128), np.asarray(truth, np.complex128)
    mask = np.asarray(active_mask, bool)
    if estimate.shape != truth.shape or estimate.shape != mask.shape or not mask.any():
        raise ValueError('matching nonempty grid/mask required')
    if not np.isfinite(estimate).all() or not np.isfinite(truth).all():
        raise ValueError('nonfinite channel')
    # Common scale prevents underflow/overflow without altering NMSE denominator.
    scale = max(float(np.max(np.abs(truth[mask]))), float(np.max(np.abs(estimate[mask]))))
    if scale == 0 or not np.any(truth[mask] != 0):
        raise ValueError('UNDEFINED_NMSE_ZERO_TRUE_ENERGY')
    qt = float(np.sum(np.abs(truth[mask]/scale)**2))
    qe = float(np.sum(np.abs((estimate[mask]/scale)-(truth[mask]/scale))**2))
    if qt == 0:
        raise ArithmeticError('true energy underflow; no denominator floor permitted')
    return qe/qt


def aggregate_nmse(frame_losses):
    values = np.asarray(frame_losses, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('nonempty finite nonnegative frame losses required')
    return float(values.mean())  # equal-weight mean of frame ratios, NOT pooled sums


def power_db(value, display_floor=None):
    a = np.asarray(value, dtype=float)
    if np.any(a < 0) or not np.isfinite(a).all():
        raise ValueError('finite nonnegative powers required')
    if display_floor is not None:
        if not np.isfinite(display_floor) or display_floor <= 0:
            raise ValueError('positive display floor required')
        a = np.maximum(a, display_floor)
    with np.errstate(divide='ignore'):
        return 10*np.log10(a)


def bit_errors(detected, transmitted):
    a, b = np.asarray(detected), np.asarray(transmitted)
    if a.shape != b.shape or not a.size or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('matching nonempty binary arrays required')
    errors = int(np.count_nonzero(a != b))
    return {'bit_errors': errors, 'bits': int(a.size), 'ber': errors/a.size}


def harm_usefulness(base_loss, candidate_loss, delta_db=0.5, useful_delta_db=0.5):
    values = [base_loss, candidate_loss, delta_db, useful_delta_db]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('finite nonnegative losses/tolerances required')
    return (candidate_loss > 10**(delta_db/10)*base_loss,
            base_loss > 10**(useful_delta_db/10)*candidate_loss)
