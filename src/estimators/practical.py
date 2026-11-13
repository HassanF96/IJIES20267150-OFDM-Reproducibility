"""No imports from generator, channel, oracle, labels or metrics namespaces."""
from dataclasses import dataclass
import numpy as np
from src.ofdm.information import require_pilots, readonly
from src.estimators.noise import estimate_variance
from src.utils.cost import count


def delay_basis(grid, support):
    if type(support) is not int or not 1 <= support <= grid.cp+1:
        raise ValueError('public delay basis must fit CP')
    b = np.exp(-2j*np.pi*np.arange(grid.nfft)[:, None]*np.arange(support)/grid.nfft)
    return np.repeat(b[:, None, :], grid.symbols, axis=1)


def design(observation, support):
    require_pilots(observation)
    basis = delay_basis(observation.grid, support)
    a = basis[observation.fit_mask]*observation.grid.pilot_grid[observation.fit_mask, None]
    return basis, a, observation.y_allowed[observation.fit_mask]


def masked_output(basis, coefficients, grid):
    out = basis @ coefficients
    out[~grid.active_mask] = 0
    if not np.isfinite(out).all():
        raise ArithmeticError('nonfinite estimate; do not drop frame')
    return out


def ls_interpolate(observation, ledger=None):
    """Per-symbol complex linear frequency interpolation, constant edge extension.

    Physical signed-bin order, not FFT storage order. No temporal smoothing.
    Each symbol must contain >=1 fitting pilot; 1 gives a constant estimate.
    """
    require_pilots(observation)
    grid = observation.grid
    out = np.zeros(grid.shape, np.complex128)
    signed = np.asarray(grid.active_signed)
    bins = grid.active_bins
    for ell in range(grid.symbols):
        m = observation.fit_mask[bins, ell]
        if not m.any():
            raise ValueError('LS interpolation requires fitting pilots in every symbol')
        z = observation.y_allowed[bins[m], ell]/grid.pilot_grid[bins[m], ell]
        out[bins, ell] = np.interp(signed, signed[m], z.real)+1j*np.interp(signed, signed[m], z.imag)
    count(ledger, 'classical_estimator_calls')
    count(ledger, 'full_fits' if np.array_equal(observation.fit_mask, grid.pilot_mask) else 'partial_fits')
    count(ledger, 'interpolated_symbols', grid.symbols)
    return out


def ridge_operator(a, regularization):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('strictly positive ridge regularization required')
    return np.linalg.solve(a.conj().T@a+regularization*np.eye(a.shape[1]), a.conj().T)


def regularized_delay(observation, support, regularization=1e-6, ledger=None):
    basis, a, y = design(observation, support)
    coefficient = ridge_operator(a, regularization)@y
    count(ledger, 'classical_estimator_calls')
    count(ledger, 'full_fits' if np.array_equal(observation.fit_mask, observation.grid.pilot_mask) else 'partial_fits')
    count(ledger, 'linear_solves')
    return masked_output(basis, coefficient, observation.grid)


@dataclass(frozen=True, slots=True)
class PilotStatistics:
    mean: np.ndarray
    covariance: np.ndarray
    support: int
    train_ids: tuple[str, ...]
    grid_signature: tuple
    provenance: str = 'allowed_train_pilot_observations_only'

    def __post_init__(self):
        mu = readonly(self.mean, np.complex128)
        cov = readonly(self.covariance, np.complex128)
        if mu.shape != (self.support,) or cov.shape != (self.support, self.support):
            raise ValueError('statistics dimensions mismatch')
        if not np.allclose(cov, cov.conj().T, atol=1e-12) or np.linalg.eigvalsh(cov).min() <= 0:
            raise ValueError('strictly positive Hermitian covariance required')
        if self.provenance != 'allowed_train_pilot_observations_only' or not self.train_ids:
            raise ValueError('practical train-pilot provenance required')
        object.__setattr__(self, 'mean', mu)
        object.__setattr__(self, 'covariance', cov)


def grid_signature(grid):
    return (grid.nfft, grid.symbols, grid.cp, grid.active_signed, 'frame_static')


def fit_pilot_statistics(observations, records, support, regularization=1e-6,
                         shrinkage=0.1, eigen_floor=1e-8, variance_floor=1e-10,
                         ledger=None):
    """Offline technical/train fit; no channel labels accepted or consulted."""
    if len(observations) != len(records) or len(records) < 2:
        raise ValueError('at least two paired train observations required')
    if not 0 <= shrinkage <= 1 or not np.isfinite(eigen_floor) or eigen_floor <= 0:
        raise ValueError('invalid covariance regularization')
    if any(r.split != 'train' for r in records):
        raise ValueError('statistics may only be fit on train realizations')
    ids = tuple(r.realization_id for r in records)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate train realization')
    sig = grid_signature(observations[0].grid)
    z, correction = [], np.zeros((support, support), np.complex128)
    for obs in observations:
        require_pilots(obs)
        if grid_signature(obs.grid) != sig:
            raise ValueError('incompatible grid statistics')
        _, a, y = design(obs, support)
        operator = ridge_operator(a, regularization)
        z.append(operator@y)
        correction += estimate_variance(obs, variance_floor)*(operator@operator.conj().T)
    z = np.asarray(z)
    mu = z.mean(axis=0)
    centered = z-mu
    cov = centered.T@centered.conj()/(len(z)-1)-correction/len(z)
    eig, u = np.linalg.eigh((cov+cov.conj().T)/2)
    cov = (u*np.maximum(eig, 0))@u.conj().T
    target = np.diag(np.diag(cov).real)
    cov = (1-shrinkage)*cov+shrinkage*target+eigen_floor*np.eye(support)
    count(ledger, 'offline_statistics_fits')
    count(ledger, 'offline_linear_solves', len(observations))
    count(ledger, 'offline_covariance_eigendecompositions')
    return PilotStatistics(mu, cov, support, ids, sig)


def lmmse_like(observation, statistics, variance_floor=1e-10, solve_floor=1e-8, ledger=None):
    require_pilots(observation)
    if type(statistics) is not PilotStatistics or grid_signature(observation.grid) != statistics.grid_signature:
        raise TypeError('compatible practical PilotStatistics required')
    if not np.isfinite(solve_floor) or solve_floor <= 0:
        raise ValueError('positive solve floor required')
    basis, a, y = design(observation, statistics.support)
    v = estimate_variance(observation, variance_floor, ledger)+solve_floor
    root = np.linalg.cholesky(statistics.covariance)
    b = a@root
    coefficient = statistics.mean + root@np.linalg.solve(
        b.conj().T@b+v*np.eye(statistics.support), b.conj().T@(y-a@statistics.mean))
    count(ledger, 'classical_estimator_calls')
    count(ledger, 'full_fits' if np.array_equal(observation.fit_mask, observation.grid.pilot_mask) else 'partial_fits')
    count(ledger, 'factorizations')
    count(ledger, 'linear_solves')
    return masked_output(basis, coefficient, observation.grid)
