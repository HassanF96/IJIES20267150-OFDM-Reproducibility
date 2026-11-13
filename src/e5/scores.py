"""Preregistered E5/S1 receiver scores; DEVELOPMENT, not final results.

Practical functions accept a restricted ReceiverObservation and candidate arrays,
never a simulator frame, transmitted payload, truth, loss, or condition label.
Larger score always predicts harm. No threshold is fitted in this module.
Diagnostic perturbations below operate only on declared candidates/decisions.
"""
import numpy as np
from scipy.special import logsumexp

from src.ofdm.discrepancy import radial_huber_mean
from src.ofdm.information import ReceiverObservation
from src.ofdm.modulation import bit_llrs, distances
from src.ofdm.receiver import mmse_equalize
from src.utils.cost import count


KAPPA = float(np.sqrt(np.log(20)))
PRIMARY_SCORES = ('g3_hard', 'g3_soft')
CONTROL_SCORES = ('g1', 'evm', 'evm_neural', 'llr', 'llr_neural', 'pilot')
DIAGNOSTIC_SCORES = ('g3_hard_mean', 'g3_soft_mean', 'g3_mean_only',
                     'pilot_mean', 'g3_anchor', 'weighted_evm_identity')
ALL_SCORES = PRIMARY_SCORES + CONTROL_SCORES + DIAGNOSTIC_SCORES


def _inputs(observation, base, neural, estimated_variance):
    if type(observation) is not ReceiverObservation:
        raise TypeError('ReceiverObservation required; labels/simulator forbidden')
    pair = tuple(np.asarray(c, np.complex128) for c in (base, neural))
    if any(c.shape != observation.grid.shape or not np.isfinite(c).all() for c in pair):
        raise ValueError('finite paired candidate grids required; no dropped frames')
    if not np.isfinite(estimated_variance) or estimated_variance <= 0:
        raise ValueError('positive finite receiver-estimated variance required')
    return pair


def _aggregate(values, aggregation, kappa, ledger=None):
    if aggregation == 'huber':
        count(ledger, 'robust_summaries')
        return radial_huber_mean(values, kappa)
    if aggregation == 'mean':
        values = np.asarray(values, float)
        if not values.size or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('finite nonnegative nonempty energies required')
        count(ledger, 'mean_summaries')
        return float(values.mean())
    raise ValueError('only preregistered huber/mean aggregators supported')


def data_components(y, h, constellation, estimated_variance, *, soft=False,
                    llrs=False, temperature=1.0, forced_hard=None, ledger=None):
    """Receive-domain detection equivalent to gain-consistent MMSE slicing.

    Diagnostic temperature changes only the receiver posterior. Temperature=inf
    is the explicit uniform limit. forced_hard is an explicit diagnostic symbol
    vector and is never accepted by score_pair, the main practical entry point.
    """
    y, h = np.broadcast_arrays(np.asarray(y, np.complex128), np.asarray(h, np.complex128))
    if not y.size:
        raise ValueError('nonempty data vector required')
    if not (temperature > 0) or np.isnan(temperature):
        raise ValueError('positive temperature or infinity required')
    z, gain, vz = mmse_equalize(y, h, estimated_variance, constellation.energy)
    d = distances(y, h, constellation)
    indices = np.argmin(d, axis=-1)
    hard = constellation.points[indices]
    if forced_hard is not None:
        hard = np.asarray(forced_hard, np.complex128)
        if hard.shape != y.shape or not np.isfinite(hard).all():
            raise ValueError('forced decisions must match finite data shape')
        membership = np.abs(hard[..., None]-constellation.points).min(axis=-1)
        if np.any(membership > 1e-12*np.sqrt(constellation.energy)):
            raise ValueError('forced decisions must be constellation points')
        indices = np.abs(hard[..., None]-constellation.points).argmin(axis=-1)
    result = {'symbols': hard, 'indices': indices,
              'bits': constellation.bits_from_indices(indices),
              'z': z, 'gain': gain, 'equalized_variance': vz,
              'hard_energy': np.abs(y-h*hard)**2/estimated_variance}
    count(ledger, 'equalizer_res', y.size)
    count(ledger, 'hard_slicer_res', y.size)
    count(ledger, 'candidate_detection_passes')
    if soft or llrs:
        if np.isinf(temperature):
            logp = np.full(d.shape, -np.log(len(constellation.points)))
        else:
            logits = -(d-d.min(axis=-1, keepdims=True))/(estimated_variance*temperature)
            logp = logits-logsumexp(logits, axis=-1, keepdims=True)
        p = np.exp(logp)
        result['probabilities'] = p
        count(ledger, 'symbol_likelihoods', y.size*len(constellation.points))
        if soft:
            mean = p @ constellation.points
            variance = p @ np.abs(constellation.points)**2 - np.abs(mean)**2
            if np.any(variance < -1e-10*constellation.energy):
                raise ArithmeticError('negative posterior variance beyond roundoff')
            variance = np.maximum(variance, 0)
            mean_only = np.abs(y-h*mean)**2/estimated_variance
            # Huber is applied to this expectation, not inside the posterior sum.
            result.update(posterior_mean=mean, posterior_variance=variance,
                          mean_only_energy=mean_only,
                          soft_energy=mean_only+np.abs(h)**2*variance/estimated_variance)
            count(ledger, 'posterior_moment_res', y.size)
        if llrs:
            llr_values = bit_llrs(logp, constellation)
            result['llrs'] = llr_values
            result['llr_confidence'] = float(np.mean(np.abs(np.tanh(llr_values/2))))
            count(ledger, 'bit_llrs', y.size*constellation.bits_per_symbol)
    return result


def _pilot_components(observation, base, neural, variance, mask, aggregation, kappa, ledger):
    mask = np.asarray(mask, bool)
    if mask.shape != observation.grid.shape or not mask.any() or np.any(mask & ~observation.grid.pilot_mask):
        raise ValueError('nonempty subset of transmitted pilots required')
    y, p = observation.y[mask], observation.grid.pilot_grid[mask]
    out = []
    for candidate in (base, neural):
        energies = np.abs(y-candidate[mask]*p)**2/variance
        out.append(_aggregate(energies, aggregation, kappa, ledger))
        count(ledger, 'pilot_reconstruction_res', int(mask.sum()))
    return out


def pilot_pair_score(observation, base, neural, estimated_variance, validation_mask,
                     *, aggregation='huber', kappa=KAPPA, ledger=None):
    """G2 score only. Caller must construct both partial candidates without heldouts.

    Refit and all-pilot labels are intentionally outside this score operation.
    Only this function accesses heldout observations after the partial fit.
    """
    base, neural = _inputs(observation, base, neural, estimated_variance)
    pair = _pilot_components(observation, base, neural, estimated_variance,
                             validation_mask, aggregation, kappa, ledger)
    count(ledger, 'gate_score_calls')
    return float(pair[1]-pair[0])


def score_pair(observation, base, neural, constellation, estimated_variance, *,
               h_floor=1e-6, kappa=KAPPA, names=None, ledger=None,
               return_details=False):
    """All registered scores, or a selected subset for honest cost measurement.

    Optional details contain already-executed base/neural detections for final
    selected-bit reuse. A single-neural control does not detect the base unless
    another requested score needs it. No unused LLR is charged to G3-soft.
    """
    return _score_pair(observation, base, neural, constellation, estimated_variance,
                       h_floor=h_floor, kappa=kappa, names=names, ledger=ledger,
                       return_details=return_details)


def diagnostic_score_pair(observation, base, neural, constellation, estimated_variance, *,
                          temperature=1.0, neural_hard_decisions=None,
                          h_floor=1e-6, kappa=KAPPA, names=None, ledger=None,
                          return_details=False):
    """Explicit challenge API, never the main scientific score entry point.

    Fixed channel candidates remain unchanged; controlled detected-symbol or
    posterior perturbations isolate the decision dependence of each score.
    """
    return _score_pair(observation, base, neural, constellation, estimated_variance,
                       h_floor=h_floor, kappa=kappa, names=names, ledger=ledger,
                       return_details=return_details, temperature=temperature,
                       neural_hard_decisions=neural_hard_decisions)


def _score_pair(observation, base, neural, constellation, estimated_variance, *,
                h_floor, kappa, names, ledger, return_details,
                temperature=1.0, neural_hard_decisions=None):
    base, neural = _inputs(observation, base, neural, estimated_variance)
    if not np.isfinite(h_floor) or h_floor <= 0:
        raise ValueError('positive finite EVM channel floor required')
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('positive finite Huber kappa required')
    names = tuple(ALL_SCORES if names is None else names)
    if not names or set(names)-set(ALL_SCORES):
        raise ValueError('unknown or empty score request')
    needed = set(names)
    out, details = {}, {}
    if 'g1' in needed:
        gamma = float(constellation.energy*np.mean(np.abs(base[observation.grid.active_mask])**2)/estimated_variance)
        out.update(g1=-float(np.log1p(gamma)), estimated_snr_linear=gamma)
    data_names = needed-{'g1', 'pilot', 'pilot_mean'}
    if data_names:
        mask = observation.grid.data_mask
        y = observation.y[mask]
        single_only = data_names <= {'evm_neural', 'llr_neural'}
        soft = bool(data_names & {'g3_soft', 'g3_soft_mean', 'g3_mean_only'})
        llrs = bool(data_names & {'llr', 'llr_neural'})
        need_evm = bool(data_names & {'evm', 'evm_neural', 'weighted_evm_identity'})
        summaries = {}
        for candidate_name, candidate in (('base', base), ('neural', neural)):
            if candidate_name == 'base' and single_only:
                continue
            h = candidate[mask]
            det = data_components(y, h, constellation, estimated_variance,
                                  soft=soft, llrs=llrs, ledger=ledger,
                                  temperature=temperature,
                                  forced_hard=neural_hard_decisions if candidate_name == 'neural' else None)
            details[candidate_name] = det
            summary = {}
            if data_names & {'g3_hard', 'g3_anchor', 'weighted_evm_identity'}:
                summary['hard'] = _aggregate(det['hard_energy'], 'huber', kappa, ledger)
            if 'g3_hard_mean' in data_names:
                summary['hard_mean'] = _aggregate(det['hard_energy'], 'mean', kappa, ledger)
            if 'g3_soft' in data_names:
                summary['soft'] = _aggregate(det['soft_energy'], 'huber', kappa, ledger)
            if 'g3_soft_mean' in data_names:
                summary['soft_mean'] = _aggregate(det['soft_energy'], 'mean', kappa, ledger)
            if 'g3_mean_only' in data_names:
                summary['mean_only'] = _aggregate(det['mean_only_energy'], 'huber', kappa, ledger)
            if need_evm:
                gain2 = np.abs(h)**2
                floor_mask = gain2 < h_floor*h_floor
                equalized = h.conj()*y/np.maximum(gain2, h_floor*h_floor)
                squared = np.abs(equalized-det['symbols'])**2
                summary['evm'] = float(np.mean(squared)/constellation.energy)
                out['evm_floor_fraction_'+candidate_name] = float(np.mean(floor_mask))
                if 'weighted_evm_identity' in data_names:
                    # Independent algebraic route away from the EVM floor.
                    weighted = gain2*squared/estimated_variance
                    weighted[floor_mask] = det['hard_energy'][floor_mask]
                    summary['weighted'] = _aggregate(weighted, 'huber', kappa, ledger)
            if llrs:
                summary['llr'] = det['llr_confidence']
            summaries[candidate_name] = summary
        for score, field in (('g3_hard', 'hard'), ('g3_soft', 'soft'),
                              ('g3_hard_mean', 'hard_mean'), ('g3_soft_mean', 'soft_mean'),
                              ('g3_mean_only', 'mean_only'), ('evm', 'evm'),
                              ('weighted_evm_identity', 'weighted')):
            if score in needed:
                out[score] = summaries['neural'][field]-summaries['base'][field]
        if 'llr' in needed:
            out['llr'] = summaries['base']['llr']-summaries['neural']['llr']
        if 'llr_neural' in needed:
            out['llr_neural'] = -summaries['neural']['llr']
        if 'evm_neural' in needed:
            out['evm_neural'] = summaries['neural']['evm']
    if needed & {'pilot', 'g3_anchor'}:
        dp = _pilot_components(observation, base, neural, estimated_variance,
                               observation.grid.pilot_mask, 'huber', kappa, ledger)
        pilot = dp[1]-dp[0]
        if 'pilot' in needed:
            out['pilot'] = pilot
        if 'g3_anchor' in needed:
            out['g3_anchor'] = summaries['neural']['hard']-summaries['base']['hard']+pilot
    if 'pilot_mean' in needed:
        dp = _pilot_components(observation, base, neural, estimated_variance,
                               observation.grid.pilot_mask, 'mean', kappa, ledger)
        out['pilot_mean'] = dp[1]-dp[0]
    if not all(np.isfinite(value) for value in out.values()):
        raise ArithmeticError('nonfinite score; implementation failure, not frame exclusion')
    count(ledger, 'gate_score_calls', len(names))
    return (out, details) if return_details else out


def transform_candidate(candidate, *, phase_degrees=0.0, amplitude=1.0):
    """Prespecified phase/scale challenges; no transmitted symbols required."""
    candidate = np.asarray(candidate, np.complex128)
    if not np.isfinite(candidate).all() or not np.isfinite(phase_degrees) or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('finite candidate/phase and positive amplitude required')
    return candidate*amplitude*np.exp(1j*np.deg2rad(phase_degrees))


def perturb_candidate(candidate, active_mask, relative_norm, seed, *, zero_mask=None):
    """Fixed-seed complex Gaussian direction with prescribed active-relative norm.

    Candidate norm uses the full active grid for both whole-grid and between-
    pilot perturbations. zero_mask can forbid any pilot perturbation exactly.
    """
    candidate = np.asarray(candidate, np.complex128)
    active = np.asarray(active_mask, bool)
    if candidate.shape != active.shape or not active.any() or not np.isfinite(candidate).all():
        raise ValueError('finite candidate and matching nonempty active mask required')
    if not np.isfinite(relative_norm) or relative_norm < 0:
        raise ValueError('nonnegative finite perturbation norm required')
    permitted = active.copy()
    if zero_mask is not None:
        zero = np.asarray(zero_mask, bool)
        if zero.shape != active.shape:
            raise ValueError('zero-mask shape mismatch')
        permitted &= ~zero
    if not permitted.any():
        raise ValueError('empty perturbation support')
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=int(permitted.sum()))+1j*rng.normal(size=int(permitted.sum()))
    norm = np.linalg.norm(direction)
    perturbation = np.zeros_like(candidate)
    perturbation[permitted] = direction*(relative_norm*np.linalg.norm(candidate[active])/norm)
    return candidate+perturbation


def corrupt_decisions(decisions, constellation, fraction, seed, *, mode='independent'):
    """Replace floor(fraction*n) symbols; wrong means relative to supplied decisions.

    Supply oracle symbols only in the separately labeled challenge harness when
    a guaranteed true symbol-error fraction is desired; main scores never do so.
    Coherent mode rotates the selected symbols by -90 degrees, allowing the
    registered +90-degree candidate/coherent-decision cancellation fixture.
    """
    decisions = np.asarray(decisions, np.complex128)
    if not decisions.size or not np.isfinite(decisions).all() or not 0 <= fraction <= 1:
        raise ValueError('finite nonempty symbols and replacement fraction in [0,1] required')
    if mode not in ('independent', 'coherent'):
        raise ValueError('independent or coherent corruption required')
    index = np.abs(decisions[..., None]-constellation.points).argmin(axis=-1)
    if np.any(np.abs(decisions-constellation.points[index]) > 1e-12*np.sqrt(constellation.energy)):
        raise ValueError('decisions must be constellation symbols')
    rng = np.random.default_rng(seed)
    selected = rng.choice(decisions.size, int(np.floor(fraction*decisions.size)), replace=False)
    result = decisions.copy().reshape(-1)
    if mode == 'independent':
        offsets = rng.integers(1, len(constellation.points), size=len(selected))
        result[selected] = constellation.points[(index.reshape(-1)[selected]+offsets) % len(constellation.points)]
    else:
        result[selected] *= -1j
    return result.reshape(decisions.shape)


def challenge_data_pair(observation, base, neural, constellation, estimated_variance,
                        *, temperature=1.0, neural_hard_decisions=None,
                        kappa=KAPPA, ledger=None):
    """Diagnostic override only; original candidate-pair normalizers remain fixed.

    Temper both candidate posteriors equally. Hard corruption affects only the
    neural branch, explicitly isolating changed decisions from changed channel.
    """
    base, neural = _inputs(observation, base, neural, estimated_variance)
    mask = observation.grid.data_mask
    summary = []
    for candidate, override in ((base, None), (neural, neural_hard_decisions)):
        d = data_components(observation.y[mask], candidate[mask], constellation,
                            estimated_variance, soft=True, llrs=True,
                            temperature=temperature, forced_hard=override, ledger=ledger)
        summary.append({name: _aggregate(d[key], 'huber', kappa, ledger)
                        for name, key in (('g3_hard', 'hard_energy'),
                                          ('g3_soft', 'soft_energy'),
                                          ('g3_mean_only', 'mean_only_energy'))})
    return {name: summary[1][name]-summary[0][name] for name in summary[0]}
