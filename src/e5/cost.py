"""Actual S1 development receiver cost, including selected deployment work.

Practical receiver entry points accept observations, public configuration, a
frozen candidate operator and receiver randomness only. Truth stays outside.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import psutil
import torch

from src.e5.common import OUT, LABEL, sha, write_json
from src.e5.model import CandidateOperator
from src.e5.scores import score_pair, pilot_pair_score, data_components
from src.estimators.noise import estimate_variance
from src.ofdm.holdout import pilot_holdout
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import demodulate
from src.utils.cost import CostLedger, count


@dataclass
class Deployment:
    candidate: np.ndarray
    accepted: bool
    score: float | None
    variance: float
    detections: dict


def _component(ledger, name, callback):
    if ledger is None:
        return callback()
    with ledger.time(name):
        return callback()


def validate_and_select(method, observation, constellation, operator, threshold,
                        g2_policy, holdout_seed, *, ledger=None):
    """Common resident FFT grid -> selected full estimate and decision (Cval)."""
    if type(observation) is not ReceiverObservation:
        raise TypeError('restricted ReceiverObservation required')
    pilots = observation.pilots()
    count(ledger, 'transmitted_pilot_res', int(observation.grid.pilot_mask.sum()))
    count(ledger, 'transmitted_pilot_energy', float(np.sum(np.abs(observation.grid.pilot_grid[observation.grid.pilot_mask])**2)))
    details = {}
    score = None
    if method == 'g2':
        fit, validation = pilot_holdout(observation.grid, float(g2_policy['fraction']),
                                        g2_policy['policy'], np.random.default_rng(holdout_seed))
        count(ledger, 'fitting_pilot_res', int(fit.sum()))
        count(ledger, 'heldout_pilot_res', int(validation.sum()))
        pb, pn, pv = _component(ledger, 'partial_candidate_seconds', lambda:
                                operator.pair(observation.pilots(fit), ledger))
        score = _component(ledger, 'heldout_score_seconds', lambda:
                            pilot_pair_score(observation, pb, pn, pv, validation,
                                             aggregation=g2_policy['aggregation'], ledger=ledger))
        accepted = bool(threshold is not None and score <= threshold)
        # The heldout decision is fixed before the all-pilot operator executes.
        base = _component(ledger, 'full_base_seconds', lambda: operator.base(pilots, ledger))
        variance = estimate_variance(pilots, operator.variance_floor, ledger)
        candidate = _component(ledger, 'full_neural_seconds', lambda:
                                operator.neural(pilots, base, variance, ledger)) if accepted else base
        count(ledger, 'validation_passes')
        count(ledger, 'refit_passes')
    else:
        count(ledger, 'fitting_pilot_res', int(observation.grid.pilot_mask.sum()))
        base = _component(ledger, 'full_base_seconds', lambda: operator.base(pilots, ledger))
        variance = estimate_variance(pilots, operator.variance_floor, ledger)
        if method == 'baseline':
            accepted, candidate = False, base
        elif method == 'g1':
            score = _component(ledger, 'score_seconds', lambda:
                score_pair(observation, base, base, constellation, variance,
                            names=('g1',), ledger=ledger))['g1']
            accepted = bool(threshold is not None and score <= threshold)
            candidate = _component(ledger, 'full_neural_seconds', lambda:
                                    operator.neural(pilots, base, variance, ledger)) if accepted else base
        else:
            neural = _component(ledger, 'full_neural_seconds', lambda:
                                  operator.neural(pilots, base, variance, ledger))
            if method == 'g0':
                accepted, candidate = True, neural
            else:
                scores, details = _component(ledger, 'score_seconds', lambda:
                    score_pair(observation, base, neural, constellation, variance,
                               names=(method,), return_details=True, ledger=ledger))
                score = scores[method]
                accepted = bool(threshold is not None and score <= threshold)
                candidate = neural if accepted else base
            count(ledger, 'validation_passes', int(method != 'g0'))
    count(ledger, 'gate_decisions', int(method not in ('baseline', 'g0')))
    return Deployment(candidate, accepted, score, variance, details)


def finish_bits(observation, constellation, deployment, *, ledger=None):
    branch = 'neural' if deployment.accepted else 'base'
    if branch in deployment.detections:
        count(ledger, 'selected_detection_reuse')
        return deployment.detections[branch]['bits']
    mask = observation.grid.data_mask
    return _component(ledger, 'final_detection_seconds', lambda:
        data_components(observation.y[mask], deployment.candidate[mask], constellation,
                         deployment.variance, ledger=ledger))['bits']


def receiver(method, raw, grid, constellation, operator, threshold, g2_policy,
             holdout_seed, *, ledger=None):
    """Resident raw samples -> hard bits; buffering/simulation/I/O excluded."""
    observation = _component(ledger, 'cp_fft_seconds', lambda:
        ReceiverObservation(grid, demodulate(raw, grid.nfft, grid.symbols, grid.cp)))
    count(ledger, 'cp_fft_passes')
    deployed = validate_and_select(method, observation, constellation, operator, threshold,
                                  g2_policy, holdout_seed, ledger=ledger)
    return finish_bits(observation, constellation, deployed, ledger=ledger), deployed


def branch_replay(raw, grid, constellation, operator, accepted, *, ledger=None):
    """Timing counterfactual: externally fixed chosen branch, no validation."""
    observation = ReceiverObservation(grid, demodulate(raw, grid.nfft, grid.symbols, grid.cp))
    count(ledger, 'cp_fft_passes')
    pilots = observation.pilots()
    base = operator.base(pilots, ledger)
    variance = estimate_variance(pilots, operator.variance_floor, ledger)
    candidate = operator.neural(pilots, base, variance, ledger) if accepted else base
    deployed = Deployment(candidate, bool(accepted), None, variance, {})
    return finish_bits(observation, constellation, deployed, ledger=ledger)


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def timed(device, callback):
    synchronize(device)
    start = perf_counter()
    result = callback()
    synchronize(device)
    return perf_counter()-start, result


def measure_resolution(operator, observation, constellation, raw, *, repeats=30, warmups=10):
    """Preassessment technical baseline jitter; no labels/scores/thresholds."""
    action = lambda: branch_replay(raw, observation.grid, constellation, operator, False)
    for _ in range(warmups):
        action()
    samples = [timed(operator.device, action)[0] for _ in range(repeats)]
    return {'label': LABEL, 'device': operator.device, 'repeats': repeats,
            'warmups': warmups, 'raw_seconds': samples,
            'receiver_allowance_seconds': float(np.quantile(np.abs(np.diff(samples)), .95)),
            'rule': '95th percentile absolute consecutive baseline-only replay difference; fixed before V-assess'}


def _summary(values):
    a = np.asarray(values, float)
    return {'n': len(a), 'mean_seconds': float(a.mean()), 'median_seconds': float(np.median(a)),
            'p95_seconds': float(np.quantile(a, .95))} if len(a) else {'n': 0}


def benchmark(operator, observations, raw_frames, constellations, holdout_seeds,
              frame_ids, thresholds, g2_policy, *, warmups=10, repetitions=3,
              order_seed=71992, methods=None):
    """Interleaved paired batch-one boundaries; instrumentation outside timings."""
    methods = tuple(methods or ('baseline', 'g0', 'g1', 'g2', 'g3_hard', 'g3_soft',
                               'evm', 'evm_neural', 'llr', 'llr_neural', 'pilot'))
    n = len(observations)
    if not n or any(len(x) != n for x in (raw_frames, constellations, holdout_seeds, frame_ids)):
        raise ValueError('paired nonempty resident timing frames required')
    device = operator.device
    rss = [psutil.Process().memory_info().rss]
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(order_seed)
    records, ledgers = [], []
    # Warm complete paths including actual selected-branch behavior.
    for j in range(warmups):
        i = j % n
        for method in rng.permutation(methods):
            receiver(method, raw_frames[i], observations[i].grid, constellations[i], operator,
                     thresholds.get(method), g2_policy, holdout_seeds[i])
    for repeat in range(repetitions):
        for i in range(n):
            obs, raw, alphabet, holdout = observations[i], raw_frames[i], constellations[i], holdout_seeds[i]
            for method in rng.permutation(methods):
                threshold = thresholds.get(method)
                val_time, deployed = timed(device, lambda: validate_and_select(
                    method, obs, alphabet, operator, threshold, g2_policy, holdout))
                rx_time, (bits, rx_deployed) = timed(device, lambda: receiver(
                    method, raw, obs.grid, alphabet, operator, threshold, g2_policy, holdout))
                replay_time, replay_bits = timed(device, lambda: branch_replay(
                    raw, obs.grid, alphabet, operator, rx_deployed.accepted))
                if deployed.accepted != rx_deployed.accepted or not np.array_equal(bits, replay_bits):
                    raise AssertionError('timing paths disagree in chosen branch/output')
                records.append({'frame_id': frame_ids[i], 'frame_index': i, 'repeat': repeat,
                                'method': str(method), 'accepted': rx_deployed.accepted,
                                'validation_seconds': val_time, 'receiver_seconds': rx_time,
                                'replay_seconds': replay_time, 'increment_seconds': rx_time-replay_time})
            rss.append(psutil.Process().memory_info().rss)
    # Counters and component timers perturb performance: separate retained replay.
    for i in range(n):
        for method in methods:
            ledger = CostLedger()
            bits, deployed = receiver(method, raw_frames[i], observations[i].grid, constellations[i],
                operator, thresholds.get(method), g2_policy, holdout_seeds[i], ledger=ledger)
            ledgers.append({'frame_id': frame_ids[i], 'method': method, 'accepted': deployed.accepted,
                            **ledger.export()})
    summaries, vectors = {}, {'validation': {}, 'receiver': {}, 'increment': {}}
    for method in methods:
        rows = [r for r in records if r['method'] == method]
        summaries[method] = {'coverage': float(np.mean([r['accepted'] for r in rows])),
                             'conditional': {}}
        for boundary in ('validation', 'receiver', 'replay', 'increment'):
            summaries[method][boundary] = _summary([r[boundary+'_seconds'] for r in rows])
            if boundary in vectors:
                vectors[boundary][method] = [float(np.mean([r[boundary+'_seconds'] for r in rows
                    if r['frame_index'] == i])) for i in range(n)]
        for branch in (False, True):
            subset = [r for r in rows if r['accepted'] == branch]
            summaries[method]['conditional']['accepted' if branch else 'rejected'] = {
                boundary: _summary([r[boundary+'_seconds'] for r in subset])
                for boundary in ('validation', 'receiver', 'increment')}
    return {'label': LABEL, 'device': device, 'batch_size': 1, 'warmups': warmups,
            'repetitions': repetitions, 'frame_ids': frame_ids, 'order_seed': order_seed,
            'raw': records, 'ledgers': ledgers, 'summary': summaries, 'frame_means': vectors,
            'ram_observed_peak_bytes': max(rss), 'ram_sampling': 'RSS after every completed frame/method block',
            'process_lifetime_peak_working_set_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', None),
            'vram_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
            'vram_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
            'dtype': 'CPU complex128/float64; neural float32', 'cpu_torch_threads': torch.get_num_threads(),
            'model_parameter_bytes': sum(p.numel()*p.element_size() for p in operator.model.parameters()),
            'statistics_resident_bytes': operator.statistics.mean.nbytes+operator.statistics.covariance.nbytes,
            'boundary_order_and_cache': 'Per method: validation, complete receiver, fixed-branch replay. Method order is randomized per frame/repetition. Observations/models resident; no cache flush.',
            'timing_scope': 'Sequential resident-frame NumPy receiver plus device neural pass with necessary transfers. Synchronization brackets both CUDA boundaries. No observation trimming. Truth, simulation and logging excluded. Negative incremental differences retained.'}


def run_cost(guard, settings, statistics, selection, certificates):
    """Benchmark the first predeclared primary assessment frames on CPU/CUDA.

    Return path/schema is shared with reporting.py. Model seed is fixed to the
    first manifest seed, never selected by timing or scientific outcomes.
    """
    from src.e5.data import simulate_e5
    from src.e5.training import load_model, load_stats
    seed = int(settings['seeds']['training'][0])
    timing = settings.get('timing', {})
    count_frames = int(timing.get('frames', 64))
    warmups = int(timing.get('warmups', 10))
    repetitions = int(timing.get('repetitions', 3))
    order_seed = int(settings['seeds'].get('timing', 71992))
    records = guard.records('V-assess')[:count_frames]
    observations, raws, constellations, holdouts = [], [], [], []
    for record in records:
        config, obs, _labels, raw = simulate_e5(guard, record)
        # No truth object is passed to any practical timing function.
        observations.append(obs)
        raws.append(raw)
        constellations.append(config.constellation)
        holdouts.append(record.seeds()['holdout'])
    thresholds = {method: by_seed[str(seed)]['threshold'] for method, by_seed in certificates.items()}
    methods = ('baseline', 'g0', 'g1', 'g2', 'g3_hard', 'g3_soft',
               'evm', 'evm_neural', 'llr', 'llr_neural', 'pilot')
    folder = OUT / 'timing'
    folder.mkdir(parents=True, exist_ok=True)
    devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    results = {}
    for device in devices:
        def cold_load():
            model = load_model(seed, device)
            stats = load_stats(OUT / 'checkpoints/pilot_statistics.npz', observations[0].grid)
            return CandidateOperator(stats, model, device)
        setup, operator = timed(device, cold_load)
        result = benchmark(operator, observations, raws, constellations, holdouts,
                            [r.realization_id for r in records], thresholds, selection['g2'],
                            warmups=warmups, repetitions=repetitions, order_seed=order_seed,
                            methods=methods)
        result.update(training_seed=seed, checkpoint_sha256=sha(OUT / 'checkpoints' / f'seed_{seed}_best.pt'),
                      statistics_sha256=sha(OUT / 'checkpoints/pilot_statistics.npz'),
                      cold_checkpoint_statistics_load_seconds=setup,
                      cold_setup_scope='First measured disk deserialization/model construction/device transfer in this process; OS file cache not flushed.',
                      amortization_horizon=count_frames*repetitions)
        for method in methods:
            result['summary'][method]['amortized_setup_plus_validation_seconds'] = (
                result['summary'][method]['validation']['mean_seconds']+setup/(count_frames*repetitions))
        device_name = device.replace(':', '')
        vectors = {'frame_ids': np.asarray(result['frame_ids'])}
        for kind in ('validation', 'receiver', 'increment'):
            prefix = {'validation': 'total_times', 'receiver': 'receiver_times', 'increment': 'increment_times'}[kind]
            vectors.update({prefix+'__'+method: np.asarray(values)
                            for method, values in result['frame_means'][kind].items()})
        np.savez_compressed(folder / f'timing_{device_name}.npz', **vectors)
        write_json(folder / f'benchmark_{device_name}.json', result)
        results[device] = result
        del operator
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    primary = timing.get('decision_device', 'cpu')
    if primary not in results:
        raise ValueError('prespecified timing decision device unavailable; no post hoc substitution')
    result = {'label': LABEL, 'primary_device': primary, 'training_seed': seed,
              'devices': results, 'total_times': results[primary]['frame_means']['validation'],
              'receiver_times': results[primary]['frame_means']['receiver'],
              'increment_times': results[primary]['frame_means']['increment'],
              'frame_ids': results[primary]['frame_ids'],
              'comparison_scope': 'Fixed seed and certified operating points; no matched-performance claim from timing alone.'}
    write_json(folder / 'benchmark.json', result)
    return result
