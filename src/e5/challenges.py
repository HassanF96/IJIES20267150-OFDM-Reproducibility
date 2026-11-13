"""Preregistered S1 score-failure diagnostics. No threshold fitting here.

Noisy cases use one fixed V-panel realization per predeclared cell and the first
training seed only; their uncertainty is frame-level and mechanism-descriptive.
Analytical fixtures explicitly construct truth, never practical estimator inputs.
"""
import csv
import hashlib
import time
from collections import defaultdict

import numpy as np

from src.e5.common import OUT, LABEL, write_json, sha
from src.e5.scores import (ALL_SCORES, score_pair, diagnostic_score_pair,
                           pilot_pair_score, transform_candidate,
                           perturb_candidate, corrupt_decisions)
from src.metrics.errors import frame_loss, harm_usefulness
from src.ofdm.grid import Grid
from src.ofdm.information import ReceiverObservation
from src.ofdm.modulation import Constellation


PHASES = (-90, 90, -30, -15, -5, 5, 15, 30)
SCALES = (.5, .75, 1.25, 2., 3.)
PERTURBATIONS = (.1, .5)
FRACTIONS = (0., .1, .5, 1.)
TEMPERATURES = (1., 2., 10., np.inf)


def challenge_specs():
    """Finite list fixed before outcomes; no result-dependent strata/selection."""
    result = [{'kind': 'phase', 'phase': p} for p in PHASES]
    result += [{'kind': 'amplitude', 'amplitude': a} for a in SCALES]
    result += [{'kind': 'phase_scale', 'phase': p, 'amplitude': a}
               for p in PHASES for a in SCALES]
    result += [{'kind': 'perturbation', 'relative_norm': n, 'data_only': d}
               for n in PERTURBATIONS for d in (False, True)]
    result += [{'kind': 'hard_corruption', 'fraction': f} for f in FRACTIONS]
    result += [{'kind': 'temperature', 'temperature': 'uniform' if np.isinf(t) else t}
               for t in TEMPERATURES]
    result += [{'kind': 'coherent_rotated_decisions', 'phase': 90, 'fraction': 1.}]
    for index, spec in enumerate(result):
        spec['challenge_id'] = index
    return result


def _seed(realization_id, challenge_id, master_seed=71993):
    # Declared deterministic receiver/diagnostic namespace, not simulator RNG.
    payload = f'E5-S1-challenge-v1/{master_seed}/{realization_id}/{challenge_id}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little')


def challenged_candidate(candidate, grid, spec, seed):
    if spec['kind'] == 'perturbation':
        return perturb_candidate(candidate, grid.active_mask, spec['relative_norm'], seed,
                                  zero_mask=grid.pilot_mask if spec['data_only'] else None)
    return transform_candidate(candidate, phase_degrees=spec.get('phase', 0),
                                amplitude=spec.get('amplitude', 1))


def _scalar_labels(base, neural, truth, mask):
    lb, ln = frame_loss(base, truth, mask), frame_loss(neural, truth, mask)
    harm, useful = harm_usefulness(lb, ln, .5, .5)
    return {'base_loss': lb, 'neural_loss': ln, 'harm': bool(harm), 'useful': bool(useful)}


def _accepted(score, threshold):
    return bool(threshold is not None and np.isfinite(score) and score <= threshold)


def evaluate_challenge_frame(observation, base, neural, partial_base, partial_neural,
                             variance, partial_variance, validation_mask,
                             constellation, truth, transmitted_symbols, metadata,
                             thresholds, *, g2_aggregation='huber'):
    """Truth joins only after practical scores/detections have been produced.

    The caller records already-fixed thresholds. Artificial candidates retain
    the same all-pilot baseline and practical variance. The selected G2 partial
    neural candidate receives the same declared defect operator, and its score
    is evaluated against unchanged partial base on the original heldout pilots.
    """
    if any(threshold is not None and not np.isfinite(threshold)
           for threshold in thresholds.values()):
        raise ValueError('challenge thresholds must be finite or None for fallback')
    start = time.perf_counter()
    initial, initial_details = score_pair(observation, base, neural, constellation,
                                          variance, return_details=True)
    initial_bundle_seconds = time.perf_counter()-start
    start = time.perf_counter()
    initial['g2'] = pilot_pair_score(observation, partial_base, partial_neural,
                                      partial_variance, validation_mask,
                                      aggregation=g2_aggregation)
    initial_g2_seconds = time.perf_counter()-start
    before = _scalar_labels(base, neural, truth, observation.grid.active_mask)
    original_decisions = initial_details['neural']['symbols']
    transmitted_symbols = np.asarray(transmitted_symbols)
    before_ser = float(np.mean(np.abs(original_decisions-transmitted_symbols) > 1e-12))
    methods = ALL_SCORES+('g2',)
    for spec in challenge_specs():
        seed = _seed(metadata['realization_id'], spec['challenge_id'], metadata.get('challenge_master_seed', 71993))
        candidate = challenged_candidate(neural, observation.grid, spec, seed)
        partial = challenged_candidate(partial_neural, observation.grid, spec, seed)
        override = None
        if spec['kind'] == 'hard_corruption':
            override = corrupt_decisions(original_decisions, constellation, spec['fraction'], seed)
        elif spec['kind'] == 'coherent_rotated_decisions':
            override = corrupt_decisions(original_decisions, constellation, 1., seed, mode='coherent')
        temperature = spec.get('temperature', 1.)
        if temperature == 'uniform':
            temperature = np.inf
        start = time.perf_counter()
        scores, details = diagnostic_score_pair(observation, base, candidate, constellation,
                                                 variance, return_details=True,
                                                 temperature=temperature,
                                                 neural_hard_decisions=override)
        bundle_seconds = time.perf_counter()-start
        start = time.perf_counter()
        scores['g2'] = pilot_pair_score(observation, partial_base, partial,
                                         partial_variance, validation_mask,
                                         aggregation=g2_aggregation)
        g2_seconds = time.perf_counter()-start
        after = _scalar_labels(base, candidate, truth, observation.grid.active_mask)
        decisions = details['neural']['symbols']
        row = {'label': LABEL, **metadata, **spec, 'perturbation_seed': seed,
               'threshold_scope': 'Primary-law thresholds, diagnostic application only; no challenge guarantee',
               'score_bundle_seconds_before': initial_bundle_seconds,
               'score_bundle_seconds_after': bundle_seconds,
               'g2_score_seconds_before': initial_g2_seconds,
               'g2_score_seconds_after': g2_seconds,
               'ser_before': before_ser,
               'ser_after': float(np.mean(np.abs(decisions-transmitted_symbols) > 1e-12)),
               'base_loss': before['base_loss'], 'loss_before': before['neural_loss'],
               'loss_after': after['neural_loss'], 'harm_before': before['harm'],
               'harm_after': after['harm'], 'useful_before': before['useful'],
               'useful_after': after['useful'],
               'partial_loss_before': frame_loss(partial_neural, truth, observation.grid.active_mask),
               'partial_loss_after': frame_loss(partial, truth, observation.grid.active_mask),
               'decision_replacement_convention': 'fraction of original detected symbols, actual true SER measured'}
        for method in methods:
            threshold = thresholds.get(method)
            # A registered None threshold is the prescribed reject-all fallback,
            # not an available operating threshold for harm discrimination.
            # Preserve its actual rejection behavior separately from eligibility.
            registered = method in thresholds
            available = registered and threshold is not None
            before_accept = _accepted(initial[method], threshold) if registered else None
            after_accept = _accepted(scores[method], threshold) if registered else None
            row.update({method+'_before': initial[method], method+'_after': scores[method],
                        method+'_change': scores[method]-initial[method],
                        method+'_threshold': threshold,
                        method+'_threshold_available': available,
                        method+'_accept_before': before_accept,
                        method+'_accept_after': after_accept,
                        method+'_accept_flip': before_accept != after_accept if registered else None})
        yield row


def _analytic_frames():
    grid = Grid(128, 8, 16, tuple(range(-48, 0))+tuple(range(1, 49)), 8)
    for modulation in ('QPSK', '16QAM'):
        constellation = Constellation(modulation)
        symbols = np.resize(constellation.points, int(grid.data_mask.sum()))
        channel = np.zeros(grid.shape, np.complex128)
        channel[grid.active_mask] = 1+.2j
        obs = ReceiverObservation(grid, channel*grid.map_data(symbols))
        yield constellation, obs, channel, symbols


def summarize_challenges(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row['source'], row['modulation'], row['kind'], row['challenge_id'])
        groups[key].append(row)
    summary = []
    for (source, modulation, kind, challenge_id), values in sorted(groups.items()):
        for method in ALL_SCORES+('g2',):
            n = len(values)
            harmful = [v for v in values if v['harm_after']]
            eligible = [v for v in values if v[method+'_threshold_available']]
            harmful_eligible = [v for v in harmful if v[method+'_threshold_available']]
            summary.append({'label': LABEL, 'source': source, 'modulation': modulation,
                            'kind': kind, 'challenge_id': challenge_id, 'method': method,
                            'frames': n, 'harm_count': len(harmful),
                            'harm_prevalence': len(harmful)/n,
                            'mean_score_change': float(np.mean([v[method+'_change'] for v in values])),
                            'numerically_unchanged_fraction': float(np.mean([abs(v[method+'_change']) <= 1e-11 for v in values])),
                            'harm_detection_fraction': (float(np.mean([not v[method+'_accept_after'] for v in harmful_eligible])) if harmful_eligible else None),
                            'harm_acceptance_fraction': (float(np.mean([v[method+'_accept_after'] for v in harmful_eligible])) if harmful_eligible else None),
                            'acceptance_flip_fraction': (float(np.mean([v[method+'_accept_flip'] for v in eligible])) if eligible else None),
                            'mean_ser_before': float(np.mean([v['ser_before'] for v in values])),
                            'mean_ser_after': float(np.mean([v['ser_after'] for v in values])),
                            'scope': 'Descriptive mechanisms; fixed primary thresholds may be inapplicable; no success inference'})
    return summary


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_challenges(guard, settings, statistics, selection, certificates):
    """Execute only after the root verifies the committed S1 manifest.

    Returns an artifact manifest. First record per panel cell is chosen by
    condition metadata and committed manifest ordering, never by score/outcome.
    """
    from src.e5.data import simulate_e5
    from src.e5.model import CandidateOperator
    from src.e5.training import load_model, mask_for
    model_seeds = settings['seeds'].get('training', settings['seeds'].get('train_seeds', [5101]))
    seed = int(model_seeds[0])
    operator = CandidateOperator(statistics, load_model(seed, 'cpu'))
    g2 = selection['g2']
    thresholds = {method: by_seed[str(seed)]['threshold'] for method, by_seed in certificates.items()
                  if str(seed) in by_seed}
    chosen = {}
    for record in guard.records('V-panel'):
        condition = guard.by_id[record.realization_id]
        key = tuple(condition[k] for k in ('family', 'modulation', 'snr_db', 'burst_cell'))
        chosen.setdefault(key, record)
    if len(chosen) != 112:
        raise ValueError(f'Expected 112 predeclared panel cells, got {len(chosen)}')
    rows = []
    start = time.perf_counter()
    for i, record in enumerate(chosen.values()):
        config, obs, labels, _ = simulate_e5(guard, record)
        build_start = time.perf_counter()
        base, neural, variance = operator.pair(obs.pilots())
        fit, heldout = mask_for(obs.grid, record, g2['fraction'], g2['policy'])
        partial_base, partial_neural, partial_variance = operator.pair(obs.pilots(fit))
        build_seconds = time.perf_counter()-build_start
        condition = guard.by_id[record.realization_id]
        metadata = {'source': 'noisy_development', 'realization_id': record.realization_id,
                    'training_seed': seed, 'modulation': config.constellation.name,
                    'challenge_master_seed': settings['seeds']['challenge'],
                    'family': condition['family'], 'snr_db': float(condition['snr_db']),
                    'burst_cell': int(condition['burst_cell']), 'burst_present': bool(labels.burst_schedule),
                    'candidate_construction_seconds': build_seconds}
        rows.extend(evaluate_challenge_frame(obs, base, neural, partial_base, partial_neural,
                                              variance, partial_variance, heldout, config.constellation,
                                              labels.h_true, labels.transmitted_grid[obs.grid.data_mask],
                                              metadata, thresholds, g2_aggregation=g2['aggregation']))
        if (i+1) % 16 == 0:
            print('S1 diagnostic challenge cells', i+1, '/', len(chosen), flush=True)
    for constellation, obs, truth, symbols in _analytic_frames():
        # Exact ideal candidate is an oracle analytical fixture only. No model
        # training or performance inference is made from these synthetic cases.
        rng = np.random.default_rng(5101)
        from src.ofdm.holdout import pilot_holdout
        _, heldout = pilot_holdout(obs.grid, g2['fraction'], g2['policy'], rng)
        metadata = {'source': 'analytic_noiseless_oracle_fixture',
                    'realization_id': 'analytic_'+constellation.name,
                    'training_seed': seed, 'modulation': constellation.name,
                    'challenge_master_seed': settings['seeds']['challenge'],
                    'family': 'deterministic_nonzero_flat', 'snr_db': 'noiseless',
                    'burst_cell': 0, 'burst_present': False,
                    'candidate_construction_seconds': 0.}
        rows.extend(evaluate_challenge_frame(obs, truth, truth, truth, truth, .1, .1,
                                              heldout, constellation, truth, symbols, metadata,
                                              thresholds, g2_aggregation=g2['aggregation']))
    raw_path = OUT/'raw/challenges.csv'
    summary_path = OUT/'summary/challenges.csv'
    _write_csv(raw_path, rows)
    _write_csv(summary_path, summarize_challenges(rows))
    manifest = {'label': LABEL, 'training_seed': seed, 'challenge_master_seed': settings['seeds']['challenge'], 'noisy_frames': len(chosen),
                'analytic_frames': 2, 'challenges_per_frame': len(challenge_specs()),
                'rows': len(rows), 'seconds': time.perf_counter()-start,
                'raw_sha256': sha(raw_path), 'summary_sha256': sha(summary_path),
                'frame_ids': [r.realization_id for r in chosen.values()],
                'g2': g2, 'challenge_specs': challenge_specs(),
                'uncertainty': 'First training seed only; descriptive one-frame-per-cell mechanism panel, not continuation evidence',
                'threshold_scope': 'Frozen primary V-cal-dev thresholds reused without tuning; no guarantee for artificial candidates or shifted/panel laws',
                'cost_scope': 'Per-case fused all-score bundle and isolated G2 score timings; full gate deployment costs are measured in the separate benchmark'}
    write_json(OUT/'manifests/challenges.json', manifest)
    return manifest
