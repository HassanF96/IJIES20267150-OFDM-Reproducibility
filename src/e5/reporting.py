"""Rebuild S1 development summaries and plots from guarded scalar raw outputs.

This module never generates observations, trains a residual model, accesses final
data, selects a threshold from assessment, or emits manuscript performance text.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.e5.common import ROOT, OUT, LABEL, sha, write_json
from src.e5.data import BoundaryGuard
from src.e5.analysis import (
    ConfidenceDiagnostic, acceptance, brier_losses, bootstrap_summary,
    discrimination, gate_metrics, labels_from_losses, mismatch_bootstrap,
    partial_aurc, partial_full_mismatch, primary_bound_table, rank_acceptance,
    rank_curve, s1_decision, timing_bound_table, weighted_discrimination,
    weighted_rank_points,
)

METHODS = ('g1', 'g2', 'g3_hard', 'g3_soft', 'evm', 'evm_neural', 'llr', 'llr_neural', 'pilot')
ORDINARY = ('g1', 'evm', 'evm_neural', 'llr', 'llr_neural', 'pilot')
DIAGNOSTICS = ('g3_hard_mean', 'g3_soft_mean', 'g3_mean_only', 'pilot_mean',
               'g3_anchor', 'weighted_evm_identity')
FIGURE_LABEL = 'S1 DEVELOPMENT — NOT FINAL'


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path, data):
    write_json(path, _clean(data))


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('empty report table')
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            clean = _clean(row)
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in clean.items()})


def load_partition(partition, training_seeds, selection, guard):
    """Fixed filename/ID allowlist; exact pairing across all fitted seeds."""
    if partition not in ('V-design', 'V-cal-dev', 'V-assess', 'V-panel'):
        raise PermissionError('S1 reporting allows validation partitions only')
    expected = [r.realization_id for r in guard.records(partition)]
    records, hashes = [], {}
    for seed in training_seeds:
        path = OUT/'raw'/f'seed_{seed}_{partition}.npz'
        if not path.resolve().is_relative_to(OUT.resolve()):
            raise PermissionError('S1 output boundary')
        with np.load(path, allow_pickle=False) as data:
            record = {key: data[key].copy() for key in data.files}
        if record['frame_id'].tolist() != expected:
            raise PermissionError(f'{partition}: raw frame IDs/order do not match committed manifest')
        for identifier in record['frame_id']:
            item = guard.allowed.get(str(identifier))
            if item is None or item.split != 'validation' or item.partition != partition:
                raise PermissionError('forbidden or mislabeled raw frame')
        if 'g2' not in record:
            record['g2'] = record[selection['g2']['name']].copy()
        e, u = labels_from_losses(record['base_loss'], record['neural_loss'])
        if not np.array_equal(e, record['harm']) or not np.array_equal(u, record['useful']):
            raise ArithmeticError('stored oracle labels fail independent strict-loss recomputation')
        for method in METHODS:
            if record[method].shape != (len(expected),) or not np.isfinite(record[method]).all():
                raise ArithmeticError('nonfinite/unpaired practical score; analysis aborted')
        if records:
            for key in ('frame_id', 'tie_key', 'family', 'modulation', 'snr_db', 'burst_cell', 'burst_present'):
                if not np.array_equal(records[0][key], record[key]):
                    raise ArithmeticError(f'seed pairing mismatch: {key}')
        records.append(record)
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return records, hashes


def _stack(records, name):
    return np.stack([record[name] for record in records])


def _primary_mask(record):
    return ((record['family'] == 'generic_A') & (record['modulation'] == 'QPSK')
            & (record['snr_db'] >= 0) & (record['snr_db'] <= 20))


def _classifier_metadata(model):
    steps = model.pipeline.named_steps
    return {'feature_names': model.feature_names, 'design_n': model.design_n,
            'imputation_medians': steps['simpleimputer'].statistics_,
            'standardization_mean': steps['standardscaler'].mean_,
            'standardization_scale': steps['standardscaler'].scale_,
            'coefficients': steps['logisticregression'].coef_,
            'intercept': steps['logisticregression'].intercept_,
            'iterations': steps['logisticregression'].n_iter_, 'C': 1.0}


def confidence_diagnostics(design, assess, training_seeds):
    metadata, arrays = {}, {}
    differences = {v: [] for v in ('g3_hard', 'g3_soft')}
    for s, seed in enumerate(training_seeds):
        mask = _primary_mask(design[s])
        x = np.column_stack([design[s][key][mask] for key in ORDINARY])
        xa = np.column_stack([assess[s][key] for key in ORDINARY])
        y, ya = design[s]['harm'][mask], assess[s]['harm']
        try:
            base = ConfidenceDiagnostic.fit(x, y, ORDINARY, seed=seed)
            probability = base.predict(xa)
            base_loss = brier_losses(ya, probability)
            metadata[f'seed{seed}/ordinary'] = {'status': 'FIT_DESIGN_ONLY', **_classifier_metadata(base)}
            arrays[f'seed{seed}_ordinary_probability'] = probability
            arrays[f'seed{seed}_ordinary_brier'] = base_loss
            for v in differences:
                plus = ConfidenceDiagnostic.fit(np.column_stack([x, design[s][v][mask]]), y,
                                                 [*ORDINARY, v], seed=seed)
                pplus = plus.predict(np.column_stack([xa, assess[s][v]]))
                losses = brier_losses(ya, pplus)
                differences[v].append(losses-base_loss)
                metadata[f'seed{seed}/{v}'] = {'status': 'FIT_DESIGN_ONLY', **_classifier_metadata(plus)}
                arrays[f'seed{seed}_{v}_probability'] = pplus
                arrays[f'seed{seed}_{v}_brier_difference'] = losses-base_loss
        except ValueError as exc:
            if str(exc) != 'INSUFFICIENT_DESIGN_CLASSES':
                raise
            metadata[f'seed{seed}'] = {'status': 'INSUFFICIENT_DESIGN_CLASSES', 'design_n': int(mask.sum())}
            for v in differences:
                differences[v].append(np.full(len(ya), np.nan))
    np.savez_compressed(OUT/'analysis/confidence_diagnostic_predictions.npz', **arrays)
    _json(OUT/'analysis/confidence_diagnostic_models.json', {'label': LABEL,
          'role': 'offline diagnostic only; not G4 or deployed gate',
          'design_subset': 'generic_A QPSK nominal SNR0-through20', 'models': metadata})
    return {key: np.asarray(value) for key, value in differences.items()}, metadata


def _ordinary_summary(records, seeds, certificates, partition, settings):
    discrimination_rows, gate_rows, curve_rows, sensitivity_rows = [], [], [], []
    coverages = np.unique(np.r_[np.linspace(.025, 1, 40), .25, .5, .75])
    for s, seed in enumerate(seeds):
        r = records[s]
        scores = [*METHODS, *[name for name in DIAGNOSTICS if name in r]]
        for name in scores:
            d = discrimination(r['harm'], r[name])
            discrimination_rows.append({'label': LABEL, 'partition': partition, 'seed': seed,
                                        'method': name, **d,
                                        'partial_aurc_25_75': partial_aurc(r[name], r['harm'], r['tie_key'])})
            for point in rank_curve(r[name], r['harm'], r['useful'], r['tie_key'], coverages,
                                    r['base_loss'], r['neural_loss']):
                curve_rows.append({'label': LABEL, 'partition': partition, 'seed': seed,
                                   'method': name, **point})
        for name in (*METHODS, 'g0', 'base', 'oracle'):
            if name == 'g0':
                gate = np.ones(len(r['harm']), bool)
            elif name == 'base':
                gate = np.zeros(len(r['harm']), bool)
            elif name == 'oracle':
                gate = r['neural_loss'] < r['base_loss']
            else:
                gate = acceptance(r[name], certificates[name][str(seed)]['threshold'])
            metrics = gate_metrics(r['harm'], r['useful'], gate, r['base_loss'], r['neural_loss'])
            metrics['bit_errors'] = int(np.where(gate, r['neural_bit_errors'], r['base_bit_errors']).sum())
            metrics['bits'] = int(r['bits'].sum())
            metrics['development_ber'] = metrics['bit_errors']/metrics['bits']
            metrics['threshold'] = (certificates[name][str(seed)]['threshold'] if name in METHODS else None)
            gate_rows.append({'label': LABEL, 'partition': partition, 'seed': seed, 'method': name, **metrics})
            for delta in settings['statistics']['sensitivity_db']:
                e, u = labels_from_losses(r['base_loss'], r['neural_loss'], delta, delta)
                sensitivity_rows.append({'label': LABEL, 'partition': partition, 'seed': seed,
                    'method': name, 'delta_db': delta,
                    'status': 'DESCRIPTIVE_RELABELING_FIXED_ORIGINAL_GATE',
                    **gate_metrics(e, u, gate, r['base_loss'], r['neural_loss'])})
    return discrimination_rows, gate_rows, curve_rows, sensitivity_rows


def panel_groups(record):
    """Only preregistered strata; estimated-SNR overflow bins preserve all frames."""
    groups = {'all_panel': np.ones(len(record['harm']), bool)}
    for name in ('family', 'modulation', 'burst_cell', 'burst_present'):
        for value in np.unique(record[name]):
            groups[f'{name}={value}'] = record[name] == value
    nominal = record['snr_db']
    estimated = 10*np.log10(np.maximum(record['estimated_snr_linear'], np.finfo(float).tiny))
    bands = [('below_-5', -np.inf, -5), ('[-5,0)', -5, 0), ('[0,10)', 0, 10),
             ('[10,20)', 10, 20), ('[20,25]', 20, np.nextafter(25., np.inf)),
             ('above_25', np.nextafter(25., np.inf), np.inf)]
    for label, lo, hi in bands:
        for name, values in [('nominal_snr', nominal), ('estimated_snr', estimated)]:
            mask = (values >= lo) & (values < hi)
            if mask.any():
                groups[f'{name}={label}'] = mask
    occupancy = record['burst_occupancy']
    groups['useful_window_burst=absent'] = occupancy == 0
    groups['useful_window_burst=present'] = occupancy > 0
    # These crossing panels are predeclared to avoid pooled QPSK masking 16-QAM.
    for family in np.unique(record['family']):
        for modulation in np.unique(record['modulation']):
            for label, lo, hi in bands[1:5]:
                groups[f'{family}/{modulation}/{label}'] = (
                    (record['family'] == family) & (record['modulation'] == modulation)
                    & (nominal >= lo) & (nominal < hi))
    return {key: value for key, value in groups.items() if value.any()}


def _panel_cells(record):
    return np.array([f'{f}/{m}/{s}/{b}' for f, m, s, b in zip(
        record['family'], record['modulation'], record['snr_db'], record['burst_cell'])])


def stratified_summary(records, seeds, settings):
    """Descriptive fixed-panel inference, resampling within original quota cells.

    Estimated-SNR groups use each model's receiver estimate. Baseline is common
    across seeds; if a future implementation changes that, do not pair subgroup
    memberships blindly. Small/absent classes explicitly withhold inference.
    """
    first = records[0]
    for r in records[1:]:
        if not np.array_equal(first['estimated_snr_linear'], r['estimated_snr_linear']):
            raise ArithmeticError('estimated-SNR strata differ across common-baseline seeds')
    groups = panel_groups(first)
    original_cells = _panel_cells(first)
    draws = int(settings['statistics'].get('stratum_bootstrap_draws', 2000))
    bootstrap_seed = int(settings['seeds'].get('strata_bootstrap', 71995))
    rng = np.random.default_rng(bootstrap_seed)
    rows, point_rows = [], []
    ns = len(seeds)
    for group, mask in groups.items():
        n = int(mask.sum())
        if n < 2:
            continue
        cell = original_cells[mask]
        cell_groups = [np.flatnonzero(cell == v) for v in np.unique(cell)]
        e, u = _stack(records, 'harm')[:, mask], _stack(records, 'useful')[:, mask]
        keys = first['tie_key'][mask]
        scores = {name: _stack(records, name)[:, mask] for name in METHODS}
        estimates, raw = {}, {}
        for name in METHODS:
            estimates[name] = []
            for s, seed in enumerate(seeds):
                d = discrimination(e[s], scores[name][s])
                g = rank_acceptance(scores[name][s], .5, keys)
                m = gate_metrics(e[s], u[s], g)
                estimates[name].append([d['auroc'], d['auprc'], m['conditional_harm'], m['joint_useful']])
                point_rows.append({'label': LABEL, 'partition': 'V-panel', 'stratum': group,
                    'method': name, 'seed': seed, **d,
                    'conditional_harm_at_50': m['conditional_harm'], 'joint_useful_at_50': m['joint_useful'],
                    'accepted_count_at_50': m['accepted_count'],
                    'accepted_evidence': 'ADEQUATE' if m['accepted_count'] >= 200 else 'INSUFFICIENT_EVIDENCE',
                    'neural_detector_ser': float(records[s]['neural_ser'][mask].mean()),
                    'base_detector_ser': float(records[s]['base_ser'][mask].mean())})
            raw[name] = np.empty((draws, 4))
        for start in range(0, draws, 64):
            stop = min(draws, start+64)
            batch = stop-start
            ix = np.concatenate([g[rng.integers(len(g), size=(batch, len(g)))] for g in cell_groups], axis=1)
            w = np.array([np.bincount(v, minlength=n) for v in ix])
            si = rng.integers(ns, size=(batch, ns))
            sw = np.array([np.bincount(v, minlength=ns) for v in si])/ns
            for name in METHODS:
                values = np.empty((batch, ns, 4))
                for s in range(ns):
                    roc, ap = weighted_discrimination(w, scores[name][s], e[s])
                    risk, useful = weighted_rank_points(w, np.lexsort((keys, scores[name][s])), e[s], u[s], (.5,))
                    values[:, s] = np.column_stack([roc, ap, risk[:, 0], useful[:, 0]])
                raw[name][start:stop] = np.einsum('bs,bsk->bk', sw, values)
        for name in METHODS:
            point = np.array([[np.nan if v is None else v for v in row] for row in estimates[name]], float).mean(axis=0)
            for j, metric in enumerate(('auroc', 'auprc', 'conditional_harm_at_50', 'joint_useful_at_50')):
                ci = bootstrap_summary(raw[name][:, j], point[j])
                rows.append({'label': LABEL, 'partition': 'V-panel', 'stratum': group, 'method': name,
                    'metric': metric, 'n_independent_frames': n, 'training_seeds': ns,
                    'harm_prevalence_mean': float(e.mean()), 'useful_prevalence_mean': float(u.mean()),
                    'inference': 'DESCRIPTIVE_STRATIFIED_PAIRED_95_PERCENT_NOT_CONTINUATION', **ci})
        print('S1 stratum analysis', group, n, flush=True)
    return rows, point_rows


def _save_figure(fig, name):
    target = OUT/'plots'
    target.mkdir(parents=True, exist_ok=True)
    fig.suptitle(FIGURE_LABEL, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(target/f'{name}.{suffix}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def make_plots(primary, summaries, curves, strata, timing_summary=None):
    """All plotted values are read/computed from retained raw output arrays."""
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'lines.linewidth': 1.2})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ('auroc', 'auprc')):
        for j, name in enumerate(METHODS):
            ci = summaries['descriptive_discrimination'][f'{name}/{metric}']
            value, low, high = ci['estimate'], ci['lower'], ci['upper']
            if value is not None:
                error = None if low is None else [[max(0, value-low)], [max(0, high-value)]]
                ax.errorbar(j, value, yerr=error, fmt='o', capsize=3)
        ax.set_xticks(range(len(METHODS)), METHODS, rotation=60, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.set_title('Primary domain; mean across training seeds')
    _save_figure(fig, 'primary_discrimination')
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, metric in zip(axes.ravel(), ('conditional_harm', 'joint_harm', 'joint_useful', 'baseline_regret')):
        for name in METHODS:
            rows = [r for r in curves if r['method'] == name and r['partition'] == 'V-assess']
            coverages = sorted(set(r['target_coverage'] for r in rows))
            values = [np.mean([r[metric] for r in rows if r['target_coverage'] == a]) for a in coverages]
            ax.plot(coverages, values, label=name)
        ax.set_xlabel('Correction coverage')
        ax.set_ylabel(metric.replace('_', ' '))
    axes[0, 0].legend(fontsize=7, ncol=2)
    _save_figure(fig, 'primary_risk_usefulness_regret')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, name in zip(axes, ('g3_hard', 'g3_soft')):
        record = primary[0]
        for event, color in [('harm', 'tab:red'), ('useful', 'tab:blue')]:
            values = record[name][record[event]]
            if len(values):
                ax.hist(values, bins=40, histtype='step', density=True, label=event, color=color)
        ax.set_xlabel(name+' score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title('Stored checkpoint seed5101; all primary frames retained')
    _save_figure(fig, 'g3_score_distributions')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, family_modulation in zip(axes.ravel(), ('generic_A/QPSK', 'generic_A/16QAM', 'generic_B/QPSK', 'generic_B/16QAM')):
        for method in ('g3_hard', 'g3_soft', 'g1', 'g2'):
            relevant = [r for r in strata if r['method'] == method and r['metric'] == 'auroc'
                        and r['stratum'].startswith(family_modulation+'/')]
            labels = ['[-5,0)', '[0,10)', '[10,20)', '[20,25]']
            values = [next((r['estimate'] for r in relevant if r['stratum'].endswith('/'+band)), np.nan)
                      for band in labels]
            ax.plot(range(4), [np.nan if v is None else v for v in values], marker='o', label=method)
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Harm AUROC')
        ax.set_title(family_modulation)
    axes[0, 0].legend(fontsize=7)
    _save_figure(fig, 'hard_soft_snr_panel')
    if timing_summary is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        methods = [name for name in METHODS if name in timing_summary['total_times']]
        for ax, key, title in zip(axes, ('total_times', 'receiver_times'),
                                 ('Validation and candidate deployment', 'Complete receiver')):
            means = [1000*np.mean(timing_summary[key][name]) for name in methods]
            ax.bar(range(len(methods)), means)
            ax.set_xticks(range(len(methods)), methods, rotation=60, ha='right')
            ax.set_ylabel('Measured mean milliseconds/frame')
            ax.set_title(title)
        _save_figure(fig, 'measured_cost')


def run_analysis(settings, selection, certificates):
    """Called only after frozen-manifest S1 execution and raw timing complete."""
    if settings['stage'] != 'E5' or settings['mode'] != 'development':
        raise PermissionError('E5 development only')
    (OUT/'analysis').mkdir(parents=True, exist_ok=True)
    seeds = settings['seeds']['training']
    guard = BoundaryGuard()
    raw, hashes = {}, {}
    for partition in ('V-design', 'V-cal-dev', 'V-assess', 'V-panel'):
        raw[partition], h = load_partition(partition, seeds, selection, guard)
        hashes.update(h)
    primary = raw['V-assess']
    if not all(_primary_mask(r).all() for r in primary):
        raise ArithmeticError('primary assessment law mismatch')
    score_arrays = {name: _stack(primary, name) for name in METHODS}
    accepted = {name: np.stack([acceptance(r[name], certificates[name][str(seed)]['threshold'])
                               for r, seed in zip(primary, seeds)]) for name in METHODS}
    brier_differences, diagnostic_metadata = confidence_diagnostics(raw['V-design'], primary, seeds)
    stats = settings['statistics']
    print('S1 primary bootstrap started', flush=True)
    result = primary_bound_table(score_arrays, _stack(primary, 'base_loss'), _stack(primary, 'neural_loss'),
        primary[0]['tie_key'], accepted, brier_differences, control=selection['control'], training_seeds=seeds,
        draws=stats['bootstrap_draws'], seed=settings['seeds']['bootstrap'],
        batch_size=stats.get('bootstrap_batch_size', 64),
        discrimination_draws=stats['discrimination_bootstrap_draws'])
    replicates = result.pop('replicates')
    np.savez_compressed(OUT/'analysis/primary_bootstrap_replicates.npz', **replicates)
    _json(OUT/'analysis/primary_inference.json', result)
    hard_soft = {}
    for metric in ('auroc', 'auprc'):
        hard = result['descriptive_discrimination'][f'g3_hard/{metric}']['estimate']
        soft = result['descriptive_discrimination'][f'g3_soft/{metric}']['estimate']
        estimate = hard-soft if hard is not None and soft is not None else np.nan
        hard_soft[metric] = bootstrap_summary(replicates[f'g3_hard/{metric}']-replicates[f'g3_soft/{metric}'], estimate)
    for metric in ('risk', 'useful'):
        for coverage in (.25, .5, .75):
            hkey, skey = f'g3_hard/g1/{metric}/{coverage}', f'g3_soft/g1/{metric}/{coverage}'
            hard_soft[f'{metric}/{coverage}'] = bootstrap_summary(replicates[hkey]-replicates[skey],
                result['bounds'][hkey]['estimate']-result['bounds'][skey]['estimate'])
    _json(OUT/'analysis/hard_soft_comparison.json', {'label': LABEL,
        'direction': 'hard minus soft; lower risk/usefulness directions remain distinct',
        'inference': 'DESCRIPTIVE_PAIRED_95_PERCENT; not an additional winner-search test', 'comparisons': hard_soft})
    print('S1 primary bootstrap complete', flush=True)
    summaries, gates, curves, sensitivities = [], [], [], []
    for partition in ('V-assess', 'V-panel'):
        d, g, c, s = _ordinary_summary(raw[partition], seeds, certificates, partition, settings)
        summaries.extend(d); gates.extend(g); curves.extend(c); sensitivities.extend(s)
    _csv(OUT/'analysis/discrimination.csv', summaries)
    _csv(OUT/'analysis/fixed_gates.csv', gates)
    _csv(OUT/'analysis/rank_curves.csv', curves)
    _csv(OUT/'analysis/delta_sensitivity.csv', sensitivities)
    mismatch = mismatch_bootstrap(_stack(primary, 'partial_base_loss'), _stack(primary, 'partial_neural_loss'),
        _stack(primary, 'base_loss'), _stack(primary, 'neural_loss'), accepted['g2'], training_seeds=seeds,
        draws=stats.get('mismatch_bootstrap_draws', 2000), seed=settings['seeds'].get('mismatch_bootstrap', 71994))
    np.savez_compressed(OUT/'analysis/mismatch_bootstrap_replicates.npz',
                        aggregate=mismatch.pop('replicates'), per_seed=mismatch.pop('per_seed_replicates'))
    _json(OUT/'analysis/partial_full_mismatch.json', mismatch)
    transfer_discrimination = []
    panel_mismatch = {}
    for partition in ('V-assess', 'V-panel'):
        for s, seed in enumerate(seeds):
            r = raw[partition][s]
            ep, _ = labels_from_losses(r['partial_base_loss'], r['partial_neural_loss'])
            for label_type, event in [('partial_labels', ep), ('final_full_labels', r['harm'])]:
                transfer_discrimination.append({'label': LABEL, 'partition': partition, 'seed': seed,
                    'score': 'same_selected_g2_partial_score', 'label_type': label_type,
                    **discrimination(event, r['g2'])})
            if partition == 'V-panel':
                g = acceptance(r['g2'], certificates['g2'][str(seed)]['threshold'])
                panel_mismatch[str(seed)] = partial_full_mismatch(r['partial_base_loss'], r['partial_neural_loss'],
                                                                 r['base_loss'], r['neural_loss'], g)
    _json(OUT/'analysis/partial_full_mismatch_panel.json', {'label': LABEL, 'status': 'DESCRIPTIVE_PANEL', 'per_seed': panel_mismatch})
    _csv(OUT/'analysis/g2_label_transfer_discrimination.csv', transfer_discrimination)
    strata, stratum_points = stratified_summary(raw['V-panel'], seeds, settings)
    _csv(OUT/'analysis/strata_intervals.csv', strata)
    _csv(OUT/'analysis/strata_per_seed.csv', stratum_points)
    # Algebraic identity is mandatory disclosure, not a distinct competitor.
    identity = [{'partition': partition, 'seed': seed,
                 'max_absolute_hard_weighted_evm_difference': float(np.max(np.abs(r['g3_hard']-r['weighted_evm_identity'])))}
                for partition in ('V-assess', 'V-panel') for r, seed in zip(raw[partition], seeds)]
    _json(OUT/'analysis/weighted_evm_identity.json', {'label': LABEL, 'results': identity,
        'interpretation': 'G3-hard is robust disturbance/gain-weighted decision EVM; no distinct information-source claim.'})
    # Timing contract is consumed below; missing data yields withheld cost bounds.
    timing_path = OUT/'timing/benchmark.json'
    timing = None
    allowance = 0.0
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding='utf8'))
        resolution = json.loads((OUT/'timing/resolution.json').read_text(encoding='utf8'))
        allowance = float(resolution['receiver_allowance_seconds'])
        if timing['primary_device'] != settings['timing']['decision_device']:
            raise ArithmeticError('timing decision device differs from frozen settings')
        timing_result = timing_bound_table(timing['total_times'], timing['receiver_times'],
            control=selection['control'], draws=stats['bootstrap_draws'], seed=settings['seeds']['timing'],
            error=result['one_sided_error'])
        np.savez_compressed(OUT/'analysis/timing_bootstrap_replicates.npz', **timing_result.pop('replicates'))
        _json(OUT/'analysis/timing_inference.json', timing_result)
        result['bounds'].update(timing_result['bounds'])
    certificate_pass = {f'{v}/seed{seed}': (certificates[v][str(seed)]['status'] == 'CERTIFIED_SCREEN_PASS'
            and result['fixed_gates'][f'{v}/seed{seed}']['accepted_count'] >= stats['accepted_min'])
            for v in ('g3_hard', 'g3_soft') for seed in seeds}
    decision = s1_decision(result['bounds'], certificate_pass, training_seeds=seeds,
                            receiver_allowance=allowance, definitive_failure=False)
    # Non-rejection of success is not a scientific falsification. Any FAILURE
    # needs the root's documented diagnosis/novelty review; no hidden optimizer.
    inequality_rows = [{'label': LABEL, 'inequality': name, **result['bounds'].get(name,
                       {'status': 'UNAVAILABLE', 'lower': None, 'upper': None}),
                       'one_sided_error': result['one_sided_error']} for name in result['family_names']]
    _csv(OUT/'analysis/decision_inequalities.csv', inequality_rows)
    decision.update(label=LABEL, selection=selection, certificates_pass=certificate_pass,
        raw_input_hashes=hashes, timing_available=timing is not None,
        ordinary_confidence_threat='Resolved only if the exact distinct-information/cost condition passes',
        final_data_accessed=False, next_stage='E6 NOT_STARTED; this report does not authorize execution',
        limitations=['Three fitted training seeds give limited seed-distribution precision.',
                     'Bonferroni-adjusted bootstrap tails are approximate, with about13 tail draws.',
                     'Incremental Brier inference is conditional on the fixed V-design logistic fits.',
                     'Panel strata are descriptive; no independent secondary-domain certificates.',
                     'A PARTIAL SUCCESS can mean insufficient evidence; no scientific FAILURE inferred from non-rejection.'])
    _json(OUT/'analysis/decision_report.json', decision)
    make_plots(primary, result, curves, strata, timing)
    _json(OUT/'analysis/artifact_manifest.json', {'label': LABEL, 'raw_input_hashes': hashes,
          'outputs': {p.relative_to(ROOT).as_posix(): sha(p) for folder in (OUT/'analysis', OUT/'plots')
                      for p in folder.glob('*') if p.is_file() and p.name != 'artifact_manifest.json'}})
    return decision
