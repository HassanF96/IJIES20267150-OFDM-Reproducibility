"""Offline S1 DEVELOPMENT analysis; no data loading or practical-receiver inputs.

All observations must be train or validation records. V-cal-dev is an independent
VALIDATION partition, not the unopened final calibration split. The caller must
enforce ID custody. Arrays are paired by frame; learned-seed arrays use [seed,frame].
"""
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.stats import beta, binom, rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL = 'DEVELOPMENT / S1 — NOT FINAL RESEARCH RESULT'
COVERAGES = (0.25, 0.50, 0.75)


def _vector(value, name, binary=False):
    a = np.asarray(value)
    if a.ndim != 1 or not len(a):
        raise ValueError(f'{name}: nonempty vector required')
    if binary and not np.isin(a, (0, 1)).all():
        raise ValueError(f'{name}: binary values required')
    if not binary and not np.isfinite(a).all():
        raise ValueError(f'{name}: finite values required')
    return a.astype(bool if binary else float, copy=False)


def labels_from_losses(base_loss, neural_loss, delta_db=0.5, useful_delta_db=0.5):
    """Strict frame-level events; no dB/log floor, including exact-zero cases."""
    b, n = np.asarray(base_loss, float), np.asarray(neural_loss, float)
    if (b.shape != n.shape or not b.size or not np.isfinite(b).all()
            or not np.isfinite(n).all() or np.any(b < 0) or np.any(n < 0)
            or not np.isfinite([delta_db, useful_delta_db]).all()
            or min(delta_db, useful_delta_db) < 0):
        raise ValueError('matching finite nonnegative losses/tolerances required')
    return n > 10**(delta_db/10)*b, b > 10**(useful_delta_db/10)*n


def discrimination(harm, score):
    """Harm-positive AUROC (half ties), AP with whole tie blocks."""
    y, t = _vector(harm, 'harm', True), _vector(score, 'score')
    if y.shape != t.shape:
        raise ValueError('unpaired arrays')
    p, n = int(y.sum()), len(y)
    result = {'n': n, 'harm_count': p, 'nonharm_count': n-p,
              'harm_prevalence': p/n, 'auroc': None, 'auprc': None}
    if p == 0 or p == n:
        result['status'] = 'UNDEFINED_MISSING_CLASS'
        # AP is well-defined at one with all positives; absent positives is undefined.
        result['auprc'] = 1.0 if p == n else None
        return result
    ranks = rankdata(t, method='average')
    result['auroc'] = float((ranks[y].sum()-p*(p+1)/2)/(p*(n-p)))
    order = np.argsort(-t, kind='stable')
    ys, ts = y[order], t[order]
    ends = np.r_[np.flatnonzero(ts[1:] != ts[:-1]), n-1]
    tp = np.cumsum(ys)[ends]
    increments = np.diff(np.r_[0, tp])
    result['auprc'] = float(np.sum((tp/(ends+1))*(increments/p)))
    result['status'] = 'OK'
    return result


def acceptance(score, threshold, valid=None):
    t = np.asarray(score, float)
    if t.ndim != 1 or not len(t):
        raise ValueError('nonempty score vector required')
    v = np.isfinite(t) if valid is None else np.asarray(valid, bool) & np.isfinite(t)
    if v.shape != t.shape:
        raise ValueError('unpaired validity')
    if threshold is None:
        return np.zeros(len(t), bool)
    if not np.isfinite(threshold):
        raise ValueError('finite threshold required; None denotes fallback')
    return v & (t <= threshold)


def gate_metrics(harm, useful, accepted, base_loss=None, neural_loss=None):
    e, u, g = (_vector(v, name, True) for v, name in
               ((harm, 'harm'), (useful, 'useful'), (accepted, 'accepted')))
    if not e.shape == u.shape == g.shape or np.any(e & u):
        raise ValueError('paired disjoint event arrays required')
    n, m, h, b = len(e), int(g.sum()), int((g & e).sum()), int((g & u).sum())
    result = {'n': n, 'accepted_count': m, 'harmful_accepts': h,
              'useful_accepts': b, 'neutral_accepts': m-h-b,
              'rejected_useful_count': int((~g & u).sum()),
              'coverage': m/n, 'joint_harm': h/n,
              'conditional_harm': h/m if m else None,
              'joint_useful': b/n, 'conditional_useful': b/m if m else None,
              'rejected_useful': float(np.mean(~g & u)),
              'harm_prevalence': float(e.mean()), 'useful_prevalence': float(u.mean())}
    if (base_loss is None) != (neural_loss is None):
        raise ValueError('both losses required')
    if base_loss is not None:
        lb, ln = _vector(base_loss, 'base'), _vector(neural_loss, 'neural')
        if lb.shape != e.shape or ln.shape != e.shape or np.any(lb < 0) or np.any(ln < 0):
            raise ValueError('paired nonnegative losses required')
        final = np.where(g, ln, lb)
        result.update(baseline_regret=float(np.mean(g*(ln-lb))),
                      oracle_regret=float(np.mean(final-np.minimum(lb, ln))),
                      mean_linear_nmse=float(final.mean()))
    return result


def rank_acceptance(score, coverage, tie_keys, valid=None):
    """Outcome-independent tie ordering; floor(a*n) accepted. No threshold tuning.

    Unavailable coverage is an error, never extrapolated. Randomized within-tie
    ranks are descriptive and need not be attainable by a deterministic gate.
    """
    t, keys = _vector(score, 'score'), _vector(tie_keys, 'tie_keys')
    v = np.ones(len(t), bool) if valid is None else np.asarray(valid, bool)
    if keys.shape != t.shape or v.shape != t.shape or not 0 <= coverage <= 1:
        raise ValueError('invalid paired rank inputs')
    m = int(np.floor(coverage*len(t)+1e-12))
    indices = np.flatnonzero(v)
    if m > len(indices):
        raise ValueError('UNATTAINABLE_COVERAGE')
    ordered = indices[np.lexsort((keys[indices], t[indices]))]
    g = np.zeros(len(t), bool)
    g[ordered[:m]] = True
    return g


def rank_curve(score, harm, useful, tie_keys, coverages=COVERAGES,
               base_loss=None, neural_loss=None, valid=None):
    rows = []
    for a in coverages:
        try:
            g = rank_acceptance(score, a, tie_keys, valid)
            row = gate_metrics(harm, useful, g, base_loss, neural_loss)
            row.update(target_coverage=float(a), status='OK')
        except ValueError as exc:
            if str(exc) != 'UNATTAINABLE_COVERAGE':
                raise
            row = {'target_coverage': float(a), 'status': 'UNATTAINABLE_COVERAGE'}
        rows.append(row)
    return rows


def partial_aurc(score, harm, tie_keys, lower=0.25, upper=0.75, valid=None):
    """Right-endpoint conditional-harm rectangle integral on a common interval."""
    t, y, keys = _vector(score, 'score'), _vector(harm, 'harm', True), _vector(tie_keys, 'keys')
    if not t.shape == y.shape == keys.shape or not 0 < lower < upper <= 1:
        raise ValueError('invalid pAURC inputs')
    v = np.ones(len(t), bool) if valid is None else np.asarray(valid, bool)
    indices = np.flatnonzero(v)
    if len(indices)/len(t) < upper:
        return None
    order = indices[np.lexsort((keys[indices], t[indices]))]
    j = np.arange(1, len(order)+1)
    risk = np.cumsum(y[order])/j
    overlap = np.maximum(0, np.minimum(j/len(t), upper)-np.maximum((j-1)/len(t), lower))
    return float(np.dot(risk, overlap)/(upper-lower))


def threshold_grid(design_score, quantiles=None):
    """Design-only quantiles, inclusive endpoints, deterministic deduplication."""
    t = _vector(design_score, 'design_score')
    q = np.linspace(0, 1, 41) if quantiles is None else np.asarray(quantiles, float)
    if q.ndim != 1 or not len(q) or np.any(np.diff(q) < 0) or np.any((q < 0) | (q > 1)):
        raise ValueError('ordered quantiles required')
    return np.unique(np.quantile(t, q, method='linear'))


def ltt_development(score, harm, useful, grid, *, partition,
                    alpha=0.05, epsilon=0.05/27, accepted_min=200,
                    coverage_min=0.05, useful_min=0.05, valid=None):
    """Independent V-cal-dev fixed-sequence LTT; never final calibration.

    All rates use n complete frames. Stop at the first non-rejection; screens
    cannot resurrect a failed prefix. The numeric grid must come from V-design.
    """
    if partition != 'V-cal-dev':
        raise ValueError('E5 permits LTT only on the validation V-cal-dev partition')
    t, e, u = _vector(score, 'score'), _vector(harm, 'harm', True), _vector(useful, 'useful', True)
    thresholds = _vector(grid, 'grid')
    if not t.shape == e.shape == u.shape or np.any(e & u) or np.any(np.diff(thresholds) <= 0):
        raise ValueError('paired data and strictly increasing frozen grid required')
    if not 0 < alpha < 1 or not 0 < epsilon < 1 or accepted_min < 1:
        raise ValueError('invalid LTT policy')
    trace, selected, prefix = [], None, 0
    for tau in thresholds:
        metrics = gate_metrics(e, u, acceptance(t, tau, valid))
        p = float(binom.cdf(metrics['harmful_accepts'], len(e), alpha))
        passed = p <= epsilon
        trace.append(dict(threshold=float(tau), p_value=p, risk_certified=passed, **metrics))
        if not passed:
            break
        prefix += 1
        if (metrics['accepted_count'] >= accepted_min and metrics['coverage'] >= coverage_min
                and metrics['joint_useful'] >= useful_min):
            selected = float(tau)
    status = ('CERTIFIED_SCREEN_PASS' if selected is not None else
              'NO_CERTIFIED_THRESHOLD' if not prefix else 'INSUFFICIENT_USEFUL_ACCEPTANCE')
    return {'label': LABEL, 'partition': partition, 'status': status,
            'threshold': selected, 'certified_prefix_length': prefix,
            'n': len(e), 'alpha': alpha, 'epsilon': epsilon, 'trace': trace}


def binomial_interval(successes, trials, error=0.05):
    """Two-sided exact CP and same-error one-sided bounds. Zero n is undefined."""
    if not isinstance(successes, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError('integer binomial counts required')
    if not 0 <= successes <= trials or not 0 < error < 1:
        raise ValueError('invalid binomial inputs')
    if trials == 0:
        return {'lower': None, 'upper': None, 'lower_one_sided': None,
                'upper_one_sided': None, 'status': 'UNDEFINED_NO_ACCEPTANCE'}
    s, n = successes, trials
    return {'lower': 0.0 if s == 0 else float(beta.ppf(error/2, s, n-s+1)),
            'upper': 1.0 if s == n else float(beta.ppf(1-error/2, s+1, n-s)),
            'lower_one_sided': 0.0 if s == 0 else float(beta.ppf(error, s, n-s+1)),
            'upper_one_sided': 1.0 if s == n else float(beta.ppf(1-error, s+1, n-s)),
            'status': 'OK'}


def fixed_gate_report(harm, useful, accepted, base_loss=None, neural_loss=None,
                      error=0.05, accepted_min=200, *, independent_iid=True):
    metrics = gate_metrics(harm, useful, accepted, base_loss, neural_loss)
    if not independent_iid:
        raise ValueError('Exact fixed-gate intervals require independent IID evaluation')
    n, m = metrics['n'], metrics['accepted_count']
    metrics['intervals'] = {
        'coverage': binomial_interval(m, n, error),
        'joint_harm': binomial_interval(metrics['harmful_accepts'], n, error),
        'joint_useful': binomial_interval(metrics['useful_accepts'], n, error),
        'conditional_harm': binomial_interval(metrics['harmful_accepts'], m, error)}
    metrics['accepted_evidence'] = ('ADEQUATE' if m >= accepted_min else
                                    'UNDEFINED_NO_ACCEPTANCE' if not m else 'INSUFFICIENT_EVIDENCE')
    return metrics


def binomial_sample_plan(n, epsilon, alpha=0.05, alternative=0.03,
                         acceptance_probability=0.10, accepted_min=200):
    possible = np.flatnonzero(binom.cdf(np.arange(n+1), n, alpha) <= epsilon)
    cutoff = int(possible[-1]) if len(possible) else -1
    return {'n': int(n), 'zero_harm_feasibility_min': int(np.ceil(np.log(epsilon)/np.log1p(-alpha))),
            'harm_count_cutoff': cutoff, 'design_alternative': alternative,
            'power': float(binom.cdf(cutoff, n, alternative)),
            'planning_acceptance': acceptance_probability,
            'probability_adequate_acceptance': float(binom.sf(accepted_min-1, n, acceptance_probability))}


def bootstrap_indices(n_seeds, n_frames, draws, seed, strata=None, hierarchical=True):
    """Crossed seed/whole-frame bootstrap for shared evaluation frames.

    Frame indices are common to every seed/method in each replicate. Resampling
    separate frames independently for each training seed would erase shared-frame
    dependence. Fixed-stratum diagnostic panels preserve stratum sample sizes.
    """
    if min(n_seeds, n_frames, draws) < 1:
        raise ValueError('positive dimensions required')
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(n_frames)]
    else:
        s = np.asarray(strata)
        if s.shape != (n_frames,):
            raise ValueError('one stratum per frame required')
        groups = [np.flatnonzero(s == x) for x in np.unique(s)]
    for _ in range(draws):
        si = rng.integers(n_seeds, size=n_seeds) if hierarchical else np.arange(n_seeds)
        fi = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        yield si, fi


def bootstrap_summary(replicates, estimate, error=0.05):
    r = np.asarray(replicates, float)
    if r.ndim != 1 or not len(r) or not 0 < error < 1:
        raise ValueError('nonempty bootstrap vector/error required')
    finite = r[np.isfinite(r)]
    # Never silently discard absent-class/coverage replicates to claim success.
    valid = len(finite) == len(r) and np.isfinite(estimate)
    if not valid:
        return {'estimate': float(estimate) if np.isfinite(estimate) else None,
                'status': 'INSUFFICIENT_BOOTSTRAP_EVIDENCE', 'draws': len(r),
                'undefined_draws': int(len(r)-len(finite)), 'lower': None, 'upper': None,
                'lower_one_sided': None, 'upper_one_sided': None}
    q = np.quantile(r, [error/2, 1-error/2, error, 1-error], method='linear')
    return {'estimate': float(estimate), 'status': 'APPROXIMATE_PERCENTILE', 'draws': len(r),
            'undefined_draws': 0, 'lower': float(q[0]), 'upper': float(q[1]),
            'lower_one_sided': float(q[2]), 'upper_one_sided': float(q[3]),
            'tail_expected_draws': float(len(r)*error)}


def paired_bootstrap(arrays: Mapping[str, np.ndarray], statistic: Callable,
                     *, draws, seed, error=0.05, strata=None, hierarchical=True):
    """Generic paired analysis; statistic receives [seed,frame] arrays and returns scalar."""
    values = {key: np.asarray(value) for key, value in arrays.items()}
    shapes = {a.shape for a in values.values()}
    if len(shapes) != 1 or len(next(iter(shapes), ())) != 2:
        raise ValueError('all arrays must share [training_seed,frame] shape')
    ns, nf = next(iter(shapes))
    estimate = statistic(values)
    samples = np.empty(draws)
    for i, (si, fi) in enumerate(bootstrap_indices(ns, nf, draws, seed, strata, hierarchical)):
        samples[i] = statistic({k: a[np.ix_(si, fi)] for k, a in values.items()})
    report = bootstrap_summary(samples, estimate, error)
    report.update(resampling='crossed training-seed and paired whole-frame' if hierarchical
                  else 'paired whole-frame conditional on fitted seeds', training_seed_count=ns,
                  independent_frame_count=nf, bootstrap_seed=int(seed))
    return report, samples


@dataclass
class ConfidenceDiagnostic:
    """V-design-only fixed C=1 L2 logistic diagnostic, not a deployment gate."""
    pipeline: object
    feature_names: tuple
    design_n: int

    @classmethod
    def fit(cls, features, harm, feature_names, *, partition='V-design', seed=1701):
        if partition != 'V-design':
            raise ValueError('Diagnostic fit is V-design only')
        x, y = np.asarray(features, float), _vector(harm, 'harm', True)
        if x.ndim != 2 or x.shape != (len(y), len(feature_names)) or np.isinf(x).any():
            raise ValueError('invalid design features')
        if np.unique(y).size < 2:
            raise ValueError('INSUFFICIENT_DESIGN_CLASSES')
        # Missingness imputation and all scaling are learned from design only.
        pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                 StandardScaler(), LogisticRegression(C=1.0,
                                 solver='lbfgs', max_iter=1000, tol=1e-8, random_state=seed))
        pipeline.fit(x, y)
        return cls(pipeline, tuple(feature_names), len(y))

    def predict(self, features):
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or np.isinf(x).any():
            raise ValueError('invalid assessment features')
        return self.pipeline.predict_proba(x)[:, 1]


def brier_losses(harm, probabilities):
    y, p = _vector(harm, 'harm', True), _vector(probabilities, 'probabilities')
    if p.shape != y.shape or np.any((p < 0) | (p > 1)):
        raise ValueError('paired probabilities required')
    return (p-y)**2


def partial_full_mismatch(partial_base, partial_neural, full_base, full_neural,
                           accepted, delta_db=0.5):
    pb, pn, fb, fn = [_vector(a, 'loss') for a in (partial_base, partial_neural, full_base, full_neural)]
    g = _vector(accepted, 'accepted', True)
    if not pb.shape == pn.shape == fb.shape == fn.shape == g.shape:
        raise ValueError('paired full/partial arrays required')
    ep, up = labels_from_losses(pb, pn, delta_db, delta_db)
    ef, uf = labels_from_losses(fb, fn, delta_db, delta_db)
    dp, df = pn-pb, fn-fb
    def correlations(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return {'pearson': None, 'spearman': None, 'status': 'UNDEFINED_CONSTANT'}
        return {'pearson': float(np.corrcoef(a, b)[0, 1]),
                'spearman': float(spearmanr(a, b).statistic), 'status': 'OK'}
    transfer = g & ~ep & ef
    reverse = g & ep & ~ef
    denom, revdenom = int((g & ~ep).sum()), int((g & ep).sum())
    return {'n': len(g), 'base_correlation': correlations(pb, fb),
            'neural_correlation': correlations(pn, fn), 'difference_correlation': correlations(dp, df),
            'partial_base_mean': float(pb.mean()), 'partial_neural_mean': float(pn.mean()),
            'full_base_mean': float(fb.mean()), 'full_neural_mean': float(fn.mean()),
            'harm_transfer_count': int(transfer.sum()), 'harm_transfer_denominator': denom,
            'harm_transfer_joint': float(transfer.mean()),
            'harm_transfer_conditional': float(transfer.sum()/denom) if denom else None,
            'reverse_transfer_count': int(reverse.sum()), 'reverse_transfer_denominator': revdenom,
            'reverse_transfer_joint': float(reverse.mean()),
            'reverse_transfer_conditional': float(reverse.sum()/revdenom) if revdenom else None,
            'harm_label_disagreement': float(np.mean(ep != ef)),
            'useful_label_disagreement': float(np.mean(up != uf)),
            'oracle_decision_disagreement': float(np.mean((dp < 0) != (df < 0))),
            'gate_full_oracle_disagreement': float(np.mean(g != (df < 0))),
            'base_refit_shift': float(np.mean(fb-pb)), 'neural_refit_shift': float(np.mean(fn-pn)),
            'fixed_decision_refit_shift': float(np.mean(np.where(g, fn-pn, fb-pb)))}


def mismatch_bootstrap(partial_base, partial_neural, full_base, full_neural, accepted,
                       *, training_seeds, draws=2000, seed=71993, delta_db=0.5, strata=None):
    """Paired descriptive 95% uncertainty for refit transfers and correlations.

    Reports each fixed training seed separately and a crossed seed/frame mean.
    Correlations are averaged across seeds (not a pooled n*seed correlation).
    These are descriptive intervals, outside the continuation inequality family.
    Undefined correlations/conditional denominators remain insufficient evidence.
    """
    arrays = [np.asarray(x) for x in (partial_base, partial_neural, full_base, full_neural, accepted)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('paired [seed,frame] full/partial arrays required')
    ns, nf = arrays[0].shape
    if len(training_seeds) != ns:
        raise ValueError('training seed identity mismatch')
    per_seed = [partial_full_mismatch(*(a[s] for a in arrays), delta_db=delta_db) for s in range(ns)]
    fields = ('harm_transfer_joint', 'harm_transfer_conditional', 'reverse_transfer_joint',
              'reverse_transfer_conditional', 'harm_label_disagreement', 'useful_label_disagreement',
              'oracle_decision_disagreement', 'gate_full_oracle_disagreement',
              'base_refit_shift', 'neural_refit_shift', 'fixed_decision_refit_shift')
    correlation_fields = ('base_correlation', 'neural_correlation', 'difference_correlation')
    flat_names = list(fields)+[f'{f}/{kind}' for f in correlation_fields for kind in ('pearson', 'spearman')]
    def flat(record):
        out = [record[name] for name in fields]
        out += [record[f][kind] for f in correlation_fields for kind in ('pearson', 'spearman')]
        return np.array([np.nan if value is None else value for value in out], float)
    values = np.stack([flat(r) for r in per_seed])
    per_replicates = np.empty((draws, ns, len(flat_names)))
    aggregate_replicates = np.empty((draws, len(flat_names)))
    for b, (si, fi) in enumerate(bootstrap_indices(ns, nf, draws, seed, strata=strata)):
        rows = [flat(partial_full_mismatch(*(a[s, fi] for a in arrays), delta_db=delta_db)) for s in range(ns)]
        per_replicates[b] = rows
        aggregate_replicates[b] = per_replicates[b, si].mean(axis=0)
    reports = {}
    for j, name in enumerate(flat_names):
        reports[name] = bootstrap_summary(aggregate_replicates[:, j], values[:, j].mean())
        for s, train_seed in enumerate(training_seeds):
            reports[f'seed{train_seed}/{name}'] = bootstrap_summary(per_replicates[:, s, j], values[s, j])
    return {'label': LABEL, 'per_seed': dict(zip(map(str, training_seeds), per_seed)),
            'intervals': reports, 'draws': draws, 'seed': seed,
            'independent_frame_count': nf, 'training_seed_count': ns,
            'replicate_names': flat_names, 'replicates': aggregate_replicates,
            'per_seed_replicates': per_replicates,
            'stratified': strata is not None,
            'status': 'DESCRIPTIVE_PAIRED_APPROXIMATE_95_PERCENT'}


def decision_inequality_names(variants=('g3_hard', 'g3_soft'), training_seeds=(0, 1, 2)):
    """Full Bonferroni family. C* is fixed from V-design, never V-assess."""
    names = []
    for v in variants:
        for seed in training_seeds:
            names.extend((f'{v}/seed{seed}/coverage', f'{v}/seed{seed}/useful'))
        names.append(f'{v}/regret')
        for c in ('g1', 'g2', 'control'):
            for a in COVERAGES:
                names.extend((f'{v}/{c}/risk/{a}', f'{v}/{c}/useful/{a}'))
        names.append(f'{v}/brier')
        for c in ('g2', 'control'):
            for field in ('latency_ratio', 'receiver_difference', 'coverage_upper',
                          'coverage_lower', 'risk', 'useful'):
                names.append(f'{v}/{c}/cost/{field}')
    return names


def select_design_control(scores, harm, useful, tie_keys, methods):
    """Common across seeds: mean R_C@.5, then larger U_J, then lexical name.

    Caller supplies only V-design arrays. No automatic method/score reversal.
    """
    e, u = np.asarray(harm, bool), np.asarray(useful, bool)
    keys = np.asarray(tie_keys, float)
    if e.ndim != 2 or e.shape != u.shape or keys.shape != (e.shape[1],):
        raise ValueError('paired design arrays [seed,frame] and frame tie keys required')
    rows = []
    for name in methods:
        t = np.asarray(scores[name], float)
        if t.shape != e.shape:
            raise ValueError('unpaired design score')
        results = [gate_metrics(e[s], u[s], rank_acceptance(t[s], .5, keys))
                   for s in range(len(e))]
        rows.append({'method': name,
                     'mean_conditional_harm': float(np.mean([r['conditional_harm'] for r in results])),
                     'mean_joint_useful': float(np.mean([r['joint_useful'] for r in results]))})
    rows.sort(key=lambda r: (r['mean_conditional_harm'], -r['mean_joint_useful'], r['method']))
    if not rows:
        raise ValueError('no design candidates')
    return rows[0]['method'], rows


def weighted_rank_points(weights, order, harm, useful, coverages=COVERAGES):
    """Exactly replicate empirical bootstrap ranks without sorting each draw.

    weights: [replicate,frame] nonnegative integer multiplicities, equal row sums.
    order: original ascending score/tie-key order. Repeated copies at a cutoff
    share the same labels, so partial tie-copy inclusion is deterministic.
    """
    w = np.asarray(weights)
    ix = np.asarray(order, int)
    e, u = np.asarray(harm, float), np.asarray(useful, float)
    if w.ndim != 2 or ix.shape != (w.shape[1],) or e.shape != ix.shape or u.shape != ix.shape:
        raise ValueError('paired weighted rank arrays required')
    n = w.sum(axis=1)
    if not np.all(n == n[0]) or n[0] < 1 or np.any(w < 0):
        raise ValueError('equal positive bootstrap sample sizes required')
    ws = w[:, ix]
    cumulative = np.cumsum(ws, axis=1)
    cumulative_harm = np.cumsum(ws*e[ix], axis=1)
    cumulative_useful = np.cumsum(ws*u[ix], axis=1)
    risks, utilities = [], []
    rows = np.arange(len(w))
    for a in coverages:
        m = int(np.floor(float(n[0])*a+1e-12))
        if m < 1:
            risks.append(np.full(len(w), np.nan))
            utilities.append(np.zeros(len(w)))
            continue
        k = np.argmax(cumulative >= m, axis=1)
        excess = cumulative[rows, k]-m
        risks.append((cumulative_harm[rows, k]-excess*e[ix[k]])/m)
        utilities.append((cumulative_useful[rows, k]-excess*u[ix[k]])/n)
    return np.column_stack(risks), np.column_stack(utilities)


def weighted_discrimination(weights, score, harm):
    """Weighted empirical AUROC/AP with score ties grouped, no class smoothing."""
    w, t, e = np.asarray(weights), _vector(score, 'score'), _vector(harm, 'harm', True)
    if w.ndim != 2 or w.shape[1] != len(t) or e.shape != t.shape:
        raise ValueError('unpaired weighted discrimination inputs')
    order = np.argsort(t, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(t[order]) != 0)+1]
    positive = np.add.reduceat(w[:, order]*e[order], starts, axis=1)
    negative = np.add.reduceat(w[:, order]*(~e[order]), starts, axis=1)
    p, q = positive.sum(axis=1), negative.sum(axis=1)
    negative_before = np.cumsum(negative, axis=1)-negative
    with np.errstate(divide='ignore', invalid='ignore'):
        roc = np.sum(positive*(negative_before+.5*negative), axis=1)/(p*q)
        posdesc, negdesc = positive[:, ::-1], negative[:, ::-1]
        cumulative_positive = np.cumsum(posdesc, axis=1)
        cumulative_total = cumulative_positive+np.cumsum(negdesc, axis=1)
        precision = np.divide(cumulative_positive, cumulative_total,
                              out=np.zeros_like(cumulative_total, dtype=float), where=cumulative_total > 0)
        ap = np.sum(precision*posdesc, axis=1)/p
    return roc, ap


def primary_bound_table(scores, base_loss, neural_loss, tie_keys, accepted,
                        brier_differences, *, control, training_seeds,
                        draws=20000, seed=71991, error_total=0.05, batch_size=64,
                        delta_db=0.5, timing_model_index=0,
                        discrimination_draws=2000):
    """76-inequality primary inference, plus descriptive rank/ranking intervals.

    All inputs are paired V-assess arrays [seed,frame]; tie_keys are stored by
    frame. `accepted` contains fixed V-cal-dev gates. `brier_differences` maps each
    G3 to frame Brier(plus-G3)-Brier(confidence). Timing ratio/receiver-difference
    bounds are supplied separately by timing_bound_table. They remain in the
    multiplicity denominator. No final data or assessment threshold fitting.

    Efficient multinomial counts exactly represent ordinary frame resampling.
    Seeds are independently resampled as clusters while frames remain common.
    Descriptive discrimination intervals use the first fixed count of replicates;
    rank/decision inference uses every replicate. Return raw replicate arrays for
    auditable storage; do not pass these arrays directly to a JSON serializer.
    """
    lb, ln = np.asarray(base_loss, float), np.asarray(neural_loss, float)
    e, u = labels_from_losses(lb, ln, delta_db, delta_db)
    if lb.ndim != 2 or len(training_seeds) != lb.shape[0]:
        raise ValueError('paired [seed,frame] losses and seed list required')
    ns, nf = lb.shape
    keys = _vector(tie_keys, 'tie_keys')
    if keys.shape != (nf,) or not 0 <= timing_model_index < ns:
        raise ValueError('frame keys / timing model mismatch')
    variants = ('g3_hard', 'g3_soft')
    comparators = {'g1': 'g1', 'g2': 'g2', 'control': control}
    selected = list(dict.fromkeys([*variants, *comparators.values()]))
    all_methods = list(scores)
    for name in all_methods:
        if np.shape(scores[name]) != lb.shape or not np.isfinite(scores[name]).all():
            raise ValueError('unpaired or invalid score arrays')
    for name in selected:
        if np.shape(accepted[name]) != lb.shape:
            raise ValueError('unpaired fixed gates')
    for name in variants:
        if np.shape(brier_differences[name]) != lb.shape:
            raise ValueError('unpaired diagnostic Brier arrays')
    names = decision_inequality_names(variants, training_seeds)
    error = error_total/len(names)
    orders = {name: [np.lexsort((keys, np.asarray(scores[name])[s])) for s in range(ns)]
              for name in selected}
    # Per-seed estimates remain separate, never multiplied into nf.
    points = {}
    for name in selected:
        points[name] = [rank_curve(scores[name][s], e[s], u[s], keys, base_loss=lb[s], neural_loss=ln[s])
                        for s in range(ns)]
    draws_discrimination = min(draws, discrimination_draws)
    raw = {name: np.empty(draws) for name in names
           if '/seed' not in name and not name.endswith(('latency_ratio', 'receiver_difference'))}
    descriptions = {f'{name}/{metric}': np.empty(draws_discrimination)
                    for name in all_methods for metric in ('auroc', 'auprc')}
    # Frame-conditional discrimination CIs expose evaluation sampling separately
    # from the resampled-seed mixture intervals.
    per_seed_discrimination = {f'{name}/seed{training_seeds[s]}/{metric}': np.empty(draws_discrimination)
                              for name in all_methods for s in range(ns) for metric in ('auroc', 'auprc')}
    rng = np.random.default_rng(seed)
    m_j = .05/10
    for start in range(0, draws, batch_size):
        stop = min(start+batch_size, draws)
        b = stop-start
        frame_indices = rng.integers(nf, size=(b, nf))
        weights = np.array([np.bincount(fi, minlength=nf) for fi in frame_indices])
        seed_indices = rng.integers(ns, size=(b, ns))
        seed_weights = np.array([np.bincount(si, minlength=ns) for si in seed_indices])/ns
        ranks = {}
        for name in selected:
            rs, us = [], []
            for s in range(ns):
                r, v = weighted_rank_points(weights, orders[name][s], e[s], u[s])
                rs.append(r)
                us.append(v)
            ranks[name] = (np.einsum('bs,sba->ba', seed_weights, rs),
                           np.einsum('bs,sba->ba', seed_weights, us))
        for v in variants:
            for alias, c in comparators.items():
                for j, a in enumerate(COVERAGES):
                    raw[f'{v}/{alias}/risk/{a}'][start:stop] = ranks[v][0][:, j]-ranks[c][0][:, j]
                    raw[f'{v}/{alias}/useful/{a}'][start:stop] = ranks[v][1][:, j]-ranks[c][1][:, j]
            regret = np.asarray(accepted[v], bool)*(ln-lb)
            raw[f'{v}/regret'][start:stop] = np.sum(seed_weights*(weights @ regret.T/nf), axis=1)
            raw[f'{v}/brier'][start:stop] = np.sum(seed_weights*(weights @ np.asarray(brier_differences[v]).T/nf), axis=1)
            s = timing_model_index
            gv = np.asarray(accepted[v], bool)[s]
            av = weights @ gv/nf
            rv = weights @ (gv & e[s])/nf
            uv = weights @ (gv & u[s])/nf
            for alias in ('g2', 'control'):
                gc = np.asarray(accepted[comparators[alias]], bool)[s]
                ac = weights @ gc/nf
                rc = weights @ (gc & e[s])/nf
                uc = weights @ (gc & u[s])/nf
                prefix = f'{v}/{alias}/cost/'
                raw[prefix+'coverage_upper'][start:stop] = av-ac
                raw[prefix+'coverage_lower'][start:stop] = av-ac
                with np.errstate(divide='ignore', invalid='ignore'):
                    raw[prefix+'risk'][start:stop] = rv/av-rc/ac-m_j/np.minimum(av, ac)
                raw[prefix+'useful'][start:stop] = uv-uc
        if start < draws_discrimination:
            limit = min(b, draws_discrimination-start)
            for name in all_methods:
                values_roc, values_ap = [], []
                for s in range(ns):
                    roc, ap = weighted_discrimination(weights[:limit], scores[name][s], e[s])
                    values_roc.append(roc)
                    values_ap.append(ap)
                    per_seed_discrimination[f'{name}/seed{training_seeds[s]}/auroc'][start:start+limit] = roc
                    per_seed_discrimination[f'{name}/seed{training_seeds[s]}/auprc'][start:start+limit] = ap
                descriptions[f'{name}/auroc'][start:start+limit] = np.sum(seed_weights[:limit]*np.array(values_roc).T, axis=1)
                descriptions[f'{name}/auprc'][start:start+limit] = np.sum(seed_weights[:limit]*np.array(values_ap).T, axis=1)
    def average_point(name, index, metric):
        return float(np.mean([r[index][metric] for r in points[name]]))
    estimates = {}
    for v in variants:
        estimates[f'{v}/regret'] = float(np.mean(np.asarray(accepted[v])*(ln-lb)))
        estimates[f'{v}/brier'] = float(np.mean(brier_differences[v]))
        for alias, c in comparators.items():
            for j, a in enumerate(COVERAGES):
                estimates[f'{v}/{alias}/risk/{a}'] = average_point(v, j, 'conditional_harm')-average_point(c, j, 'conditional_harm')
                estimates[f'{v}/{alias}/useful/{a}'] = average_point(v, j, 'joint_useful')-average_point(c, j, 'joint_useful')
        s = timing_model_index
        mv = gate_metrics(e[s], u[s], accepted[v][s])
        for alias in ('g2', 'control'):
            mc = gate_metrics(e[s], u[s], accepted[comparators[alias]][s])
            prefix = f'{v}/{alias}/cost/'
            estimates[prefix+'coverage_upper'] = mv['coverage']-mc['coverage']
            estimates[prefix+'coverage_lower'] = mv['coverage']-mc['coverage']
            estimates[prefix+'useful'] = mv['joint_useful']-mc['joint_useful']
            estimates[prefix+'risk'] = (mv['conditional_harm']-mc['conditional_harm']-m_j/min(mv['coverage'], mc['coverage'])
                                         if min(mv['coverage'], mc['coverage']) > 0 else np.nan)
    bounds, intervals, fixed = {}, {}, {}
    for name, samples in raw.items():
        report = bootstrap_summary(samples, estimates[name], error)
        bounds[name] = {'estimate': report['estimate'], 'lower': report['lower_one_sided'],
                        'upper': report['upper_one_sided'], 'status': report['status']}
        intervals[name] = report
    for v in variants:
        for s, train_seed in enumerate(training_seeds):
            report = fixed_gate_report(e[s], u[s], accepted[v][s], lb[s], ln[s], error=error)
            fixed[f'{v}/seed{train_seed}'] = report
            for field, metric in [('coverage', 'coverage'), ('useful', 'joint_useful')]:
                ci = report['intervals'][metric]
                bounds[f'{v}/seed{train_seed}/{field}'] = {
                    'estimate': report[metric], 'lower': ci['lower_one_sided'],
                    'upper': ci['upper_one_sided'], 'status': 'EXACT_FIXED_GATE'}
    descriptives = {}
    for name in all_methods:
        measured = [discrimination(e[s], scores[name][s]) for s in range(ns)]
        for metric in ('auroc', 'auprc'):
            vals = [d[metric] for d in measured]
            avg = float(np.mean(vals)) if all(v is not None for v in vals) else np.nan
            descriptives[f'{name}/{metric}'] = bootstrap_summary(descriptions[f'{name}/{metric}'], avg)
            for s, train_seed in enumerate(training_seeds):
                key = f'{name}/seed{train_seed}/{metric}'
                value = measured[s][metric]
                descriptives[key] = bootstrap_summary(per_seed_discrimination[key], value if value is not None else np.nan)
    return {'label': LABEL, 'bounds': bounds, 'intervals': intervals, 'fixed_gates': fixed,
            'descriptive_discrimination': descriptives, 'per_seed_rank_points': points,
            'family_names': names, 'family_size': len(names), 'one_sided_error': error,
            'bootstrap_draws': draws, 'discrimination_draws': draws_discrimination,
            'bootstrap_seed': seed, 'bootstrap_batch_size': batch_size,
            'training_seed_count': ns, 'independent_frame_count': nf,
            'timing_model_index': timing_model_index,
            'replicates': {**raw, **descriptions, **per_seed_discrimination}}


def timing_bound_table(total_times, receiver_times, *, control, draws=20000,
                       seed=71992, error=0.05/76):
    """Paired bootstrap of means over timing frames, not unpaired kernel samples.

    Each method vector is a frame mean over its retained repeated measurements.
    All methods must use the same ordered frames. Times include complete declared
    boundaries. A fixed single checkpoint is a hardware benchmark, not training-
    seed performance replication. Receiver allowance is applied by s1_decision.
    """
    variants = ('g3_hard', 'g3_soft')
    methods = list(dict.fromkeys([*variants, 'g2', control]))
    arrays = {name: _vector(total_times[name], name) for name in methods}
    receivers = {name: _vector(receiver_times[name], name) for name in methods}
    lengths = {len(v) for v in [*arrays.values(), *receivers.values()]}
    if len(lengths) != 1 or any(np.any(v <= 0) for v in arrays.values()):
        raise ValueError('positive paired frame timings required')
    n = next(iter(lengths))
    rng = np.random.default_rng(seed)
    raw = {f'{v}/{alias}/cost/{metric}': np.empty(draws)
           for v in variants for alias in ('g2', 'control')
           for metric in ('latency_ratio', 'receiver_difference')}
    for start in range(0, draws, 256):
        stop = min(draws, start+256)
        ix = rng.integers(n, size=(stop-start, n))
        for v in variants:
            for alias, c in [('g2', 'g2'), ('control', control)]:
                raw[f'{v}/{alias}/cost/latency_ratio'][start:stop] = arrays[v][ix].mean(axis=1)/arrays[c][ix].mean(axis=1)
                raw[f'{v}/{alias}/cost/receiver_difference'][start:stop] = (receivers[v][ix]-receivers[c][ix]).mean(axis=1)
    bounds, intervals = {}, {}
    for v in variants:
        for alias, c in [('g2', 'g2'), ('control', control)]:
            for metric, estimate in [('latency_ratio', arrays[v].mean()/arrays[c].mean()),
                                     ('receiver_difference', np.mean(receivers[v]-receivers[c]))]:
                name = f'{v}/{alias}/cost/{metric}'
                report = bootstrap_summary(raw[name], estimate, error)
                intervals[name] = report
                bounds[name] = {'estimate': report['estimate'], 'lower': report['lower_one_sided'],
                                'upper': report['upper_one_sided'], 'status': report['status']}
    return {'bounds': bounds, 'intervals': intervals, 'replicates': raw,
            'bootstrap_seed': seed, 'timing_frames': n, 'draws': draws}


def s1_decision(bounds, certificate_pass, *, variants=('g3_hard', 'g3_soft'),
                training_seeds=(0, 1, 2), alpha=0.05, useful_min=0.05,
                coverage_min=0.05, receiver_allowance=0.0,
                definitive_failure=False):
    """Evaluate frozen inequalities. Missing/underpowered evidence is PARTIAL.

    bounds is keyed by decision_inequality_names; each record provides lower and
    upper ONE-SIDED multiplicity-adjusted bounds. Cost-risk entries already
    subtract m_J/min(A3,Ac); cost-useful entries are raw U3-Uc. Certification is
    per variant/seed and includes >=200 assessment accepts. Failure can only be
    asserted by a separately justified, documented scientific diagnostic review;
    mere non-rejection of success never proves FAILURE.
    """
    required = decision_inequality_names(variants, training_seeds)
    unknown = set(bounds)-set(required)
    if unknown:
        raise ValueError(f'unregistered inequalities: {sorted(unknown)}')
    def bound(name, side, target, strict=False):
        value = bounds.get(name, {}).get(side)
        if value is None or not np.isfinite(value):
            return False
        return bool((value < target if strict else value <= target) if side == 'upper'
                    else (value > target if strict else value >= target))
    m_j, m_u = alpha/10, useful_min/5
    outcomes = {}
    for v in variants:
        evidence = all(certificate_pass.get(f'{v}/seed{s}', False)
                       and bound(f'{v}/seed{s}/coverage', 'lower', coverage_min, True)
                       and bound(f'{v}/seed{s}/useful', 'lower', useful_min, True)
                       for s in training_seeds)
        regret = bound(f'{v}/regret', 'upper', 0)
        def comparable(c):
            return all(bound(f'{v}/{c}/risk/{a}', 'upper', m_j/a)
                       and bound(f'{v}/{c}/useful/{a}', 'lower', -m_u) for a in COVERAGES)
        def cheaper(c):
            prefix = f'{v}/{c}/cost/'
            return (bound(prefix+'latency_ratio', 'upper', 0.90, True)
                    and bound(prefix+'receiver_difference', 'upper', receiver_allowance)
                    and bound(prefix+'coverage_upper', 'upper', 0.02)
                    and bound(prefix+'coverage_lower', 'lower', -0.02)
                    and bound(prefix+'risk', 'upper', 0)
                    and bound(prefix+'useful', 'lower', -m_u))
        distinct = bound(f'{v}/brier', 'upper', 0, True) or cheaper('control')
        common = evidence and regret and comparable('g2') and comparable('control') and distinct
        reliability = (bound(f'{v}/g1/risk/0.5', 'upper', 0, True)
                       and all(bound(f'{v}/g1/risk/{a}', 'upper', m_j/a) for a in (0.25, 0.75)))
        cost = comparable('g1') and cheaper('g2')
        outcomes[v] = {'useful_operating_evidence': evidence, 'regret': regret,
                       'g2_comparable': comparable('g2'), 'control_comparable': comparable('control'),
                       'distinct_information_or_cost': distinct, 'common': common,
                       'reliability_route': common and reliability, 'cost_route': common and cost,
                       'success': common and (reliability or cost)}
    winners = [v for v, o in outcomes.items() if o['success']]
    decision = 'SUCCESS' if winners else 'FAILURE' if definitive_failure else 'PARTIAL SUCCESS'
    return {'decision': decision, 'variants': outcomes, 'successful_variants': winners,
            'missing_inequalities': sorted(set(required)-set(bounds)),
            'family_size': len(required), 'one_sided_error': 0.05/len(required)}
