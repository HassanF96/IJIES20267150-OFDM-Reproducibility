"""Frozen definitions usable on supplied arrays; no observation materialization."""
import hashlib
import json
import numpy as np
import yaml
from scipy.stats import beta, binom
from src.e7.common import ROOT, FINAL, canonical, sha, read_json


def config(name):
    return yaml.safe_load((FINAL/name).read_text(encoding='utf8'))


def verify_bundle():
    bundle = read_json(ROOT/'E7_PROTOCOL_HASHES.json')
    expected = hashlib.sha256(canonical(bundle['files'])).hexdigest()
    if expected != bundle['protocol_sha256']:
        raise RuntimeError('Protocol root hash mismatch')
    for name, digest in bundle['files'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Frozen file changed: '+name)
    manifest = yaml.safe_load((ROOT/'E7_FREEZE_MANIFEST.yaml').read_text(encoding='utf8'))
    for group in ('source_files', 'evidence_files'):
        for name, digest in manifest[group].items():
            if sha(ROOT/name) != digest:
                raise RuntimeError('Source/evidence custody mismatch: '+name)
    return bundle['protocol_sha256']


def calibrate_arrays(score, harm, useful, grid, settings, *, singleton=False):
    """Exact fixed-sequence procedure, independent of file/split loading.

    Caller enforces frozen split identity. Can be tested on hand-built arrays
    without opening any real calibration/test outcomes.
    """
    t = np.asarray(score, float); e = np.asarray(harm); u = np.asarray(useful)
    if t.ndim != 1 or not len(t) or not t.shape == e.shape == u.shape:
        raise ValueError('Paired nonempty vectors required')
    if not np.isfinite(t).all() or not np.isin(e,(0,1)).all() or not np.isin(u,(0,1)).all() or np.any(np.logical_and(e,u)):
        raise ValueError('Finite scores and disjoint binary events required')
    e=e.astype(bool);u=u.astype(bool)
    values = [None] if singleton else list(grid)
    if not singleton and (not values or not np.isfinite(values).all() or np.any(np.diff(values)<=0)):
        raise ValueError('Strictly increasing finite frozen grid required')
    result=[];selected=None;selected_index=None
    for index,tau in enumerate(values):
        g=np.ones(len(t),bool) if singleton else t<=tau
        h=int(np.count_nonzero(g&e));b=int(np.count_nonzero(g&u));m=int(g.sum())
        p=float(binom.cdf(h,len(t),settings['alpha']))
        rejected=p<=settings['epsilon_per_certificate']
        screen=(m>=settings['n_min'] and m/len(t)>=settings['A_min'] and b/len(t)>=settings['U_joint_min'])
        result.append(dict(index=index,threshold=tau,n=len(t),accepted=m,harmful=h,useful=b,
                           p_value=p,rejected_null=rejected,useful_screen=screen))
        if not rejected:
            break
        if screen:
            selected=tau;selected_index=index
    passed=selected_index is not None
    return dict(threshold=selected,selected_index=selected_index,certified=passed,
                singleton=singleton,rows=result,
                status='CERTIFIED_USEFUL_POLICY' if passed else 'REJECT_ALL_NO_USEFUL_CERTIFICATE')


def event_interval(x,n,error=.05):
    if not isinstance(x,(int,np.integer)) or not isinstance(n,(int,np.integer)) or n<0 or x<0 or x>n:
        raise ValueError('Valid integer event counts required')
    if n==0:
        return dict(estimate=None,lower=None,upper=None,upper_one_sided=None,status='UNDEFINED_NO_ACCEPTANCE')
    return dict(estimate=x/n,lower=0. if x==0 else float(beta.ppf(error/2,x,n-x+1)),
                upper=1. if x==n else float(beta.ppf(1-error/2,x+1,n-x)),
                upper_one_sided=1. if x==n else float(beta.ppf(1-error,x+1,n-x)),status='OK')


def design_condition(block, record, distributions):
    """Metadata-only fixed-cell lookup; IID conditions are drawn ONLY in E8."""
    if 'cell' in block:
        return block['cell']
    from src.utils.random import rng_for
    r=rng_for(record['master_seed'],record['namespace'],record['realization_id'],'split')
    d=distributions[block['distribution']]
    return dict(family=d['family'],modulation=d['modulation'],stride=d['stride'],
                snr_db=int(r.choice(d['snr_db_uniform'])),
                burst=distributions['bursts'][int(r.integers(len(d['burst_uniform'])))])
