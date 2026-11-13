"""Public-geometry G2 masks; no observed values or labels."""
from functools import lru_cache
import numpy as np
from src.ofdm.holdout import pilot_holdout
from src.estimators.practical import delay_basis
from src.utils.random import rng_for

POLICIES = ('random','interleaved','block','geometry')
HOLDOUTS = tuple((f,p) for f in (.25,.5) for p in POLICIES)

@lru_cache(maxsize=24)
def geometry_mask(grid, fraction):
    if fraction not in (.25,.5): raise ValueError('Unregistered fraction')
    fit=grid.pilot_mask.copy(); hold=np.zeros(grid.shape,bool)
    basis=delay_basis(grid,17); gram=basis[fit].conj().T@basis[fit]+1e-6*np.eye(17)
    # Removal preserves determinant as much as possible; exact candidate logdet.
    for ell in range(grid.symbols):
        bins=grid.active_bins[fit[grid.active_bins,ell]]
        count=min(len(bins)-1,max(1,int(np.floor(fraction*len(bins)))))
        for _ in range(count):
            choices=[]
            for k in bins:
                if not fit[k,ell]: continue
                a=basis[k,ell]; candidate=gram-np.outer(a.conj(),a)
                sign,logdet=np.linalg.slogdet(candidate)
                choices.append((float(logdet.real) if sign.real>0 else -np.inf,int(k),candidate))
            # choices preserve signed-frequency order for exact numerical ties.
            _,k,gram=max(choices,key=lambda x:x[0]);fit[k,ell]=False;hold[k,ell]=True
    fit.setflags(write=False);hold.setflags(write=False)
    return fit,hold

def masks(grid, fraction, policy, seed):
    if policy=='geometry': return geometry_mask(grid,float(fraction))
    return pilot_holdout(grid,float(fraction),policy,np.random.default_rng(seed))

def holdout_seed(record):
    return record.seeds()['holdout']

def name(fraction,policy,aggregation): return f'g2_{int(fraction*100)}_{policy}_{aggregation}'

def conditioning(grid, fit):
    a=delay_basis(grid,17)[fit]*grid.pilot_grid[fit,None]
    gram=a.conj().T@a
    gaps=[]
    for ell in range(grid.symbols):
        locations=np.asarray(grid.active_signed)[fit[grid.active_bins,ell]]
        gaps.append(float(np.diff(locations).max()) if len(locations)>1 else float('inf'))
    return dict(fit_pilots=int(fit.sum()),heldout_pilots=int(grid.pilot_mask.sum()-fit.sum()),
                rank=int(np.linalg.matrix_rank(a)),condition=float(np.linalg.cond(a)),
                regularized_condition=float(np.linalg.cond(gram+1e-6*np.eye(17))),max_interpolation_gap=max(gaps))
