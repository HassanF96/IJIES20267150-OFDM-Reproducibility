"""Receiver-only EVM controls; exact existing hard/soft/anchor definitions."""
import numpy as np
from src.e5.scores import score_pair as prior_scores, ALL_SCORES as PRIOR_NAMES, data_components, KAPPA
from src.ofdm.discrepancy import radial_huber_mean
from src.utils.cost import count

EXTRA_NAMES=('evm_robust','evm_robust_neural','weighted_mean','weighted_mean_neural',
             'weighted_normalized','weighted_normalized_neural','weighted_robust_neural')
NAMES=tuple(PRIOR_NAMES)+EXTRA_NAMES
CONTROLS=('g1','evm','evm_neural','evm_robust','evm_robust_neural','weighted_mean','weighted_mean_neural',
          'weighted_normalized','weighted_normalized_neural','weighted_robust_neural','llr','llr_neural','pilot','pilot_mean')

def scores(obs,base,neural,alphabet,variance,*,names=None,ledger=None,return_details=False):
    names=tuple(NAMES if names is None else names)
    if not names or set(names)-set(NAMES):raise ValueError('Unregistered score')
    prior=[n for n in names if n in PRIOR_NAMES];extra=[n for n in names if n in EXTRA_NAMES]
    out,details={},{}
    if prior:out,details=prior_scores(obs,base,neural,alphabet,variance,names=prior,ledger=ledger,return_details=True)
    if extra:
        single=all(n.endswith('_neural') for n in extra);summ={}
        for c,hgrid in (('base',base),('neural',neural)):
            if c=='base' and single:continue
            h=hgrid[obs.grid.data_mask];y=obs.y[obs.grid.data_mask]
            det=details.get(c)
            if det is None:
                det=data_components(y,h,alphabet,variance,ledger=ledger);details[c]=det
            summary={}
            if any(n.startswith('evm_robust') for n in extra):
                equalized=h.conj()*y/np.maximum(np.abs(h)**2,1e-12)
                summary['evm_robust']=radial_huber_mean(np.abs(equalized-det['symbols'])**2/alphabet.energy,KAPPA)
                count(ledger,'robust_summaries')
            if any(n.startswith('weighted_mean') for n in extra):summary['weighted_mean']=float(det['hard_energy'].mean());count(ledger,'mean_summaries')
            if any(n.startswith('weighted_normalized') for n in extra):
                summary['weighted_normalized']=float(np.mean(np.abs(y-h*det['symbols'])**2)/max(float(np.mean(np.abs(h)**2))*alphabet.energy,1e-12));count(ledger,'mean_summaries')
            if 'weighted_robust_neural' in extra and c=='neural':summary['weighted_robust']=radial_huber_mean(det['hard_energy'],KAPPA);count(ledger,'robust_summaries')
            summ[c]=summary
        for n in extra:
            if n.endswith('_neural'):out[n]=summ['neural'][n[:-7]]
            else:out[n]=summ['neural'][n]-summ['base'][n]
        count(ledger,'gate_score_calls',len(extra))
    if not all(np.isfinite(v) for v in out.values()):raise ArithmeticError('Nonfinite E6 score')
    return (out,details) if return_details else out
