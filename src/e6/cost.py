"""Actual matched-coverage receiver timing; cutoff never uses harm labels."""
import time
import numpy as np
import psutil
import torch
from src.e6.common import OUT,LABEL,COVERAGES,write_json,read_json
from src.e6.holdout import masks,holdout_seed,geometry_mask
from src.e6.scores import scores
from src.e6.data import simulate
from src.e6.training import load_model
from src.e5.model import CandidateOperator
from src.e5.cost import Deployment,finish_bits,branch_replay,timed,measure_resolution
from src.e5.scores import pilot_pair_score
from src.e5.analysis import rank_acceptance,gate_metrics
from src.estimators.noise import estimate_variance
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import demodulate
from src.utils.cost import CostLedger,count

def matched_cutoff(score,keys,coverage):
    score=np.asarray(score,float);keys=np.asarray(keys,float)
    if score.shape!=keys.shape or score.ndim!=1 or not len(score) or not np.isfinite(score).all() or not np.isfinite(keys).all():raise ValueError('Finite paired scores/keys')
    if not 0<coverage<1:raise ValueError('Interior coverage required')
    if len(np.unique(keys))!=len(keys):raise ValueError('Tie keys must be unique')
    m=int(np.floor(len(score)*coverage+1e-12))
    if m<1 or m>=len(score):raise ValueError('Unattainable interior coverage')
    order=np.lexsort((keys,score));i=order[m-1]
    return float(score[i]),float(keys[i])

def accepts(score,key,cutoff):
    if cutoff is None:return False
    if not np.isfinite(score) or not np.isfinite(key):return False
    return (float(score),float(key))<=tuple(cutoff)

def component(ledger,name,fn):
    if ledger is None:return fn()
    with ledger.time(name):return fn()

def validate(method,obs,alphabet,op,cutoff,key,g2,seed,ledger=None):
    if type(obs) is not ReceiverObservation:raise TypeError('Restricted receiver observation')
    p=obs.pilots();detail={};score=None
    if ledger is not None:
        count(ledger,'transmitted_pilot_res',int(obs.grid.pilot_mask.sum()))
        count(ledger,'transmitted_pilot_energy',float(np.sum(np.abs(obs.grid.pilot_grid[obs.grid.pilot_mask])**2)))
    if method=='g2':
        fit,hold=component(ledger,'holdout_seconds',lambda:masks(obs.grid,g2['fraction'],g2['policy'],seed))
        if ledger is not None:
            count(ledger,'fitting_pilot_res',int(fit.sum()));count(ledger,'heldout_pilot_res',int(hold.sum()))
        pb,pn,pv=component(ledger,'partial_candidate_seconds',lambda:op.pair(obs.pilots(fit),ledger))
        score=component(ledger,'score_seconds',lambda:pilot_pair_score(obs,pb,pn,pv,hold,aggregation=g2['aggregation'],ledger=ledger))
        accepted=accepts(score,key,cutoff)
        b=component(ledger,'full_base_seconds',lambda:op.base(p,ledger));v=estimate_variance(p,op.variance_floor,ledger)
        candidate=component(ledger,'full_neural_seconds',lambda:op.neural(p,b,v,ledger)) if accepted else b
        count(ledger,'refit_passes');count(ledger,'validation_passes')
    else:
        if ledger is not None:count(ledger,'fitting_pilot_res',int(obs.grid.pilot_mask.sum()))
        b=component(ledger,'full_base_seconds',lambda:op.base(p,ledger));v=estimate_variance(p,op.variance_floor,ledger)
        if method=='baseline':accepted=False;candidate=b
        elif method=='g1':
            score=component(ledger,'score_seconds',lambda:scores(obs,b,b,alphabet,v,names=('g1',),ledger=ledger))['g1']
            accepted=accepts(score,key,cutoff)
            candidate=component(ledger,'full_neural_seconds',lambda:op.neural(p,b,v,ledger)) if accepted else b
        else:
            n=component(ledger,'full_neural_seconds',lambda:op.neural(p,b,v,ledger))
            if method=='g0':accepted=True;candidate=n
            else:
                val,detail=component(ledger,'score_seconds',lambda:scores(obs,b,n,alphabet,v,names=(method,),ledger=ledger,return_details=True))
                score=val[method];accepted=accepts(score,key,cutoff);candidate=n if accepted else b
                count(ledger,'validation_passes')
    count(ledger,'gate_decisions',int(method not in ('baseline','g0')))
    return Deployment(candidate,accepted,score,v,detail)

def receiver(method,raw,grid,alphabet,op,cutoff,key,g2,seed,ledger=None):
    obs=component(ledger,'cp_fft_seconds',lambda:ReceiverObservation(grid,demodulate(raw,grid.nfft,grid.symbols,grid.cp)))
    count(ledger,'cp_fft_passes')
    selected=validate(method,obs,alphabet,op,cutoff,key,g2,seed,ledger)
    return finish_bits(obs,alphabet,selected,ledger=ledger),selected

def benchmark(guard,stats,width,g2,control,data,partition,binding):
    path=OUT/'timing'/f'{partition}.json'
    if path.exists():
        result=read_json(path)
        if result['binding']!=binding:raise RuntimeError('Stale timing')
        return result
    count_frames=64 if partition=='V-select' else 128
    records=guard.records(partition)[:count_frames]
    start=time.perf_counter();model=load_model(width,5101);op=CandidateOperator(stats,model)
    cold=time.perf_counter()-start
    frames=[simulate(guard,r) for r in records]
    methods=list(dict.fromkeys(('g1','g2','g3_hard','g3_soft','g3_anchor','pilot','weighted_mean','evm','llr',control)))
    keys=data['tie_key'][:count_frames]
    ds=dict(data);ds['g2']=data[g2['name']] if g2['name'] in data else data['g2']
    cuts={(m,a):matched_cutoff(ds[m][:count_frames],keys,a) for m in methods for a in COVERAGES}
    # Fixed public-geometry setup charged separately, including cache construction.
    geometry_mask.cache_clear();t=time.perf_counter()
    if g2['policy']=='geometry':masks(frames[0][1].grid,g2['fraction'],g2['policy'],holdout_seed(records[0]))
    setup=time.perf_counter()-t
    resolution=measure_resolution(op,frames[0][1],frames[0][0].constellation,frames[0][3],repeats=30,warmups=10)
    rng=np.random.default_rng(61992);order=[(m,a) for m in methods for a in COVERAGES]
    for j in range(10):
        cfg,obs,lab,raw=frames[j%count_frames];r=records[j%count_frames]
        for z in rng.permutation(len(order)):
            m,a=order[z];receiver(m,raw,obs.grid,cfg.constellation,op,cuts[m,a],keys[j%count_frames],g2,holdout_seed(r))
    rows=[];ledgers=[];rss=psutil.Process().memory_info().rss
    for repeat in range(3):
        for i,(cfg,obs,lab,raw) in enumerate(frames):
            r=records[i]
            for z in rng.permutation(len(order)):
                m,a=order[z];cut=cuts[m,a]
                cv,deployed=timed('cpu',lambda:validate(m,obs,cfg.constellation,op,cut,keys[i],g2,holdout_seed(r)))
                cr,(bits,dr)=timed('cpu',lambda:receiver(m,raw,obs.grid,cfg.constellation,op,cut,keys[i],g2,holdout_seed(r)))
                cp,replayed=timed('cpu',lambda:branch_replay(raw,obs.grid,cfg.constellation,op,dr.accepted))
                if deployed.accepted!=dr.accepted or not np.array_equal(bits,replayed):raise AssertionError('Timing receiver disagreement')
                expected=accepts(ds[m][i],keys[i],cut)
                if expected!=dr.accepted:raise AssertionError('Cached and timed gate differ')
                rows.append(dict(frame_id=r.realization_id,index=i,method=m,coverage=a,repeat=repeat,accepted=dr.accepted,
                                 validation_seconds=cv,receiver_seconds=cr,replay_seconds=cp,increment_seconds=cr-cp))
            rss=max(rss,psutil.Process().memory_info().rss)
        print('timing',partition,'repeat',repeat+1,flush=True)
    for i,(cfg,obs,lab,raw) in enumerate(frames):
        for m,a in order:
            ledger=CostLedger();_,d=receiver(m,raw,obs.grid,cfg.constellation,op,cuts[m,a],keys[i],g2,holdout_seed(records[i]),ledger)
            ledgers.append(dict(frame_id=records[i].realization_id,index=i,method=m,coverage=a,accepted=d.accepted,**ledger.export()))
    summaries=[]
    for m,a in order:
        subset=[r for r in rows if r['method']==m and r['coverage']==a]
        acceptance=np.asarray([r['accepted'] for r in subset if r['repeat']==0])
        if int(acceptance.sum())!=int(count_frames*a):raise AssertionError('Unmatched realized coverage')
        metric=gate_metrics(data['harm'][:count_frames],data['useful'][:count_frames],acceptance,data['base_loss'][:count_frames],data['neural_loss'][:count_frames])
        s=dict(method=m,target_coverage=a,**metric)
        for b in ('validation','receiver','increment','replay'):
            vals=[r[b+'_seconds'] for r in subset]
            s[b+'_mean']=float(np.mean(vals));s[b+'_p95']=float(np.quantile(vals,.95))
        summaries.append(s)
    result=dict(label=LABEL,binding=binding,partition=partition,device='cpu',threads=1,batch_size=1,warmups=10,repetitions=3,model_seed=5101,
                dtype='complex128 receiver/float32 neural',frames=count_frames,rows=rows,ledgers=ledgers,summaries=summaries,
                cutoffs=[dict(method=m,coverage=a,cutoff=cuts[m,a]) for m,a in order],cold_model_seconds=cold,geometry_setup_seconds=setup,
                setup_amortization_frames=384,peak_sampled_rss_bytes=rss,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                statistics_bytes=stats.mean.nbytes+stats.covariance.nbytes,allocated_vram=0,reserved_vram=0,resolution=resolution,
                cutoff_scope='label-free timing-panel rank cutoffs; empirical diagnostic, no deployable certificate')
    write_json(path,result);return result
