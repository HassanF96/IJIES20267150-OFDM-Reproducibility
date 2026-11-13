"""Frozen E8 CPU cost experiment; no calls from E7 validation."""
import threading
import time
import numpy as np
import psutil
from src.e7.common import ROOT,write_json,read_json
from src.e7.protocol import config
from src.e6.cost import validate,receiver,matched_cutoff
from src.e5.cost import Deployment,finish_bits,branch_replay,timed
from src.e6.scores import scores
from src.e6.holdout import masks,geometry_mask
from src.e5.scores import pilot_pair_score
from src.estimators.noise import estimate_variance
from src.estimators.practical import ls_interpolate,regularized_delay
from src.e6.comparators import helena_estimate,block_huber
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import demodulate
from src.utils.random import rng_for
from src.utils.cost import CostLedger,count


def run_cost(run):
    path=ROOT/'outputs/e8/P0-COST-01/timing.json'
    if path.exists():
        data=read_json(path)
        if data['protocol_sha256']!=run.protocol:raise RuntimeError('Timing protocol mismatch')
        return data
    spec=config('cost.yaml');seed=spec['residual_seed']
    begin=time.perf_counter();run.current=None;run.models(seed);cold=time.perf_counter()-begin
    geometry_mask.cache_clear();begin=time.perf_counter();masks(run.grid(8),.5,'geometry',0);setup=time.perf_counter()-begin
    records=run.records('test','P0-COST-01')
    frames=[run.frame('test',r) for r in records]
    methods=next(e for e in run.experiments if e['id']=='P0-COST-01')['methods']
    gates=[m for m in methods if m.startswith('g') and m!='g0']+['weighted_mean','pilot']
    keys=np.asarray([rng_for(2026100803,'e8-ties',r['realization_id'],'split').random() for r in records])
    values={m:[] for m in gates};g2=dict(fraction=.5,policy='geometry',aggregation='huber')
    for _,obs,_,_,a in frames:
        b,n,v=run.op.pair(obs.pilots());s=scores(obs,b,n,a,v,names=tuple(m for m in gates if m!='g2'))
        fit,hold=masks(obs.grid,.5,'geometry',0);pb,pn,pv=run.op.pair(obs.pilots(fit))
        s['g2']=pilot_pair_score(obs,pb,pn,pv,hold,aggregation='huber')
        for m in gates:values[m].append(s[m])
    cert=read_json(ROOT/'outputs/e8/calibration/certificates.json')['certificates'][str(seed)]
    cases=[]
    for method in methods:
        if method in gates:
            tau=cert[method]['threshold'];cases.append((method,'calibrated',None if tau is None else (tau,1.)))
            for a in (.25,.5,.75):cases.append((method,str(a),matched_cutoff(values[method],keys,a)))
        else:cases.append((method,'ungated',None))
    def deploy(m,obs,alpha,cut,key,ledger=None):
        if m in ('helena','ls','delay','block_huber'):
            p=obs.pilots();v=estimate_variance(p,ledger=ledger)
            if m=='helena':h=helena_estimate(p,run.helena,ledger=ledger)
            elif m=='ls':h=ls_interpolate(p,ledger)
            elif m=='delay':h=regularized_delay(p,17,ledger=ledger)
            else:h=block_huber(p,run.stats,ledger=ledger)
            return Deployment(h,False,None,v,{})
        return validate('baseline' if m=='base' else m,obs,alpha,run.op,cut,key,g2,0,ledger)
    def full(m,raw,grid,alpha,cut,key,ledger=None):
        obs=ReceiverObservation(grid,demodulate(raw,grid.nfft,grid.symbols,grid.cp))
        count(ledger,'cp_fft_passes')
        selected=deploy(m,obs,alpha,cut,key,ledger)
        return finish_bits(obs,alpha,selected,ledger=ledger),selected
    for m,mode,cut in cases:
        for i in range(spec['warmup_per_method_mode']):
            _,obs,_,raw,a=frames[i%len(frames)]
            sel=deploy(m,obs,a,cut,keys[i%len(keys)]);full(m,raw,obs.grid,a,cut,keys[i%len(keys)])
            if m in gates or m in ('base','g0'):
                branch_replay(raw,obs.grid,a,run.op,sel.accepted)
    rss=[];stop=threading.Event()
    def sample():
        while not stop.is_set():
            rss.append(psutil.Process().memory_info().rss);stop.wait(.01)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    rows=[];rng=np.random.default_rng(71992);ledgers={}
    try:
        for rep in range(spec['timed_repetitions']):
            for i,(_,obs,_,raw,a) in enumerate(frames):
                for j in rng.permutation(len(cases)):
                    m,mode,cut=cases[j]
                    val,sel=timed('cpu',lambda:deploy(m,obs,a,cut,keys[i]))
                    rx,result=timed('cpu',lambda:full(m,raw,obs.grid,a,cut,keys[i]))
                    if not np.isfinite([val,rx]).all():raise ArithmeticError('Nonfinite timing')
                    inc=None
                    if m in gates or m in ('base','g0'):
                        replay,_=timed('cpu',lambda:branch_replay(raw,obs.grid,a,run.op,sel.accepted))
                        inc=rx-replay
                    rows.append(dict(frame_id=records[i]['realization_id'],repeat=rep,method=m,mode=mode,
                        C_val=val,C_rx=rx,C_inc=inc,accepted=bool(sel.accepted)))
        for m,mode,cut in cases:
            for i,(_,obs,_,raw,a) in enumerate(frames):
                ledger=CostLedger();full(m,raw,obs.grid,a,cut,keys[i],ledger)
                ledgers[f'{m}/{mode}/{i}']=ledger.export()
    finally:
        stop.set();thread.join()
    result=dict(**run.meta,experiment_id='P0-COST-01',rows=rows,ledgers=ledgers,
        setup_seconds=setup,cold_model_seconds=cold,setup_amortization_frames=384,
        statistics_load_seconds=run.statistics_load_seconds,
        statistics_bytes=run.stats.mean.nbytes+run.stats.covariance.nbytes,
        peak_sampled_rss_bytes=max(rss),residual_parameter_bytes=44738*4,helena_parameter_bytes=97714*4,
        allocated_vram=0,reserved_vram=0,
        cutoffs=[dict(method=m,mode=mode,value=cut) for m,mode,cut in cases],specification=spec)
    write_json(path,result);return result
