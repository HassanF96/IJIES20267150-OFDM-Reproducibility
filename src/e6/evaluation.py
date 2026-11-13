"""Retained paired validation evidence; hidden labels join after receiver work."""
import time
import numpy as np
from src.e6.common import OUT, LABEL, SEEDS, sha, write_json, bound_npz, checked_npz
from src.e6.data import simulate
from src.e6.training import load_model
from src.e6.holdout import HOLDOUTS, masks, holdout_seed, name, conditioning
from src.e6.scores import scores, CONTROLS, NAMES
from src.e5.model import CandidateOperator
from src.e5.scores import pilot_pair_score, diagnostic_score_pair, transform_candidate, perturb_candidate
from src.e5.analysis import labels_from_losses, rank_curve
from src.metrics.errors import frame_loss
from src.utils.random import seed_for

def raw_path(seed,partition,role=None):return OUT/'raw'/f's{seed}_{partition}{"_"+role if role else ""}.npz'

def load_raw(seed,partition,role=None,binding=None):return checked_npz(raw_path(seed,partition,role),binding)

def evaluate(guard,stats,width,seed,partition,binding,selection=None,role=None):
    path=raw_path(seed,partition,role);records=guard.records(partition,role)
    if path.exists():
        d=checked_npz(path,binding)
        if d['frame_id'].tolist()!=[r.realization_id for r in records]:raise PermissionError('Raw ID mismatch')
        for r in records:guard.check(r)
        return d
    operator=CandidateOperator(stats,load_model(width,seed))
    choices=HOLDOUTS if selection is None else [(selection['fraction'],selection['policy'])]
    shard_dir=OUT/'shards'/path.stem;shard_dir.mkdir(parents=True,exist_ok=True)
    pred_dir=OUT/'predictions'/path.stem;pred_dir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for off in range(0,len(records),128):
        shard=shard_dir/f'{off:06d}.npz';subset=records[off:off+128]
        if shard.exists():
            d=checked_npz(shard,binding)
            if d['frame_id'].tolist()!=[r.realization_id for r in subset]:raise PermissionError('Shard IDs')
            checked_npz(pred_dir/shard.name,binding)
            for r in subset:guard.check(r)
            continue
        rows=[];bases=[];neurals=[]
        for r in subset:
            cfg,obs,lab,_=simulate(guard,r);b,n,v=operator.pair(obs.pilots())
            row,detail=scores(obs,b,n,cfg.constellation,v,return_details=True)
            for f,p in choices:
                fit,hold=masks(obs.grid,f,p,holdout_seed(r));pb,pn,pv=operator.pair(obs.pilots(fit))
                for agg in ('huber','mean'):row[name(f,p,agg)]=pilot_pair_score(obs,pb,pn,pv,hold,aggregation=agg)
                key=f'{int(f*100)}_{p}'
                row['partial_base_'+key]=frame_loss(pb,lab.h_true,obs.grid.active_mask)
                row['partial_neural_'+key]=frame_loss(pn,lab.h_true,obs.grid.active_mask)
                if selection is None:
                    row.update({'g2_'+k+'_'+key:value for k,value in conditioning(obs.grid,fit).items()})
                else:
                    row['g2']=row[name(f,p,selection['aggregation'])]
                    row['partial_base_loss']=row['partial_base_'+key];row['partial_neural_loss']=row['partial_neural_'+key]
            # Inference is complete; truth used only from this point in full-pair reporting.
            lb=frame_loss(b,lab.h_true,obs.grid.active_mask);ln=frame_loss(n,lab.h_true,obs.grid.active_mask)
            harm,use=labels_from_losses(np.array([lb]),np.array([ln]));meta=guard.by_id[r.realization_id]
            row.update(base_loss=lb,neural_loss=ln,harm=bool(harm[0]),useful=bool(use[0]),frame_id=r.realization_id,
                       tie_key=float(np.random.default_rng(seed_for(61990,'e6-ties',r.realization_id,'split')).random()),
                       family=meta['family'],modulation=meta['modulation'],snr_db=float(meta['snr_db']),stride=int(meta['stride']),
                       burst_cell=int(meta['burst_cell']),inr_db=float(meta['inr_db']),duration=int(meta['duration']),probability=float(meta['probability']),
                       cell=int(meta['cell']),role=meta['role'],burst_present=bool(lab.burst_schedule),
                       burst_occupancy=float(np.mean(lab.burst_variance_time.reshape(obs.grid.symbols,-1)[:,obs.grid.cp:]>0)),
                       bits=len(lab.transmitted_bits),estimated_variance=v)
            for c in ('base','neural'):
                row[c+'_bit_errors']=int(np.count_nonzero(detail[c]['bits']!=lab.transmitted_bits))
                row[c+'_ser']=float(np.mean(np.abs(detail[c]['symbols']-lab.transmitted_grid[obs.grid.data_mask])>1e-10))
            rows.append(row);bases.append(b);neurals.append(n)
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        bound_npz(pred_dir/shard.name,dict(frame_id=arrays['frame_id'],base=np.asarray(bases),neural=np.asarray(neurals)),dict(binding=binding,seed=seed))
        bound_npz(shard,arrays,dict(binding=binding,seed=seed))
        print('score',seed,partition,role or '',off+len(subset),'/',len(records),flush=True)
    parts=[checked_npz(p,binding) for p in sorted(shard_dir.glob('*.npz'))]
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if d['frame_id'].tolist()!=[r.realization_id for r in records]:raise RuntimeError('Pair order')
    bound_npz(path,d,dict(binding=binding,width=width,seed=seed,partition=partition,role=role,seconds=time.perf_counter()-start))
    return d

def select_g2_control(binding):
    data=[load_raw(s,'V-select',binding=binding) for s in SEEDS]
    def select(candidates):
        rows=[]
        for candidate in candidates:
            points=[rank_curve(d[candidate],d['harm'],d['useful'],d['tie_key'],[.5],d['base_loss'],d['neural_loss'])[0] for d in data]
            row=dict(name=candidate,conditional_harm=float(np.mean([x['conditional_harm'] for x in points])),joint_useful=float(np.mean([x['joint_useful'] for x in points])))
            if candidate.startswith('g2_'):
                _,fraction,policy,agg=candidate.split('_');key=f'{fraction}_{policy}'
                row['partial_loss']=float(np.mean([d['partial_neural_'+key].mean() for d in data]))
            rows.append(row)
        rows.sort(key=lambda x:(x['conditional_harm'],-x['joint_useful'],x.get('partial_loss',0),x['name']))
        return rows[0]['name'],rows
    chosen,g2=select([name(f,p,a) for f,p in HOLDOUTS for a in ('huber','mean')])
    control,controls=select(CONTROLS)
    _,f,p,a=chosen.split('_')
    result=dict(label=LABEL,binding=binding,g2=dict(name=chosen,fraction=int(f)/100,policy=p,aggregation=a),control=control,all_g2=g2,all_controls=controls)
    write_json(OUT/'g2_control_selection.json',result)
    return result

def challenges(guard,stats,width,seed,role,binding):
    path=OUT/'raw'/f'challenges_s{seed}_{role}.npz'
    if path.exists():return checked_npz(path,binding)
    records=guard.records('V-panel',role);records=[r for r in records if r.index%16==(0 if role=='selection' else 8)]
    op=CandidateOperator(stats,load_model(width,seed));rows=[]
    specs=[('phase',x) for x in (-90,90,-30,-15,-5,5,15,30)]+[('scale',x) for x in (.5,.75,1.25,2.,3.)]+[('temperature',10.),('temperature',float('inf')),('data_perturb',.5)]
    for r in records:
        cfg,obs,lab,_=simulate(guard,r);b,n,v=op.pair(obs.pilots())
        original=scores(obs,b,n,cfg.constellation,v,names=('g3_hard','g3_soft','g3_anchor','pilot','weighted_mean'))
        for kind,value in specs:
            nn=n;temperature=1.
            if kind=='phase':nn=transform_candidate(n,phase_degrees=value)
            elif kind=='scale':nn=transform_candidate(n,amplitude=value)
            elif kind=='temperature':temperature=value
            else:nn=perturb_candidate(n,obs.grid.active_mask,value,seed_for(61993,'e6-challenge',r.realization_id,'split'),zero_mask=obs.grid.pilot_mask)
            vals=diagnostic_score_pair(obs,b,nn,cfg.constellation,v,temperature=temperature,names=('g3_hard','g3_soft','g3_anchor','pilot','g3_hard_mean'))
            lb=frame_loss(b,lab.h_true,obs.grid.active_mask);ln=frame_loss(nn,lab.h_true,obs.grid.active_mask)
            e,u=labels_from_losses([lb],[ln]);meta=guard.by_id[r.realization_id]
            row=dict(frame_id=r.realization_id,seed=seed,kind=kind,value=str(value),harm=bool(e[0]),useful=bool(u[0]),base_loss=lb,neural_loss=ln,
                     family=meta['family'],modulation=meta['modulation'],snr_db=float(meta['snr_db']),stride=int(meta['stride']),cell=int(meta['cell']))
            row.update(vals)
            for method in ('g3_hard','g3_soft','g3_anchor','pilot'):row['original_'+method]=original[method]
            rows.append(row)
    result={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    bound_npz(path,result,dict(binding=binding,seed=seed,role=role,independent_source_frames=len(records),interventions_per_frame=len(specs)))
    print('challenges',seed,role,len(rows),flush=True)
    return result
