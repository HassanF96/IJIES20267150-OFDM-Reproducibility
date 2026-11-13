"""Paired scalar/candidate evidence, with labels joined after receiver output."""
import json
import time
import numpy as np
from src.e5.common import OUT,LABEL,write_json,sha
from src.e5.data import simulate_e5
from src.e5.model import CandidateOperator
from src.e5.training import load_model,mask_for,HOLDOUTS
from src.e5.scores import score_pair,pilot_pair_score
from src.e5.analysis import labels_from_losses,select_design_control,threshold_grid,ltt_development,rank_curve
from src.metrics.errors import frame_loss
from src.estimators.practical import design
from src.utils.random import seed_for

CONTROLS=('g1','evm','evm_neural','llr','llr_neural','pilot')
METHODS=('g1','g2','g3_hard','g3_soft','evm','evm_neural','llr','llr_neural','pilot','g0')

def g2_name(fraction,policy,aggregation):return f'g2_{int(100*fraction)}_{policy}_{aggregation}'

def load_raw(seed,partition):
    with np.load(OUT/'raw'/f'seed_{seed}_{partition}.npz') as z:return {k:z[k] for k in z.files}

def model_sanity(guard,settings,statistics):
    rows=[]
    for seed in settings['seeds']['training']:
        operator=CandidateOperator(statistics,load_model(seed))
        losses=[];relative=[]
        for record in guard.records('V-design'):
            _,obs,labels,_=simulate_e5(guard,record)
            b,n,_=operator.pair(obs.pilots())
            if b.shape!=n.shape or not np.isfinite(n).all():raise ArithmeticError('Sanity finite/shape')
            losses.append((frame_loss(b,labels.h_true,obs.grid.active_mask),frame_loss(n,labels.h_true,obs.grid.active_mask)))
            relative.append(float(np.linalg.norm(n-b)/max(np.linalg.norm(b),1e-12)))
        losses=np.asarray(losses)
        row={'label':LABEL,'seed':seed,'n':len(losses),'improve_count':int(np.sum(losses[:,1]<losses[:,0])),
             'worsen_count':int(np.sum(losses[:,1]>losses[:,0])),'mean_base_loss':float(losses[:,0].mean()),
             'mean_neural_loss':float(losses[:,1].mean()),'median_loss_ratio':float(np.median(losses[:,1]/losses[:,0])),
             'residual_relative_quantiles':np.quantile(relative,[0,.5,.95,1]).tolist()}
        row['pass']=bool(row['improve_count']>0 and row['worsen_count']>0 and row['median_loss_ratio']<10 and max(relative)>1e-7)
        rows.append(row)
    write_json(OUT/'sanity.json',{'label':LABEL,'results':rows,'status':'PASS' if all(x['pass'] for x in rows) else 'FAIL'})
    if not all(x['pass'] for x in rows):raise RuntimeError('Initial model sanity failed; diagnostic required before scores')
    return rows

def evaluate_partition(guard,settings,statistics,seed,partition,selection=None):
    path=OUT/'raw'/f'seed_{seed}_{partition}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    records=guard.records(partition)
    if path.exists():
        for r in records:guard.check(r)
        cached=load_raw(seed,partition)
        if cached['frame_id'].tolist()!=[r.realization_id for r in records]:raise PermissionError('Cached result ID mismatch')
        saved=json.loads((OUT/'run_manifests'/f'{seed}_{partition}.json').read_text())
        if sha(path)!=saved['raw_sha256']:raise RuntimeError('Cached result checksum mismatch')
        return cached
    operator=CandidateOperator(statistics,load_model(seed,'cpu'))
    policies=HOLDOUTS if partition=='V-design' else [(selection['g2']['fraction'],selection['g2']['policy'])]
    shards=OUT/'raw/shards'/f'seed_{seed}_{partition}';shards.mkdir(parents=True,exist_ok=True)
    candidates=OUT/'predictions'/f'seed_{seed}_{partition}';candidates.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for off in range(0,len(records),256):
        shard=shards/f'{off:06d}.npz'
        subset=records[off:off+256]
        if shard.exists():
            for r in subset:guard.check(r)
            continue
        rows=[];bs=[];ns=[]
        for r in subset:
            config,obs,labels,_=simulate_e5(guard,r)
            b,n,v=operator.pair(obs.pilots())
            values,details=score_pair(obs,b,n,config.constellation,v,return_details=True)
            row=dict(values)
            for fraction,policy in policies:
                fit,hold=mask_for(obs.grid,r,fraction,policy)
                p=obs.pilots(fit);pb,pn,pv=operator.pair(p)
                for agg in ('huber','mean'):
                    key=g2_name(fraction,policy,agg)
                    row[key]=pilot_pair_score(obs,pb,pn,pv,hold,aggregation=agg)
                if partition=='V-design':
                    _,a,_=design(p,statistics.support)
                    row[f'condition_{int(100*fraction)}_{policy}']=float(np.linalg.cond(a))
                    row[f'partial_base_{int(100*fraction)}_{policy}']=frame_loss(pb,labels.h_true,obs.grid.active_mask)
                    row[f'partial_neural_{int(100*fraction)}_{policy}']=frame_loss(pn,labels.h_true,obs.grid.active_mask)
                else:
                    row['g2']=row[selection['g2']['name']]
                    row['partial_base_loss']=frame_loss(pb,labels.h_true,obs.grid.active_mask)
                    row['partial_neural_loss']=frame_loss(pn,labels.h_true,obs.grid.active_mask)
            # Oracle-only evidence joins AFTER all practical score/candidate calls.
            row.update(base_loss=frame_loss(b,labels.h_true,obs.grid.active_mask),neural_loss=frame_loss(n,labels.h_true,obs.grid.active_mask))
            e,u=labels_from_losses(np.asarray([row['base_loss']]),np.asarray([row['neural_loss']]))
            row.update(harm=bool(e[0]),useful=bool(u[0]),g0=0.,frame_id=r.realization_id,
                       tie_key=float(np.random.default_rng(seed_for(settings['seeds']['tie'],'e5-ties',r.realization_id,'split')).random()),
                       family=labels.channel_family,modulation=config.constellation.name,snr_db=labels.nominal_snr_db,
                       burst_cell=int(guard.by_id[r.realization_id]['burst_cell']),burst_present=bool(labels.burst_schedule),
                       burst_occupancy=float(np.mean(labels.burst_variance_time.reshape(obs.grid.symbols,-1)[:,obs.grid.cp:]>0)),
                       bits=len(labels.transmitted_bits),estimated_variance=v)
            for c in ('base','neural'):
                row[c+'_bit_errors']=int(np.count_nonzero(details[c]['bits']!=labels.transmitted_bits))
                row[c+'_ser']=float(np.mean(np.abs(details[c]['symbols']-labels.transmitted_grid[obs.grid.data_mask])>1e-10))
            rows.append(row);bs.append(b);ns.append(n)
        # Complex128 original candidates retained locally. Scalar evidence stays compact for Git.
        np.savez_compressed(candidates/f'{off:06d}.npz',label=LABEL,frame_id=np.asarray([r['frame_id'] for r in rows]),base=np.asarray(bs),neural=np.asarray(ns))
        np.savez_compressed(shard,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        print('score',seed,partition,min(off+256,len(records)),'/',len(records),flush=True)
    parts=[]
    for shard in sorted(shards.glob('*.npz')):
        with np.load(shard) as z:parts.append({k:z[k] for k in z.files})
    result={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if list(result['frame_id'])!=[r.realization_id for r in records]:raise AssertionError('Candidate pairing/ID order changed')
    np.savez_compressed(path,**result)
    write_json(OUT/'run_manifests'/f'{seed}_{partition}.json',{'label':LABEL,'seed':seed,'partition':partition,'frames':len(records),
               'elapsed_seconds':time.perf_counter()-start,'raw_sha256':sha(path),'candidate_files':len(parts),'dtype':'complex128 candidates, float64 scores/labels'})
    return result

def design_selection(settings):
    datasets=[load_raw(s,'V-design') for s in settings['seeds']['training']]
    harm=np.stack([d['harm'] for d in datasets]);use=np.stack([d['useful'] for d in datasets])
    keys=datasets[0]['tie_key']
    primary=(datasets[0]['modulation']=='QPSK')&(datasets[0]['snr_db']>=0)&(datasets[0]['snr_db']<=20)
    def choose(names):
        records=[]
        for name in names:
            points=[rank_curve(d[name][primary],d['harm'][primary],d['useful'][primary],keys[primary],[.5])[0] for d in datasets]
            records.append({'name':name,'conditional_harm':float(np.mean([r['conditional_harm'] for r in points])),
                            'joint_useful':float(np.mean([r['joint_useful'] for r in points]))})
        records.sort(key=lambda x:(x['conditional_harm'],-x['joint_useful'],x['name']))
        return records[0]['name'],records
    name,g2results=choose([g2_name(f,p,a) for f,p in HOLDOUTS for a in ('huber','mean')])
    control,controls=choose(CONTROLS)
    _,fraction,policy,aggregation=name.split('_')
    result={'label':LABEL,'g2':{'name':name,'fraction':int(fraction)/100,'policy':policy,'aggregation':aggregation},
            'control':control,'all_g2_attempts':g2results,'all_control_attempts':controls,
            'selection_data':'V-design primary-domain subset; mean across all three training seeds','selection_rule':'min conditional harm@0.5, max usefulness tie, lexical name tie'}
    write_json(OUT/'design_selection.json',result)
    grids={}
    for seed,d in zip(settings['seeds']['training'],datasets):
        primary=(d['modulation']=='QPSK')&(d['snr_db']>=0)&(d['snr_db']<=20)
        d['g2']=d[name]
        for method in METHODS:
            grids.setdefault(method,{})[str(seed)]=threshold_grid(d[method][primary]).tolist()
    write_json(OUT/'design_grids.json',{'label':LABEL,'grids':grids,'source':'V-design primary-domain subset only'})
    return result,grids

def certify_development(settings,grids):
    result={m:{} for m in METHODS}
    for seed in settings['seeds']['training']:
        d=load_raw(seed,'V-cal-dev')
        for method in METHODS:
            result[method][str(seed)]=ltt_development(d[method],d['harm'],d['useful'],grids[method][str(seed)],
                         partition='V-cal-dev',epsilon=settings['statistics']['epsilon_total']/settings['statistics']['certificate_count'])
    write_json(OUT/'development_certificates.json',{'label':LABEL,'split':'validation','partition':'V-cal-dev','certificates':result,
               'scope':'Development demonstration only; final calibration and test unopened'})
    return result
