"""E8 raw-derived analysis. Not executed on real final outcomes in E7."""
import numpy as np
from scipy.special import expit
from scipy.stats import beta
from src.e7.common import ROOT,FINAL,SEEDS,METHODS,read_json,write_json
from src.e7.protocol import config,calibrate_arrays,event_interval
from src.e5.analysis import (gate_metrics,rank_acceptance,rank_curve,discrimination,
    partial_aurc,weighted_rank_points)


def bounds(values,error=.05):
    a=np.asarray(values,float);valid=np.isfinite(a);bad=int((~valid).sum());a=a[valid]
    if bad>.05*len(valid) or not len(a):
        return dict(status='INSUFFICIENT_EVIDENCE',undefined_replicas=bad)
    return dict(status='OK',undefined_replicas=bad,lower=float(np.quantile(a,error/2)),
        upper=float(np.quantile(a,1-error/2)),lower_one_sided=float(np.quantile(a,error)),
        upper_one_sided=float(np.quantile(a,1-error)))


def mean_bootstrap(values,draws=2000,error=.05,strata=None):
    v=np.asarray(values,float)
    if v.ndim==1:v=v[None,:]
    ns,n=v.shape;rng=np.random.default_rng(71991);reps=[]
    groups=[np.arange(n)] if strata is None else [np.flatnonzero(strata==k) for k in np.unique(strata)]
    for off in range(0,draws,32):
        size=min(32,draws-off);w=np.zeros((size,n),np.int32)
        for g in groups:w[:,g]=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=size)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=size)/ns
        reps.extend(np.einsum('bs,sb->b',sw,v@w.T/n))
    return dict(estimate=float(v.mean()),seed_sd=float(v.mean(1).std(ddof=1)) if ns>1 else None,
                per_seed=v.mean(1).tolist(),**bounds(reps,error))


def rank_difference(data,a,b,coverage,draws=50000,error=.05/28):
    n=len(data[0]['harm']);ns=len(data);rng=np.random.default_rng(71991);rr=[];uu=[]
    orders={m:[np.lexsort((d['tie_key'],d[m])) for d in data] for m in (a,b)}
    for off in range(0,draws,16):
        size=min(16,draws-off);w=rng.multinomial(n,np.full(n,1/n),size=size)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=size)/ns;vals={}
        for m in (a,b):
            ps=[weighted_rank_points(w,order,d['harm'],d['useful'],[coverage]) for order,d in zip(orders[m],data)]
            vals[m]=tuple(np.einsum('bs,sb->b',sw,np.asarray([p[j][:,0] for p in ps])) for j in (0,1))
        rr.extend(vals[a][0]-vals[b][0]);uu.extend(vals[a][1]-vals[b][1])
    points={m:[gate_metrics(d['harm'],d['useful'],rank_acceptance(d[m],coverage,d['tie_key'])) for d in data] for m in (a,b)}
    return dict(coverage=coverage,risk=dict(estimate=float(np.mean([x['conditional_harm'] for x in points[a]])-np.mean([x['conditional_harm'] for x in points[b]])),**bounds(rr,error)),
        useful=dict(estimate=float(np.mean([x['joint_useful'] for x in points[a]])-np.mean([x['joint_useful'] for x in points[b]])),**bounds(uu,error)))


def collect(run,experiment,seed,split='test'):
    directory=ROOT/'outputs/e8'/('calibration' if split=='calibration' else experiment)/f's{seed}'
    paths=sorted(directory.glob('*.npz'));records=run.records(split,experiment if split=='test' else None)
    if not paths:raise RuntimeError('Missing final experiment: '+experiment)
    parts=[run.load(p) for p in paths];d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if d['frame_id'].tolist()!=[r['realization_id'] for r in records]:raise RuntimeError('Incomplete or reordered experiment: '+experiment)
    return d


def accepted(d,method,certificate):
    if method=='g0':return np.ones(len(d['harm']),bool)
    if method=='oracle_gate':return d['neural_loss']<d['base_loss']
    if method.endswith('_uncalibrated'):return d[method[:-13]]<=0
    if method=='weighted_evm_identity':method='g3_hard'
    tau=certificate[method]['threshold']
    return np.zeros(len(d['harm']),bool) if tau is None else d[method]<=tau


def summary(d,method,certificate):
    gated=method in METHODS or method in ('g0','oracle_gate','weighted_evm_identity') or method.endswith('_uncalibrated')
    result={}
    if gated:
        g=accepted(d,method,certificate)
        result.update(gate_metrics(d['harm'],d['useful'],g,d['base_loss'],d['neural_loss']))
        n=len(g);m=int(g.sum());h=int((g&d['harm']).sum());u=int((g&d['useful']).sum())
        result['intervals']=dict(RJ=event_interval(h,n),A=event_interval(m,n),U_joint=event_interval(u,n),RC=event_interval(h,m))
        result['RC_status']='UNDEFINED_NO_ACCEPTANCE' if not m else ('INSUFFICIENT_ACCEPTED_FRAMES' if m<200 else 'OK')
        result['accepted_gain']=float(np.mean((d['base_loss']-d['neural_loss'])[g])) if m else None
        loss=np.where(g,d['neural_loss'],d['base_loss']);errors=np.where(g,d['neural_bit_errors'],d['base_bit_errors'])
        serr=np.where(g,d['neural_symbol_errors'],d['base_symbol_errors'])
    else:
        loss=d[method+'_loss'];errors=d[method+'_bit_errors'];serr=d[method+'_symbol_errors']
    mean=float(np.mean(loss));result.update(mean_linear_nmse=mean,nmse_db_of_linear_mean=float(10*np.log10(mean)) if mean>0 else None,
        nmse_p95=float(np.quantile(loss,.95)),bit_errors=int(errors.sum()),bits=int(d['bits'].sum()),
        ber=float(errors.sum()/d['bits'].sum()),ser=float(serr.sum()/d['symbols'].sum()),
        nmse_ci=mean_bootstrap(loss),ber_frame_ci=mean_bootstrap(errors/d['bits']))
    if not errors.any():
        result['zero_ber_upper']=event_interval(0,len(errors))['upper_one_sided'];result['zero_ber_bound_unit']='error-containing frame probability bounds BER'
    elif np.ptp(errors/d['bits'])==0:
        radius=float(np.sqrt(np.log(2/.05)/(2*len(errors))));p=result['ber']
        result['ber_conservative_hoeffding']=[max(0.,p-radius),min(1.,p+radius)]
    return result


def analyze(run):
    base=ROOT/'outputs/e8';cert=run.calibrate()['certificates'];matched=[collect(run,'P0-GATES-01',s) for s in SEEDS]
    output=base/'analysis';output.mkdir(parents=True,exist_ok=True)
    all_rows=[];aggregate_rows=[];failure_rows=[]
    for ex in run.experiments:
        if ex['kind'] not in ('iid','fixed_cells') or ex['id']=='P0-COST-01':continue
        data=matched if ex['id']=='P0-GATES-01' else [collect(run,ex['id'],s) for s in SEEDS]
        for block in np.unique(data[0]['block_id']):
            mask=data[0]['block_id']==block
            ds=[{k:v[mask] for k,v in d.items()} for d in data]
            for seed,d in zip(SEEDS,ds):
                g=accepted(d,'g3_hard',cert[str(seed)])
                for category,select in [('harmful_accept',g&d['harm']),('useful_reject',~g&d['useful'])]:
                    ids=d['frame_id'][select]
                    failure_rows.append(dict(experiment_id=ex['id'],block_id=str(block),seed=seed,
                        category=category,frame_id=str(min(ids)) if len(ids) else None,
                        status='SELECTED_FIRST_LEXICOGRAPHIC_ID' if len(ids) else 'NA_EMPTY_CATEGORY'))
            for method in ex['methods']:
                rows=[];losses=[];bers=[];rjs=[]
                for s,d in zip(SEEDS,ds):
                    row=summary(d,method,cert[str(s)]);rows.append(row)
                    all_rows.append(dict(experiment_id=ex['id'],block_id=str(block),method=method,seed=s,**row))
                    gated=method in METHODS or method in ('g0','oracle_gate','weighted_evm_identity')
                    if gated:
                        g=accepted(d,method,cert[str(s)]);losses.append(np.where(g,d['neural_loss'],d['base_loss']))
                        bers.append(np.where(g,d['neural_bit_errors'],d['base_bit_errors'])/d['bits']);rjs.append(g&d['harm'])
                    else:losses.append(d[method+'_loss']);bers.append(d[method+'_bit_errors']/d['bits'])
                agg=dict(experiment_id=ex['id'],block_id=str(block),method=method,
                    NMSE=mean_bootstrap(losses),BER=mean_bootstrap(bers),seeds=list(SEEDS),
                    NMSE_db_of_linear_mean=float(10*np.log10(np.mean(losses))) if np.mean(losses)>0 else None)
                if rjs:agg['RJ']=mean_bootstrap(rjs)
                if not np.any(bers):agg['zero_BER_simultaneous_seed_upper']=event_interval(0,len(ds[0]['harm']),.05/5)['upper_one_sided']
                aggregate_rows.append(agg)
        print('analyzed',ex['id'],flush=True)
    write_json(output/'metrics_per_seed.json',dict(**run.meta,rows=all_rows))
    write_json(output/'metrics_aggregate.json',dict(**run.meta,rows=aggregate_rows))
    write_json(output/'failure_catalog.json',dict(**run.meta,rows=failure_rows))
    decisions={};error=.05/28;reg=[];curves=[]
    for s,d in zip(SEEDS,matched):
        g=accepted(d,'g3_hard',cert[str(s)]);reg.append(g*(d['neural_loss']-d['base_loss']))
        for key,values in [('A',g),('U',g&d['useful'])]:
            count=int(values.sum());low=0. if count==0 else float(beta.ppf(error,count,len(g)-count+1))
            decisions[f'USE-s{s}-{key}']=dict(lower=low,margin=.05,passed=low>.05)
        for m in METHODS:
            curves.append(dict(seed=s,method=m,rank_points=rank_curve(d[m],d['harm'],d['useful'],d['tie_key'],(.25,.5,.75),d['base_loss'],d['neural_loss']),
                discrimination=discrimination(d['harm'],d[m]),pAURC=partial_aurc(d[m],d['harm'],d['tie_key'],.05,.95)))
    rr=mean_bootstrap(reg,50000,error);decisions['REGRET-BASE']=dict(**rr,passed=rr.get('upper_one_sided',np.inf)<=0)
    for m,margin in [('g1',0.),('g2',.01),('weighted_mean',0.)]:
        r=rank_difference(matched,'g3_hard',m,.5)
        upper=r['risk'].get('upper_one_sided',np.inf);lower=r['useful'].get('lower_one_sided',-np.inf)
        decisions['RANK-'+m+'-R']=dict(**r['risk'],margin=margin,passed=upper<=margin if m=='g2' else upper<margin)
        decisions['RANK-'+m+'-U']=dict(**r['useful'],margin=-.01,passed=lower>=-.01)
    models=read_json(FINAL/'incremental_models.json')['models'];probs={}
    for model in models:
        j=list(SEEDS).index(model['seed']);d=matched[j];name=model['name']
        x=np.column_stack([d[k] for k in model['features']]);z=(x-np.array(model['mean']))/np.array(model['scale'])
        probs[(name,model['seed'])]=expit(z@np.array(model['coefficient'])+model['intercept'])
    for label in ('ordinary','pilot','expanded'):
        diff=[(probs[(label+'_hard',s)]-d['harm'])**2-(probs[(label,s)]-d['harm'])**2 for s,d in zip(SEEDS,matched)]
        r=mean_bootstrap(diff,50000,error);decisions['INFO-'+label]=dict(**r,passed=r.get('upper_one_sided',np.inf)<0)
    timing=read_json(base/'P0-COST-01/timing.json');cs=config('cost.yaml')['comparison']
    for a in (.25,.5):
        rank=rank_difference([matched[0]],'g3_hard','g2',a)
        decisions[f'COST{a}-R']=dict(**rank['risk'],margin=.005/a,passed=rank['risk'].get('upper_one_sided',np.inf)<=.005/a)
        decisions[f'COST{a}-U']=dict(**rank['useful'],margin=-.01,passed=rank['useful'].get('lower_one_sided',-np.inf)>=-.01)
        tr=[r for r in timing['rows'] if r['mode']==str(a) and r['method'] in ('g3_hard','g2')]
        ids=sorted({r['frame_id'] for r in tr});arr={}
        for m in ('g3_hard','g2'):
            for metric in ('C_val','C_rx'):
                arr[(m,metric)]=np.array([np.mean([r[metric] for r in tr if r['method']==m and r['frame_id']==i]) for i in ids])
        rng=np.random.default_rng(71991);reps=[]
        for off in range(0,50000,64):
            size=min(64,50000-off);w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=size)/len(ids)
            reps.extend((w@arr[('g3_hard','C_val')])/(w@arr[('g2','C_val')]))
        ratio=bounds(reps,error);rx=mean_bootstrap(arr[('g3_hard','C_rx')]-arr[('g2','C_rx')],50000,error)
        decisions[f'COST{a}-VAL']=dict(**ratio,margin=.90,passed=ratio.get('upper_one_sided',np.inf)<.90)
        decisions[f'COST{a}-RX']=dict(**rx,margin=cs['receiver_jitter_seconds'],passed=rx.get('upper_one_sided',np.inf)<=cs['receiver_jitter_seconds'])
    if len(decisions)!=28:raise RuntimeError('Decision family changed')
    write_json(output/'registered_decisions.json',dict(**run.meta,error_family=.05,inequalities=28,decisions=decisions))
    write_json(output/'rank_curves.json',dict(**run.meta,rows=curves))
    ablations=[];sensitivity=[];grids=read_json(FINAL/'threshold_grids.json')['grids']
    for seed,d in zip(SEEDS,matched):
        for m in list(METHODS)+['g0','weighted_evm_identity','g2_uncalibrated','g3_hard_uncalibrated','g3_soft_uncalibrated']:
            ablations.append(dict(experiment_id='P1-ABLATION-01',seed=seed,method=m,**summary(d,m,cert[str(seed)])))
        cd=collect(run,'CAL-MATCHED-01',seed,'calibration')
        for n in (2048,4096,8192):
            for m in ('g2','g3_hard','g3_soft','weighted_mean','pilot'):
                c=calibrate_arrays(cd[m][:n],cd['harm'][:n],cd['useful'][:n],grids[str(seed)][m],run.calibration)
                sensitivity.append(dict(experiment_id='P1-CALSIZE-01',seed=seed,n_cal=n,method=m,certificate=c,
                    guarantee='primary only at8192; extra prefixes empirical',test=summary(d,m,{m:c})))
    write_json(output/'ablations.json',dict(**run.meta,rows=ablations))
    write_json(output/'calibration_size.json',dict(**run.meta,rows=sensitivity))
    phase=[]
    for seed in SEEDS:
        d=run.load(base/'P0-PHASE-01'/f's{seed}.npz')
        for mod in ('QPSK','16QAM'):
            for intervention in ('all_active','data_only'):
                for degree in (-90,90):
                    mask=(d['modulation']==mod)&(d['intervention']==intervention)&(d['degrees']==degree)
                    for m in ('g3_hard','g3_soft','g3_anchor','pilot'):
                        h=d['harm'][mask];g=d[m+'_accepted'][mask];n=int(mask.sum())
                        phase.append(dict(seed=seed,modulation=mod,intervention=intervention,degrees=degree,method=m,frames=n,
                            unchanged_fraction=float(np.mean(np.abs(d[m][mask]-d[m+'_original'][mask])<=1e-10)),
                            harmful_candidate_acceptance=float(g[h].mean()) if h.any() else None,
                            harmful_candidates=int(h.sum()),accepted=int(g.sum()),
                            uncalibrated_acceptance=float(d[m+'_uncalibrated'][mask].mean())))
    write_json(output/'phase_limitation.json',dict(**run.meta,experiment_id='P0-PHASE-01',rows=phase))
    write_json(output/'completed.json',dict(**run.meta,status='ANALYSIS_GENERATED_REQUIRES_E8_AUDIT',
        scientific_success_not_assumed=True,claim_scope='FINAL_CLAIM_SCOPE.md'))
