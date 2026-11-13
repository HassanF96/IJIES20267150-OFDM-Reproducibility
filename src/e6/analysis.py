"""Prespecified interpretable development comparisons, never calibration."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.e6.common import OUT,LABEL,SEEDS,COVERAGES,write_json,read_json,bound_npz
from src.e6.evaluation import load_raw
from src.e6.scores import CONTROLS,NAMES
from src.e5.analysis import (rank_acceptance,rank_curve,gate_metrics,discrimination,partial_aurc,
                            weighted_rank_points,bootstrap_summary,labels_from_losses,partial_full_mismatch,binomial_sample_plan)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,lineterminator='\n')

def with_g2(d,selection):
    d=dict(d);g=selection['g2'];d['g2']=d[g['name']] if g['name'] in d else d['g2']
    key=f'{int(g["fraction"]*100)}_{g["policy"]}'
    for c in ('base','neural'):
        if 'partial_'+c+'_loss' not in d:d['partial_'+c+'_loss']=d['partial_'+c+'_'+key]
    return d

def rank_comparison(data,a,b,coverages=COVERAGES,draws=5000,error=.05,strata=None):
    """Crossed seed/frame bootstrap, ranks reconstructed using multiplicities."""
    n=len(data[0]['harm']);ns=len(data);rng=np.random.default_rng(61991)
    points={m:[rank_curve(d[m],d['harm'],d['useful'],d['tie_key'],coverages) for d in data] for m in (a,b)}
    estimate_r=np.mean([[x['conditional_harm'] for x in row] for row in points[a]],axis=0)-np.mean([[x['conditional_harm'] for x in row] for row in points[b]],axis=0)
    estimate_u=np.mean([[x['joint_useful'] for x in row] for row in points[a]],axis=0)-np.mean([[x['joint_useful'] for x in row] for row in points[b]],axis=0)
    orders={m:[np.lexsort((d['tie_key'],d[m])) for d in data] for m in (a,b)}
    reps_r=[];reps_u=[]
    groups=[np.arange(n)] if strata is None else [np.flatnonzero(strata==x) for x in np.unique(strata)]
    for off in range(0,draws,32):
        size=min(32,draws-off);weights=np.zeros((size,n),np.int32)
        for group in groups:weights[:,group]=rng.multinomial(len(group),np.full(len(group),1/len(group)),size=size)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=size)/ns
        vals={}
        for m in (a,b):
            rr=[];uu=[]
            for j,d in enumerate(data):
                r,u=weighted_rank_points(weights,orders[m][j],d['harm'],d['useful'],coverages);rr.append(r);uu.append(u)
            vals[m]=(np.einsum('bs,sbc->bc',sw,np.asarray(rr)),np.einsum('bs,sbc->bc',sw,np.asarray(uu)))
        reps_r.append(vals[a][0]-vals[b][0]);reps_u.append(vals[a][1]-vals[b][1])
    rr=np.concatenate(reps_r);uu=np.concatenate(reps_u)
    return [dict(coverage=a0,risk=bootstrap_summary(rr[:,i],estimate_r[i],error),useful=bootstrap_summary(uu[:,i],estimate_u[i],error)) for i,a0 in enumerate(coverages)]

def mean_difference(values,draws=5000,error=.05):
    v=np.asarray(values,float);ns,n=v.shape;rng=np.random.default_rng(61991);result=[]
    for off in range(0,draws,64):
        m=min(64,draws-off);w=rng.multinomial(n,np.full(n,1/n),size=m)/n
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=m)/ns
        result.extend(np.einsum('bs,sb->b',sw,v@w.T))
    return bootstrap_summary(result,float(v.mean()),error)

def group_masks(d):
    return {'all':np.ones(len(d['harm']),bool),'low_snr':d['snr_db']<=0,'16QAM':d['modulation']=='16QAM',
            'QPSK':d['modulation']=='QPSK','actual_burst':d['burst_present'],'A':d['family']=='generic_A','B':d['family']=='generic_B',
            **{f'pilot_1_{k}':d['stride']==k for k in (4,8,16)},
            **{f'snr_{x}':d['snr_db']==x for x in (-5,0,5,10,15,20,25)}}

def subset(data,mask):return [{k:v[mask] for k,v in d.items()} for d in data]

def point(data,method,coverage=.5):
    rows=[rank_curve(d[method],d['harm'],d['useful'],d['tie_key'],[coverage],d['base_loss'],d['neural_loss'])[0] for d in data]
    return {k:float(np.mean([r[k] for r in rows])) for k in ('conditional_harm','joint_harm','joint_useful','baseline_regret','oracle_regret','coverage','accepted_count','harm_prevalence')}

def challenge_rates(challenge_sets,design_sets):
    rows=[]
    for seed,d,select in zip(SEEDS,challenge_sets,design_sets):
        for mod in ('QPSK','16QAM'):
            for kind in ('quarter_turn','small_phase','scale','temperature','data_perturb'):
                mask=d['modulation']==mod
                if kind=='quarter_turn':mask&=(d['kind']=='phase')&np.isin(d['value'],('-90','90'))
                elif kind=='small_phase':mask&=(d['kind']=='phase')&~np.isin(d['value'],('-90','90'))
                else:mask&=d['kind']==kind
                for method in ('g3_hard','g3_soft','g3_anchor','pilot'):
                    threshold=float(np.quantile(select[method],.5));e=d['harm'][mask];g=d[method][mask]<=threshold
                    metric=gate_metrics(e,d['useful'][mask],g,d['base_loss'][mask],d['neural_loss'][mask])
                    rows.append(dict(seed=seed,modulation=mod,kind=kind,method=method,threshold=threshold,
                                     harmful_candidate_acceptance=float(g[e].mean()) if e.any() else None,
                                     score_change_mean=float(np.mean(d[method][mask]-d['original_'+method][mask])),
                                     unchanged_fraction=float(np.mean(np.abs(d[method][mask]-d['original_'+method][mask])<=1e-10)),**metric))
    return rows

def choose_variant(selection,timing,binding):
    data=[with_g2(load_raw(s,'V-select',binding=binding),selection) for s in SEEDS]
    panel=[with_g2(load_raw(s,'V-panel','selection',binding),selection) for s in SEEDS]
    ch=[load_raw_challenge(s,'selection',binding) for s in SEEDS]
    rates=challenge_rates(ch,data);write_csv(OUT/'analysis/challenges_selection.csv',rates)
    comparisons={v:rank_comparison(data,v,'g3_hard') for v in ('g3_anchor','g3_soft')}
    checks={};groups={}
    for v in ('g3_anchor','g3_soft'):
        ni=all(x['risk']['upper_one_sided']<=.005/x['coverage'] and x['useful']['lower_one_sided']>=-.01 for x in comparisons[v])
        group=[]
        for label in ('low_snr','16QAM','actual_burst','B'):
            ds=subset(panel,group_masks(panel[0])[label]);vp=point(ds,v);hp=point(ds,'g3_hard')
            enough=all(len(d['harm'])//2>=32 and len(np.unique(d['harm']))==2 for d in ds)
            group.append(dict(group=label,candidate=vp,hard=hp,adequate=enough,
                              pass_margins=enough and vp['conditional_harm']-hp['conditional_harm']<=.01 and vp['joint_useful']-hp['joint_useful']>=-.01 and vp['joint_useful']>=.05 and vp['baseline_regret']<=hp['baseline_regret']))
        groups[v]=group
        tr=[]
        for a in COVERAGES:
            vr=next(r for r in timing['summaries'] if r['method']==v and r['target_coverage']==a)
            hr=next(r for r in timing['summaries'] if r['method']=='g3_hard' and r['target_coverage']==a)
            tr.append(dict(coverage=a,validation_ratio=vr['validation_mean']/hr['validation_mean'],receiver_ratio=vr['receiver_mean']/hr['receiver_mean']))
        phase=[]
        for mod in ('QPSK','16QAM'):
            vv=np.mean([r['harmful_candidate_acceptance'] for r in rates if r['method']==v and r['modulation']==mod and r['kind']=='quarter_turn'])
            hh=np.mean([r['harmful_candidate_acceptance'] for r in rates if r['method']=='g3_hard' and r['modulation']==mod and r['kind']=='quarter_turn'])
            phase.append(dict(modulation=mod,candidate=float(vv),hard=float(hh),half_or_less=bool(vv<=.5*hh),not_worse=bool(vv<=hh)))
        vp=point(data,v);pp=point(data,'pilot')
        distinct=(vp['conditional_harm']<pp['conditional_harm'] and vp['joint_useful']>=pp['joint_useful']-.01) or (vp['joint_useful']>pp['joint_useful'] and vp['conditional_harm']<=pp['conditional_harm']+.01)
        checks[v]=dict(primary_ni=ni,group_checks=group,phase=phase,timing=tr,pilot_distinct=distinct)
    ac=checks['g3_anchor'];anchor=bool(ac['primary_ni'] and all(x['pass_margins'] for x in ac['group_checks']) and all(x['half_or_less'] for x in ac['phase']) and all(x['validation_ratio']<=1.20 and x['receiver_ratio']<=1.20 for x in ac['timing']) and ac['pilot_distinct'])
    sc=checks['g3_soft']
    low_improve=all(x['adequate'] and x['candidate']['conditional_harm']<=.75*x['hard']['conditional_harm'] for x in sc['group_checks'] if x['group'] in ('low_snr','16QAM'))
    soft=bool(sc['primary_ni'] and low_improve and all(x['not_worse'] for x in sc['phase']) and all(x['validation_ratio']<=1.25 and x['receiver_ratio']<=1.25 for x in sc['timing']))
    # If anchor is retained, soft must also satisfy NI to that retained family.
    if anchor and soft:
        cross=rank_comparison(data,'g3_soft','g3_anchor');comparisons['soft_vs_anchor']=cross
        soft=all(x['risk']['upper_one_sided']<=.005/x['coverage'] and x['useful']['lower_one_sided']>=-.01 for x in cross)
    primary='g3_soft' if soft else ('g3_anchor' if anchor else 'g3_hard')
    result=dict(label=LABEL,binding=binding,primary=primary,anchor_retained=primary=='g3_anchor',soft_primary=soft,
                checks=checks,comparisons=comparisons,source='V-select and selection panel half only; before V-assess',g4='NOT_ADOPTED')
    write_json(OUT/'variant_selection.json',result);return result

def load_raw_challenge(seed,role,binding):
    from src.e6.common import checked_npz
    return checked_npz(OUT/'raw'/f'challenges_s{seed}_{role}.npz',binding)

ORDINARY=('g1','evm','evm_neural','llr','llr_neural','pilot')
EXPANDED=ORDINARY+('weighted_mean','weighted_mean_neural','weighted_normalized','weighted_normalized_neural','evm_robust','evm_robust_neural','weighted_robust_neural')
FEATURES={'ordinary':ORDINARY,'ordinary_hard':ORDINARY+('g3_hard',),'pilot':('pilot',),'pilot_hard':('pilot','g3_hard'),
          'expanded':EXPANDED,'expanded_hard':EXPANDED+('g3_hard',),
          'identity':('weighted_evm_identity',),'identity_hard':('weighted_evm_identity','g3_hard')}

def fit_logistic(data,names,partition='V-select'):
    if partition!='V-select':raise PermissionError('Regression fitting allowed on V-select only')
    x=np.column_stack([data[k] for k in names]);y=data['harm'].astype(int);cut=int(.7*len(y))
    if not np.isfinite(x).all() or any(len(np.unique(y[s]))<2 for s in (slice(None,cut),slice(cut,None))):raise ValueError('Insufficient design classes/features')
    rows=[]
    for c in (.1,1.,10.):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,tol=1e-8,random_state=61991))
        model.fit(x[:cut],y[:cut]);prob=model.predict_proba(x[cut:])[:,1]
        rows.append((float(np.mean((prob-y[cut:])**2)),c))
    _,chosen=min(rows)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=chosen,max_iter=2000,tol=1e-8,random_state=61991));model.fit(x,y)
    scaler=model.named_steps['standardscaler'];logit=model.named_steps['logisticregression']
    info=dict(features=names,chosen_C=chosen,attempts=[dict(brier=v,C=c) for v,c in rows],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=logit.coef_[0].tolist(),intercept=float(logit.intercept_[0]),fit_frame_ids=data['frame_id'].tolist())
    return model,info

def regression(design,review,binding,panel=None):
    infos=[];predictions={};results=[];panel_predictions={}
    for label,names in FEATURES.items():
        requested_names=names
        if label=='identity_hard':names=('weighted_evm_identity',)
        probs=[];panel_probs=[]
        for seed,ds,rv in zip(SEEDS,design,review):
            model,info=fit_logistic(ds,names);info.update(seed=seed,name=label,requested_features=requested_names,
                duplicate_identity_removed=label=='identity_hard');infos.append(info)
            probs.append(model.predict_proba(np.column_stack([rv[k] for k in names]))[:,1])
            if panel is not None:
                pd=panel[list(SEEDS).index(seed)]
                panel_probs.append(model.predict_proba(np.column_stack([pd[k] for k in names]))[:,1])
        predictions[label]=np.asarray(probs)
        if panel is not None:panel_predictions[label]=np.asarray(panel_probs)
    y=np.stack([d['harm'] for d in review])
    for base,plus in (('ordinary','ordinary_hard'),('pilot','pilot_hard'),('expanded','expanded_hard'),('identity','identity_hard')):
        diff=(predictions[plus]-y)**2-(predictions[base]-y)**2
        report=mean_difference(diff,error=.05/10)
        results.append(dict(base=base,plus=plus,brier_base=float(np.mean((predictions[base]-y)**2)),brier_plus=float(np.mean((predictions[plus]-y)**2)),difference=report))
    write_json(OUT/'analysis/regression.json',dict(label=LABEL,binding=binding,results=results,models=infos,meaning='associative validation-only diagnostic; identity duplicate can change L2 penalty, not information'))
    bound_npz(OUT/'analysis/regression_predictions.npz',dict(harm=y,frame_id=review[0]['frame_id'],**predictions),dict(binding=binding))
    if panel is not None:
        py=np.stack([d['harm'] for d in panel]);rows=[]
        for group,mask in group_masks(panel[0]).items():
            if not mask.any():continue
            for base,plus in (('ordinary','ordinary_hard'),('pilot','pilot_hard'),('expanded','expanded_hard'),('identity','identity_hard')):
                for j,seed in enumerate(SEEDS):
                    b=float(np.mean((panel_predictions[base][j,mask]-py[j,mask])**2));p=float(np.mean((panel_predictions[plus][j,mask]-py[j,mask])**2))
                    rows.append(dict(group=group,seed=seed,base=base,plus=plus,brier_base=b,brier_plus=p,difference=p-b,n=int(mask.sum())))
        write_csv(OUT/'analysis/regression_strata.csv',rows)
        bound_npz(OUT/'analysis/regression_panel_predictions.npz',dict(harm=py,frame_id=panel[0]['frame_id'],**panel_predictions),dict(binding=binding))
    return results

def summaries(data,methods,partition,group='all'):
    rows=[]
    for seed,d in zip(SEEDS,data):
        for method in methods:
            disc=discrimination(d['harm'],d[method]);aurc=partial_aurc(d[method],d['harm'],d['tie_key'],.05,.95)
            for coverage in COVERAGES:
                g=rank_acceptance(d[method],coverage,d['tie_key']);r=gate_metrics(d['harm'],d['useful'],g,d['base_loss'],d['neural_loss'])
                rows.append(dict(partition=partition,group=group,seed=seed,method=method,target_coverage=coverage,
                                 auroc=disc['auroc'],auprc=disc['auprc'],pAURC=aurc,
                                 accepted_mean_gain=float(np.mean((d['base_loss']-d['neural_loss'])[g])) if g.any() else None,
                                 gated_ber=float(np.where(g,d['neural_bit_errors'],d['base_bit_errors']).sum()/d['bits'].sum()),
                                 loss_p95=float(np.quantile(np.where(g,d['neural_loss'],d['base_loss']),.95)),**r))
    return rows

def cost_inference(timing,review,primary,g2name):
    frame_n=timing['frames'];rng=np.random.default_rng(61992);results=[]
    # Three timing claims plus risk/usefulness for each coverage: nine one-sided statements.
    for a in COVERAGES:
        methods=(primary,'g2');vectors={}
        for m in methods:
            rows=[r for r in timing['rows'] if r['method']==m and r['coverage']==a]
            vectors[m]={b:np.asarray([np.mean([r[b+'_seconds'] for r in rows if r['index']==i]) for i in range(frame_n)]) for b in ('validation','receiver')}
        idx=rng.integers(frame_n,size=(5000,frame_n))
        ratio=vectors[primary]['validation'][idx].mean(1)/vectors['g2']['validation'][idx].mean(1)
        rx=vectors[primary]['receiver'][idx].mean(1)-vectors['g2']['receiver'][idx].mean(1)
        rbound=bootstrap_summary(ratio,float(vectors[primary]['validation'].mean()/vectors['g2']['validation'].mean()),.05/12)
        xbound=bootstrap_summary(rx,float(vectors[primary]['receiver'].mean()-vectors['g2']['receiver'].mean()),.05/12)
        # Accuracy matched to the timed checkpoint, not averaged over different models.
        perf=rank_comparison([review[0]],primary,'g2',[a],error=.05/12)[0]
        p=point([review[0]],primary,a)
        ok=(rbound['upper_one_sided']<.90 and xbound['upper_one_sided']<=timing['resolution']['receiver_allowance_seconds'] and
            perf['risk']['upper_one_sided']<=.005/a and perf['useful']['lower_one_sided']>=-.01 and p['joint_useful']>=.05)
        results.append(dict(coverage=a,ratio=rbound,receiver_difference=xbound,performance=perf,primary_point=p,supported=bool(ok)))
    passed=sum(x['supported'] for x in results)
    classification='SUPPORTED' if passed==3 else ('CONDITIONALLY_SUPPORTED' if passed else 'NOT_SUPPORTED')
    return dict(label=LABEL,classification=classification,comparisons=results,scope='CPU, seed5101, exact empirical matched coverage; validation evidence only',g2=g2name)

def analyze(selection,variant,timing,binding):
    design=[with_g2(load_raw(s,'V-select',binding=binding),selection) for s in SEEDS]
    review=[with_g2(load_raw(s,'V-assess',binding=binding),selection) for s in SEEDS]
    panel=[with_g2(load_raw(s,'V-panel','review',binding),selection) for s in SEEDS]
    burst=[with_g2(load_raw(s,'V-burst','review',binding),selection) for s in SEEDS]
    primary=variant['primary'];methods=list(dict.fromkeys(('g1','g2','g3_hard','g3_soft','g3_anchor',*CONTROLS,'g3_hard_mean','g3_soft_mean','g3_mean_only','weighted_evm_identity')))
    rows=summaries(review,methods,'V-assess')
    for group,mask in group_masks(panel[0]).items():
        if mask.any():rows+=summaries(subset(panel,mask),methods,'V-panel-review',group)
    for cell in np.unique(burst[0]['cell']):
        mask=burst[0]['cell']==cell
        cellrows=summaries(subset(burst,mask),('g2','g3_hard','g3_soft','g3_anchor','pilot','weighted_mean'),'V-burst-review',f'cell_{cell}')
        for r in cellrows:
            for k in ('family','snr_db','inr_db','duration','probability'):r[k]=burst[0][k][mask][0]
        rows+=cellrows
    write_csv(OUT/'analysis/metrics.csv',rows)
    # Full risk curves; all serious candidates, no smoothing.
    curves=[]
    for seed,d in zip(SEEDS,review):
        for m in methods:
            for r in rank_curve(d[m],d['harm'],d['useful'],d['tie_key'],np.arange(.05,.951,.025),d['base_loss'],d['neural_loss']):curves.append(dict(seed=seed,method=m,**r))
    write_csv(OUT/'analysis/curves.csv',curves)
    fixed=[]
    for seed,ds,rv in zip(SEEDS,design,review):
        for m in methods:
            for a in COVERAGES:
                threshold=float(np.quantile(ds[m],a));g=rv[m]<=threshold
                fixed.append(dict(seed=seed,method=m,target_design_quantile=a,threshold=threshold,**gate_metrics(rv['harm'],rv['useful'],g,rv['base_loss'],rv['neural_loss'])))
    write_csv(OUT/'analysis/fixed_validation_thresholds.csv',fixed)
    bounds={c:rank_comparison(review,primary,c,[.5],error=.05/10) for c in ('g1','g2',selection['control'])}
    write_json(OUT/'analysis/primary_comparisons.json',dict(label=LABEL,primary=primary,bounds=bounds))
    regs=regression(design,review,binding,panel)
    ch=challenge_rates([load_raw_challenge(s,'review',binding) for s in SEEDS],design)
    write_csv(OUT/'analysis/challenges_review.csv',ch)
    mismatches=[]
    for seed,d in zip(SEEDS,review):
        g=rank_acceptance(d['g2'],.5,d['tie_key'])
        mismatches.append(dict(seed=seed,**partial_full_mismatch(d['partial_base_loss'],d['partial_neural_loss'],d['base_loss'],d['neural_loss'],g)))
    write_json(OUT/'analysis/mismatch.json',dict(label=LABEL,per_seed=mismatches))
    cost=cost_inference(timing,review,primary,selection['g2']['name']);write_json(OUT/'analysis/cost_conclusion.json',cost)
    identities=[float(np.max(np.abs(d['weighted_evm_identity']-d['g3_hard']))) for d in review+panel+burst]
    write_json(OUT/'analysis/equivalence.json',dict(label=LABEL,max_error=max(identities),floor_fraction_max=max(float(d[k].max()) for d in review+panel+burst for k in ('evm_floor_fraction_base','evm_floor_fraction_neural')),mean_identity_max=max(float(np.max(np.abs(d['g3_hard_mean']-d['weighted_mean']))) for d in review+panel+burst)))
    sensitivity=[]
    for seed,d in zip(SEEDS,review):
        for delta in (0.,.5,1.):
            e,u=labels_from_losses(d['base_loss'],d['neural_loss'],delta,delta)
            for m in (primary,'g2',selection['control']):
                for a in COVERAGES:sensitivity.append(dict(seed=seed,delta=delta,method=m,target_coverage=a,**gate_metrics(e,u,rank_acceptance(d[m],a,d['tie_key']),d['base_loss'],d['neural_loss'])))
    write_csv(OUT/'analysis/delta_sensitivity.csv',sensitivity)
    plans=[binomial_sample_plan(n,.05/40,acceptance_probability=.05) for n in (2048,4096,8192,16384)]
    write_json(OUT/'analysis/calibration_size_planning.json',dict(label=LABEL,analytic_only=True,calibration_data_accessed=False,certificate_family_proposal=40,plans=plans))
    # First IDs per outcome category/condition: deterministic catalog examples.
    examples=[]
    for seed,d in zip(SEEDS,panel):
        ds=design[list(SEEDS).index(seed)];threshold=float(np.quantile(ds[primary],.5));g=d[primary]<=threshold
        for cell in np.unique(d['cell']):
            for label,mask in (('false_accept',g&d['harm']),('false_reject',~g&d['useful'])):
                ids=np.flatnonzero((d['cell']==cell)&mask)
                if len(ids):
                    i=ids[0];examples.append(dict(seed=seed,cell=int(cell),type=label,frame_id=d['frame_id'][i],score=float(d[primary][i]),threshold=threshold,base_loss=float(d['base_loss'][i]),neural_loss=float(d['neural_loss'][i]),family=d['family'][i],modulation=d['modulation'][i],snr=float(d['snr_db'][i]),stride=int(d['stride'][i])))
    write_csv(OUT/'analysis/failure_examples.csv',examples)
    write_json(OUT/'analysis/completed.json',dict(label=LABEL,binding=binding,primary=primary,cost=cost['classification'],incremental=regs,complete=True))
