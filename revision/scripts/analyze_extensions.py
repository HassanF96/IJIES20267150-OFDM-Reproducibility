"""Prespecified post-review descriptive summaries; reads completed immutable shards."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'revision/scripts'))
import numpy as np
from scipy.stats import beta
from audit_certificates import arrays,read,write,csvout,sha,ReferenceCDF
from summarize_retained import metrics,proportion
OUT=ROOT/'revision/experiments';DEST=OUT/'analysis';SPEC=read(ROOT/'revision/configs/extension.json');SEEDS=list(range(5101,5106))
SOURCES=[]
MANIFEST=list(csv.DictReader((ROOT/'revision/configs/frame_manifest.csv').open(encoding='utf-8')))

def cohort(cohort_name,condition):
    result={};sources=[]
    for seed in SEEDS:
        folder=OUT/f'raw/{cohort_name}/{condition}/s{seed}'
        first=next(folder.glob('*.npz'))
        with np.load(first,allow_pickle=False) as z:keys=z.files
        d,files=arrays(folder,keys)
        for p in files:
            meta=read(p.with_suffix('.sha.json'));assert sha(p)==meta['sha256'];assert meta['protocol_sha256']==sha(ROOT/'revision/configs/protocol_seal.json')
            sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        assert len(d['frame_id'])==SPEC['cohorts'][cohort_name][condition] and len(set(d['frame_id']))==len(d['frame_id'])
        assert d['frame_id'].tolist()==[r['realization_id'] for r in MANIFEST if r['cohort']==cohort_name and r['condition']==condition]
        if result:np.testing.assert_array_equal(d['frame_id'],result[SEEDS[0]]['frame_id'])
        result[seed]=d
    SOURCES.extend(sources)
    return result,sources

def boot_means(x,rng):
    # Resampling complete paired frames; all columns/models share each draw.
    n=len(x);out=[]
    for start in range(0,5000,100):
        counts=rng.multinomial(n,np.full(n,1/n),size=min(100,5000-start)).astype(float)
        out.append(counts@x/n)
    return np.concatenate(out)

def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975]).tolist() if len(x) else [None,None]

def decision(d,c,score):return np.zeros(len(d['frame_id']),bool) if not c['certified'] else d[score]<=c['threshold']

def certificates():
    path=OUT/'stronger_tdl/certificates.json';seal=read(OUT/'stronger_tdl/calibration.seal.json');assert sha(path)==seal['sha256'];allc=read(path)['certificates'];cdf=ReferenceCDF();rows=[]
    for condition in SPEC['cohorts']['CAL']:
        data,_=cohort('CAL',condition)
        for seed,d in data.items():
            harm=d['neural_loss']>10**.05*d['base_loss'];useful=d['base_loss']>10**.05*d['neural_loss']
            for score in ['g3_hard','g2_50']:
                c=allc[f'{condition}/{seed}/{score}'];selected=None
                assert len(c['rows'])==len(SPEC['grids'][str(seed)][score])
                for index,r in enumerate(c['rows']):
                    g=d[score]<=r['threshold'];n=len(g);m=int(g.sum());h=int(np.sum(g&harm));u=int(np.sum(g&useful));p=cdf(h,n)
                    assert r['threshold']==SPEC['grids'][str(seed)][score][index]
                    assert (n,m,h,u)==(r['n'],r['m'],r['h'],r['u'])
                    assert np.isclose(p,r['p_value'],rtol=2e-10,atol=1e-300)
                    screen=m>=200 and m/n>=.05 and u/n>=.05;ok=screen and p<=SPEC['error_each']
                    assert r['screen']==screen and r['passed']==ok
                    if ok:selected=r
                    rows.append(dict(condition=condition,seed=seed,score=score,grid_index=index,n=n,m=m,h=h,u=u,p_value=p,error_each=SPEC['error_each'],passed=ok))
                assert selected==c['selected'] and c['certified']==(selected is not None)
                assert c['target']=='joint' and c['threshold']==(selected['threshold'] if selected is not None else None)
    assert len(rows)==2040
    write(DEST/'calibration_verification.json',dict(status='PASS',tested_rows=len(rows),policies=len(allc),source_sha256=sha(path),independent_binomial_and_raw_replay=True))
    return allc

def accuracy(allc):
    old=read(ROOT/'outputs/revision_r2/calibration/certificates.json')['certificates'];summaries=[];intervals=[];phase_rows=[];phase_boot=[];sources=[];aggregate=[]
    rng=np.random.default_rng(SPEC['bootstrap_seed'])
    for name,condition in [('TEST','TDLA30'),('TEST','TDLA300'),('PHASE','SYN_A')]:
        data,files=cohort(name,condition);sources.extend(files);vectors=[];keys=[];structural_reject={}
        for seed,d in data.items():
            n=len(d['frame_id']);mask=np.ones(n,bool)
            if name=='TEST':
                policies=[('transfer_g3',old[f'{seed}/tdl_joint'],'g3_hard'),('matched_g3',allc[f'{condition}/{seed}/g3_hard'],'g3_hard'),('matched_g2',allc[f'{condition}/{seed}/g2_50'],'g2_50')]
            else:policies=[('frozen_g3',old[f'{seed}/joint_d0.5'],'g3_hard'),('pilot_anchor',old[f'{seed}/pilot_anchor'],'g3_anchor'),('frozen_g2',old[f'{seed}/g2_50'],'g2_50')]
            decisions={p:decision(d,c,score) for p,c,score in policies};decisions.update(fallback=np.zeros(n,bool),always_neural=np.ones(n,bool))
            structural_reject.update({(seed,p):not c['certified'] for p,c,score in policies})
            structural_reject[(seed,'fallback')]=True
            structural_reject[(seed,'always_neural')]=False
            h=d['neural_loss']>10**.05*d['base_loss'];u=d['base_loss']>10**.05*d['neural_loss'];ratio=10*np.log10(d['neural_loss']/d['base_loss'])
            for policy,g in decisions.items():
                for snr in ['ALL',0,5,10,15,20]:
                    subset=mask if snr=='ALL' else d['snr_db']==snr
                    row=metrics(d,g,subset);row.pop('paired_ber_difference_ci')
                    summaries.append(dict(cohort=name,condition=condition,seed=seed,policy=policy,snr_db=snr,**row))
                e=np.where(g,d['neural_bit_errors'],d['base_bit_errors'])/d['bits'];loss=np.where(g,d['neural_loss'],d['base_loss']);delta=e-d['base_bit_errors']/d['bits']
                v=np.column_stack([g,g&h,g&u,e,loss,delta,np.where(g&h,ratio,0)])
                if name=='PHASE':
                    event=d['phase_event'].astype(bool);valid=d['phase_valid'].all(axis=1)
                    extras=np.column_stack([event,valid,event&valid,event&g,event&g&h,event&g&u,d['phase_severe']])
                else:extras=np.zeros((n,7))
                v=np.column_stack([v,extras])
                vectors.append(v);keys.append((seed,policy))
                if name=='PHASE':
                    event=d['phase_event'].astype(bool);valid=d['phase_valid'].all(axis=1);count=int(event.sum());validn=int(valid.sum())
                    phase_rows.append(dict(seed=seed,policy=policy,n=n,valid_frames=validn,invalid_frames=n-validn,event_frames=count,valid_event_frames=int(np.sum(event&valid)),valid_event_ci=proportion(int(np.sum(event&valid)),validn),event_given_accepted=float(np.sum(event&g)/g.sum()) if g.any() else None,event_given_accepted_ci=proportion(int(np.sum(event&g)),int(g.sum())),sparse_event_count_under20=count<20,severe_frames=int(d['phase_severe'].sum()),incidence=count/n,incidence_ci=proportion(count,n),valid_frame_incidence=float(event[valid].mean()) if validn else None,zero_event_upper95=1-.05**(1/n) if count==0 else None,
                        event_accepted=int(np.sum(event&g)),event_harmful_accepted=int(np.sum(event&g&h)),event_useful_accepted=int(np.sum(event&g&u)),event_extra_bit_errors=int(np.sum(np.where(g,d['neural_bit_errors']-d['base_bit_errors'],0)[event])),accepted_frames=int(g.sum()),event_conditional_acceptance=float(g[event].mean()) if count else None,event_acceptance_ci=proportion(int(np.sum(event&g)),count),event_ber=float(np.sum(np.where(g,d['neural_bit_errors'],d['base_bit_errors'])[event])/np.sum(d['bits'][event])) if count else None))
            export=DEST/f'decisions/{name}_{condition}_s{seed}.npz';export.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(export,**d,**{'accept__'+k:v for k,v in decisions.items()})
        x=np.concatenate(vectors,axis=1);boot=boot_means(x,rng).reshape(5000,len(keys),14);point=x.mean(axis=0).reshape(len(keys),14)
        for policy in dict.fromkeys(k[1] for k in keys):
            indices=[i for i,k in enumerate(keys) if k[1]==policy]
            for i in indices+[-1]:
                bm=boot[:,i,:] if i>=0 else boot[:,indices,:].mean(axis=1);pm=point[i] if i>=0 else point[indices].mean(axis=0);seed=keys[i][0] if i>=0 else 'FIXED_MODEL_MIXTURE'
                with np.errstate(divide='ignore',invalid='ignore'):
                    conditional=bm[:,1]/bm[:,0];severity=bm[:,6]/bm[:,1]
                values=dict(coverage=pm[0],joint_harm=pm[1],joint_useful=pm[2],ber=pm[3],mean_nmse=pm[4],paired_ber_difference=pm[5],conditional_harm=pm[1]/pm[0] if pm[0] else None,mean_accepted_harm_db=pm[6]/pm[1] if pm[1] else None)
                intervals.append(dict(cohort=name,condition=condition,seed=seed,policy=policy,**{k:float(v) if v is not None else None for k,v in values.items()},
                    zero_joint_harm_upper95=(0. if all(structural_reject[keys[j]] for j in ([i] if i>=0 else indices)) else 1-.05**(1/n)) if pm[1]==0 else None,
                    zero_event_scope='A zero empirical bootstrap interval is not a population guarantee. Zero-event bound uses n independent frames; frame-union bound is conservative for an all-model zero-event mixture. Reject-all has structural zero intervention harm.',
                    undefined_conditional_bootstrap_draws=int(np.sum(~np.isfinite(conditional))),undefined_severity_bootstrap_draws=int(np.sum(~np.isfinite(severity))),
                    intervals={**{k:ci(bm[:,j]) for j,k in enumerate(['coverage','joint_harm','joint_useful','ber','mean_nmse','paired_ber_difference'])},'conditional_harm':ci(conditional),'mean_accepted_harm_db':ci(severity)}))
                if name=='PHASE':
                    rates={name:(num,den) for name,num,den in [('incidence',7,None),('valid_frame_incidence',9,8),('event_given_accepted',10,0),('acceptance_given_event',10,7),('harmful_acceptance_given_event',11,7),('useful_acceptance_given_event',12,7),('severe_incidence',13,None)]}
                    result={};bounds={}
                    for label,(num,den) in rates.items():
                        result[label]=float(pm[num]) if den is None else (float(pm[num]/pm[den]) if pm[den] else None)
                        with np.errstate(divide='ignore',invalid='ignore'):values=bm[:,num] if den is None else bm[:,num]/bm[:,den]
                        bounds[label]=ci(values)
                    phase_boot.append(dict(seed=seed,policy=policy,point=result,intervals=bounds,independent_frames=n,
                        degenerate_empirical_bootstrap_fields=[k for k,v in bounds.items() if v[0] is not None and v[0]==v[1]],
                        zero_incidence_upper95=1-.05**(1/n) if pm[7]==0 else None,zero_severe_incidence_upper95=1-.05**(1/n) if pm[13]==0 else None,
                        zero_event_scope='Degenerate empirical bootstrap does not imply zero population uncertainty. For an all-model zero-event mixture the frame-union one-sided bound is conservative; models are not independent replicates.'))
        # Paired method comparisons on the same fixed model mixture and frames.
        main='matched_g3' if name=='TEST' else 'frozen_g3';comparators=['matched_g2','transfer_g3'] if name=='TEST' else ['pilot_anchor','frozen_g2']
        for comparator in comparators:
            a=[i for i,k in enumerate(keys) if k[1]==main];b=[i for i,k in enumerate(keys) if k[1]==comparator]
            delta=boot[:,a,:].mean(axis=1)-boot[:,b,:].mean(axis=1);pointdiff=point[a].mean(axis=0)-point[b].mean(axis=0)
            aggregate.append(dict(cohort=name,condition=condition,comparison=main+' minus '+comparator,point=pointdiff[:7].tolist(),ci=[ci(delta[:,j]) for j in range(7)],columns=['coverage','joint_harm','joint_useful','ber','mean_nmse','paired_ber_difference_vs_fallback','harm_db_sum_per_frame']))
        print('Accuracy bootstrap complete',condition,flush=True)
    csvout(DEST/'per_seed_snr_condition.csv',summaries);write(DEST/'accuracy_summary.json',dict(scope='POST-REVIEW EXTENSION; descriptive pointwise 95% paired whole-frame percentile intervals, 5000 draws; fixed-model mixture samples one of the five retained models uniformly. No new-model or superiority guarantee. SNR slices descriptive; severity percentiles descriptive.',rows=intervals,paired_comparisons=aggregate))
    csvout(DEST/'phase_summary.csv',phase_rows);write(DEST/'phase_summary.json',dict(scope='Evaluation-only coherent pilot/data relative phase inconsistency; not all natural phase errors. Global rotations do not trigger this definition. Exact per-model frame intervals; descriptive whole-frame paired bootstrap fixed-model mixture. Fewer than 20 observed events flagged as sparse, not a validation or decision threshold.',rows=phase_rows,paired_frame_bootstrap=phase_boot))

def timing():
    files=sorted((OUT/'timing/raw').glob('*.npz'));assert len(files)==128
    keys=None;parts=[]
    for p in files:
        meta=read(p.with_suffix('.sha.json'));assert sha(p)==meta['sha256'] and meta['protocol_sha256']==sha(ROOT/'revision/configs/protocol_seal.json')
        SOURCES.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        with np.load(p,allow_pickle=False) as z:parts.append({k:z[k].copy() for k in z.files})
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};assert len(d['frame_id'])==128*5*4*30
    expected=[r['realization_id'] for r in MANIFEST if r['cohort']=='TIMING']
    assert all(str(rid)==expected[int(i)] for rid,i in zip(d['frame_id'],d['frame_index']))
    assert np.all(d['protocol_sha256']==sha(ROOT/'revision/configs/protocol_seal.json'))
    rows=[];vectors=[];labels=[];rng=np.random.default_rng(SPEC['bootstrap_seed']+1)
    for seed in SEEDS:
        for study,policy in [('Study1','g3_hard'),('Study1','g2'),('Study2','joint_d0.5'),('Study2','g2_50')]:
            mask=(d['seed']==seed)&(d['study']==study)&(d['policy']==policy);ix=np.flatnonzero(mask)
            order=np.lexsort((d['repeat'][ix],d['frame_index'][ix]));ix=ix[order]
            assert np.array_equal(d['frame_index'][ix].reshape(128,30)[:,0],np.arange(128))
            assert np.array_equal(d['repeat'][ix].reshape(128,30),np.broadcast_to(np.arange(30),(128,30)))
            assert np.all(d['accepted'][ix].reshape(128,30)==d['accepted'][ix].reshape(128,30)[:,0,None])
            one=ix.reshape(128,30)[:,0];raw={k:d[k][one] for k in ['frame_id','base_loss','neural_loss','base_bit_errors','neural_bit_errors','bits']};g=d['accepted'][one].astype(bool)
            row=dict(seed=seed,study=study,policy=policy,**metrics(raw,g,np.ones(128,bool)))
            vectors.append(np.where(g,raw['neural_bit_errors']-raw['base_bit_errors'],0)/raw['bits']);labels.append((seed,study,policy,'paired_ber_difference'))
            for boundary in ['validation','receiver']:
                values=d[boundary+'_seconds'][ix].reshape(128,30);means=values.mean(axis=1)
                row.update({boundary+'_mean_seconds':float(values.mean()),boundary+'_median_seconds':float(np.median(values)),boundary+'_p95_seconds':float(np.quantile(values,.95)),boundary+'_sd_seconds':float(values.std(ddof=1)),boundary+'_between_frame_mean_sd':float(means.std(ddof=1)),boundary+'_mean_within_frame_repeat_sd':float(values.std(axis=1,ddof=1).mean())})
                vectors.append(means);labels.append((seed,study,policy,boundary))
            row['receiver_frame_period_ratio']=row['receiver_mean_seconds']/.0006;row['sequential_frames_per_second']=1/row['receiver_mean_seconds'];rows.append(row)
    x=np.column_stack(vectors);boot=boot_means(x,rng);intervals=[]
    for j,key in enumerate(labels):
        if key[3]=='paired_ber_difference':
            row=next(r for r in rows if (r['seed'],r['study'],r['policy'])==key[:3]);row['paired_ber_difference_ci']=ci(boot[:,j]);row['paired_ber_interval_method']='5000-draw paired whole-frame percentile bootstrap'
        else:intervals.append(dict(seed=key[0],study=key[1],policy=key[2],boundary=key[3],mean_seconds=float(x[:,j].mean()),ci_seconds=ci(boot[:,j])))
    mixture=[]
    for study,p1,p2 in [('Study1','g3_hard','g2'),('Study2','joint_d0.5','g2_50')]:
        for boundary in ['validation','receiver']:
            a=[j for j,k in enumerate(labels) if k[1:]==(study,p1,boundary)];b=[j for j,k in enumerate(labels) if k[1:]==(study,p2,boundary)]
            aa=boot[:,a].mean(axis=1);bb=boot[:,b].mean(axis=1)
            mixture.append(dict(study=study,boundary=boundary,g3_seconds=float(x[:,a].mean()),g2_seconds=float(x[:,b].mean()),g3_ci=ci(aa),g2_ci=ci(bb),g3_to_g2_ratio=float(x[:,a].mean()/x[:,b].mean()),ratio_ci=ci(aa/bb),difference_ci=ci(aa-bb),g3_frame_period_ratio=float(x[:,a].mean()/.0006),g2_frame_period_ratio=float(x[:,b].mean()/.0006)))
    csvout(DEST/'timing_per_seed.csv',rows);write(DEST/'timing_summary.json',dict(scope='Actual fixed calibrated thresholds; 128 independent fresh frames, 30 repeats, all five fixed models; descriptive paired frame-cluster 5000-draw intervals for latency and paired BER difference; exact pointwise Clopper-Pearson event-rate intervals. No real-time claim.',rows=rows,intervals=intervals,mixture=mixture,raw_measurements=len(d['frame_id'])))

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--accuracy-only',action='store_true');args=parser.parse_args()
    if not args.accuracy_only:assert read(OUT/'execution_complete.json')['status']=='EXECUTED'
    DEST.mkdir(parents=True,exist_ok=True);allc=certificates();accuracy(allc)
    if args.accuracy_only:
        write(DEST/'accuracy_source_manifest.json',SOURCES)
        write(DEST/'accuracy_completion.json',dict(status='ACCURACY_VERIFIED_TIMING_PENDING',analysis_script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'revision/configs/protocol_seal.json')))
        print('ACCURACY_ANALYSIS_PASS; TIMING REMAINS INCOMPLETE',flush=True);return
    timing()
    for rel in ['revision/configs/extension.json','revision/configs/protocol_seal.json','revision/configs/frame_manifest.csv','revision/experiments/stronger_tdl/certificates.json','revision/experiments/stronger_tdl/calibration.seal.json','revision/experiments/timing/policies.json','revision/experiments/timing/operations.json','revision/experiments/execution_environment.json','revision/experiments/execution_complete.json','outputs/e8/calibration/certificates.json','outputs/revision_r2/calibration/certificates.json']:
        SOURCES.append(dict(path=rel,sha256=sha(ROOT/rel)))
    write(DEST/'source_manifest.json',SOURCES)
    write(DEST/'completion.json',dict(status='PASS',protocol_sha256=sha(ROOT/'revision/configs/protocol_seal.json'),analysis_script_sha256=sha(Path(__file__)),execution_receipt_sha256=sha(OUT/'execution_complete.json')))
    print('EXTENSION_ANALYSIS_PASS',flush=True)

if __name__=='__main__':main()
