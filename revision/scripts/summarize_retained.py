"""Lossless decisions and descriptive, frame-unit retained-data summaries."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'revision/scripts'))
import numpy as np
from scipy.stats import beta,t
from audit_certificates import arrays,read,write,csvout,sha
OUT=ROOT/'revision/verification/retained'

def proportion(k,n):
    if n==0:return [None,None]
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def metrics(d,g,mask,fallback='base',delta=.5):
    b=d[fallback+'_loss'][mask];nn=d['neural_loss'][mask];g=g[mask];n=len(g);m=int(g.sum())
    bh=d[fallback+'_bit_errors'][mask];nh=d['neural_bit_errors'][mask];bits=d['bits'][mask]
    harm=nn>10**(delta/10)*b;useful=b>10**.05*nn;h=int(np.sum(harm&g));u=int(np.sum(useful&g))
    errors=np.where(g,nh,bh);loss=np.where(g,nn,b);diff=(errors-bh)/bits
    se=float(np.std(diff,ddof=1)/np.sqrt(n)) if n>1 else 0.;half=float(t.ppf(.975,n-1))*se if n>1 else 0.
    severity=10*np.log10(nn[g&harm]/b[g&harm])
    return dict(n=n,accepted=m,harmful=h,useful=u,coverage=m/n if n else None,joint_harm=h/n if n else None,conditional_harm=h/m if m else None,joint_useful=u/n if n else None,
      coverage_ci=proportion(m,n),joint_harm_ci=proportion(h,n),conditional_harm_ci=proportion(h,m),joint_useful_ci=proportion(u,n),
      bit_errors=int(errors.sum()),bits=int(bits.sum()),ber=float(errors.sum()/bits.sum()) if n else None,fallback_ber=float(bh.sum()/bits.sum()) if n else None,
      mean_nmse=float(loss.mean()) if n else None,p95_nmse=float(np.quantile(loss,.95)) if n else None,paired_ber_difference=float(diff.mean()) if n else None,
      paired_ber_difference_ci=[float(diff.mean()-half),float(diff.mean()+half)] if n else [None,None],
      accepted_harm_median_db=float(np.median(severity)) if h else None,accepted_harm_p95_db=float(np.quantile(severity,.95)) if h else None,accepted_harm_max_db=float(severity.max()) if h else None)

def main():
    OUT.mkdir(parents=True,exist_ok=True);s1=read(ROOT/'outputs/e8/calibration/certificates.json')['certificates'];s2=read(ROOT/'outputs/revision_r2/calibration/certificates.json')['certificates'];spec=read(ROOT/'configs/revision_r2/study.json')
    rows=[];receipts=[];unavailable=[]
    for study,env,folder in [('Study1','SYN_A','outputs/e8/P0-GATES-01'),*[('Study2',e,f'outputs/revision_r2/raw/R2_TEST/{e}') for e in ['SYN_A','SYN_C','TDLA','TDLC']]]:
        for seed in range(5101,5106):
            f=ROOT/f'{folder}/s{seed}';first=next(f.glob('*.npz'))
            with np.load(first,allow_pickle=False) as z:keys=z.files
            d,files=arrays(f,keys);n=len(d['frame_id']);assert n==len(set(d['frame_id']))
            policies=[]
            if study=='Study1':
                for name,c in s1[str(seed)].items():policies.append((name,name,c,'base',None,.5))
            else:
                source_env='SYN_A' if env in ['SYN_A','SYN_C'] else 'TDLA'
                for ident,c in s2.items():
                    p=c['policy']
                    if c['seed']==seed and p['environment']==source_env:
                        if p['score'] not in d:
                            unavailable.append(dict(study=study,environment=env,seed=seed,policy=ident.split('/',1)[1],reason='Required score was not retained in this historical transfer cohort; no reconstructed or invented outcome'))
                            continue
                        policies.append((ident.split('/',1)[1],p['score'],c,p['fallback'],p['stratum'],p['delta_db']))
            export=dict(d)
            for name,score,c,fallback,stratum,delta in policies+[('fallback','',None,'base',None,.5),('always_neural','',None,'base',None,.5)]:
                mask=np.ones(n,bool) if stratum is None else np.digitize(d['estimated_snr_db'],spec['estimated_snr_boundaries_db'])==stratum
                if name=='fallback':g=np.zeros(n,bool)
                elif name=='always_neural':g=np.ones(n,bool)
                elif not c['certified']:g=np.zeros(n,bool)
                elif score=='g0':g=np.ones(n,bool)
                else:g=d[score]<=c['threshold']
                g&=mask;export['accept__'+name]=g
                for snr in ['ALL',0,5,10,15,20]:
                    subset=mask if snr=='ALL' else mask&(d['snr_db']==snr)
                    if not subset.any():continue
                    rows.append(dict(study=study,environment=env,seed=seed,policy=name,stratum=stratum,snr_db=snr,scope='matched' if env in ['SYN_A','TDLA'] else 'transfer_no_certificate',fallback=fallback,delta_db=delta,**metrics(d,g,subset,fallback,delta)))
            path=OUT/f'frame_decisions/{study}_{env}_s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**export)
            receipts.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),independent_frames=n,policies=len(policies),inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files]))
    csvout(OUT/'per_seed_snr_condition.csv',rows)
    write(OUT/'summary.json',dict(scope='POST-HOC DESCRIPTIVE. Exact pointwise Clopper-Pearson intervals for Bernoulli frame events. Paired BER difference uses a pointwise Student-t interval across complete-frame error-rate differences, not independent bits. No multiplicity-adjusted superiority claim; no pooling five models as fivefold independent n. Conditional harm undefined at zero accepts.',rows=rows))
    write(OUT/'export_manifest.json',dict(status='PASS',exports=receipts,unavailable_historical_transfer_scores=unavailable,summary_sha256=sha(OUT/'summary.json')))
    print('RETAINED_SUMMARIES_PASS',len(rows),'rows',len(receipts),'lossless exports',flush=True)

if __name__=='__main__':main()
