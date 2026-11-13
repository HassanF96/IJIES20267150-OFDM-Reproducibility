"""Export and independently replay historical certificates without rewriting them.

The reference CDF uses a compensated binomial-PMF recurrence, not SciPy's CDF.
All outputs are under revision/verification. Original arrays/seals are read-only.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys,time
from decimal import Decimal,localcontext
from collections import defaultdict
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class ReferenceCDF:
    def __init__(self,alpha=.05):self.alpha=alpha;self.cache={}
    def __call__(self,h,n):
        assert 0<=h<=n and int(h)==h and int(n)==n
        if n==0:return 1.
        if n not in self.cache:
            p=math.exp(n*math.log1p(-self.alpha));s=p;c=0.;values=[p]
            for k in range(n):
                p*=((n-k)/(k+1))*(self.alpha/(1-self.alpha))
                y=p-c;t=s+y;c=(t-s)-y;s=t
                values.append(min(1.,s))
            self.cache[n]=values
        return self.cache[n][h]

def decimal_cdf(h,n,alpha='0.05'):
    with localcontext() as ctx:
        ctx.prec=80;a=Decimal(alpha);q=1-a;p=q**n;s=p
        for k in range(h):p=p*Decimal(n-k)/Decimal(k+1)*a/q;s+=p
        return float(s)

def arrays(folder,keys):
    files=sorted(folder.glob('*.npz'));assert files,folder
    parts=[]
    for p in files:
        with np.load(p,allow_pickle=False) as z:parts.append({k:z[k].copy() for k in keys})
    return {k:np.concatenate([d[k] for d in parts]) for k in keys},files

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--no-raw-replay',action='store_true');ap.add_argument('--output-dir',type=Path,default=None);args=ap.parse_args()
    root=args.root.resolve();out=(args.output_dir.resolve() if args.output_dir else root/'reproduction_outputs/certificates');out.mkdir(parents=True,exist_ok=True)
    cdf=ReferenceCDF();checks=[]
    for n,h in [(0,0),(1,0),(1,1),(200,0),(200,10),(8192,0),(8192,345),(8192,346),(8192,8192)]:
        actual=cdf(h,n);expected=decimal_cdf(h,n)
        assert math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-300),(n,h,actual,expected)
        checks.append(dict(n=n,h=h,reference=actual,decimal80=expected))
    sources=[root/'outputs/e8/calibration/certificates.json',root/'outputs/revision_r2/calibration/certificates.json']
    hashes={str(p.relative_to(root)):sha(p) for p in sources}
    for p in sources:assert sha(p)==read(p.with_suffix('.seal.json'))['sha256']
    s1=read(sources[0]);s2=read(sources[1]);settings=yaml.safe_load((root/'configs/final/calibration.yaml').read_text());spec=read(root/'configs/revision_r2/study.json')
    grids1=read(root/'configs/final/threshold_grids.json')['grids'];grids2=read(root/'configs/revision_r2/threshold_grids.json')['grids']
    rows=[];policies=[];replay=[];max_relative=0.
    for study,certs in [('Study1',{f'{s}/{m}':dict(seed=int(s),method=m,**c) for s,d in s1['certificates'].items() for m,c in d.items()}),('Study2',s2['certificates'])]:
        for ident,c in certs.items():
            seed=c['seed'];method=c['method'] if study=='Study1' else c['policy']['score'];policy=ident.split('/',1)[1]
            target='joint' if study=='Study1' else c['target'];singleton=c.get('singleton',False)
            fallback='base' if study=='Study1' else c['policy']['fallback'];stratum=None if study=='Study1' else c['policy']['stratum'];env='SYN_A' if study=='Study1' else c['policy']['environment']
            grid=[None] if singleton else (grids1 if study=='Study1' else grids2)[str(seed)][method]
            assert singleton or all(grid[i]<grid[i+1] for i in range(len(grid)-1))
            gridid=hashlib.sha256(canonical(grid)).hexdigest();error=settings['epsilon_per_certificate'] if study=='Study1' else spec['error_each']
            bounds=['-inf',*spec['estimated_snr_boundaries_db'],'inf'];low='' if stratum is None else bounds[stratum];high='' if stratum is None else bounds[stratum+1]
            selected=None;certrows=[];stop=False
            for index,r in enumerate(c['rows']):
                assert not stop,'Fixed sequence contains rows after first non-rejection'
                assert index<len(grid) and r['threshold']==grid[index]
                n=r['n'];m=r['accepted'] if study=='Study1' else r['m'];h=r['harmful'] if study=='Study1' else r['h'];u=r['useful'] if study=='Study1' else r['u']
                assert 0<=h<=m<=n and 0<=u<=m and h+u<=m
                trials=n if target=='joint' else m;p=cdf(h,trials) if trials else 1.
                assert math.isclose(p,r['p_value'],rel_tol=2e-10,abs_tol=1e-300),(ident,index,p,r['p_value'])
                max_relative=max(max_relative,abs(p-r['p_value'])/max(p,1e-300))
                reject=p<=error;screen=n>0 and m>=200 and m/n>=.05 and u/n>=.05
                if study=='Study1':assert reject==r['rejected_null'] and screen==r['useful_screen']
                else:assert screen==r['screen'] and (reject and screen)==r['passed']
                if reject and screen:selected=index
                if study=='Study1' and not reject:stop=True
                row=dict(study=study,seed=seed,policy=policy,score=method,fallback=fallback,environment=env,risk_target=target,stratum='' if stratum is None else stratum,stratum_lower=low,stratum_upper=high,
                    threshold_grid_id=gridid,grid_order=index,grid_size=len(grid),threshold='' if singleton else r['threshold'],singleton_accept_all=singleton,
                    calibration_frames=n,accepted=m,harmful_accepted=h,useful_accepted=u,binomial_denominator=trials,p_value=r['p_value'],independent_p_value=p,
                    multiplicity_allocation=error,rejected_null=reject,screen_eligible=screen,sequence_stop=stop,selected=False,source_path=str(sources[study=='Study2'].relative_to(root)),source_sha256=hashes[str(sources[study=='Study2'].relative_to(root))])
                certrows.append(row)
            assert len(certrows)==len(grid) if study=='Study2' else (len(certrows)==len(grid) or stop)
            expected_index=c['selected_index'] if study=='Study1' else next((i for i,r in enumerate(c['rows']) if c['selected'] is not None and r['threshold']==c['selected']['threshold']),None)
            assert selected==expected_index and (selected is not None)==c['certified'],ident
            if selected is not None:certrows[selected]['selected']=True
            assert c['threshold']==(grid[selected] if selected is not None else None)
            rows.extend(certrows)
            for normalized in certrows:
                normalized.update(alpha=.05,harm_delta_db=settings['delta_db'] if study=='Study1' else c['policy']['delta_db'],useful_delta_db=.5,family_id='original_105_fixed_sequences' if study=='Study1' else 'R2_9180_threshold_tests',family_error=.05,family_confidence=.95,screen_min_accepted=200,screen_min_coverage=.05,screen_min_joint_useful=.05)
            policies.append(dict(study=study,seed=seed,policy=policy,score=method,fallback=fallback,environment=env,risk_target=target,stratum='' if stratum is None else stratum,
                stratum_lower=low,stratum_upper=high,selected_index='' if selected is None else selected,threshold='' if c['threshold'] is None else c['threshold'],singleton_accept_all=singleton,
                certified=c['certified'],status=c['status'],grid_id=gridid,tested_thresholds=len(certrows),grid_size=len(grid),
                certificate_scope='Matched IID simulation law conditional on fixed trained models and fixed grids; no BER/shift/new-model guarantee; empirical usefulness screens',source_sha256=certrows[0]['source_sha256']))
        print(study,'exported and independently checked',flush=True)
    assert len(rows)==3780+9180 and len(policies)==195
    csvout(out/'machine_readable_certificates/tested_thresholds.csv',rows);csvout(out/'policy_summary.csv',policies)
    # Replay every retained certificate from actual complete calibration arrays.
    if not args.no_raw_replay:
        for seed in range(5101,5106):
            methods=list(s1['certificates'][str(seed)])
            keys=['frame_id','harm','useful']+[m for m in methods if m!='g0']
            d,files=arrays(root/f'outputs/e8/calibration/s{seed}',keys);assert len(d['frame_id'])==8192 and len(set(d['frame_id']))==8192
            for m,c in s1['certificates'][str(seed)].items():
                for r in c['rows']:
                    g=np.ones(8192,bool) if m=='g0' else d[m]<=r['threshold']
                    assert (int(g.sum()),int(np.sum(g&d['harm'])),int(np.sum(g&d['useful'])))==(r['accepted'],r['harmful'],r['useful'])
            replay.append(dict(study='Study1',seed=seed,environment='SYN_A',certificates=len(methods),frames=len(d['frame_id']),source_files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in files]))
            for env in ['SYN_A','TDLA']:
                certs=[c for c in s2['certificates'].values() if c['seed']==seed and c['policy']['environment']==env]
                keys=sorted({'frame_id','estimated_snr_db','neural_loss',*[c['policy']['fallback']+'_loss' for c in certs],*[c['policy']['score'] for c in certs]})
                d,files=arrays(root/f'outputs/revision_r2/raw/R2_CAL/{env}/s{seed}',keys);assert len(d['frame_id'])==8192 and len(set(d['frame_id']))==8192
                for c in certs:
                    pol=c['policy'];mask=np.ones(8192,bool) if pol['stratum'] is None else np.digitize(d['estimated_snr_db'],spec['estimated_snr_boundaries_db'])==pol['stratum']
                    h=d['neural_loss']>10**(pol['delta_db']/10)*d[pol['fallback']+'_loss'];u=d[pol['fallback']+'_loss']>10**(.5/10)*d['neural_loss']
                    for r in c['rows']:
                        g=mask&(d[pol['score']]<=r['threshold'])
                        assert (int(mask.sum()),int(g.sum()),int(np.sum(g&h)),int(np.sum(g&u)))==(r['n'],r['m'],r['h'],r['u'])
                replay.append(dict(study='Study2',seed=seed,environment=env,certificates=len(certs),frames=len(d['frame_id']),source_files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in files]))
            print('Full calibration-array replay complete',seed,flush=True)
    for p in sources:assert sha(p)==hashes[str(p.relative_to(root))]
    write(out/'certificate_verification.json',dict(status='PASS',threshold_rows=len(rows),policies=len(policies),reference_algorithm='Compensated PMF recurrence; independent Decimal80 edge references; no scipy.stats.binom CDF used',edge_checks=checks,
        maximum_relative_cdf_discrepancy=max_relative,full_raw_replay=not args.no_raw_replay,replay=replay,source_hashes=hashes,historical_outputs_unchanged=True))
    print(json.dumps(dict(status='PASS',rows=len(rows),policies=len(policies),raw_certificates_replayed=sum(x['certificates'] for x in replay))),flush=True)

if __name__=='__main__':main()
