"""Historical ID/custody audit and independent receiver engineering fixtures."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv,hashlib,json,subprocess,sys
from datetime import datetime
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.stats import gamma
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from src.e7.common import structural_records
from src.utils.random import seed_for,STREAMS
from src.revision_r2.core import frame,record,operator,grid,holdout
from src.e6.comparators import block_huber,Helena,helena_estimate
from src.ofdm.information import ReceiverObservation
from src.estimators.features import pilot_features
from src.e6.holdout import geometry_mask
from audit_certificates import sha,read,write,csvout
import yaml

OUT=ROOT/'revision/verification'
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def boundaries():
    fields=['study','partition','frame_id','namespace','master_seed','index','channel_seed',*('rng_'+s for s in STREAMS),'source']
    rows=[];seen={};seeds={};groups=Counter();source_hashes={}
    def add(study,part,r,source):
        rid=r['realization_id'];master=int(r['master_seed']);ns=r['namespace'];rng={s:str(seed_for(master,ns,rid,s)) for s in STREAMS}
        assert rid not in seen,(rid,seen.get(rid),source)
        assert rng['channel'] not in seeds,'Channel seed reused'
        if 'channel_seed' in r:assert str(r['channel_seed'])==rng['channel']
        seen[rid]=(study,part);seeds[rng['channel']]=rid;groups[(study,part)]+=1
        rows.append(dict(study=study,partition=part,frame_id=rid,namespace=ns,master_seed=master,index=r['index'],channel_seed=rng['channel'],**{'rng_'+s:v for s,v in rng.items()},source=source))
    for name in ['configs/final/train_manifest.csv','configs/final/validation_manifest.csv']:
        source_hashes[name]=sha(ROOT/name)
        for r in csvread(ROOT/name):add('Development',r['partition'],r,name)
    for split in ['calibration','test']:
        name=f'configs/final/{split}_manifest.json';source_hashes[name]=sha(ROOT/name)
        for r in structural_records(read(ROOT/name)):add('Study1',split+'/'+r['block_id'],r,name)
    name='configs/revision_r2/data_manifest.csv';source_hashes[name]=sha(ROOT/name)
    for r in csvread(ROOT/name):add('Study2',r['namespace'],r,name)
    # Cross-model raw arrays must match the unique manifest identities, never add replicates.
    checked=0
    folders=[(ROOT/'outputs/e8/calibration','Study1','calibration/CAL-MATCHED-01'),(ROOT/'outputs/e8/P0-GATES-01','Study1','test/P0-GATES-01')]
    for split,env in [('R2_CAL','SYN_A'),('R2_CAL','TDLA'),('R2_TEST','SYN_A'),('R2_TEST','SYN_C'),('R2_TEST','TDLA'),('R2_TEST','TDLC')]:folders.append((ROOT/f'outputs/revision_r2/raw/{split}/{env}','Study2',split+'/'+env))
    for folder,study,part in folders:
        expected={rid for rid,key in seen.items() if key==(study,part)}
        for seed in range(5101,5106):
            ids=[]
            for p in sorted((folder/f's{seed}').glob('*.npz')):
                with np.load(p,allow_pickle=False) as z:ids.extend(z['frame_id'].tolist())
            assert len(ids)==len(expected) and set(ids)==expected,(str(folder),seed,len(ids),len(expected))
            checked+=len(ids)
    csvout(OUT/'frame_manifest.csv',rows)
    csvout(OUT/'split_indices.csv',[{k:r[k] for k in ('study','partition','frame_id','index','channel_seed')} for r in rows])
    write(OUT/'split_boundary_audit.json',dict(status='PASS',unique_frames=len(rows),unique_channel_seeds=len(seeds),raw_model_frame_identifiers_checked=checked,groups=[dict(study=a,partition=b,frames=n) for (a,b),n in groups.items()],source_hashes=source_hashes,scope='All original final manifest IDs and all Study2 IDs; actual main calibration/test arrays checked across five fixed models; no model multiplier in independent n.'))
    print('Frame/seed disjointness PASS',len(rows),flush=True)

def chronology():
    checks=[]
    for study,commit,gridpath,certificate in [('Study1','89f89e861dfbd965bb42283f2069129371affaf8','configs/final/threshold_grids.json','outputs/e8/calibration/certificates.json'),('Study2','18b3880deaf0811e891706cdce914c238dccc165','configs/revision_r2/threshold_grids.json','outputs/revision_r2/calibration/certificates.json')]:
        blob=subprocess.check_output(['git','show',commit+':'+gridpath],cwd=ROOT)
        # Git text normalization may differ, so compare parsed values as well as digest.
        assert json.loads(blob)==read(ROOT/gridpath)
        subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
        stamp=subprocess.check_output(['git','show','-s','--format=%cI',commit],cwd=ROOT,text=True).strip()
        assert datetime.fromisoformat(stamp) < datetime.fromisoformat(read(ROOT/certificate)['created_utc'])
        checks.append(dict(study=study,freeze_commit=commit,commit_time=stamp,calibration_creation=read(ROOT/certificate)['created_utc'],grid_matches_committed_values=True,grid_sha256=sha(ROOT/gridpath)))
    bundles=[]
    for name in ['E7_PROTOCOL_HASHES.json','configs/revision_r2/freeze.json']:
        mismatches=[]
        for path,digest in read(ROOT/name)['files'].items():
            if not (ROOT/path).is_file() or sha(ROOT/path)!=digest:mismatches.append(path)
        assert not mismatches,(name,mismatches)
        bundles.append(dict(path=name,files=len(read(ROOT/name)['files']),status='PASS'))
    retained=read(ROOT/'outputs/revision_r2/manifests/execution_audit.json')
    seal=read(ROOT/'outputs/revision_r2/calibration/certificates.seal.json')
    assert datetime.fromisoformat(seal['utc']) == datetime.fromisoformat(retained['calibration_completed_utc'])
    assert datetime.fromisoformat(seal['utc']) < datetime.fromisoformat(retained['first_test_access_utc'])
    write(OUT/'freeze_chronology.json',dict(status='PASS',checks=checks,bundles=bundles,study2_seal_before_first_test=True,retained_execution_audit_sha256=sha(ROOT/'outputs/revision_r2/manifests/execution_audit.json'),scope='Asserted UTC freeze-before-calibration ordering and committed grid ancestry for both studies; retained seal-before-test evidence asserted for Study2. Study1 seal has no timestamp: its seal-before-test guarantee relies on retained guarded runner logic, not an independently timestamped external audit. Not public preregistration.'))

def receiver():
    training=yaml.safe_load((ROOT/'configs/final/training.yaml').read_text());g=grid();op=operator(5101)
    train=csvread(ROOT/'configs/final/train_manifest.csv');ids=[r['realization_id'] for r in train]
    assert list(op.statistics.train_ids)==ids and len(ids)==4096
    assert sha(ROOT/training['statistics']['path'])==training['statistics']['sha256']
    checks=[]
    for seed in range(5101,5106):
        for name,reference in training['checkpoints'][str(seed)].items():
            p=ROOT/reference['path'];assert sha(p)==reference['sha256']
            saved=torch.load(p,map_location='cpu',weights_only=False)
            model=operator(seed).model if name=='residual' else Helena()
            model.load_state_dict(saved['model']);model.eval()
            checks.append(dict(seed=seed,model=name,parameters=sum(p.numel() for p in model.parameters()),sha256=sha(p),checkpoint_epoch=saved.get('epoch')))
    obs,lab,_,alphabet,_=frame(record('R2_ENGINEERING','SYN_A',2026110301),engineering=True)
    geometry=read(ROOT/'configs/final/g2_geometry.json')['8'];fit,held=geometry_mask(g,.5)
    assert np.flatnonzero(fit).tolist()==geometry['fit_flat_indices']
    assert np.flatnonzero(held).tolist()==geometry['heldout_flat_indices']
    # Perturb every held-out pilot and every data observation; allowed nulls unchanged.
    changed=obs.y.copy();changed[held]=1e6+2e6j;changed[g.data_mask]=-9e5+8e5j
    other=ReceiverObservation(g,changed);a=obs.pilots(fit);b=other.pilots(fit)
    for x,y in zip(op.pair(a),op.pair(b)):np.testing.assert_array_equal(x,y)
    ba,na,va=op.pair(a);xa,sa=pilot_features(a,ba,va);xb,sb=pilot_features(b,ba,va)
    np.testing.assert_array_equal(xa,xb);assert sa==sb
    indexrows=[]
    for frac in [.25,1/3,.5]:
        f,h=holdout(frac);reference=read(ROOT/'configs/revision_r2/g2_geometry.json')[str(frac)]
        assert np.argwhere(f).tolist()==reference['fit'] and np.argwhere(h).tolist()==reference['validation']
        for role,mask in [('fit',f),('heldout',h)]:
            indexrows.extend(dict(fraction=frac,role=role,fft_bin=int(k),symbol=int(l),flat_index=int(k*g.symbols+l)) for k,l in np.argwhere(mask))
    csvout(OUT/'g2_split_indices.csv',indexrows)
    # Force an active robust branch while preserving legal receiver information.
    # Original benign receiver fixture above still checks masking, separately.
    outlier_y=obs.y.copy();outlier_mask=g.pilot_mask.copy();outlier_mask[:,1:]=False
    outlier_y[outlier_mask]+=30+20j
    robust_obs=ReceiverObservation(g,outlier_y)
    # Independent augmented least-squares IRLS, avoiding the production normal solver.
    pp=robust_obs.pilots();k,l=np.nonzero(pp.fit_mask);support=op.statistics.support
    A=np.exp(-2j*np.pi*k[:,None]*np.arange(support)/g.nfft)*g.pilot_grid[pp.fit_mask,None]
    C=np.linalg.cholesky(op.statistics.covariance);B=A@C;t=pp.y_allowed[pp.fit_mask]-A@op.statistics.mean
    v=max(float(np.median(abs(pp.null_samples)**2)/np.log(2)),1e-10)+1e-8
    groups=[np.flatnonzero(l==s) for s in range(8)];sizes=np.array([len(x) for x in groups]);assert np.all(sizes==12)
    kappas=np.sqrt(gamma.ppf(.95,sizes)/sizes)
    def solve(weights):
        return np.linalg.lstsq(np.vstack([np.sqrt(weights)[:,None]*B,np.sqrt(v)*np.eye(support)]),np.r_[np.sqrt(weights)*t,np.zeros(support)],rcond=None)[0]
    u=solve(np.ones(len(t)));history=[]
    for step in range(6):
        e=np.array([np.mean(abs((t-B@u)[group])**2)/v for group in groups]);rho=np.where(e<=kappas**2,e,2*kappas*np.sqrt(e)-kappas**2)
        history.append(float(np.vdot(u,u).real+np.sum(sizes*rho)))
        if step<5:
            w=np.empty(len(t))
            for idx,group in enumerate(groups):w[group]=min(1.,kappas[idx]/np.sqrt(max(e[idx],1e-30)))
            u=solve(w)
    expected=np.exp(-2j*np.pi*np.arange(128)[:,None]*np.arange(support)/128)@(op.statistics.mean+C@u)
    expected=np.repeat(expected[:,None],8,axis=1);expected[~g.active_mask]=0
    actual,detail=block_huber(pp,op.statistics,details=True)
    np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-9)
    np.testing.assert_allclose(detail['objective'],history,rtol=2e-9,atol=2e-8)
    assert np.all(np.diff(history)<=1e-8)
    assert np.min(w)<.5 and history[-1]<history[0]-1e-3
    write(OUT/'receiver_verification.json',dict(status='PASS',engineering_only=True,train_statistics_ids=4096,checkpoints=checks,g2_mask_indices_match=True,heldout_and_data_perturbation_leakage_test='PASS exact equality',block_huber=dict(groups=8,group_sizes=sizes.tolist(),initial_solve_plus_updates=6,fixture='Fixed +30+20j corruption on all first-symbol pilots; synthetic engineering only',minimum_last_update_weight=float(np.min(w)),reference='Independently formed basis and augmented least squares with group IRLS',maximum_channel_difference=float(np.max(abs(actual-expected))),objective=history)))
    print('Receiver/covariance/G2/Block-Huber verification PASS',flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    boundaries();chronology();receiver()
