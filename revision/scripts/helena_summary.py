"""Descriptive fixed-model paired analysis of retained Study 1 frames only.

No training, waveform generation, threshold fitting or certificate changes.
"""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[2]; P=R
D=R/'reproduction_outputs/helena';D.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cert=json.loads((P/'outputs/e8/calibration/certificates.json').read_text())['certificates']
keys=['frame_id','bits','helena_loss','helena_bit_errors','neural_loss','neural_bit_errors','base_loss','base_bit_errors','g3_hard']
per=[];nm=[];err=[];inputs=[];ids=None;bits=None
for seed in range(5101,5106):
    parts=[]
    for f in sorted((P/f'outputs/e8/P0-GATES-01/s{seed}').glob('*.npz')):
        with np.load(f,allow_pickle=False) as a:parts.append({k:a[k].copy() for k in keys})
        inputs.append(dict(path=f.relative_to(P).as_posix(),sha256=sha(f)))
    assert parts,seed
    a={k:np.concatenate([z[k] for z in parts]) for k in keys}
    assert len(a['frame_id'])==8192 and len(set(a['frame_id']))==8192
    order=np.argsort(a['frame_id']);a={k:v[order] for k,v in a.items()}
    if ids is None:ids=a['frame_id'];bits=a['bits']
    else:np.testing.assert_array_equal(ids,a['frame_id']);np.testing.assert_array_equal(bits,a['bits'])
    c=cert[str(seed)]['g3_hard'];g=a['g3_hard']<=c['threshold'] if c['certified'] else np.zeros(8192,bool)
    losses=[a['neural_loss'],np.where(g,a['neural_loss'],a['base_loss'])]
    errors=[a['neural_bit_errors'],np.where(g,a['neural_bit_errors'],a['base_bit_errors'])]
    assert all(np.isfinite(x).all() for x in losses+[a['helena_loss']])
    nm.append(np.stack([x-a['helena_loss'] for x in losses]))
    err.append(np.stack([x-a['helena_bit_errors'] for x in errors]))
    per.append(dict(seed=seed,frames=8192,helena_nmse=float(a['helena_loss'].mean()),helena_ber=float(a['helena_bit_errors'].sum()/bits.sum()),cnn_nmse=float(losses[0].mean()),cnn_ber=float(errors[0].sum()/bits.sum()),g3_nmse=float(losses[1].mean()),g3_ber=float(errors[1].sum()/bits.sum())))
nm=np.mean(nm,axis=0);err=np.mean(err,axis=0)
rng=np.random.default_rng(20261113);B=10000;draw=np.empty((B,4))
for start in range(0,B,25):
    ix=rng.integers(0,8192,size=(min(25,B-start),8192))
    draw[start:start+len(ix),:2]=nm[:,ix].mean(axis=2).T
    draw[start:start+len(ix),2:]=err[:,ix].sum(axis=2).T/bits[ix].sum(axis=1)[:,None]
rows=[]
for i,method in enumerate(['Always-neural CNN','Calibrated G3-hard']):
    for metric,col,value in [('Mean linear NMSE',i,nm[i].mean()),('BER',i+2,err[i].sum()/bits.sum())]:
        lo,hi=np.quantile(draw[:,col],[.025,.975]);rows.append(dict(method=method,contrast='method minus local HELENA adaptation',metric=metric,estimate=float(value),lower=float(lo),upper=float(hi)))
result=dict(scope='POST-HOC DESCRIPTIVE. Pointwise 95% paired whole-frame percentile bootstrap; 10000 resamples, RNG seed 20261113; the same 8192 frame identities are resampled jointly for all five fixed models. All five HELENA seeds, including 5104, retained. No trained-population inference or multiplicity-controlled superiority claim.',matching='Exact sorted frame IDs and bit counts match across all five seeds; each endpoint pair is retained in the same raw shard.',per_seed=per,comparisons=rows,inputs=inputs)
(D/'helena_paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['scope','matching','per_seed','comparisons']},indent=2))
