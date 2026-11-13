"""Post-review v1 receiver kernels; no historical writers are called."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys,hashlib,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from src.revision_r2.core import (grid,operator,holdout,tdl_waveform,channel_laws,rng,calibrate)
from src.utils.random import seed_for
from src.ofdm.modulation import Constellation
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import modulate,demodulate
from src.channel.tapped_delay import ChannelLaw,convolve,frequency_response
from src.channel.noise import complex_awgn,variance_from_snr
from src.interference.bursts import BurstConfig,generate
from src.e5.scores import score_pair,pilot_pair_score
from src.metrics.errors import frame_loss
from src.e6.cost import validate,receiver
from src.utils.cost import CostLedger
from types import SimpleNamespace
from audit_certificates import read,write,sha,csvout

CFG=ROOT/'revision/configs';OUT=ROOT/'revision/experiments';MASTER=202611030267
SEEDS=tuple(range(5101,5106));G2={'fraction':.5,'policy':'geometry','aggregation':'huber'}
_VERIFIED=False

def record(cohort,condition,index):
    ns=f'IJIES20267150_POSTREVIEW_V1/{cohort}/{condition}'
    rid=hashlib.sha256(json.dumps([MASTER,ns,index],separators=(',',':')).encode()).hexdigest()
    return dict(namespace=ns,cohort=cohort,condition=condition,index=index,realization_id=rid,master_seed=MASTER,channel_seed=str(seed_for(MASTER,ns,rid,'channel')))

def verify_protocol():
    global _VERIFIED
    seal=read(CFG/'protocol_seal.json')
    for p,h in seal['files'].items():assert sha(ROOT/p)==h,('Frozen extension changed',p)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--','revision/configs/protocol_seal.json'],cwd=ROOT,text=True).strip()
    assert commit
    assert json.loads(subprocess.check_output(['git','show',commit+':revision/configs/protocol_seal.json'],cwd=ROOT))==seal
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
    _VERIFIED=True
    return commit

def frame(r,*,engineering=False,check=False,speed_override=None):
    if engineering:assert r['cohort']=='ENGINEERING'
    else:
        assert _VERIFIED,'Commit/hash verification required before final access'
        assert speed_override is None,'Final channel law cannot be overridden'
        spec=read(CFG/'extension.json');count=spec['cohorts'][r['cohort']][r['condition']]
        assert 0<=r['index']<count and r==record(r['cohort'],r['condition'],r['index'])
        if r['cohort']!='CAL':
            seal=read(OUT/'stronger_tdl/calibration.seal.json')
            assert sha(OUT/'stronger_tdl/certificates.json')==seal['sha256']
    g=grid();alphabet=Constellation('QPSK');prng=rng(r,'split')
    snr=int(prng.choice([0,5,10,15,20]));bc=int(prng.integers(4));burst=((0,1,0.),(10,8,.5),(10,32,.5),(20,32,.5))[bc]
    bits=rng(r,'bits').integers(0,2,int(g.data_mask.sum())*2,dtype=np.uint8)
    xg=g.map_data(alphabet.modulate(bits));x=modulate(xg,g.cp)
    if r['condition'].startswith('TDLA'):
        speed=(300 if r['condition']=='TDLA300' else 30)/3.6 if speed_override is None else speed_override
        y,truth,phase,taps=tdl_waveform(x,'A',int(r['channel_seed']),speed=speed,check=check)
    else:
        law=channel_laws()['A'];taps=ChannelLaw('generic_A',tuple(law['delays']),tuple(law['powers'])).draw(rng(r,'channel'))
        y=convolve(x,taps);truth=frequency_response(taps,g.nfft,g.symbols);phase=np.ones(128,complex)
    clean=demodulate(y,128,8,16)*phase[:,None];ici=float(np.sum(abs(clean-truth*xg)**2)/np.sum(abs(clean)**2))
    var=variance_from_snr(snr,g.data_energy);y=y+complex_awgn(rng(r,'awgn'),y.shape,var)
    inr,duration,prob=burst
    inter,_,schedule=generate(rng(r,'interference'),BurstConfig(prob,duration,var*10**(inr/10) if prob else 0),len(y),144)
    y+=inter;obs=ReceiverObservation(g,demodulate(y,128,8,16)*phase[:,None])
    lab=SimpleNamespace(h_true=truth,bits=bits,xg=xg,variance=var,snr_db=snr,burst_cell=bc,ici_ratio=ici,burst_present=bool(schedule))
    return obs,lab,y,alphabet,phase

def phase_diagnostics(n,truth,g):
    angles=[];coherences=[];energies=[]
    for mask in [g.pilot_mask,g.data_mask]:
        aa=[];cc=[];ee=[]
        for ell in range(8):
            nn=n[mask[:,ell],ell];tt=truth[mask[:,ell],ell];cross=np.vdot(tt,nn)
            en=float(np.vdot(nn,nn).real);et=float(np.vdot(tt,tt).real);den=np.sqrt(en*et)
            valid=den>0 and abs(cross)>0
            aa.append(float(np.angle(cross)) if valid else np.nan);cc.append(float(abs(cross)/den) if valid else np.nan);ee.append([en,et])
        angles.append(aa);coherences.append(cc);energies.append(ee)
    angles=np.array(angles);coherences=np.array(coherences);delta=np.angle(np.exp(1j*(angles[1]-angles[0])))
    valid=np.isfinite(delta);coherent=valid&(coherences.min(axis=0)>=.5)
    return dict(phase_angles=angles,phase_coherences=coherences,phase_energies=np.array(energies),phase_delta=delta,phase_valid=valid,
                phase_event=bool(np.any(coherent&(abs(delta)>=np.pi/4))),phase_severe=bool(np.any(coherent&(abs(delta)>=np.pi/2))))

def evaluate(op,obs,lab,alphabet,phase=False):
    b,n,v=op.pair(obs.pilots());vals,details=score_pair(obs,b,n,alphabet,v,names=('g3_hard','g3_anchor'),return_details=True)
    fit,held=holdout(.5);pb,pn,pv=op.pair(obs.pilots(fit));g2=pilot_pair_score(obs,pb,pn,pv,held)
    row=dict(g3_hard=vals['g3_hard'],g3_anchor=vals['g3_anchor'],g2_50=g2,base_loss=frame_loss(b,lab.h_true,obs.grid.active_mask),neural_loss=frame_loss(n,lab.h_true,obs.grid.active_mask),
             base_bit_errors=int(np.count_nonzero(details['base']['bits']!=lab.bits)),neural_bit_errors=int(np.count_nonzero(details['neural']['bits']!=lab.bits)),bits=len(lab.bits),snr_db=lab.snr_db,burst_cell=lab.burst_cell,ici_ratio=lab.ici_ratio)
    if phase:row.update(phase_diagnostics(n,lab.h_true,obs.grid))
    return row

def atomic_npz(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**{k:np.asarray([np.nan if r[k] is None else r[k] for r in rows]) for k in rows[0]})
    tmp.replace(path);write(path.with_suffix('.sha.json'),dict(sha256=sha(path),rows=len(rows),protocol_sha256=sha(CFG/'protocol_seal.json'),ordered_frame_ids_sha256=hashlib.sha256(json.dumps([r['frame_id'] for r in rows],separators=(',',':')).encode()).hexdigest()))

def valid_shard(path,expected_ids):
    if not path.exists():return False
    assert path.with_suffix('.sha.json').exists(),('Incomplete shard requires explicit recovery',str(path))
    meta=read(path.with_suffix('.sha.json'))
    assert sha(path)==meta['sha256'] and meta['protocol_sha256']==sha(CFG/'protocol_seal.json'),('Stale/corrupt shard',str(path))
    assert meta['ordered_frame_ids_sha256']==hashlib.sha256(json.dumps(expected_ids,separators=(',',':')).encode()).hexdigest()
    with np.load(path,allow_pickle=False) as z:assert z['frame_id'].tolist()==expected_ids
    return True
