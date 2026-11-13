"""Receiver, data-boundary and calibration kernels for reviewer R2."""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import hashlib,json,subprocess,time,csv
import numpy as np
import torch
from scipy.stats import binom,beta
from src.e7.common import ROOT,sha,write_json,read_json,canonical
from src.e7.protocol import config
from src.e5.model import CandidateOperator,ResidualCNN
from src.e5.training import load_stats
from src.e5.scores import score_pair,pilot_pair_score
from src.e6.comparators import block_huber
from src.e6.holdout import conditioning
from src.estimators.practical import delay_basis
from src.ofdm.grid import Grid
from src.ofdm.modulation import Constellation
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import modulate,demodulate
from src.ofdm.receiver import detect
from src.channel.tapped_delay import ChannelLaw,convolve,frequency_response
from src.channel.noise import complex_awgn,variance_from_snr
from src.interference.bursts import BurstConfig,generate
from src.metrics.errors import frame_loss
from src.utils.random import rng_for,seed_for

OUT=ROOT/'outputs/revision_r2'
CFG=ROOT/'configs/revision_r2'
SEEDS=(5101,5102,5103,5104,5105)
MASTER=202610050201
def grid(): return Grid(128,8,16,tuple(range(-48,0))+tuple(range(1,49)),8)
def record(namespace,environment,index):
    ns=f'{namespace}/{environment}'
    rid=hashlib.sha256(f'IJIES-REVIEWER-R2/{MASTER}/{ns}/{index}'.encode()).hexdigest()
    return dict(namespace=ns,environment=environment,index=int(index),realization_id=rid,master_seed=MASTER,
                channel_seed=str(seed_for(MASTER,ns,rid,'channel')))
def rng(r,stream):return rng_for(r['master_seed'],r['namespace'],r['realization_id'],stream)

@lru_cache(maxsize=3)
def holdout(fraction):
    if fraction not in (.25,1/3,.5):raise ValueError('R2 fixed holdout fraction only')
    g=grid();fit=g.pilot_mask.copy();held=np.zeros(g.shape,bool)
    basis=delay_basis(g,17);gram=basis[fit].conj().T@basis[fit]+1e-6*np.eye(17)
    for ell in range(g.symbols):
        bins=g.active_bins[fit[g.active_bins,ell]]
        for _ in range(int(np.floor(fraction*len(bins)))):
            choices=[]
            for k in bins:
                if not fit[k,ell]:continue
                a=basis[k,ell];candidate=gram-np.outer(a.conj(),a)
                sign,logdet=np.linalg.slogdet(candidate)
                choices.append((float(logdet.real) if sign.real>0 else -np.inf,int(k),candidate))
            _,k,gram=max(choices,key=lambda x:x[0]);fit[k,ell]=False;held[k,ell]=True
    fit.setflags(write=False);held.setflags(write=False)
    return fit,held

def operator(seed):
    spec=config('training.yaml');stats=load_stats(ROOT/spec['statistics']['path'],grid())
    m=ResidualCNN(48)
    m.load_state_dict(torch.load(ROOT/spec['checkpoints'][str(seed)]['residual']['path'],map_location='cpu',weights_only=False)['model'])
    return CandidateOperator(stats,m.eval())

@lru_cache(maxsize=4)
def tdl_model(profile,speed):
    from sionna.phy.channel.tr38901 import TDL
    return TDL(profile,delay_spread=100e-9,carrier_frequency=3.5e9,min_speed=speed,max_speed=speed,
               num_sinusoids=20,precision='double',device='cpu',spec_version='16.1')

def tdl_waveform(x,profile,channel_seed,speed=30/3.6,*,check=False):
    """Sample-rate fading, sinc fractional delays, linear time-varying convolution.

    Sionna's causal array starts at physical lag -6. Its known timing phase is
    removed after FFT, as in OFDMDemodulator. NMSE targets the exact diagonal
    of this time-varying OFDM operator, not a static approximation to received data.
    """
    from sionna.phy import config as sc
    from sionna.phy.channel import cir_to_time_channel,ApplyTimeChannel
    sc.seed=int(channel_seed)%(2**63-1)
    lmin,lmax=-6,8;nt=len(x)+lmax-lmin
    a,tau=tdl_model(profile,float(speed))(1,nt,1.92e6)
    ht=cir_to_time_channel(1.92e6,a,tau,lmin,lmax,normalize=False)
    taps=ht.numpy().reshape(nt,lmax-lmin+1)
    indices=np.arange(nt)[:,None]-np.arange(lmax-lmin+1)[None,:]
    xp=np.concatenate([x,np.zeros(1,complex)])
    indices[(indices<0)|(indices>=len(x))]=len(x)
    y=np.sum(taps*xp[indices],axis=1)
    if check:
        official=ApplyTimeChannel(len(x),lmax-lmin+1,precision='double',device='cpu')(
            torch.from_numpy(x[None,None,None]),ht).numpy().ravel()
        np.testing.assert_allclose(y,official,atol=1e-12,rtol=1e-12)
    # Window exactly matches the existing CP/FFT framing; compensate known l_min.
    phase=np.exp(-2j*np.pi*np.arange(128)*lmin/128)
    means=taps[:len(x)].reshape(8,144,-1)[:,16:].mean(axis=1)
    truth=(np.fft.fft(means,n=128,axis=1)*phase[None,:]).T
    return y[:len(x)],truth,phase,taps

def frame(r,*,engineering=False,tdl_speed=30/3.6,check=False):
    if not engineering:
        verify_freeze()
        if r != allowed_records().get(r['realization_id']):raise PermissionError('Frame is not in the committed R2 allowlist')
        if not r['namespace'].startswith(('R2_CAL/','R2_TEST/')):raise PermissionError('Unknown final namespace')
        if r['namespace'].startswith('R2_TEST/') and not (OUT/'calibration/certificates.seal.json').exists():
            raise PermissionError('All R2 calibration must be sealed before any R2 test access')
    else:
        if not r['namespace'].startswith('R2_ENGINEERING/'):raise PermissionError('Engineering cannot read final IDs')
    g=grid();alphabet=Constellation('QPSK');prng=rng(r,'split')
    snr=int(prng.choice([0,5,10,15,20]));bc=int(prng.integers(4))
    burst=((0,1,0.),(10,8,.5),(10,32,.5),(20,32,.5))[bc]
    bits=rng(r,'bits').integers(0,2,int(g.data_mask.sum())*2,dtype=np.uint8)
    xg=g.map_data(alphabet.modulate(bits));x=modulate(xg,g.cp);env=r['environment']
    if env in ('TDLA','TDLC'):
        y,truth,phase,taps=tdl_waveform(x,env[-1],int(r['channel_seed']),speed=tdl_speed,check=check)
    else:
        family='C' if env=='SYN_C' else 'A';law=channel_laws()[family]
        taps=ChannelLaw('generic_'+family,tuple(law['delays']),tuple(law['powers'])).draw(rng(r,'channel'))
        y=convolve(x,taps);truth=frequency_response(taps,g.nfft,g.symbols);phase=np.ones(128,complex)
    clean_grid=demodulate(y,g.nfft,g.symbols,g.cp)*phase[:,None]
    ici=float(np.sum(np.abs(clean_grid-truth*xg)**2)/np.sum(np.abs(clean_grid)**2))
    var=variance_from_snr(snr,g.data_energy);y=y+complex_awgn(rng(r,'awgn'),y.shape,var)
    inr,duration,probability=burst
    inter,_,schedule=generate(rng(r,'interference'),BurstConfig(probability,duration,var*10**(inr/10) if probability else 0),len(y),144)
    y+=inter;obs=ReceiverObservation(g,demodulate(y,128,8,16)*phase[:,None])
    lab=SimpleNamespace(h_true=truth,bits=bits,xg=xg,variance=var,snr_db=snr,burst_cell=bc,ici_ratio=ici,burst_present=bool(schedule))
    return obs,lab,y,alphabet,phase

@lru_cache(maxsize=1)
def channel_laws():return config('distributions.yaml')['families']

@lru_cache(maxsize=1)
def allowed_records():
    rows=list(csv.DictReader((CFG/'data_manifest.csv').open(encoding='utf8')))
    for r in rows:
        r['master_seed']=int(r['master_seed']);r['index']=int(r['index'])
    return {r['realization_id']:r for r in rows}

def evaluate_frame(op,obs,lab,alphabet,*,fractions=(.25,1/3,.5),perturb=False):
    b,n,v=op.pair(obs.pilots())
    vals,details=score_pair(obs,b,n,alphabet,v,names=('g3_hard','g3_anchor','g1'),return_details=True)
    bh=block_huber(obs.pilots(),op.statistics)
    bhvals,bhd=score_pair(obs,bh,n,alphabet,v,names=('g3_hard',),return_details=True)
    vals['block_score']=bhvals['g3_hard'];vals['estimated_snr_db']=float(10*np.log10(vals.pop('estimated_snr_linear')))
    partial={}
    for f in fractions:
        key={.25:'g2_25',1/3:'g2_33',.5:'g2_50'}[f];fit,held=holdout(f)
        pb,pn,pv=op.pair(obs.pilots(fit));vals[key]=pilot_pair_score(obs,pb,pn,pv,held)
        partial[key+'_base_loss']=frame_loss(pb,lab.h_true,obs.grid.active_mask)
        partial[key+'_neural_loss']=frame_loss(pn,lab.h_true,obs.grid.active_mask)
    # Receiver work finishes before true labels are joined here.
    row={**vals,**partial,'base_loss':frame_loss(b,lab.h_true,obs.grid.active_mask),
         'neural_loss':frame_loss(n,lab.h_true,obs.grid.active_mask),'block_loss':frame_loss(bh,lab.h_true,obs.grid.active_mask),
         'base_bit_errors':int(np.count_nonzero(details['base']['bits']!=lab.bits)),
         'neural_bit_errors':int(np.count_nonzero(details['neural']['bits']!=lab.bits)),
         'block_bit_errors':int(np.count_nonzero(bhd['base']['bits']!=lab.bits)),
         'bits':len(lab.bits),'snr_db':lab.snr_db,'burst_cell':lab.burst_cell,'estimated_variance':v,
         'ici_ratio':lab.ici_ratio,'channel_power':float(np.mean(np.abs(lab.h_true[obs.grid.active_mask])**2))}
    if perturb:
        for name,nn in perturbations(n,obs.grid):
            pv=score_pair(obs,b,nn,alphabet,v,names=('g3_hard','g3_anchor'))
            row[name+'_hard']=pv['g3_hard'];row[name+'_anchor']=pv['g3_anchor']
            row[name+'_loss']=frame_loss(nn,lab.h_true,obs.grid.active_mask)
            row[name+'_bit_errors']=int(np.count_nonzero(detect(obs,nn,alphabet,v)['bits']!=lab.bits))
    return row

def perturbations(n,g):
    patterns=[('global_p90',np.full(g.shape,1j)),('global_m90',np.full(g.shape,-1j)),
      ('carrier_quarters',np.broadcast_to((1j**(np.arange(128)%4))[:,None],g.shape)),
      ('small_p5',np.full(g.shape,np.exp(1j*np.deg2rad(5)))),('small_m5',np.full(g.shape,np.exp(-1j*np.deg2rad(5)))),
      ('small_p15',np.full(g.shape,np.exp(1j*np.deg2rad(15)))),('small_m15',np.full(g.shape,np.exp(-1j*np.deg2rad(15)))),
      ('amplitude_075',np.full(g.shape,.75)),('amplitude_125',np.full(g.shape,1.25))]
    for name,factor in patterns:
        for maskname,mask in [('active',g.active_mask),('data',g.data_mask)]:
            nn=n.copy();nn[mask]*=factor[mask];yield name+'_'+maskname,nn

def calibrate(score,harm,useful,grid_values,*,target,alpha,error,min_accepted=200,min_coverage=.05):
    """Bonferroni over independently fixed thresholds; conditional h|m is binomial.
    No monotonicity assumption for conditional risk. Largest passing threshold.
    error already divides the GLOBAL R2 threshold/seed/policy/stratum family.
    """
    score=np.asarray(score,float);harm=np.asarray(harm,bool);useful=np.asarray(useful,bool)
    if not np.isfinite(score).all() or target not in ('joint','conditional'):raise ValueError('Invalid calibration inputs')
    rows=[];chosen=None;n=len(score)
    for tau in grid_values:
        g=score<=tau;m=int(g.sum());h=int(np.sum(g&harm));u=int(np.sum(g&useful))
        trials=n if target=='joint' else m
        p=float(binom.cdf(h,trials,alpha)) if trials else 1.
        screen=n>0 and m>=min_accepted and m/n>=min_coverage and u/n>=.05
        ok=screen and p<=error
        row=dict(threshold=float(tau),n=n,m=m,h=h,u=u,p_value=p,screen=screen,passed=ok)
        rows.append(row)
        if ok:chosen=row
    return dict(target=target,threshold=None if chosen is None else chosen['threshold'],selected=chosen,
                certified=chosen is not None,status='CERTIFIED_USEFUL_POLICY' if chosen else 'REJECT_ALL_NO_USEFUL_CERTIFICATE',rows=rows)

@lru_cache(maxsize=1)
def verify_freeze():
    seal=read_json(CFG/'freeze.json')
    for path,digest in seal['files'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('R2 frozen file changed: '+path)
    # A committed,ancestor protocol is required. No self-referential commit hash.
    commit=subprocess.check_output(['git','log','-1','--format=%H','--','REVISION_PROTOCOL_R2.md'],cwd=ROOT,text=True).strip()
    if not commit:raise PermissionError('R2 protocol not committed')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
    blob=subprocess.check_output(['git','show',f'{commit}:configs/revision_r2/freeze.json'],cwd=ROOT)
    if json.loads(blob)!=seal:raise PermissionError('R2 configuration not committed with protocol')
    return hashlib.sha256(canonical(seal)).hexdigest(),commit
