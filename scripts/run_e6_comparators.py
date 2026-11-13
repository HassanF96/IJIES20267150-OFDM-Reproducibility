"""Sequential bounded E6 comparator addendum, train/validation only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.e6.common import configure
configure()
import argparse
import hashlib
import time
import os
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import psutil
import torch
from src.e6.common import ROOT,OUT,LABEL,SEEDS,sha,git,write_json,read_json,bound_npz,checked_npz,install_file_guard
from src.e6.data import BoundaryGuard,simulate
from src.e6.training import load_stats
from src.e6.comparators import Helena,helena_input,helena_estimate,block_huber
from src.estimators.practical import ls_interpolate,regularized_delay,lmmse_like
from src.estimators.noise import estimate_variance
from src.metrics.errors import frame_loss
from src.e5.scores import data_components
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import demodulate
from src.utils.cost import CostLedger
from scripts.run_e6 import verify_manifest

BASE=OUT/'comparators'; MANIFEST=ROOT/'configs/e6/comparator_manifest.json'
TRAIN=dict(max_epochs=120,patience=50,batch_size=64,learning_rate=.01,adam_epsilon=1e-7,
           lr_patience=40,lr_factor=.8,lr_min=1e-5,loss='unnormalized MSE')

def init():
    if MANIFEST.exists():raise RuntimeError('Comparator manifest exists')
    assert not (BASE/'execution.json').exists()
    sources=[ROOT/'E6_COMPARATOR_ADDENDUM.md',ROOT/'src/e6/comparators.py',Path(__file__),
             ROOT/'outputs/e6/sources/HELENA_official_config.json',ROOT/'configs/e6/execution_manifest.json']
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    binding=hashlib.sha256(str(sorted(files.items())).encode()).hexdigest()
    write_json(MANIFEST,dict(label=LABEL,binding=binding,files=files,training=TRAIN,seeds=SEEDS,
                           addendum_commit=git('log','-1','--format=%H','--','E6_COMPARATOR_ADDENDUM.md')))
    print(binding)

def prepare(guard,binding):
    for partition in ('train','V-design'):
        path=OUT/'cache'/f'helena_{partition}.npz'
        if path.exists():checked_npz(path,binding);continue
        xs=[];ys=[];ids=[]
        for r in guard.records(partition):
            _,obs,lab,_=simulate(guard,r);xs.append(helena_input(obs.pilots()))
            h=lab.h_true[obs.grid.active_bins];ys.append(np.stack((h.real,h.imag)).astype(np.float32));ids.append(r.realization_id)
        bound_npz(path,dict(x=np.asarray(xs),y=np.asarray(ys),frame_id=np.asarray(ids)),dict(binding=binding,partition=partition))

def checkpoint(seed):return OUT/'checkpoints'/f'helena_s{seed}_best.pt'

def train(seed,binding,guard):
    summary=BASE/'training'/f's{seed}.json'
    if summary.exists():
        s=read_json(summary)
        if s['binding']!=binding or s['checkpoint_sha256']!=sha(checkpoint(seed)):raise RuntimeError('Comparator checkpoint binding')
        return s
    arrays=[checked_npz(OUT/'cache'/f'helena_{p}.npz',binding) for p in ('train','V-design')]
    for a,p in zip(arrays,('train','V-design')):
        if a['frame_id'].tolist()!=[r.realization_id for r in guard.records(p)]:raise PermissionError('Comparator data IDs')
        for r in guard.records(p):guard.check(r)
    tx,ty,vx,vy=[torch.from_numpy(a[k]) for a in arrays for k in ('x','y')];del arrays
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model=Helena().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01,eps=1e-7)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.8,patience=40,min_lr=1e-5)
    rng=np.random.default_rng(seed);best=float('inf');bad=0;history=[];begin=0;prior_seconds=0.
    resume=OUT/'checkpoints'/f'helena_s{seed}_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved['binding']!=binding:raise RuntimeError('Comparator resume binding')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];bad=saved['bad'];begin=saved['epoch']+1;prior_seconds=saved['seconds']
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    start=time.perf_counter();rss=psutil.Process().memory_info().rss
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(begin,120):
        model.train();total=0.;order=rng.permutation(len(tx))
        for off in range(0,len(order),64):
            idx=order[off:off+64];x=tx[idx].to(device);y=ty[idx].to(device)
            opt.zero_grad(set_to_none=True);loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite HELENA training')
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        model.eval();mse=[];nmse=[]
        with torch.inference_mode():
            for off in range(0,len(vx),64):
                y=vy[off:off+64].to(device);error=(model(vx[off:off+64].to(device))-y).square()
                mse.extend(error.mean((1,2,3)).cpu().tolist());nmse.extend((error.sum((1,2,3))/y.square().sum((1,2,3))).cpu().tolist())
        val=float(np.mean(mse));sched.step(val)
        history.append(dict(epoch=epoch+1,train_mse=total/len(tx),validation_mse=val,validation_frame_nmse=float(np.mean(nmse)),validation_nmse_p95=float(np.quantile(nmse,.95)),lr=opt.param_groups[0]['lr'],seconds=prior_seconds+time.perf_counter()-start))
        if val<best:
            best=val;bad=0;torch.save(dict(label=LABEL,binding=binding,seed=seed,model=model.state_dict(),epoch=epoch+1,validation_mse=val),checkpoint(seed))
        else:bad+=1
        torch.save(dict(binding=binding,epoch=epoch,best=best,bad=bad,history=history,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],seconds=prior_seconds+time.perf_counter()-start),resume)
        write_json(BASE/'training'/f's{seed}_epochs.json',dict(label=LABEL,history=history))
        rss=max(rss,psutil.Process().memory_info().rss)
        print('HELENA',seed,'epoch',epoch+1,'V-design MSE',val,flush=True)
        if device.startswith('cuda') and torch.cuda.max_memory_reserved()>.7*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('HELENA safe VRAM envelope exceeded')
        if bad>=50:break
    result=dict(label=LABEL,binding=binding,seed=seed,parameters=sum(p.numel() for p in model.parameters()),history=history,
                best_epoch=torch.load(checkpoint(seed),map_location='cpu',weights_only=False)['epoch'],training=TRAIN,
                seconds=prior_seconds+time.perf_counter()-start,peak_rss_bytes=rss,peak_allocated_vram=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                peak_reserved_vram=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,device=device,checkpoint_sha256=sha(checkpoint(seed)),
                stop='patience' if bad>=50 else 'registered epoch cap; convergence not guaranteed')
    write_json(summary,result);return result

def load_model(seed):
    model=Helena();model.load_state_dict(torch.load(checkpoint(seed),map_location='cpu',weights_only=False)['model']);return model.eval()

def estimate(method,pilots,stats,model,ledger=None):
    if method=='helena':return helena_estimate(pilots,model,ledger=ledger)
    if method=='ls':return ls_interpolate(pilots,ledger)
    if method=='delay':return regularized_delay(pilots,17,1e-6,ledger)
    if method=='block_huber':return block_huber(pilots,stats,ledger=ledger)
    if method=='base':return lmmse_like(pilots,stats,ledger=ledger)
    raise ValueError(method)

METHODS=('helena','ls','delay','block_huber','base')

def evaluate(seed,guard,stats,binding):
    model=load_model(seed)
    for partition,role in (('V-assess',None),('V-panel','review'),('V-burst','review')):
        records=guard.records(partition,role);path=OUT/'raw'/f'comparator_s{seed}_{partition}.npz'
        if path.exists():checked_npz(path,binding);continue
        shard_dir=OUT/'shards'/path.stem;shard_dir.mkdir(exist_ok=True,parents=True)
        for off in range(0,len(records),128):
            sp=shard_dir/f'{off:06d}.npz'
            if sp.exists():checked_npz(sp,binding);continue
            rows=[];pred=[]
            for r in records[off:off+128]:
                cfg,obs,lab,_=simulate(guard,r);p=obs.pilots();v=estimate_variance(p)
                candidates={m:estimate(m,p,stats,model) for m in METHODS}
                detail=block_huber(p,stats,details=True)[1]
                row=dict(frame_id=r.realization_id,objective_max_increase=float(np.max(np.diff(detail['objective']))),minimum_group_weight=float(np.min(detail['weights'])))
                for m,h in candidates.items():
                    det=data_components(obs.y[obs.grid.data_mask],h[obs.grid.data_mask],cfg.constellation,v)
                    row[m+'_loss']=frame_loss(h,lab.h_true,obs.grid.active_mask)
                    row[m+'_bit_errors']=int(np.count_nonzero(det['bits']!=lab.transmitted_bits))
                rows.append(row);pred.append(np.stack(list(candidates.values())))
            bound_npz(sp,{k:np.asarray([x[k] for x in rows]) for k in rows[0]},dict(binding=binding,seed=seed,partition=partition))
            bound_npz(OUT/'predictions'/path.stem/sp.name,dict(frame_id=np.asarray([x['frame_id'] for x in rows]),estimates=np.asarray(pred)),dict(binding=binding,seed=seed,methods=METHODS))
            print('comparator score',seed,partition,off+len(rows),'/',len(records),flush=True)
        parts=[checked_npz(p,binding) for p in sorted(shard_dir.glob('*.npz'))];arrays={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
        if arrays['frame_id'].tolist()!=[r.realization_id for r in records]:raise PermissionError('Comparator review IDs')
        bound_npz(path,arrays,dict(binding=binding,seed=seed,partition=partition,role=role))

def benchmark(guard,stats,binding):
    path=BASE/'timing.json'
    if path.exists():return read_json(path)
    model=load_model(5101);records=guard.records('V-assess')[:128];frames=[simulate(guard,r) for r in records]
    def receiver(m,cfg,raw):
        o=ReceiverObservation(cfg.grid,demodulate(raw,cfg.grid.nfft,cfg.grid.symbols,cfg.grid.cp))
        p=o.pilots();h=estimate(m,p,stats,model);v=estimate_variance(p)
        bits=data_components(o.y[o.grid.data_mask],h[o.grid.data_mask],cfg.constellation,v)['bits']
        return h,bits
    for m in METHODS:
        for i in range(10):receiver(m,frames[i][0],frames[i][3])
    rng=np.random.default_rng(61992);rows=[];rss=psutil.Process().memory_info().rss
    for repeat in range(3):
        for i,(cfg,obs,lab,raw) in enumerate(frames):
            for m in rng.permutation(METHODS):
                t=time.perf_counter();h=estimate(m,obs.pilots(),stats,model);e=time.perf_counter()-t
                t=time.perf_counter();rh,bits=receiver(m,cfg,raw);rtime=time.perf_counter()-t
                if not np.allclose(h,rh,rtol=0,atol=1e-12):raise AssertionError('Comparator receiver estimate mismatch')
                rows.append(dict(method=m,repeat=repeat,frame_id=records[i].realization_id,estimator_seconds=e,receiver_seconds=rtime))
            rss=max(rss,psutil.Process().memory_info().rss)
    ledgers={}
    for m in METHODS:
        ledger=CostLedger();estimate(m,frames[0][1].pilots(),stats,model,ledger);ledgers[m]=dict(ledger.counts)
    table=pd.DataFrame(rows);summaries=table.groupby('method')[['estimator_seconds','receiver_seconds']].agg(['mean','median','std']).to_dict()
    result=dict(label=LABEL,binding=binding,rows=rows,ledgers=ledgers,peak_sampled_rss_bytes=rss,model_resident_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                device='CPU one thread batch1 float32 model/complex128 receiver',warmups=10,repeats=3,frames=128,
                summaries={str(k):v for k,v in summaries.items()},note='Separate addendum process/time; do not pool core timing repetitions')
    write_json(path,result);return result

def run(action):
    core=verify_manifest();manifest=read_json(MANIFEST)
    for path,value in manifest['files'].items():
        if sha(ROOT/path)!=value:raise RuntimeError('Comparator source changed: '+path)
    progress=read_json(OUT/'progress.json')
    if progress.get('running') and psutil.pid_exists(progress['pid']):raise RuntimeError('Core heavy worker still running; comparator must wait')
    binding=manifest['binding'];install_file_guard();torch.set_num_threads(1)
    guard=BoundaryGuard(expected_hash=core['files']['configs/e6/split_manifest.csv'])
    if (BASE/'boundary.json').exists():guard.accessed.update(read_json(BASE/'boundary.json')['accessed_ids'])
    write_json(BASE/'progress.json',dict(label=LABEL,running=True,pid=os.getpid(),action=action))
    if not (BASE/'execution.json').exists():write_json(BASE/'execution.json',dict(label=LABEL,binding=binding,commit=git('rev-parse','HEAD'),core_binding=core['binding'],start_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,pid=os.getpid(),training=TRAIN))
    try:
        cfg,_,_,_=simulate(guard,guard.records('train')[0]);stats=load_stats(OUT/'checkpoints/pilot_statistics.npz',cfg.grid)
        if action in ('all','train'):
            prepare(guard,binding)
            for seed in SEEDS:train(seed,binding,guard)
        if action in ('all','evaluate'):
            for seed in SEEDS:evaluate(seed,guard,stats,binding)
            benchmark(guard,stats,binding)
        write_json(BASE/'progress.json',dict(label=LABEL,running=False,pid=os.getpid(),completed=True,action=action))
    except BaseException as e:
        write_json(BASE/'progress.json',dict(label=LABEL,running=False,pid=os.getpid(),completed=False,error=str(e)));raise
    finally:write_json(BASE/'boundary.json',guard.audit())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('init','all','train','evaluate'));a=parser.parse_args()
    if a.action=='init':init()
    else:run(a.action)
