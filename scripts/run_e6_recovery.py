"""Final bounded E6 training recovery. Core/source-recipe evidence stays unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.e6.common import configure
configure()
import argparse
import hashlib
import os
import time
from datetime import datetime,timezone
import numpy as np
import psutil
import torch
from src.e6.common import ROOT,OUT as MAIN_OUT,LABEL,SEEDS,sha,git,write_json,read_json,bound_npz,checked_npz,install_file_guard
from src.e6.data import BoundaryGuard,simulate
from src.e6.comparators import Helena,helena_input,same_conv
from src.e5.training import load_stats
from scripts.run_e6 import verify_manifest
import scripts.run_e6_comparators as original

OUT=MAIN_OUT/'modern_recovery'
BASE=OUT
TRAIN=dict(original.TRAIN,learning_rate=.001)
MANIFEST=ROOT/'configs/e6/recovery_manifest.json'

def checkpoint(seed):return OUT/'checkpoints'/f'helena_s{seed}_best.pt'

# Training loop copied from the local, already tested comparator implementation.
# Only the declared initial Adam rate differs; same stopping/checkpoint/RNG rules.
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
    model=Helena().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
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

def init():
    if MANIFEST.exists():raise RuntimeError('Recovery manifest already exists')
    assert not (OUT/'execution.json').exists()
    paths=[Path(__file__),ROOT/'scripts/run_e6_comparators.py',ROOT/'src/e6/comparators.py',ROOT/'E6_TRAINING_RECOVERY_PROTOCOL.md',ROOT/'configs/e6/comparator_manifest.json',ROOT/'configs/e6/execution_manifest.json']
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    binding=hashlib.sha256(str(sorted(files.items())).encode()).hexdigest()
    write_json(MANIFEST,dict(label=LABEL,binding=binding,files=files,training=TRAIN,seeds=SEEDS,
               protocol_commit=git('log','-1','--format=%H','--','E6_TRAINING_RECOVERY_PROTOCOL.md')))
    print(binding)

def choose_recipe():
    rates={}
    for label,folder in (('published_0.01',MAIN_OUT/'comparators/training'),('recovery_0.001',OUT/'training')):
        rates[label]=float(np.mean([min(x['validation_mse'] for x in read_json(folder/f's{s}.json')['history']) for s in SEEDS]))
    best=min(rates.values());chosen='published_0.01' if rates['published_0.01']<=1.01*best else 'recovery_0.001'
    paths={str(s):(MAIN_OUT/'checkpoints'/f'helena_s{s}_best.pt' if chosen=='published_0.01' else checkpoint(s)).relative_to(ROOT).as_posix() for s in SEEDS}
    selection=dict(label=LABEL,recipe=chosen,mean_best_v_design_mse=rates,paths=paths,
                   checkpoint_hashes={s:sha(ROOT/p) for s,p in paths.items()},
                   rule='one common recipe across all five seeds; V-design MSE only; within1% retain published rate',before_recovery_review=True)
    path=OUT/'selection.json'
    if path.exists():
        if read_json(path)!=selection:raise RuntimeError('Recovery selection changed after snapshot')
    else:write_json(path,selection)
    return selection

def load_selected(seed):
    selection=read_json(OUT/'selection.json');path=ROOT/selection['paths'][str(seed)]
    if sha(path)!=selection['checkpoint_hashes'][str(seed)]:raise RuntimeError('Selected comparator checkpoint changed')
    model=Helena();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return model.eval()

def selected_activation_check(guard):
    records=guard.records('train')[:32];xs=[]
    for r in records:
        _,obs,_,_=simulate(guard,r);xs.append(helena_input(obs.pilots()))
    x=torch.from_numpy(np.asarray(xs));rows=[]
    for seed in SEEDS:
        model=load_selected(seed)
        with torch.inference_mode():
            f=torch.relu(same_conv(torch.relu(same_conv(x,model.conv1)),model.conv2));residual=model(x)-x
        rows.append(dict(seed=seed,conv2_zero_fraction=float((f==0).float().mean()),conv2_all_zero_frames=int((f.reshape(32,-1)==0).all(1).sum()),residual_across_frame_variance=float(residual.var(0).mean())))
    write_json(OUT/'activation_check.json',dict(label=LABEL,split='train',frame_ids=[r.realization_id for r in records],rows=rows))

def run():
    core=verify_manifest();manifest=read_json(MANIFEST)
    for p,s in manifest['files'].items():
        if sha(ROOT/p)!=s:raise RuntimeError('Recovery binding changed: '+p)
    prior=read_json(MAIN_OUT/'comparators/progress.json')
    if prior.get('running') and psutil.pid_exists(prior['pid']):raise RuntimeError('Original comparator still running; sequential only')
    if not prior.get('completed'):raise RuntimeError('Complete/preserve original comparator first')
    binding=manifest['binding'];install_file_guard();torch.set_num_threads(1)
    guard=BoundaryGuard(expected_hash=core['files']['configs/e6/split_manifest.csv'])
    if (OUT/'boundary.json').exists():guard.accessed.update(read_json(OUT/'boundary.json')['accessed_ids'])
    (OUT/'checkpoints').mkdir(exist_ok=True,parents=True)
    write_json(OUT/'progress.json',dict(label=LABEL,running=True,pid=os.getpid()))
    if not (OUT/'execution.json').exists():write_json(OUT/'execution.json',dict(label=LABEL,binding=binding,commit=git('rev-parse','HEAD'),python=sys.executable,pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),training=TRAIN))
    # Reuse exactly the tested data preparation/evaluation/timing functions with
    # a separate output root. No original file or original model is overwritten.
    original.OUT=OUT;original.BASE=OUT
    try:
        original.prepare(guard,binding)
        for seed in SEEDS:train(seed,binding,guard)
        choose_recipe();selected_activation_check(guard)
        cfg,_,_,_=simulate(guard,guard.records('train')[0]);stats=load_stats(MAIN_OUT/'checkpoints/pilot_statistics.npz',cfg.grid)
        original.load_model=load_selected
        for seed in SEEDS:original.evaluate(seed,guard,stats,binding)
        original.benchmark(guard,stats,binding)
        write_json(OUT/'progress.json',dict(label=LABEL,running=False,pid=os.getpid(),completed=True))
    except BaseException as e:
        write_json(OUT/'progress.json',dict(label=LABEL,running=False,pid=os.getpid(),completed=False,error=str(e)));raise
    finally:write_json(OUT/'boundary.json',guard.audit())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','all'));args=p.parse_args()
    if args.action=='init':init()
    else:run()
