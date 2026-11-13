"""Finite sequential model development; selection uses channel loss only."""
import time
import numpy as np
import psutil
import torch
from src.e6.common import OUT, LABEL, SEEDS, sha, write_json, read_json, bound_npz, checked_npz
from src.e6.data import simulate
from src.e6.holdout import HOLDOUTS, masks, holdout_seed
from src.e5.model import ResidualCNN, CandidateOperator
from src.e5.training import save_stats, load_stats
from src.estimators.practical import fit_pilot_statistics
from src.estimators.features import pilot_features
from src.estimators.noise import estimate_variance

TRAINING = dict(learning_rate=.001,weight_decay=.00001,gradient_clip=1.,batch_size=16,max_epochs=20,patience=4,min_delta=1e-6)

def prepare(guard,binding):
    path=OUT/'checkpoints/pilot_statistics.npz';path.parent.mkdir(parents=True,exist_ok=True)
    records=guard.records('train')
    if path.exists():
        meta=read_json(path.with_suffix('.json'))
        if meta['binding']!=binding or meta['sha256']!=sha(path):raise RuntimeError('Stale statistics')
        cfg,_,_,_=simulate(guard,records[0]);stats=load_stats(path,cfg.grid)
    else:
        start=time.perf_counter();obs=[simulate(guard,r)[1].pilots() for r in records]
        stats=fit_pilot_statistics(obs,records,17,regularization=1e-6,shrinkage=.1,eigen_floor=1e-8)
        save_stats(path,stats)
        write_json(path.with_suffix('.json'),dict(label=LABEL,binding=binding,sha256=sha(path),seconds=time.perf_counter()-start,train_ids=[r.realization_id for r in records],truth_used=False))
        del obs
    for partition in ('train','V-design'):
        path=OUT/'cache'/f'{partition}.npz'
        if path.exists():checked_npz(path,binding);continue
        start=time.perf_counter();xs=[];ys=[];energies=[];ids=[];views=[]
        operator=CandidateOperator(stats,ResidualCNN())
        for i,r in enumerate(guard.records(partition)):
            _,obs,lab,_=simulate(guard,r); fit=[obs.grid.pilot_mask]
            if partition=='train':fit.append(masks(obs.grid,*HOLDOUTS[i%len(HOLDOUTS)],holdout_seed(r))[0])
            for j,mask in enumerate(fit):
                p=obs.pilots(mask);base=operator.base(p);v=estimate_variance(p);x,s=pilot_features(p,base,v)
                target=(lab.h_true-base)*obs.grid.active_mask/s
                xs.append(x[0].astype(np.float32));ys.append(np.stack([target.real,target.imag]).astype(np.float32))
                energies.append(float(np.sum(np.abs(lab.h_true[obs.grid.active_mask]/s)**2)))
                ids.append(r.realization_id);views.append('full' if j==0 else str(HOLDOUTS[i%len(HOLDOUTS)]))
            if (i+1)%512==0:print('prepare',partition,i+1,flush=True)
        bound_npz(path,dict(x=np.asarray(xs),target=np.asarray(ys),energy=np.asarray(energies,np.float32),ids=np.asarray(ids),views=np.asarray(views)),dict(binding=binding,partition=partition,seconds=time.perf_counter()-start))
    return stats

def checkpoint_path(width,seed):return OUT/'checkpoints'/f'w{width}_s{seed}_best.pt'

def load_model(width,seed,device='cpu'):
    model=ResidualCNN(width).to(device)
    saved=torch.load(checkpoint_path(width,seed),map_location=device,weights_only=False)
    model.load_state_dict(saved['model']);return model.eval()

def train_one(width,seed,device,binding,guard):
    summary=OUT/'training'/f'w{width}_s{seed}.json'
    if summary.exists():
        prior=read_json(summary)
        if prior['binding']!=binding or sha(checkpoint_path(width,seed))!=prior['checkpoint_sha256']:raise RuntimeError('Checkpoint binding')
        return prior
    tx=checked_npz(OUT/'cache/train.npz',binding);vx=checked_npz(OUT/'cache/V-design.npz',binding)
    if tx['ids'].tolist()!=[r.realization_id for r in guard.records('train') for _ in range(2)]:raise PermissionError('Training cache ID')
    if vx['ids'].tolist()!=[r.realization_id for r in guard.records('V-design')]:raise PermissionError('Validation cache ID')
    train={k:torch.from_numpy(tx[k]) for k in ('x','target','energy')}
    valid={k:torch.from_numpy(vx[k]) for k in ('x','target','energy')};del tx,vx
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model=ResidualCNN(width).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=TRAINING['learning_rate'],weight_decay=TRAINING['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=float('inf');bad=0;begin=0;prior_time=0.
    resume=OUT/'checkpoints'/f'w{width}_s{seed}_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved['binding']!=binding:raise RuntimeError('Resume binding')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];bad=saved['bad'];begin=saved['epoch']+1
        rng.bit_generator.state=saved['rng'];prior_time=saved['elapsed']
    start=time.perf_counter();rss=psutil.Process().memory_info().rss
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    batch=TRAINING['batch_size']
    for epoch in range(begin,TRAINING['max_epochs']):
        model.train();total=0.;order=rng.permutation(len(train['x']))
        for off in range(0,len(order),batch):
            idx=order[off:off+batch];x=train['x'][idx].to(device);y=train['target'][idx].to(device);en=train['energy'][idx].to(device)
            optimizer.zero_grad(set_to_none=True);loss=((model(x)-y).square().sum((1,2,3))/en).mean()
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(idx);rss=max(rss,psutil.Process().memory_info().rss)
        model.eval();values=[]
        with torch.inference_mode():
            for off in range(0,len(valid['x']),batch):
                v=((model(valid['x'][off:off+batch].to(device))-valid['target'][off:off+batch].to(device)).square().sum((1,2,3))/valid['energy'][off:off+batch].to(device))
                values.extend(v.cpu().tolist())
        vl=float(np.mean(values));improved=vl<best-TRAINING['min_delta']
        history.append(dict(epoch=epoch+1,train_loss=total/len(order),validation_loss=vl,validation_p95=float(np.quantile(values,.95)),elapsed=prior_time+time.perf_counter()-start))
        if improved:
            best=vl;bad=0
            torch.save(dict(label=LABEL,binding=binding,width=width,seed=seed,epoch=epoch+1,model=model.state_dict(),validation_loss=vl),checkpoint_path(width,seed))
        else:bad+=1
        torch.save(dict(binding=binding,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,bad=bad,epoch=epoch,rng=rng.bit_generator.state,elapsed=prior_time+time.perf_counter()-start),resume)
        write_json(OUT/'training'/f'w{width}_s{seed}_epochs.json',dict(label=LABEL,history=history))
        print(f'train w{width} seed{seed} epoch{epoch+1} V-design={vl:.6g}',flush=True)
        if device.startswith('cuda') and torch.cuda.max_memory_reserved()>.7*torch.cuda.get_device_properties(0).total_memory:
            raise MemoryError('Registered safe VRAM envelope exceeded; stop for documented resource diagnosis')
        if bad>=TRAINING['patience']:break
    result=dict(label=LABEL,binding=binding,width=width,seed=seed,parameters=sum(p.numel() for p in model.parameters()),best_validation_loss=best,
                best_epoch=torch.load(checkpoint_path(width,seed),map_location='cpu',weights_only=False)['epoch'],epochs=len(history),history=history,
                seconds=prior_time+time.perf_counter()-start,peak_rss_bytes=rss,device=device,
                peak_allocated_vram=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                peak_reserved_vram=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
                checkpoint_sha256=sha(checkpoint_path(width,seed)),training=TRAINING)
    write_json(summary,result);return result

def develop(guard,binding,device):
    rows=[]
    for width in (24,32,48):
        for seed in SEEDS[:3]:rows.append(train_one(width,seed,device,binding,guard))
    means={w:float(np.mean([x['best_validation_loss'] for x in rows if x['width']==w])) for w in (24,32,48)}
    best=min(means.values());width=min(w for w in means if means[w]<=1.01*best)
    write_json(OUT/'model_selection.json',dict(label=LABEL,width=width,mean_losses=means,selection='V-design channel loss only; within1% choose smaller',binding=binding))
    for seed in SEEDS[3:]:train_one(width,seed,device,binding,guard)
    return width
