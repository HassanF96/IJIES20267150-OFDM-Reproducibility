"""Train-only fitting; V-design-only stopping; sequential resumable seeds."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from src.e5.common import OUT,LABEL,write_json,sha
from src.e5.data import simulate_e5
from src.e5.model import ResidualCNN,CandidateOperator
from src.estimators.practical import fit_pilot_statistics,PilotStatistics,grid_signature
from src.estimators.noise import estimate_variance
from src.estimators.features import pilot_features
from src.ofdm.holdout import pilot_holdout
from src.utils.random import rng_for

HOLDOUTS=[(f,p) for f in (.25,.5) for p in ('random','interleaved','block')]

def mask_for(grid,record,fraction,policy):
    return pilot_holdout(grid,fraction,policy,rng_for(record.master_seed,record.namespace,record.realization_id,'holdout'))

def save_stats(path,stats):
    np.savez_compressed(path,mean=stats.mean,covariance=stats.covariance,train_ids=np.asarray(stats.train_ids))

def load_stats(path,grid):
    with np.load(path) as z:return PilotStatistics(z['mean'],z['covariance'],len(z['mean']),tuple(z['train_ids']),grid_signature(grid))

def prepare_training(guard,settings):
    target=OUT/'cache';target.mkdir(parents=True,exist_ok=True)
    stats_path=OUT/'checkpoints/pilot_statistics.npz';stats_path.parent.mkdir(parents=True,exist_ok=True)
    records=guard.records('train')
    if stats_path.exists():
        config,_,_,_=simulate_e5(guard,records[0]);stats=load_stats(stats_path,config.grid)
    else:
        start=time.perf_counter();observations=[]
        for r in records:
            _,obs,_,_=simulate_e5(guard,r);observations.append(obs.pilots())
        stats=fit_pilot_statistics(observations,records,17,regularization=1e-6,shrinkage=.1,eigen_floor=1e-8)
        save_stats(stats_path,stats)
        write_json(OUT/'training/statistics_fit.json',{'label':LABEL,'n_train':len(records),'seconds':time.perf_counter()-start,
                                                     'statistics_hash':sha(stats_path),'target_truth_used':False})
    for partition in ('train','V-design'):
        path=target/(partition+'.npz')
        if path.exists():continue
        start=time.perf_counter();xs=[];targets=[];energies=[];ids=[];views=[]
        operator=CandidateOperator(stats,ResidualCNN())
        for i,r in enumerate(guard.records(partition)):
            config,obs,labels,_=simulate_e5(guard,r)
            masks=[obs.grid.pilot_mask]
            if partition=='train':masks.append(mask_for(obs.grid,r,*HOLDOUTS[i%6])[0])
            for j,mask in enumerate(masks):
                p=obs.pilots(mask);b=operator.base(p);v=estimate_variance(p)
                x,s=pilot_features(p,b,v)
                residual=(labels.h_true-b)*obs.grid.active_mask/s
                xs.append(x[0].astype(np.float32));targets.append(np.stack([residual.real,residual.imag]).astype(np.float32))
                energies.append(np.sum(np.abs(labels.h_true[obs.grid.active_mask]/s)**2))
                ids.append(r.realization_id);views.append('full' if j==0 else str(HOLDOUTS[i%6]))
            if (i+1)%1024==0:print('prepare',partition,i+1,flush=True)
        np.savez_compressed(path,x=np.asarray(xs),target=np.asarray(targets),energy=np.asarray(energies,dtype=np.float32),ids=np.asarray(ids),views=np.asarray(views))
        write_json(OUT/'training'/f'{partition}_preparation.json',{'label':LABEL,'frames':len(guard.records(partition)),
                   'views':len(xs),'seconds':time.perf_counter()-start,'cache_sha256':sha(path)})
    return stats

def train_seed(seed,settings,device,guard=None):
    folder=OUT/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    bestpath=folder/f'seed_{seed}_best.pt';lastpath=folder/f'seed_{seed}_resume.pt'
    summarypath=OUT/'training'/f'seed_{seed}.json'
    if summarypath.exists() and json.loads(summarypath.read_text())['completed']:
        summary=json.loads(summarypath.read_text())
        if sha(bestpath)!=summary['checkpoint_sha256']:raise RuntimeError('Checkpoint hash mismatch on resume')
        return summary
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model=ResidualCNN(settings['model']['width']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['training']['learning_rate'],weight_decay=settings['training']['weight_decay'])
    rng=np.random.default_rng(seed)
    with np.load(OUT/'cache/train.npz') as z:
        if guard is not None and z['ids'].tolist()!=[r.realization_id for r in guard.records('train') for _ in range(2)]:raise PermissionError('Train cache ID mismatch')
        tx=torch.from_numpy(z['x']);ty=torch.from_numpy(z['target']);te=torch.from_numpy(z['energy'])
    with np.load(OUT/'cache/V-design.npz') as z:
        if guard is not None and z['ids'].tolist()!=[r.realization_id for r in guard.records('V-design')]:raise PermissionError('Validation cache ID mismatch')
        vx=torch.from_numpy(z['x']);vy=torch.from_numpy(z['target']);ve=torch.from_numpy(z['energy'])
    for partition in ('train','V-design'):
        expected=json.loads((OUT/'training'/f'{partition}_preparation.json').read_text())['cache_sha256']
        if sha(OUT/'cache'/f'{partition}.npz')!=expected:raise RuntimeError('Training cache checksum mismatch')
    batch=settings['training']['batch_size'];history=[];best=float('inf');bad=0;begin=0;elapsed_prior=0.
    if lastpath.exists():
        saved=torch.load(lastpath,map_location=device,weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];bad=saved['bad'];begin=saved['epoch']+1;rng.bit_generator.state=saved['rng'];elapsed_prior=saved['elapsed']
    start=time.perf_counter();peak=psutil.Process().memory_info().rss
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(begin,settings['training']['max_epochs']):
        model.train();total=0.;order=rng.permutation(len(tx))
        for off in range(0,len(tx),batch):
            idx=order[off:off+batch];x=tx[idx].to(device);y=ty[idx].to(device);energy=te[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=((model(x)-y).square().sum((1,2,3))/energy).mean()
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['training']['gradient_clip']);optimizer.step()
            total+=float(loss.detach())*len(idx)
            peak=max(peak,psutil.Process().memory_info().rss)
        model.eval();values=[]
        with torch.inference_mode():
            for off in range(0,len(vx),batch):
                loss=(model(vx[off:off+batch].to(device))-vy[off:off+batch].to(device)).square().sum((1,2,3))/ve[off:off+batch].to(device)
                values.extend(loss.cpu().tolist())
        vl=float(np.mean(values));improved=vl<best-settings['training']['min_delta']
        row={'epoch':epoch+1,'train_loss':total/len(tx),'validation_loss':vl,'selected':improved,'elapsed_seconds':elapsed_prior+time.perf_counter()-start}
        history.append(row)
        if improved:
            best=vl;bad=0
            torch.save({'label':LABEL,'model':model.state_dict(),'seed':seed,'epoch':epoch+1,'development_loss':vl,'architecture':settings['model']},bestpath)
        else:bad+=1
        torch.save({'label':LABEL,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'best':best,'bad':bad,
                    'epoch':epoch,'rng':rng.bit_generator.state,'elapsed':elapsed_prior+time.perf_counter()-start},lastpath)
        write_json(OUT/'training'/f'seed_{seed}_epochs.json',{'label':LABEL,'history':history})
        print(f'seed {seed} epoch {epoch+1}: train={row["train_loss"]:.6g} V-design={vl:.6g}',flush=True)
        if bad>=settings['training']['patience']:break
    result={'label':LABEL,'completed':True,'seed':seed,'parameter_count':sum(p.numel() for p in model.parameters()),
            'device':device,'epochs':len(history),'best_epoch':torch.load(bestpath,map_location='cpu',weights_only=False)['epoch'],
            'best_validation_loss':best,'training_seconds':elapsed_prior+time.perf_counter()-start,'peak_rss_bytes':peak,
            'peak_allocated_vram':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
            'peak_reserved_vram':torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
            'checkpoint_sha256':sha(bestpath),'history':history,'optimization':settings['training']}
    write_json(summarypath,result)
    return result

def load_model(seed,device='cpu'):
    model=ResidualCNN();checkpoint=torch.load(OUT/'checkpoints'/f'seed_{seed}_best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.to(device).eval();return model
