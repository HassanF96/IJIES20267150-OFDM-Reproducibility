"""E8-only execution adapter for the frozen protocol. NOT RUN DURING E7.

Pure E7 validation does not import this module. All scientific kernels are
unchanged E4-E6 implementations; this adapter binds final IDs and provenance.
"""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import hashlib
import importlib.metadata
import os
import platform
import shutil
import time
import numpy as np
import torch
from src.e7.common import ROOT,FINAL,SEEDS,METHODS,canonical,sha,read_json,write_json,git,structural_records
from src.e7.protocol import config,verify_bundle,calibrate_arrays,design_condition,event_interval
from src.ofdm.grid import Grid
from src.ofdm.modulation import Constellation
from src.channel.tapped_delay import ChannelLaw,convolve,frequency_response
from src.channel.noise import complex_awgn,variance_from_snr
from src.interference.bursts import BurstConfig,generate
from src.ofdm.transform import modulate,demodulate
from src.ofdm.information import ReceiverObservation
from src.utils.random import rng_for,seed_for
from src.e5.model import CandidateOperator,ResidualCNN
from src.e5.training import load_stats
from src.e6.comparators import Helena,helena_estimate,block_huber
from src.e6.holdout import masks,conditioning
from src.e6.scores import scores
from src.e5.scores import pilot_pair_score
from src.e5.analysis import labels_from_losses,gate_metrics,rank_acceptance,discrimination,partial_aurc
from src.estimators.practical import ls_interpolate,regularized_delay
from src.estimators.oracle import matched_lmmse
from src.ofdm.receiver import detect
from src.metrics.errors import frame_loss

E8=ROOT/'outputs/e8'


class Run:
    def __init__(self):
        self.protocol=verify_bundle()
        import yaml
        environment=yaml.safe_load((ROOT/'E7_FREEZE_MANIFEST.yaml').read_text(encoding='utf8'))['environment']
        if platform.python_version()!=environment['python'] or any(importlib.metadata.version(k)!=v for k,v in environment['packages'].items()):
            raise RuntimeError('Frozen Python/package environment changed')
        if shutil.disk_usage(ROOT).free<30*1024**3:
            raise RuntimeError('Frozen E8 free-space reserve30GiB unavailable; preserve evidence and resolve storage')
        if git('diff','--name-only') or git('diff','--cached','--name-only'):
            raise RuntimeError('Commit any authorized state/implementation change before E8; frozen files must verify')
        self.distributions=config('distributions.yaml');self.training=config('training.yaml')
        self.calibration=config('calibration.yaml');self.experiments=config('experiments.yaml')['experiments']
        self.manifests={s:read_json(FINAL/(s+'_manifest.json')) for s in ('calibration','test')}
        self.blocks={s:{b['block_id']:b for b in m['blocks']} for s,m in self.manifests.items()}
        self.ids={s:{} for s in self.manifests}
        for s,m in self.manifests.items():
            for r in structural_records(m):self.ids[s][r['realization_id']]=r
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
        if any(p['num_threads']!=1 for p in pools):raise RuntimeError('Native thread limit failure')
        self.meta=dict(protocol_sha256=self.protocol,git_commit=git('rev-parse','HEAD'),
            python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            native_thread_pools=pools,cpu=platform.processor(),device='cpu',
            run_id='E8-'+self.protocol[:16],created_utc=datetime.now(timezone.utc).isoformat())
        E8.mkdir(parents=True,exist_ok=True)
        if (E8/'run.json').exists():
            old=read_json(E8/'run.json')
            if old['protocol_sha256']!=self.protocol:raise RuntimeError('Existing run uses different frozen protocol')
            if any(old[k]!=self.meta[k] for k in ('python','numpy','torch','cpu','device')):
                raise RuntimeError('Environment changed on resume; explicit provenance correction required')
            self.meta=old
        else:write_json(E8/'run.json',self.meta)
        start=time.perf_counter();self.stats=load_stats(ROOT/self.training['statistics']['path'],self.grid(8))
        self.statistics_load_seconds=time.perf_counter()-start
        self.current=None

    def grid(self,stride):
        return Grid(128,8,16,tuple(self.distributions['grid']['active_signed']),int(stride))

    def models(self,seed):
        if self.current==seed:return
        refs=self.training['checkpoints'][str(seed)]
        model=ResidualCNN(48);model.load_state_dict(torch.load(ROOT/refs['residual']['path'],map_location='cpu',weights_only=False)['model'])
        self.op=CandidateOperator(self.stats,model.eval())
        self.helena=Helena();self.helena.load_state_dict(torch.load(ROOT/refs['helena']['path'],map_location='cpu',weights_only=False)['model']);self.helena.eval()
        self.current=seed

    def records(self,split,experiment=None):
        result=[]
        for r in self.ids[split].values():
            b=self.blocks[split][r['block_id']]
            if experiment is None or b.get('experiment_id')==experiment:result.append(r)
        return result

    def frame(self,split,r):
        if r!=self.ids[split].get(r['realization_id']):raise PermissionError('Record not in frozen allowlist')
        if split=='test' and not (E8/'calibration/certificates.json').exists():
            raise PermissionError('Calibration must finish and seal before any test draw')
        self.active_record=dict(split=split,frame_id=r['realization_id'],block_id=r['block_id'],index=r['index'])
        c=design_condition(self.blocks[split][r['block_id']],r,self.distributions)
        g=self.grid(c['stride']);a=Constellation(c['modulation']);law=self.distributions['families'][c['family']]
        channel=ChannelLaw('generic_'+c['family'],tuple(law['delays']),tuple(law['powers']))
        def rng(stream):return rng_for(r['master_seed'],r['namespace'],r['realization_id'],stream)
        bits=rng('bits').integers(0,2,int(g.data_mask.sum())*a.bits_per_symbol,dtype=np.uint8)
        xg=g.map_data(a.modulate(bits));x=modulate(xg,g.cp);taps=channel.draw(rng('channel'));y=convolve(x,taps)
        v=variance_from_snr(c['snr_db'],g.data_energy);y+=complex_awgn(rng('awgn'),y.shape,v)
        b=c['burst'];bc=BurstConfig(probability=b['probability'],duration=b['duration'],power=v*10**(b['inr_db']/10) if b['probability'] else 0.)
        inter,q,schedule=generate(rng('interference'),bc,len(y),g.nfft+g.cp);y+=inter
        obs=ReceiverObservation(g,demodulate(y,g.nfft,g.symbols,g.cp))
        labels=SimpleNamespace(h_true=frequency_response(taps,g.nfft,g.symbols),bits=bits,xg=xg,variance=v,
            burst_present=bool(schedule),occupancy=float(np.mean(q.reshape(g.symbols,-1)[:,g.cp:]>0)))
        return c,obs,labels,y,a

    def save(self,path,arrays,experiment,seed):
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**arrays)
        temp.replace(path)
        write_json(path.with_suffix('.json'),dict(**self.meta,experiment_id=experiment,seed=seed,
                   sha256=sha(path),frames=len(arrays['frame_id'])))

    def load(self,path):
        m=read_json(path.with_suffix('.json'))
        if m['protocol_sha256']!=self.protocol or m['sha256']!=sha(path):raise RuntimeError('Shard custody failure')
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

    def evaluate(self,split,experiment,seed):
        self.models(seed);records=self.records(split,experiment if split=='test' else None)
        directory=E8/('calibration' if split=='calibration' else experiment)/f's{seed}'
        parts=[]
        for off in range(0,len(records),128):
            rs=records[off:off+128];path=directory/f'{off:06d}.npz'
            if path.exists():
                data=self.load(path)
                if data['frame_id'].tolist()!=[r['realization_id'] for r in rs]:raise RuntimeError('Shard ID mismatch')
                parts.append(data);continue
            rows=[]
            for r in rs:
                c,obs,lab,_,a=self.frame(split,r);b,n,v=self.op.pair(obs.pilots())
                values,detail=scores(obs,b,n,a,v,return_details=True)
                fit,hold=masks(obs.grid,.5,'geometry',0);pb,pn,pv=self.op.pair(obs.pilots(fit))
                values['g2']=pilot_pair_score(obs,pb,pn,pv,hold,aggregation='huber')
                values['identity_arithmetic_error']=values['weighted_evm_identity']-values['g3_hard']
                if abs(values['identity_arithmetic_error'])>1e-10*(1+abs(values['g3_hard'])):
                    raise ArithmeticError('Exact weighted-EVM identity mismatch')
                values['weighted_evm_identity']=values['g3_hard']
                # Practical predictions complete before evaluation labels are joined.
                practical={}
                if split=='test':
                    practical=dict(ls=ls_interpolate(obs.pilots()),delay=regularized_delay(obs.pilots(),17),
                        block_huber=block_huber(obs.pilots(),self.stats),helena=helena_estimate(obs.pilots(),self.helena))
                lb=frame_loss(b,lab.h_true,obs.grid.active_mask);ln=frame_loss(n,lab.h_true,obs.grid.active_mask)
                harm,use=labels_from_losses([lb],[ln])
                row=dict(**values,frame_id=r['realization_id'],block_id=r['block_id'],index=r['index'],
                    family=c['family'],modulation=c['modulation'],stride=c['stride'],snr_db=c['snr_db'],
                    inr_db=c['burst']['inr_db'],duration=c['burst']['duration'],probability=c['burst']['probability'],
                    bits=len(lab.bits),symbols=int(obs.grid.data_mask.sum()),base_loss=lb,neural_loss=ln,
                    harm=bool(harm[0]),useful=bool(use[0]),partial_base_loss=frame_loss(pb,lab.h_true,obs.grid.active_mask),
                    partial_neural_loss=frame_loss(pn,lab.h_true,obs.grid.active_mask),burst_present=lab.burst_present,
                    burst_occupancy=lab.occupancy,estimated_variance=v,
                    tie_key=float(rng_for(2026100803,'e8-ties',r['realization_id'],'split').random()))
                for name in ('base','neural'):
                    row[name+'_bit_errors']=int(np.count_nonzero(detail[name]['bits']!=lab.bits))
                    row[name+'_symbol_errors']=int(np.count_nonzero(np.abs(detail[name]['symbols']-lab.xg[obs.grid.data_mask])>1e-10))
                if split=='test':
                    law=self.distributions['families'][c['family']]
                    ideals=dict(perfect_csi=lab.h_true,matched_lmmse=matched_lmmse(obs.pilots(),law['delays'],law['powers'],lab.variance))
                    for name,h in {**practical,**ideals}.items():
                        dec=detect(obs,h,a,lab.variance if name in ideals else v)
                        row[name+'_loss']=frame_loss(h,lab.h_true,obs.grid.active_mask)
                        row[name+'_bit_errors']=int(np.count_nonzero(dec['bits']!=lab.bits))
                        row[name+'_symbol_errors']=int(np.count_nonzero(np.abs(dec['symbols']-lab.xg[obs.grid.data_mask])>1e-10))
                rows.append(row)
            data={k:np.asarray([x[k] for x in rows]) for k in rows[0]}
            self.save(path,data,experiment,seed);parts.append(data)
            print(experiment,seed,off+len(rs),'/',len(records),flush=True)
        return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def calibrate(self):
        path=E8/'calibration/certificates.json'
        if path.exists():
            cert=read_json(path)
            if cert['protocol_sha256']!=self.protocol or sha(path)!=read_json(path.with_suffix('.seal.json'))['sha256']:
                raise RuntimeError('Certificate seal changed')
            return cert
        grids=read_json(FINAL/'threshold_grids.json')['grids'];cs={}
        for seed in SEEDS:
            d=self.evaluate('calibration','CAL-MATCHED-01',seed);per={}
            for method in METHODS:
                per[method]=calibrate_arrays(d[method],d['harm'],d['useful'],grids[str(seed)][method],self.calibration)
            per['g0']=calibrate_arrays(np.zeros(len(d['harm'])),d['harm'],d['useful'],[],self.calibration,singleton=True)
            cs[str(seed)]=per
        cert=dict(**self.meta,experiment_id='CAL-MATCHED-01',certificates=cs)
        write_json(path,cert);write_json(path.with_suffix('.seal.json'),dict(sha256=sha(path),protocol_sha256=self.protocol))
        return cert

    def tier(self,tier):
        if not (E8/'calibration/certificates.json').exists():raise RuntimeError('Run and seal calibration before a test tier')
        self.calibrate()  # sealed reuse; never alternate selection
        if tier=='P1':
            for e in self.experiments:
                if e['tier']=='P0' and e['frames'] and e['id']!='P0-COST-01':
                    for s in e['seeds']:
                        expected=(e['frames']-1)//128*128
                        if not (E8/e['id']/f's{s}'/f'{expected:06d}.npz').exists():
                            raise RuntimeError('P0 must complete before P1')
            if not (E8/'P0-COST-01/timing.json').exists():raise RuntimeError('P0 cost incomplete')
        for e in self.experiments:
            if e['tier']!=tier:continue
            if e['id']=='P0-PHASE-01':self.phase();continue
            if e['id']=='P0-COST-01':
                from src.e7.final_cost import run_cost
                run_cost(self);continue
            if not e['frames']:continue
            for seed in e['seeds']:self.evaluate('test',e['id'],seed)

    def phase(self):
        self.calibrate();spec=next(e for e in self.experiments if e['id']=='P0-PHASE-01')
        cert=read_json(E8/'calibration/certificates.json')['certificates']
        for seed in SEEDS:
            path=E8/'P0-PHASE-01'/f's{seed}.npz'
            if path.exists():self.load(path);continue
            self.models(seed);rows=[]
            for source in spec['sources']:
                for r in self.records('test',source):
                    if r['index']>=spec['first_frames_per_cell']:continue
                    c,obs,lab,_,a=self.frame('test',r);b,n,v=self.op.pair(obs.pilots())
                    names=('g3_hard','g3_soft','g3_anchor','pilot')
                    before=scores(obs,b,n,a,v,names=names)
                    for intervention in spec['interventions']:
                        nn=n.copy();mask=obs.grid.active_mask if intervention['mask']=='all_active' else obs.grid.data_mask
                        nn[mask]*=np.exp(1j*np.deg2rad(intervention['degrees']))
                        after=scores(obs,b,nn,a,v,names=names)
                        lb=frame_loss(b,lab.h_true,obs.grid.active_mask);ln=frame_loss(nn,lab.h_true,obs.grid.active_mask)
                        e,u=labels_from_losses([lb],[ln]);dec=detect(obs,nn,a,v)
                        row=dict(frame_id=r['realization_id'],source=source,modulation=c['modulation'],
                            intervention=intervention['mask'],degrees=intervention['degrees'],harm=bool(e[0]),useful=bool(u[0]),
                            base_loss=lb,neural_loss=ln,bits=len(lab.bits),neural_bit_errors=int(np.count_nonzero(dec['bits']!=lab.bits)))
                        for m in names:
                            tau=cert[str(seed)][m]['threshold']
                            row[m]=after[m];row[m+'_original']=before[m]
                            row[m+'_accepted']=tau is not None and after[m]<=tau
                            row[m+'_uncalibrated']=after[m]<=0
                        row['weighted_evm_identity']=after['g3_hard'];rows.append(row)
            self.save(path,{k:np.asarray([r[k] for r in rows]) for k in rows[0]},'P0-PHASE-01',seed)


def run(stage):
    context=Run()
    try:
        if stage in ('calibrate','all'):context.calibrate()
        if stage in ('p0','all'):context.tier('P0')
        if stage in ('p1','all'):context.tier('P1')
        if stage in ('analyze','all'):
            if not (E8/'calibration/certificates.json').exists():raise RuntimeError('Analysis requires existing sealed calibration; no simulation fallback')
            from src.e7.final_analysis import analyze
            analyze(context)
    except Exception as exc:
        error=dict(**context.meta,stage=stage,seed=context.current,last_record=getattr(context,'active_record',None),
                   exception=repr(exc),occurred_utc=datetime.now(timezone.utc).isoformat(),scientific_outcome_imputed=False)
        with (E8/'execution_errors.jsonl').open('a',encoding='utf8') as f:
            f.write(canonical(error).decode('utf8')+'\n')
        raise
