"""Frozen, sequential, resumable post-review execution. Outputs only in revision."""
from extension_core import *
from datetime import datetime,timezone
import platform,psutil,importlib.metadata
from threadpoolctl import threadpool_info
from audit_certificates import arrays

def utc():return datetime.now(timezone.utc).isoformat()
def run_accuracy(cohort,condition,n,ops):
    for start in range(0,n,128):
        paths={s:OUT/f'raw/{cohort}/{condition}/s{s}/{start:06d}.npz' for s in SEEDS}
        expected_ids=[record(cohort,condition,i)['realization_id'] for i in range(start,min(start+128,n))]
        needed=[s for s,p in paths.items() if not valid_shard(p,expected_ids)]
        if not needed:continue
        rows={s:[] for s in needed};begin=time.perf_counter()
        for i in range(start,min(start+128,n)):
            r=record(cohort,condition,i);obs,lab,raw,abc,phase=frame(r)
            for s in needed:
                rows[s].append(dict(frame_id=r['realization_id'],**evaluate(ops[s],obs,lab,abc,phase=cohort=='PHASE')))
        for s in needed:atomic_npz(paths[s],rows[s])
        print(cohort,condition,start+len(rows[needed[0]]),'/',n,'seconds',round(time.perf_counter()-begin,2),flush=True)

def seal_calibration(spec):
    path=OUT/'stronger_tdl/certificates.json';sealpath=path.with_name('calibration.seal.json')
    if sealpath.exists():assert sha(path)==read(sealpath)['sha256'];return
    certs={};tests=[]
    for condition in spec['cohorts']['CAL']:
        for s in SEEDS:
            d,_=arrays(OUT/f'raw/CAL/{condition}/s{s}',['frame_id','base_loss','neural_loss','g3_hard','g2_50'])
            assert d['frame_id'].tolist()==[record('CAL',condition,i)['realization_id'] for i in range(8192)]
            h=d['neural_loss']>10**.05*d['base_loss'];u=d['base_loss']>10**.05*d['neural_loss']
            for method in ['g3_hard','g2_50']:
                c=calibrate(d[method],h,u,spec['grids'][str(s)][method],target='joint',alpha=spec['alpha'],error=spec['error_each'])
                certs[f'{condition}/{s}/{method}']=c
                tests.extend(dict(condition=condition,seed=s,method=method,error_each=spec['error_each'],**r) for r in c['rows'])
    assert len(tests)==spec['family_tests']
    write(path,dict(created_utc=utc(),protocol_sha256=sha(CFG/'protocol_seal.json'),certificates=certs))
    csvout(OUT/'stronger_tdl/tested_thresholds.csv',tests)
    write(sealpath,dict(utc=utc(),sha256=sha(path),tests=len(tests),family_error=.05))
    print('All extension calibration sealed',flush=True)

def timing(ops,spec):
    from src.e5.cost import finish_bits
    c1=read(ROOT/'outputs/e8/calibration/certificates.json')['certificates'];c2=read(ROOT/'outputs/revision_r2/calibration/certificates.json')['certificates']
    policies=[]
    for s in SEEDS:
        for study,pol,method in [('Study1','g3_hard','g3_hard'),('Study1','g2','g2'),('Study2','joint_d0.5','g3_hard'),('Study2','g2_50','g2')]:
            c=c1[str(s)][pol] if study=='Study1' else c2[f'{s}/{pol}']
            policies.append(dict(seed=s,study=study,policy=pol,method=method,threshold=c['threshold'],certified=c['certified']))
    folder=OUT/'timing';folder.mkdir(parents=True,exist_ok=True)
    observations=[frame(record('TIMING','SYN_A',i)) for i in range(128)]
    for p in policies:
        cut=None if not p['certified'] else (p['threshold'],1.)
        for i in range(spec['timing_warmups']):
            obs,lab,raw,abc,_=observations[i%128]
            validate(p['method'],obs,abc,ops[p['seed']],cut,0.,G2,0)
            receiver(p['method'],raw,obs.grid,abc,ops[p['seed']],cut,0.,G2,0)
    write(folder/'policies.json',policies)
    operations=[]
    for i,(obs,lab,raw,abc,_) in enumerate(observations):
        path=folder/f'raw/{i:06d}.npz'
        if valid_shard(path,[record('TIMING','SYN_A',i)['realization_id']]*(spec['timing_repetitions']*len(policies))):continue
        labels={s:evaluate(ops[s],obs,lab,abc) for s in SEEDS};rows=[]
        journal=folder/f'journal/{i:06d}.jsonl';journal.parent.mkdir(parents=True,exist_ok=True)
        if journal.exists():
            # Every completed measurement is durable. Corrupt/partial journals fail
            # explicitly rather than silently dropping a possibly slow observation.
            rows=[json.loads(line) for line in journal.read_text().splitlines()]
            assert all(r['protocol_sha256']==sha(CFG/'protocol_seal.json') for r in rows)
        done={(r['repeat'],r['seed'],r['study'],r['policy']) for r in rows};assert len(done)==len(rows)
        order_rng=np.random.default_rng(202611030269+i)
        for rep in range(spec['timing_repetitions']):
            for order,j in enumerate(order_rng.permutation(len(policies))):
                p=policies[j];op=ops[p['seed']];cut=None if not p['certified'] else (p['threshold'],1.)
                if (rep,p['seed'],p['study'],p['policy']) in done:continue
                start=time.perf_counter_ns();d=validate(p['method'],obs,abc,op,cut,0.,G2,0);cv=(time.perf_counter_ns()-start)*1e-9
                start=time.perf_counter_ns();bits,dr=receiver(p['method'],raw,obs.grid,abc,op,cut,0.,G2,0);cr=(time.perf_counter_ns()-start)*1e-9
                label=labels[p['seed']];key='g2_50' if p['method']=='g2' else p['method']
                expected=p['certified'] and label[key]<=p['threshold']
                assert d.accepted==dr.accepted==expected
                assert abs(d.score-label[key])<1e-9 and abs(dr.score-label[key])<1e-9
                assert np.count_nonzero(bits!=lab.bits)==label['neural_bit_errors' if expected else 'base_bit_errors']
                measured=dict(frame_id=record('TIMING','SYN_A',i)['realization_id'],frame_index=i,seed=p['seed'],study=p['study'],policy=p['policy'],repeat=rep,execution_order=order,validation_seconds=cv,receiver_seconds=cr,accepted=bool(expected),score=d.score,threshold=p['threshold'],protocol_sha256=sha(CFG/'protocol_seal.json'),measured_utc=utc(),**label)
                with journal.open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(measured,default=lambda v:v.item(),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(measured)
        rows.sort(key=lambda r:(r['repeat'],r['execution_order']))
        atomic_npz(path,rows)
        print('TIMING',i+1,'/ 128',flush=True)
    # Counters outside timing. Retained independently of resume state.
    anchor_policies=[]
    for s in SEEDS:
        c=c2[f'{s}/pilot_anchor']
        anchor_policies.append(dict(seed=s,study='Study2',policy='pilot_anchor',method='g3_anchor',threshold=c['threshold'],certified=c['certified']))
    for i,(obs,lab,raw,abc,_) in enumerate(observations):
        for p in policies+anchor_policies:
            ledger=CostLedger();cut=None if not p['certified'] else (p['threshold'],1.)
            receiver(p['method'],raw,obs.grid,abc,ops[p['seed']],cut,0.,G2,0,ledger)
            operations.append(dict(frame_index=i,**p,ledger=ledger.export()))
    write(folder/'operations.json',operations)

def main():
    commit=verify_protocol();spec=read(CFG/'extension.json');OUT.mkdir(parents=True,exist_ok=True)
    assert psutil.disk_usage(str(ROOT)).free>=8*1024**3,'8 GiB reserve required'
    assert all(p['num_threads']==1 for p in threadpool_info()),'Native single-thread control required'
    metadata=dict(start_utc=utc(),git_commit=commit,protocol_sha256=sha(CFG/'protocol_seal.json'),python=sys.version,os=platform.platform(),cpu=platform.processor(),cpu_count=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,threads=torch.get_num_threads(),thread_pools=threadpool_info(),affinity=psutil.Process().cpu_affinity(),clock=str(time.get_clock_info('perf_counter')),cpu_frequency=psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,libraries={k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','sionna-no-rt','pandas']},processes=[dict(pid=p.pid,name=p.name()) for p in psutil.process_iter() if p.name().lower() in ['python.exe']],power_plan=subprocess.check_output(['powercfg','/getactivescheme'],text=True).strip(),scope='CPU batch 1; no affinity or global power changes; receiver complex128 and neural float32; simulation/truth excluded from timing; OS scheduling uncontrolled.')
    envpath=OUT/'execution_environment.json'
    if envpath.exists():
        previous=read(envpath)
        for key in ['git_commit','protocol_sha256','python','os','cpu','threads','libraries','affinity','power_plan']:assert previous[key]==metadata[key],('Resume environment mismatch',key)
    else:write(envpath,metadata)
    segment=OUT/f'run_segments/{time.time_ns()}.json'
    write(segment,dict(status='STARTED',environment=metadata,resume_policy='Durable timing journal retains every completed measurement; incomplete atomic accuracy shards may be recomputed deterministically. A STARTED segment without a completion receipt denotes interruption.'))
    ops={s:operator(s) for s in SEEDS}
    for c,n in spec['cohorts']['CAL'].items():run_accuracy('CAL',c,n,ops)
    seal_calibration(spec)
    for cohort in ['TEST','PHASE']:
        if not (OUT/f'{cohort.lower()}_first_access.json').exists():write(OUT/f'{cohort.lower()}_first_access.json',dict(utc=utc(),calibration_seal_sha256=sha(OUT/'stronger_tdl/calibration.seal.json')))
        for c,n in spec['cohorts'][cohort].items():run_accuracy(cohort,c,n,ops)
    timing(ops,spec)
    write(OUT/'execution_complete.json',dict(status='EXECUTED',utc=utc(),protocol_commit=commit,protocol_sha256=sha(CFG/'protocol_seal.json')))
    write(segment.with_name(segment.stem+'_complete.json'),dict(status='COMPLETE',utc=utc()))

if __name__=='__main__':main()
