"""E6-only entry point. Never calls final calibration, test or final experiments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.e6.common import configure
configure()
import argparse
import csv
import hashlib
import os
import platform
import time
import traceback
from datetime import datetime,timezone
import numpy as np
import psutil
import torch
from src.e6.common import ROOT,OUT,LABEL,BASELINE,PROTOCOL_COMMIT,SEEDS,sha,write_json,read_json,git,install_file_guard
from src.e6.data import create_manifest,MANIFEST,COUNTS,BoundaryGuard,condition
from src.e6.training import prepare,develop,load_stats
from src.e6.evaluation import evaluate,select_g2_control,challenges,load_raw
from src.e6.analysis import choose_variant,analyze
from src.e6.cost import benchmark

MANIFEST_PATH=ROOT/'configs/e6/execution_manifest.json'
EDITABLE={'STATUS.md','PLAN.md','DECISIONS.md','RESEARCH_LOG.md','REQUIREMENTS_TRACE.md','AI_USE_DISCLOSURE_DRAFT.md','REPRODUCIBILITY.md','RESEARCH_QUESTIONS.md','.gitignore'}

def init():
    if MANIFEST_PATH.exists():raise RuntimeError('Manifest already exists; do not reset registered IDs')
    committed=git('show',PROTOCOL_COMMIT+':E6_DEVELOPMENT_PROTOCOL.md')
    if committed!= (ROOT/'E6_DEVELOPMENT_PROTOCOL.md').read_text(encoding='utf8').strip():raise RuntimeError('Protocol changed')
    rows=create_manifest();configs={}
    for r in rows:
        args=(r['family'],r['modulation'],float(r['snr_db']),int(r['stride']),float(r['inr_db']),int(r['duration']),float(r['probability']))
        if str(args) not in configs:configs[str(args)]=condition(*args).raw
    write_json(ROOT/'configs/e6/conditions.json',dict(label=LABEL,conditions=configs))
    sources=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/run_e6.py',ROOT/'E6_DEVELOPMENT_PROTOCOL.md',ROOT/'requirements-installed.txt',MANIFEST,ROOT/'configs/e6/conditions.json']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    binding=hashlib.sha256(str(sorted(hashes.items())).encode()).hexdigest()
    write_json(MANIFEST_PATH,dict(label=LABEL,protocol_commit=PROTOCOL_COMMIT,baseline=BASELINE,binding=binding,files=hashes,counts=COUNTS,seeds=SEEDS,created_utc=datetime.now(timezone.utc).isoformat()))
    protected={}
    for line in git('ls-tree','-r',BASELINE).splitlines():
        metadata,path=line.split('\t',1)
        if path not in EDITABLE:protected[path]=dict(blob=metadata.split()[2],sha256=sha(ROOT/path))
    write_json(OUT/'protected_baseline.json',dict(label=LABEL,baseline=BASELINE,files=protected,editable=sorted(EDITABLE)))
    write_json(OUT/'boundary_initial.json',BoundaryGuard(expected_hash=hashes[MANIFEST.relative_to(ROOT).as_posix()]).audit())
    print('registered',binding,len(rows),'train/validation IDs',flush=True)

def verify_manifest():
    manifest=read_json(MANIFEST_PATH)
    for path,value in manifest['files'].items():
        if sha(ROOT/path)!=value:raise RuntimeError('Unregistered source/config change: '+path)
    return manifest

def custody():
    prior=read_json(OUT/'protected_baseline.json');bad=[]
    for path,item in prior['files'].items():
        if sha(ROOT/path)!=item['sha256'] or git('hash-object',path)!=item['blob']:bad.append(path)
    result=dict(label=LABEL,status='PASS' if not bad else 'FAIL',protected_files=len(prior['files']),changed=bad)
    write_json(OUT/'protected_verification.json',result)
    if bad:raise RuntimeError('Protected prior evidence modified')
    print(result,flush=True)

def run(action):
    manifest=verify_manifest();binding=manifest['binding'];install_file_guard();torch.set_num_threads(1)
    guard=BoundaryGuard(expected_hash=manifest['files']['configs/e6/split_manifest.csv'])
    previous=OUT/'boundary_latest.json'
    if previous.exists():guard.accessed.update(read_json(previous)['accessed_ids'])
    runtime=OUT/'execution.json'
    if not runtime.exists():
        write_json(runtime,dict(label=LABEL,binding=binding,protocol_commit=PROTOCOL_COMMIT,execution_commit=git('rev-parse','HEAD'),
                   start_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,python_version=sys.version,platform=platform.platform(),
                   cpu=platform.processor(),ram_bytes=psutil.virtual_memory().total,torch=torch.__version__,numpy=np.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,cuda=torch.version.cuda,pid=os.getpid(),mode='development'))
    write_json(OUT/'progress.json',dict(label=LABEL,action=action,pid=os.getpid(),running=True))
    try:
        stats=prepare(guard,binding)
        if action in ('all','train'):width=develop(guard,binding,'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:width=read_json(OUT/'model_selection.json')['width']
        if action in ('all','select'):
            for seed in SEEDS:evaluate(guard,stats,width,seed,'V-select',binding)
            selection=select_g2_control(binding)
            for seed in SEEDS:
                evaluate(guard,stats,width,seed,'V-panel',binding,role='selection')
                evaluate(guard,stats,width,seed,'V-burst',binding,selection['g2'],role='selection')
                challenges(guard,stats,width,seed,'selection',binding)
            timing=benchmark(guard,stats,width,selection['g2'],selection['control'],load_raw(5101,'V-select',binding=binding),'V-select',binding)
            variant=choose_variant(selection,timing,binding)
            write_json(OUT/'selection_snapshot.json',dict(label=LABEL,binding=binding,before_review=True,
                hashes={p.name:sha(p) for p in (OUT/'g2_control_selection.json',OUT/'variant_selection.json',OUT/'model_selection.json')}))
        else:selection=read_json(OUT/'g2_control_selection.json') if (OUT/'g2_control_selection.json').exists() else None
        if action in ('all','review'):
            if not (OUT/'selection_snapshot.json').exists():raise RuntimeError('Review cannot precede selection')
            for filename,value in read_json(OUT/'selection_snapshot.json')['hashes'].items():
                if sha(OUT/filename)!=value:raise RuntimeError('Selection changed after snapshot')
            for seed in SEEDS:
                evaluate(guard,stats,width,seed,'V-assess',binding,selection['g2'])
                evaluate(guard,stats,width,seed,'V-panel',binding,selection['g2'],role='review')
                evaluate(guard,stats,width,seed,'V-burst',binding,selection['g2'],role='review')
                challenges(guard,stats,width,seed,'review',binding)
            benchmark(guard,stats,width,selection['g2'],selection['control'],load_raw(5101,'V-assess',binding=binding),'V-assess',binding)
        if action in ('all','analyze'):
            variant=read_json(OUT/'variant_selection.json');timing=read_json(OUT/'timing/V-assess.json')
            analyze(selection,variant,timing,binding)
        write_json(OUT/'progress.json',dict(label=LABEL,action=action,pid=os.getpid(),running=False,completed=True))
    except BaseException as exc:
        write_json(OUT/'progress.json',dict(label=LABEL,action=action,pid=os.getpid(),running=False,completed=False,error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:write_json(OUT/'boundary_latest.json',guard.audit())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('init','all','train','select','review','analyze','custody'))
    args=parser.parse_args()
    if args.action=='init':init()
    elif args.action=='custody':custody()
    else:run(args.action)
