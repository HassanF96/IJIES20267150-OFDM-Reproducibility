"""Reproduce E5 only. Run `prepare` before the preregistration commit, then `run`."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from datetime import datetime,timezone
import numpy as np
import psutil
import torch
import yaml
from src.e5.common import OUT,LABEL,BASELINE,sha,write_json,git
from src.e5.data import create_records,BoundaryGuard,manifest_path,condition_config,simulate_e5
from src.e5.analysis import decision_inequality_names,binomial_sample_plan

def source_paths():
    return sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('tests/e5/*.py'),*ROOT.glob('scripts/*e5*.py'),
                   ROOT/'configs/e5/settings.yaml',ROOT/'configs/e5/split_manifest.csv',ROOT/'configs/e5/condition_configs.json',
                   *[ROOT/p for p in ('S1_PROTOCOL.md','SCORE_THEORY.md','SYSTEM_MODEL.md','CALIBRATION_SPEC.md','VALIDATION_COST_SPEC.md','INFORMATION_AVAILABILITY.md','RESEARCH_SPEC.md')]])

def prepare(settings):
    from src.e5.model import ResidualCNN
    rows=create_records(settings);guard=BoundaryGuard()
    configs={}
    for row in rows:
        key=f"{row['family']}/{row['modulation']}/{row['snr_db']}/{row['burst_cell']}"
        if key not in configs:
            c=condition_config(row['family'],row['modulation'],float(row['snr_db']),int(row['burst_cell']))
            configs[key]={'config':c.raw,'config_hash':c.config_hash}
    write_json(ROOT/'configs/e5/condition_configs.json',{'label':LABEL,'conditions':configs})
    audit=guard.audit();write_json(OUT/'boundary_pre_run.json',audit)
    inequalities=decision_inequality_names(training_seeds=settings['seeds']['training'])
    assert len(inequalities)==settings['statistics']['decision_inequalities']
    plan=binomial_sample_plan(settings['samples']['V-cal-dev'],settings['statistics']['epsilon_total']/30)
    # Source/config/ID bytes fixed before any scientific outputs. No self hash or future commit inside.
    manifest={'label':LABEL,'stage':'E5','s1_protocol':'S1_PROTOCOL.md unchanged E3 preregistration',
        'git_baseline':BASELINE,'simulator_version':'E4 physical primitives at '+BASELINE+'; E5 guarded materializer',
        'settings':settings,'model_parameter_count':sum(p.numel() for p in ResidualCNN(settings['model']['width']).parameters()),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths()},
        'split_manifest_hashes':{'outputs/e4/splits/manifest.json':sha(ROOT/'outputs/e4/splits/manifest.json'),
                                 'configs/e5/split_manifest.csv':sha(manifest_path())},
        'data_boundary':audit,'sample_plan':plan,
        'precision_limit':'At nominal coverage .05, 4096 validation-threshold frames yield P(accepted>=200) about .645; this can produce insufficient evidence. No optional sample extension.',
        'decision_inequalities':inequalities,'decision_rule_source':'S1_PROTOCOL.md section 7 verbatim governs SUCCESS/PARTIAL SUCCESS/FAILURE',
        'diagnostic_rule':'At most 3 logged cycles; fresh V-assess IDs for changed scientific choices; no final data',
        'pilot_budget':{'transmitted_RE':96,'transmitted_energy':96.0,'data_RE':672,'null_RE':256,'heldout_RE':[24,48]},
        'training_sanity_rule':'Finite/shapes; at least one improved and worsened V-design frame; median neural/base loss ratio <10; maximum relative residual >1e-7. Otherwise diagnose before score evaluation.',
        'output_policy':'All outputs DEVELOPMENT/S1. Original complex128 validation candidates saved locally in predictions shards; scalar evidence, checkpoints and hashes retained. No manuscript claims.',
        'hash_policy':'Manifest commit read into untracked-at-launch execution metadata after commit; final completion hash only in final response.'}
    (ROOT/'S1_EXECUTION_MANIFEST.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False,allow_unicode=True),encoding='utf8',newline='\n')
    report=f'''# E5 data boundary report

{LABEL}

Pre-run guard: PASS. Existing E4 manifest SHA-256 `{audit['existing_manifest_sha256']}`; E5 manifest SHA-256 `{audit['e5_manifest_sha256']}`.
All E4 and E5 IDs and channel seeds are disjoint. The old calibration/test entries are metadata only. E5 adds train/validation IDs only. Exact IDs, partitions, master/channel seeds and conditions are in `configs/e5/split_manifest.csv`.
The E5 generator checks split, partition, purpose, scenario and equality against the hashed allowlist before any random samples are drawn. Calibration/test IDs, forged records and final-shift families abort. V-cal-dev belongs to validation and supports only the preregistered development exercise; the final calibration split is never opened.
No scientific frames were materialized before this report. The completed access record will be `outputs/e5/s1/boundary_final.json`, with exact accessed IDs. This guards accidental pipeline leakage, not malicious arbitrary Python access.
'''
    (ROOT/'E5_DATA_BOUNDARY_REPORT.md').write_text(report,encoding='utf8',newline='\n')
    print('Prepared manifest; no scientific data materialized. Commit before run.',flush=True)

def verify_manifest():
    path=ROOT/'S1_EXECUTION_MANIFEST.yaml';manifest=yaml.safe_load(path.read_text(encoding='utf8'))
    expected=dict(manifest['source_sha256'])
    correction_path=ROOT/'S1_IMPLEMENTATION_CORRECTIONS.yaml'
    if correction_path.exists():
        correction=yaml.safe_load(correction_path.read_text(encoding='utf8'))
        if correction['original_manifest_sha256']!=sha(path):raise RuntimeError('Correction belongs to another manifest')
        committed_correction=subprocess.check_output(['git','show','HEAD:S1_IMPLEMENTATION_CORRECTIONS.yaml'],cwd=ROOT)
        if committed_correction!=correction_path.read_bytes():raise RuntimeError('Implementation correction must be committed')
        for name,hashes in correction['files'].items():
            if name not in expected or expected[name]!=hashes['original_sha256']:raise RuntimeError('Invalid original correction hash: '+name)
            expected[name]=hashes['corrected_sha256']
    for name,digest in expected.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Uncommitted/preregistration source mismatch: '+name)
    committed=subprocess.check_output(['git','show','HEAD:S1_EXECUTION_MANIFEST.yaml'],cwd=ROOT)
    if committed!=path.read_bytes():raise RuntimeError('Execution manifest must be committed before S1')
    if git('remote'):raise RuntimeError('Remote forbidden')
    return manifest

def repair_challenges(settings):
    """Genuine cycle1 accounting fix only; never refit or retune scientific choices."""
    import shutil
    manifest=verify_manifest()
    guard=BoundaryGuard(expected_hash=manifest['split_manifest_hashes']['configs/e5/split_manifest.csv'])
    from src.e5.training import load_stats
    from src.e5.challenges import run_challenges
    selection=json.loads((OUT/'design_selection.json').read_text(encoding='utf8'))
    certificates=json.loads((OUT/'development_certificates.json').read_text(encoding='utf8'))['certificates']
    config,_,_,_=simulate_e5(guard,guard.records('train')[0])
    statistics=load_stats(OUT/'checkpoints/pilot_statistics.npz',config.grid)
    retained={}
    for name in ('raw/challenges.csv','summary/challenges.csv','manifests/challenges.json'):
        source=OUT/name;target=OUT/'diagnostics/cycle1/pre_correction'/name
        if not target.resolve().is_relative_to(OUT.resolve()):raise RuntimeError('Project output boundary')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
        retained[name]=sha(target)
    torch.set_num_threads(settings['training']['cpu_threads'])
    run_challenges(guard,settings,statistics,selection,certificates)
    write_json(OUT/'diagnostics/cycle1/correction.json',{'label':LABEL,'cycle':1,
        'reason':'None threshold is fallback, not an operating harm-detection threshold',
        'git_correction_commit':git('rev-parse','HEAD'),'original_manifest_sha256':sha(ROOT/'S1_EXECUTION_MANIFEST.yaml'),
        'correction_manifest_sha256':sha(ROOT/'S1_IMPLEMENTATION_CORRECTIONS.yaml'),
        'retained_original_outputs':retained,
        'rerun_scope':'All112noisy+2analytic frames and66registered challenges; same checkpoint/IDs/scores/thresholds; no primary analysis/model rerun',
        'corrected_outputs':{name:sha(OUT/name) for name in retained}})

def run(settings):
    manifest=verify_manifest()
    torch.set_num_threads(settings['training']['cpu_threads'])
    guard=BoundaryGuard(expected_hash=manifest['split_manifest_hashes']['configs/e5/split_manifest.csv'])
    guard.audit()
    meta_path=OUT/'execution.json'
    if not meta_path.exists():
        metadata={'label':LABEL,'run_id':'e5-s1-cycle0','start_utc':datetime.now(timezone.utc).isoformat(),
            'git_manifest_commit':git('rev-parse','HEAD'),'execution_manifest_sha256':sha(ROOT/'S1_EXECUTION_MANIFEST.yaml'),
            'python_executable':sys.executable,'python':sys.version,'software':{p:md.version(p) for p in ('numpy','scipy','torch','pandas','scikit-learn','matplotlib','psutil','PyYAML')},
            'hardware':{'os':platform.platform(),'cpu':platform.processor(),'ram_bytes':psutil.virtual_memory().total,
                        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()},
            'training_device':'cuda:0' if torch.cuda.is_available() else 'cpu','scoring_device':'cpu','threads':1,
            'source_sha256':manifest['source_sha256'],'final_split_access':False}
        write_json(meta_path,metadata)
    elif json.loads(meta_path.read_text())['execution_manifest_sha256']!=sha(ROOT/'S1_EXECUTION_MANIFEST.yaml'):
        raise RuntimeError('Cannot resume changed manifest into existing outputs')
    from src.e5.training import prepare_training,train_seed,load_model
    from src.e5.model import CandidateOperator
    from src.e5.evaluation import model_sanity,evaluate_partition,design_selection,certify_development
    from src.e5.cost import measure_resolution,run_cost
    from src.e5.challenges import run_challenges
    from src.e5.reporting import run_analysis
    print('Training/statistics preparation',flush=True)
    statistics=prepare_training(guard,settings)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for seed in settings['seeds']['training']:train_seed(seed,settings,device,guard)
    if not (OUT/'sanity.json').exists():model_sanity(guard,settings,statistics)
    elif json.loads((OUT/'sanity.json').read_text())['status']!='PASS':raise RuntimeError('Unresolved model sanity failure')
    for seed in settings['seeds']['training']:evaluate_partition(guard,settings,statistics,seed,'V-design')
    selection,grids=design_selection(settings)
    # Independent of assessment; only the first V-design observation establishes timing resolution.
    resolution_path=OUT/'timing/resolution.json'
    if not resolution_path.exists():
        config,obs,_,raw=simulate_e5(guard,guard.records('V-design')[0])
        operator=CandidateOperator(statistics,load_model(settings['seeds']['training'][0],'cpu'),'cpu')
        resolution=measure_resolution(operator,obs,config.constellation,raw,repeats=settings['timing']['resolution_repetitions'],warmups=settings['timing']['warmups'])
        write_json(resolution_path,{'label':LABEL,**resolution})
    if not (OUT/'design_freeze.json').exists():write_json(OUT/'design_freeze.json',{'label':LABEL,'git_manifest_commit':json.loads(meta_path.read_text())['git_manifest_commit'],
              'design_selection_sha256':sha(OUT/'design_selection.json'),'design_grids_sha256':sha(OUT/'design_grids.json'),
              'resolution_sha256':sha(resolution_path),'before_assessment':not (OUT/'raw/seed_5101_V-assess.npz').exists()})
    for seed in settings['seeds']['training']:evaluate_partition(guard,settings,statistics,seed,'V-cal-dev',selection)
    certificates=certify_development(settings,grids)
    for partition in ('V-assess','V-panel'):
        for seed in settings['seeds']['training']:evaluate_partition(guard,settings,statistics,seed,partition,selection)
    if not (OUT/'timing/benchmark.json').exists():run_cost(guard,settings,statistics,selection,certificates)
    if not (OUT/'manifests/challenges.json').exists():run_challenges(guard,settings,statistics,selection,certificates)
    run_analysis(settings,selection,certificates)
    write_json(OUT/'boundary_final.json',guard.audit())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and 'cache' not in p.parts and 'shards' not in p.parts and p.name!='hashes.json' and p.suffix not in ('.tmp',)}
    write_json(OUT/'hashes.json',{'label':LABEL,'sha256':hashes})
    print('E5 execution complete. Stop before E6.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','repair-challenges']);args=parser.parse_args()
    settings=yaml.safe_load((ROOT/'configs/e5/settings.yaml').read_text(encoding='utf8'))
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Run from project root')
    if settings['stage']!='E5' or settings['mode']!='development':raise RuntimeError('E5 only')
    if args.action=='prepare':prepare(settings)
    elif args.action=='repair-challenges':repair_challenges(settings)
    else:run(settings)

if __name__=='__main__':main()
