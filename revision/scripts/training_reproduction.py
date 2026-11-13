"""Isolated training entry point; smoke outputs never substitute fixed checkpoints.

Run from the extracted reproduction package. Full mode retrains the selected
width-48 architecture/HELENA adaptation; it does not repeat architecture search.
"""
import os,sys,argparse,hashlib,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','full'],required=True);ap.add_argument('--model',choices=['residual','helena','both'],default='both');ap.add_argument('--seed',type=int,choices=range(5101,5106),default=5101);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2]
assert (ROOT/'MANIFEST.csv').exists() and (ROOT/'README.md').exists(),'Execute only from the reproduction package'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def file_hash(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
dependencies={p.relative_to(ROOT).as_posix():file_hash(p) for folder in ['src','configs/development','configs/e5','configs/e6'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
for name in ['scripts/run_e6_comparators.py','scripts/run_e6_recovery.py','revision/scripts/training_reproduction.py']:
    dependencies[name]=file_hash(ROOT/name)
binding=hashlib.sha256(json.dumps(dict(mode=args.mode,model=args.model,seed=args.seed,device=args.device,dependencies=dependencies),sort_keys=True).encode()).hexdigest()
DEST=ROOT/f'revision/training_replay/{args.mode}/{args.model}_s{args.seed}/{binding[:12]}';DEST.mkdir(parents=True,exist_ok=True)
(DEST/'tmp').mkdir(exist_ok=True)
for key in ['TMP','TEMP','TMPDIR']:os.environ[key]=str(DEST/'tmp')
import tempfile
tempfile.tempdir=str(DEST/'tmp')
os.environ['MPLCONFIGDIR']=str(DEST/'matplotlib')
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import src.e6.common as common
common.OUT=DEST
from src.e6.data import BoundaryGuard
import src.e6.training as residual
import scripts.run_e6_comparators as comparator
import scripts.run_e6_recovery as recovery
from src.e6.comparators import Helena
from src.e6.common import sha,write_json,checked_npz
for module in [residual,comparator,recovery]:module.OUT=DEST;module.BASE=DEST
manifest=ROOT/'configs/e6/split_manifest.csv';guard=BoundaryGuard(expected_hash=sha(manifest))
class SmokeGuard:
    def records(self,partition):return guard.records(partition)[:16 if partition=='train' else 8]
    def __getattr__(self,name):return getattr(guard,name)
selected=SmokeGuard() if args.mode=='smoke' else guard
def protect(event,argv):
    if event!='open' or not isinstance(argv[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(argv[0])).resolve();mode=argv[1];flags=argv[2]
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
    if writing and not p.is_relative_to(DEST):raise PermissionError('Training writes restricted to new replay folder')
    if p.is_relative_to(ROOT/'outputs/e8') or p.is_relative_to(ROOT/'outputs/revision_r2'):raise PermissionError('Training cannot read retained final outcomes')
sys.addaudithook(protect)
results=[]
if args.model in ['residual','both']:
    residual.prepare(selected,binding)
    if args.mode=='smoke':residual.TRAINING=dict(residual.TRAINING,max_epochs=1)
    results.append(residual.train_one(48,args.seed,'cpu' if args.device=='cpu' else 'cuda:0',binding,selected))
if args.model in ['helena','both']:
    comparator.prepare(selected,binding);(DEST/'checkpoints').mkdir(parents=True,exist_ok=True)
    if args.mode=='full':results.append(recovery.train(args.seed,binding,selected))
    else:
        a=checked_npz(DEST/'cache/helena_train.npz',binding);torch.manual_seed(args.seed);model=Helena();model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
        x=torch.from_numpy(a['x']);y=torch.from_numpy(a['y']);optimizer.zero_grad(set_to_none=True);loss=(model(x)-y).square().mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
        results.append(dict(model='HELENA',mode='smoke',training_frames=len(x),loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters()),optimizer_step='PASS'))
write_json(DEST/'training_receipt.json',dict(status='PASS',scope='SMOKE/DEVELOPMENT ONLY; not scientific results or replacement weights' if args.mode=='smoke' else 'Isolated retraining of selected architecture/adaptation; no architecture search, thresholds or final test used',args=vars(args),binding=binding,dependencies=dependencies,output=str(DEST),results=results))
print('TRAINING_REPRODUCTION_PASS',args.mode,flush=True)
