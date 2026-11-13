"""Set native thread limits before NumPy/Torch import, then run an E6 entry point."""
import os
import sys
import runpy
from pathlib import Path

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
root=Path(__file__).resolve().parents[1]
allowed={'run_e6.py','run_e6_comparators.py','run_e6_recovery.py','retime_e6.py','audit_e6.py','report_e6.py','plot_e6.py'}
if len(sys.argv)<2 or sys.argv[1] not in allowed:raise SystemExit('Expected an allowlisted E6 script basename')
script=root/'scripts'/sys.argv[1]
# Record actual pools in this same process, not merely environment requests.
import json
import hashlib
from datetime import datetime,timezone
import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_info
torch.set_num_threads(1)
np.linalg.solve(np.eye(17),np.ones(17))
pools=threadpool_info()
if any(x['num_threads']!=1 for x in pools):raise RuntimeError('Native thread limit not achieved')
receipts=root/'outputs/e6/launch_receipts';receipts.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
receipt=dict(label='DEVELOPMENT — NOT FINAL RESEARCH RESULT',script=script.name,arguments=sys.argv[2:],
             pid=os.getpid(),utc=stamp,python=sys.executable,threadpools=pools,
             script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(receipts/(script.stem+'_'+stamp+'.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
sys.argv=[str(script),*sys.argv[2:]]
runpy.run_path(str(script),run_name='__main__')
