from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/e5/s1'
LABEL = 'DEVELOPMENT / S1 -- NOT FINAL RESEARCH RESULT'
BASELINE = '7ee159dac7c28c098369592141b368ec3ab9afbb'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Project boundary')
    path.parent.mkdir(parents=True, exist_ok=True)
    def native(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=native)+'\n', encoding='utf8', newline='\n')
    temporary.replace(path)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def configure_threads():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[k] = '1'
