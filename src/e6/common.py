from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/e6'
LABEL = 'DEVELOPMENT — NOT FINAL RESEARCH RESULT'
BASELINE = 'efe2fd47d4057fab86dc45e259ea6000d15eee89'
PROTOCOL_COMMIT = 'ae757011fbed9a2b8d3927fc3688edd62310e3b4'
SEEDS = (5101, 5102, 5103, 5104, 5105)
COVERAGES = (.25, .5, .75)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT): raise PermissionError('Project boundary')
    path.parent.mkdir(parents=True, exist_ok=True)
    def native(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=native)+'\n', encoding='utf8', newline='\n')
    tmp.replace(path)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))

def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def configure():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
    (OUT/'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT/'tmp/matplotlib')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def install_file_guard():
    """Accidental final-data access/write guard; not an adversarial sandbox."""
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT): return  # package/OS reads are legitimate
        parts = [p.lower() for p in path.relative_to(ROOT).parts]
        if any(p in ('calibration', 'test', 'final_shift', 'e8') for p in parts):
            raise PermissionError('E6 prohibits final-data file access: '+str(path))
    sys.addaudithook(audit)

def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}

def bound_npz(path, arrays, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix('.json'), dict(label=LABEL, sha256=sha(path), **metadata))

def checked_npz(path, binding=None):
    path = Path(path); meta = read_json(path.with_suffix('.json'))
    if sha(path) != meta['sha256']: raise RuntimeError('NPZ custody failure: '+str(path))
    if binding is not None and meta.get('binding') != binding: raise RuntimeError('Stale run binding')
    return load_npz(path)
