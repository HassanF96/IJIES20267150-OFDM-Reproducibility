import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import psutil
import numpy as np
from src.utils.config import LABEL

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('outputs must stay inside project')
    path.parent.mkdir(parents=True, exist_ok=True)
    def numeric(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Unsupported evidence type: {type(value)}')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=numeric)+'\n',
                    encoding='utf-8', newline='\n')


def source_hashes():
    paths = [ROOT/'pytest.ini', ROOT/'tests/conftest.py', *ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*e4*.py'),
             *ROOT.glob('scripts/verify_awgn.py'), *ROOT.glob('scripts/verify_estimators.py'),
             *ROOT.glob('scripts/characterize_interference.py'), *ROOT.glob('scripts/run_ofdm_smoke.py'),
             *ROOT.glob('scripts/generate_split_manifest.py'), *ROOT.glob('configs/**/*.yaml'),
             *ROOT.glob('tests/unit/*.py'), *ROOT.glob('tests/integration/*.py'), *ROOT.glob('tests/reference/*.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def manifest(config, purpose, protocol=None):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return {
        'label': LABEL, 'purpose': purpose, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': f'e4-{purpose}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}',
        'config': config.raw, 'config_hash': config.config_hash, 'master_seed': config.raw['master_seed'],
        'verification_protocol': protocol,
        'verification_protocol_hash': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() if protocol else None,
        'git_commit_at_execution': git('rev-parse', 'HEAD'),
        'working_tree_dirty_at_execution': bool(git('status', '--porcelain')),
        'source_sha256': source_hashes(), 'python_executable': sys.executable, 'python': sys.version,
        'software': {p: md.version(p) for p in ('numpy', 'scipy', 'PyYAML', 'pytest', 'psutil', 'torch', 'sionna-no-rt')},
        'hardware': {'os': platform.platform(), 'cpu': platform.processor(),
                     'physical_cores': psutil.cpu_count(logical=False), 'logical_cores': psutil.cpu_count(),
                     'ram_bytes': psutil.virtual_memory().total,
                     'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                              '--format=csv,noheader'], text=True).strip()},
        'cpu_thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'device': 'CPU', 'dtype': 'complex128/float64', 'research_training': False, 'S1': 'NOT_RUN'}
