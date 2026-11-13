"""Read-only protocol utilities and explicit final-observation access denial."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '1b21c1f32deaeb7b9fc0312b88a35ff88e906e17'
FINAL = ROOT / 'configs/final'
OUT = ROOT / 'outputs/e7'
SEEDS = (5101, 5102, 5103, 5104, 5105)
METHODS = ('g1', 'g2', 'g3_hard', 'g3_soft', 'g3_anchor', 'evm',
           'evm_neural', 'evm_robust', 'evm_robust_neural', 'weighted_mean',
           'weighted_mean_neural', 'weighted_normalized', 'weighted_normalized_neural',
           'weighted_robust_neural', 'llr', 'llr_neural', 'pilot', 'pilot_mean',
           'g3_soft_mean', 'g3_mean_only')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise PermissionError('Project-only writes')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf8', newline='\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def install_e7_guard():
    """Permit manifest metadata; deny any final observation tree before opening.

    A process audit guard, not a claim of system-wide filesystem auditing.
    E7 never imports/calls a final materializer, and all its programmatic opens
    are logged here. Parent shell commands are separately recorded in the log.
    """
    opened = set()
    denied = []

    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT):
            return
        rel = p.relative_to(ROOT).as_posix()
        parts = rel.lower().split('/')
        # configs/final contains only structural specifications, never outcomes.
        if parts[0] in ('data', 'datasets') or (parts[0] == 'outputs' and any(
                x in ('e8', 'calibration', 'test', 'final', 'final_shift') for x in parts[1:])):
            denied.append(rel)
            raise PermissionError('E7 denies final/data observation access: ' + rel)
        opened.add(rel)

    sys.addaudithook(hook)
    return opened, denied


def structural_records(manifest):
    """Expand IDs/seeds only. No condition/channel/bit/noise RNG is invoked."""
    from src.utils.random import seed_for, STREAMS
    for block in manifest['blocks']:
        namespace = 'e8-frozen-v1/' + manifest['split'] + '/' + block['block_id']
        for i in range(block['frames']):
            rid = hashlib.sha256(canonical([manifest['master_seed'], namespace, i])).hexdigest()
            streams = {s: str(seed_for(manifest['master_seed'], namespace, rid, s)) for s in STREAMS}
            yield dict(realization_id=rid, split=manifest['split'], namespace=namespace,
                       master_seed=manifest['master_seed'], index=i, block_id=block['block_id'],
                       stream_seeds=streams)
