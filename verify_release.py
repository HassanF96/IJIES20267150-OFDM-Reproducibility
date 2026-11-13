"""Verify release hashes and serialized checkpoints without training or inference."""
from pathlib import Path
import ast,csv,hashlib,json,zipfile,pickletools
import yaml
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'MANIFEST.csv').open(encoding='utf8',newline='')))
for r in rows:
    p=ROOT/r['path'];assert p.is_file() and p.stat().st_size==int(r['bytes']),r['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
source=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'revision/scripts').glob('*.py'))+[Path(__file__)]
for p in source:ast.parse(p.read_text(encoding='utf8'),filename=p.relative_to(ROOT).as_posix())
spec=yaml.safe_load((ROOT/'configs/final/training.yaml').read_text())
checkpoints=[]
for seed,models in spec['checkpoints'].items():
    for model,ref in models.items():
        p=ROOT/ref['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            names=z.namelist();data=next(n for n in names if n.endswith('/data.pkl'))
            # Inspect pickle syntax without executing it or importing PyTorch.
            ops=list(pickletools.genops(z.read(data)));assert ops[-1][0].name=='STOP'
            assert any('/data/' in n for n in names)
        checkpoints.append(dict(seed=int(seed),model=model,sha256=ref['sha256'],serialized_archive_integrity='PASS'))
assert len(checkpoints)==10
print(json.dumps(dict(status='PASS',files=len(rows),parsed_sources=len(source),verified_checkpoint_archives=checkpoints,training=False,inference=False,simulation=False)))
