"""E6 allowlisted independent train/validation realizations only."""
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import product
import csv
import hashlib
import numpy as np
from src.e6.common import ROOT, OUT, LABEL, sha, write_json
from src.datasets.splits import SplitRecord, validate_manifest
from src.datasets.simulator import EvaluationLabels
from src.e5.data import condition_config as e5_config, row_record
from src.utils.random import rng_for, seed_for
from src.utils.config import SimulatorConfig
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import modulate, demodulate, energy_accounting
from src.channel.tapped_delay import convolve, frequency_response
from src.channel.noise import complex_awgn, variance_from_snr
from src.interference.bursts import BurstConfig, generate

PURPOSE = 'E6_P0_DEV_train_validation_only'
PARTITIONS = ('train','V-design','V-select','V-assess','V-panel','V-burst')
COUNTS = dict(zip(PARTITIONS, (4096,1536,2048,4096,5376,1728)))
MANIFEST = ROOT/'configs/e6/split_manifest.csv'

def create_manifest():
    rows = []
    panel = list(product(('generic_A','generic_B'), ('QPSK','16QAM'), (-5,0,5,10,15,20,25), (4,8,16), range(4)))
    burst = list(product(('generic_A','generic_B'), (5,15), (0,10,20), (8,32,128), (.25,.5,1.)))
    for partition in PARTITIONS:
        split = 'train' if partition == 'train' else 'validation'
        master = 2026100601 if split == 'train' else 2026100602
        ns = 'e6-p0dev-v1/'+partition
        rng = np.random.default_rng(seed_for(2026100603, ns, 'conditions', 'split'))
        for i in range(COUNTS[partition]):
            cell, role = -1, 'selection' if partition in ('train','V-design','V-select') else 'review'
            if partition == 'V-panel':
                cell = i//16; family, modulation, snr, stride, bc = panel[cell]
                role = 'selection' if i%16 < 8 else 'review'
            elif partition == 'V-burst':
                cell = i//16; family, snr, inr, duration, probability = burst[cell]
                modulation, stride, bc = 'QPSK', 8, -1
                role = 'selection' if i%16 < 8 else 'review'
            else:
                family = 'generic_A'
                if partition in ('V-select','V-assess'):
                    modulation, stride, snr = 'QPSK', 8, int(rng.choice((0,5,10,15,20)))
                else:
                    modulation = str(rng.choice(('QPSK','16QAM')))
                    stride, snr = int(rng.choice((4,8,16))), int(rng.choice((-5,0,5,10,15,20,25)))
                bc = int(rng.integers(4))
            if partition != 'V-burst':
                inr, duration, probability = ((0,1,0.), (10,8,.5), (10,32,.5), (20,32,.5))[bc]
            scenario = 'matched' if family == 'generic_A' else 'development_shift'
            rid = hashlib.sha256(f'{master}/{ns}/{i}'.encode()).hexdigest()
            record = SplitRecord(rid, split, scenario, ns, master, i, PURPOSE, partition)
            rows.append({**asdict(record), 'channel_seed':str(record.seeds()['channel']),
                         'family':family,'modulation':modulation,'snr_db':snr,'stride':stride,
                         'burst_cell':bc,'inr_db':inr,'duration':duration,'probability':probability,
                         'cell':cell,'role':role})
    validate_manifest([row_record(x) for x in rows])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

class BoundaryGuard:
    def __init__(self, path=MANIFEST, expected_hash=None):
        self.path = path
        if expected_hash is not None and sha(path) != expected_hash: raise PermissionError('Manifest hash mismatch')
        with path.open(encoding='utf8') as f: self.rows=list(csv.DictReader(f))
        records=[row_record(x) for x in self.rows];validate_manifest(records)
        self.allowed={r.realization_id:r for r in records}
        self.by_id={r['realization_id']:r for r in self.rows}
        self.accessed=set()
        for record in records: self.check(record, False)

    def check(self, record, record_access=True):
        if record.split not in ('train','validation') or record.partition not in PARTITIONS or record.purpose != PURPOSE:
            raise PermissionError('E6 forbids calibration/test/unknown purpose')
        if record.scenario_id not in ('matched','development_shift') or record.namespace != 'e6-p0dev-v1/'+record.partition:
            raise PermissionError('E6 namespace/scenario forbidden')
        if record != self.allowed.get(record.realization_id): raise PermissionError('Forged/nonallowlisted record')
        if (record.partition == 'train') != (record.split == 'train'): raise PermissionError('Split contradiction')
        row=self.by_id[record.realization_id]
        if row['family'] not in ('generic_A','generic_B'): raise PermissionError('Final channel family forbidden')
        if record_access: self.accessed.add(record.realization_id)

    def records(self, partition, role=None):
        if partition not in PARTITIONS: raise PermissionError('E6 partition forbidden')
        return [row_record(x) for x in self.rows if x['partition']==partition and (role is None or x['role']==role)]

    def audit(self):
        # Only already-authorized E5 train/validation metadata is consulted here.
        with (ROOT/'configs/e5/split_manifest.csv').open(encoding='utf8') as f: prior=list(csv.DictReader(f))
        ids=[r['realization_id'] for r in prior]+list(self.allowed)
        seeds=[r['channel_seed'] for r in prior]+[r['channel_seed'] for r in self.rows]
        if len(ids)!=len(set(ids)) or len(seeds)!=len(set(seeds)): raise RuntimeError('Cross-stage overlap')
        return dict(label=LABEL, manifest_sha256=sha(self.path), counts=COUNTS,
                    accessed_ids=sorted(self.accessed), final_data_materializations=0,
                    final_family_used=False, disjoint_from_e5=True,
                    boundary='exact record allowlist before RNG, path guard; accidental-leakage guard, not sandbox')

@lru_cache(maxsize=512)
def condition(family, modulation, snr, stride, inr, duration, probability):
    if family not in ('generic_A','generic_B'): raise PermissionError('Final family forbidden')
    if stride not in (4,8,16): raise ValueError('Unregistered pilot ratio')
    original=e5_config(family,modulation,snr,0)
    grid=replace(original.grid,pilot_stride=stride)
    v=variance_from_snr(snr,grid.data_energy)
    burst=BurstConfig(probability=probability,duration=duration,power=v*10**(inr/10) if probability else 0.)
    raw=dict(original.raw,label=LABEL,stage='E6',mode='development',grid=asdict(grid),burst=asdict(burst))
    return SimulatorConfig(raw,grid,original.constellation,original.channel,burst)

def simulate(guard, record):
    guard.check(record)  # before condition lookup and every RNG
    r=guard.by_id[record.realization_id]
    cfg=condition(r['family'],r['modulation'],float(r['snr_db']),int(r['stride']),float(r['inr_db']),int(r['duration']),float(r['probability']))
    grid=cfg.grid; alphabet=cfg.constellation
    def rng(stream): return rng_for(record.master_seed,record.namespace,record.realization_id,stream)
    bits=rng('bits').integers(0,2,int(grid.data_mask.sum())*alphabet.bits_per_symbol,dtype=np.uint8)
    xg=grid.map_data(alphabet.modulate(bits)); x=modulate(xg,grid.cp)
    taps=cfg.channel.draw(rng('channel')); y=convolve(x,taps)
    v=variance_from_snr(float(r['snr_db']),grid.data_energy)
    y+=complex_awgn(rng('awgn'),y.shape,v)
    inter,q,schedule=generate(rng('interference'),cfg.burst,len(y),grid.nfft+grid.cp);y+=inter
    obs=ReceiverObservation(grid,demodulate(y,grid.nfft,grid.symbols,grid.cp))
    lab=EvaluationLabels(record.realization_id,frequency_response(taps,grid.nfft,grid.symbols),bits,xg,v,float(r['snr_db']),cfg.channel.family_id,schedule,q,energy_accounting(xg,x,grid.cp,len(bits)))
    return cfg,obs,lab,y
