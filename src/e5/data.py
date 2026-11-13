"""Manifest-authorized E5 data. The physical chain calls the verified E4 primitives.

No relabeling scientific records as E4 technical fixtures; this separate entry point
has an allowlist checked before any random samples are drawn.
"""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
from functools import lru_cache
import numpy as np
import yaml
from src.e5.common import ROOT, OUT, LABEL, sha, write_json
from src.datasets.splits import SplitRecord, validate_manifest
from src.datasets.simulator import EvaluationLabels
from src.utils.random import rng_for, seed_for
from src.utils.config import SimulatorConfig
from src.ofdm.grid import Grid
from src.ofdm.modulation import Constellation
from src.ofdm.information import ReceiverObservation
from src.ofdm.transform import modulate, demodulate, energy_accounting
from src.channel.tapped_delay import ChannelLaw, convolve, frequency_response
from src.channel.noise import complex_awgn, variance_from_snr
from src.interference.bursts import BurstConfig, generate

PURPOSE = 'E5_S1_development_only'
PARTITIONS = ('train','V-design','V-cal-dev','V-assess','V-panel')

def manifest_path(): return ROOT/'configs/e5/split_manifest.csv'

def create_records(settings):
    """IDs/conditions only: no channel, bits, disturbance or output materialized."""
    rows=[]
    for partition in PARTITIONS:
        count = settings['samples'][partition]
        split = 'train' if partition == 'train' else 'validation'
        master = settings['seeds']['train_master' if split=='train' else 'validation_master']
        ns = 'e5-s1-cycle0/'+partition
        rng=np.random.default_rng(seed_for(settings['seeds']['condition_master'],ns,'conditions','split'))
        for i in range(count):
            if partition in ('V-cal-dev','V-assess'):
                family,mod='generic_A','QPSK'
                snr=float(rng.choice(settings['conditions']['primary_snr_db']))
                burst=int(rng.integers(4))
            elif partition=='V-panel':
                cell=i//settings['conditions']['panel_frames_per_cell']
                burst=cell%4; cell//=4
                snr=float(settings['conditions']['snr_db'][cell%7]);cell//=7
                mod=settings['conditions']['modulations'][cell%2];cell//=2
                family=('generic_A','generic_B')[cell]
            else:
                family='generic_A'
                mod=str(rng.choice(settings['conditions']['modulations']))
                snr=float(rng.choice(settings['conditions']['snr_db']))
                burst=int(rng.integers(4))
            scenario='matched' if family=='generic_A' else 'development_shift'
            rid=hashlib.sha256(f'{master}/{ns}/{i}'.encode()).hexdigest()
            record=SplitRecord(rid,split,scenario,ns,master,i,PURPOSE,partition)
            rows.append({**asdict(record),'channel_seed':str(record.seeds()['channel']),
                         'family':family,'modulation':mod,'snr_db':snr,'burst_cell':burst})
    records=[row_record(x) for x in rows]
    validate_manifest(records)
    path=manifest_path();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

def row_record(row):
    fields={k:row[k] for k in ('realization_id','split','scenario_id','namespace','master_seed','index','purpose','partition')}
    fields['master_seed']=int(fields['master_seed']);fields['index']=int(fields['index'])
    return SplitRecord(**fields)

class BoundaryGuard:
    def __init__(self,path=None,expected_hash=None):
        path=manifest_path() if path is None else Path(path)
        if expected_hash is not None and sha(path)!=expected_hash: raise ValueError('split hash mismatch')
        self.path=path
        with path.open(encoding='utf8') as f: self.rows=list(csv.DictReader(f))
        records=[row_record(x) for x in self.rows];validate_manifest(records)
        self.allowed={r.realization_id:r for r in records}
        self.by_id={x['realization_id']:x for x in self.rows}
        self.accessed=set()
        for r in records: self.check(r,record_access=False)

    def check(self,record,record_access=True):
        if (record.split not in ('train','validation') or record.partition not in PARTITIONS
            or record.scenario_id not in ('matched','development_shift') or record.purpose!=PURPOSE):
            raise PermissionError('E5 abort: calibration/test/final shift or unapproved purpose')
        if record != self.allowed.get(record.realization_id):
            raise PermissionError('E5 abort: ID or record not in committed allowlist')
        if (record.partition=='train') != (record.split=='train'):
            raise PermissionError('E5 abort: split/partition contradiction')
        if record_access:self.accessed.add(record.realization_id)

    def records(self,partition):
        if partition not in PARTITIONS: raise PermissionError('E5 partition prohibited')
        return [row_record(x) for x in self.rows if x['partition']==partition]

    def audit(self):
        # Manifest-only final IDs are read solely to prove disjointness, never materialized.
        old=json.loads((ROOT/'outputs/e4/splits/manifest.json').read_text(encoding='utf8'))
        old_records=old['records']
        ids=[r['realization_id'] for r in old_records]+list(self.allowed)
        seeds=[str(r['stream_seeds']['channel']) for r in old_records]+[str(r.seeds()['channel']) for r in self.allowed.values()]
        if len(set(ids))!=len(ids) or len(set(seeds))!=len(seeds):raise ValueError('cross-stage realization overlap')
        return {'label':LABEL,'status':'PASS','existing_manifest_sha256':sha(ROOT/'outputs/e4/splits/manifest.json'),
                'e5_manifest_sha256':sha(self.path),'all_ids_unique':True,'all_channel_seeds_unique':True,
                'e4_manifest_counts':{s:sum(r['split']==s for r in old_records) for s in ('train','validation','calibration','test')},
                'e5_counts':{p:sum(r.partition==p for r in self.allowed.values()) for p in PARTITIONS},
                'accessed_ids':sorted(self.accessed),'final_split_materializations':0,
                'boundary_scope':'Typed allowlist at generator boundary; not a security sandbox against arbitrary Python.'}

@lru_cache(maxsize=128)
def condition_config(family,modulation,snr_db,burst_cell):
    base=yaml.safe_load((ROOT/'configs/development/ofdm.yaml').read_text(encoding='utf8'))
    if family not in ('generic_A','generic_B'):raise PermissionError('Final/unknown channel family forbidden')
    g=dict(base['grid']);g['active_signed']=tuple(g['active_signed']);grid=Grid(**g)
    delays,powers=((0,1,3,5),(.5,.25,.15,.1)) if family=='generic_A' else ((0,2,5,10),(.35,.3,.2,.15))
    channel=ChannelLaw(family,delays,powers)
    if burst_cell not in range(4):raise ValueError('unknown burst cell')
    duration=(1,8,32,32)[burst_cell]
    v=variance_from_snr(snr_db,grid.data_energy)
    power=0. if burst_cell==0 else v*10**((10.,10.,20.)[burst_cell-1]/10)
    burst=BurstConfig(probability=0. if burst_cell==0 else .5,duration=duration,power=power)
    base.update(label=LABEL,mode='development',modulation=modulation,snr_db=snr_db,
                channel=asdict(channel),burst=asdict(burst),e5_burst_cell=burst_cell)
    return SimulatorConfig(base,grid,Constellation(modulation),channel,burst)

def simulate_e5(guard,record):
    guard.check(record)  # MUST precede config selection and random draws.
    row=guard.by_id[record.realization_id]
    config=condition_config(row['family'],row['modulation'],float(row['snr_db']),int(row['burst_cell']))
    grid,constellation=config.grid,config.constellation
    def rng(stream):return rng_for(record.master_seed,record.namespace,record.realization_id,stream)
    bits=rng('bits').integers(0,2,int(grid.data_mask.sum())*constellation.bits_per_symbol,dtype=np.uint8)
    x_grid=grid.map_data(constellation.modulate(bits));x=modulate(x_grid,grid.cp)
    taps=config.channel.draw(rng('channel'));y=convolve(x,taps)
    v=variance_from_snr(config.raw['snr_db'],grid.data_energy)
    y+=complex_awgn(rng('awgn'),y.shape,v)
    interference,q,schedule=generate(rng('interference'),config.burst,len(y),grid.nfft+grid.cp)
    y+=interference
    obs=ReceiverObservation(grid,demodulate(y,grid.nfft,grid.symbols,grid.cp))
    labels=EvaluationLabels(record.realization_id,frequency_response(taps,grid.nfft,grid.symbols),bits,x_grid,v,
                            config.raw['snr_db'],config.channel.family_id,schedule,q,
                            energy_accounting(x_grid,x,grid.cp,bits.size))
    return config,obs,labels,y
