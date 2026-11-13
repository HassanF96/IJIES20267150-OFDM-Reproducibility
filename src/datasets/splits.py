from dataclasses import dataclass, asdict
import hashlib
from src.utils.random import seed_for, STREAMS

SPLITS = ('train', 'validation', 'calibration', 'test')
SCENARIOS = ('matched', 'development_shift', 'final_shift')


@dataclass(frozen=True, slots=True)
class SplitRecord:
    realization_id: str
    split: str
    scenario_id: str
    namespace: str
    master_seed: int
    index: int
    purpose: str = 'technical_fixture_only'
    partition: str = ''

    def seeds(self):
        return {s: seed_for(self.master_seed, self.namespace, self.realization_id, s) for s in STREAMS}

    def export(self):
        return {**asdict(self), 'stream_seeds': self.seeds(), 'materialized': False}


def generate_manifest(master, counts, scenarios=('matched', 'development_shift'), namespace='e4-fixture-v1'):
    if not set(counts) <= set(SPLITS) or not set(scenarios) <= set(SCENARIOS) or len(set(scenarios)) != len(scenarios):
        raise ValueError('unknown/duplicate split or scenario')
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError('nonnegative integer counts required')
    out = []
    for scenario in scenarios:
        for split in SPLITS:
            if scenario == 'final_shift' and split != 'test' and counts.get(split, 0):
                raise ValueError('reserved final shift may only have test manifest entries')
            ns = f'{namespace}/{scenario}/{split}'
            for i in range(counts.get(split, 0)):
                # Split RNG namespace, not random row splitting of a generated trajectory.
                key = seed_for(master, ns, str(i), 'split')
                rid = hashlib.sha256(f'{ns}/{key:032x}'.encode()).hexdigest()
                partition = ('V-design', 'V-cal-dev', 'V-assess')[i % 3] if split == 'validation' else ''
                out.append(SplitRecord(rid, split, scenario, ns, master, i, partition=partition))
    validate_manifest(out)
    return out


def validate_manifest(records):
    ids, channel_seeds = set(), set()
    for r in records:
        if r.split not in SPLITS or r.scenario_id not in SCENARIOS:
            raise ValueError('invalid split/scenario')
        if r.realization_id in ids or r.seeds()['channel'] in channel_seeds:
            raise ValueError('overlapping channel realization IDs or channel seeds')
        ids.add(r.realization_id)
        channel_seeds.add(r.seeds()['channel'])
    return True
