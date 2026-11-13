from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import yaml
from src.ofdm.grid import Grid
from src.ofdm.modulation import Constellation
from src.channel.tapped_delay import ChannelLaw
from src.interference.bursts import BurstConfig, validate_geometry

LABEL = 'TECHNICAL VERIFICATION — NOT RESEARCH RESULT'


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    raw: dict
    grid: Grid
    constellation: Constellation
    channel: ChannelLaw
    burst: BurstConfig

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse_config(raw):
    expected = {'label', 'mode', 'master_seed', 'frames', 'grid', 'modulation', 'channel',
                'snr_db', 'snr_definition', 'burst', 'receiver', 'statistics_frames'}
    if set(raw) != expected:
        raise ValueError(f'configuration fields differ: {set(raw)^expected}')
    if raw['label'] != LABEL or raw['mode'] not in ('smoke', 'development'):
        raise ValueError('E4 permits technical smoke/development only, no full mode')
    if any(type(raw[k]) is not int or raw[k] < 1 for k in ('frames', 'statistics_frames')):
        raise ValueError('positive integer frame counts required')
    if type(raw['master_seed']) is not int or raw['master_seed'] < 0:
        raise ValueError('invalid master seed')
    if raw['snr_definition'] != 'nominal_active_Es_ensemble_gain_over_N0' or not np.isfinite(raw['snr_db']):
        raise ValueError('unsupported SNR definition or value')
    g = dict(raw['grid'])
    g['active_signed'] = tuple(g['active_signed'])
    grid = Grid(**g)
    constellation = Constellation(raw['modulation'], grid.data_energy)
    c = dict(raw['channel'])
    c['delays'], c['powers'] = tuple(c['delays']), tuple(c['powers'])
    channel = ChannelLaw(**c)
    if max(channel.delays) > grid.cp:
        raise ValueError('CP insufficient for no-ICI channel')
    burst = BurstConfig(**raw['burst'])
    validate_geometry(burst, grid.frame_samples, grid.nfft+grid.cp)
    r = raw['receiver']
    if set(r) != {'support', 'ridge', 'shrinkage', 'covariance_floor', 'variance_floor', 'solve_floor'}:
        raise ValueError('unknown/missing receiver settings')
    if type(r['support']) is not int or not 1 <= r['support'] <= grid.cp+1:
        raise ValueError('invalid public CP-compatible basis')
    if any(not np.isfinite(r[k]) or r[k] <= 0 for k in ('ridge', 'covariance_floor', 'variance_floor', 'solve_floor')):
        raise ValueError('positive finite numerical regularizers required')
    if not np.isfinite(r['shrinkage']) or not 0 <= r['shrinkage'] <= 1:
        raise ValueError('invalid shrinkage')
    # Defensive copy, so YAML changes do not mutate a running simulator config.
    return SimulatorConfig(json.loads(json.dumps(raw)), grid, constellation, channel, burst)


def load_config(path):
    return parse_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
