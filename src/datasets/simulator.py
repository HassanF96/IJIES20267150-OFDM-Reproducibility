"""Generator side: returns receiver observations and labels as separate objects."""
from dataclasses import dataclass
import numpy as np
from src.utils.random import rng_for
from src.ofdm.transform import modulate, demodulate, energy_accounting
from src.ofdm.information import ReceiverObservation
from src.channel.tapped_delay import convolve, frequency_response
from src.channel.noise import complex_awgn, variance_from_snr
from src.interference.bursts import generate


@dataclass(frozen=True, slots=True)
class EvaluationLabels:
    realization_id: str
    h_true: np.ndarray
    transmitted_bits: np.ndarray
    transmitted_grid: np.ndarray
    noise_variance_true: float
    nominal_snr_db: float
    channel_family: str
    burst_schedule: tuple
    burst_variance_time: np.ndarray
    energy: dict


def simulate(config, record):
    if record.purpose != 'technical_fixture_only' or record.split not in ('train', 'validation'):
        raise ValueError('E4 materialization restricted to technical train/validation fixtures')
    if record.scenario_id == 'final_shift':
        raise ValueError('final shift materialization prohibited in E4')
    grid, constellation = config.grid, config.constellation
    def rng(stream):
        return rng_for(record.master_seed, record.namespace, record.realization_id, stream)
    bits = rng('bits').integers(0, 2, int(grid.data_mask.sum())*constellation.bits_per_symbol, dtype=np.uint8)
    x_grid = grid.map_data(constellation.modulate(bits))
    x = modulate(x_grid, grid.cp)
    taps = config.channel.draw(rng('channel'))
    y = convolve(x, taps)
    v = variance_from_snr(config.raw['snr_db'], grid.data_energy)
    y += complex_awgn(rng('awgn'), y.shape, v)
    interference, q, schedule = generate(rng('interference'), config.burst, len(y), grid.nfft+grid.cp)
    y += interference
    obs = ReceiverObservation(grid, demodulate(y, grid.nfft, grid.symbols, grid.cp))
    labels = EvaluationLabels(record.realization_id, frequency_response(taps, grid.nfft, grid.symbols),
                              bits, x_grid, v, config.raw['snr_db'], config.channel.family_id,
                              schedule, q, energy_accounting(x_grid, x, grid.cp, bits.size))
    return obs, labels, y
