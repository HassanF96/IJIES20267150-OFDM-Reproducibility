"""Finite nonwrapping Gaussian burst process from SYSTEM_MODEL section 3."""
from dataclasses import dataclass
import numpy as np
from src.channel.noise import complex_awgn


@dataclass(frozen=True, slots=True)
class BurstConfig:
    probability: float = 0.5
    duration: int = 32
    power: float = 1.0
    count_law: str = 'bernoulli'
    max_count: int = 1
    overlap: bool = True
    cross_symbols: bool = True
    fixed_start: int | None = None
    distribution: str = 'proper_complex_gaussian'

    def __post_init__(self):
        if not np.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError('invalid arrival probability')
        if type(self.duration) is not int or self.duration < 1 or type(self.max_count) is not int or self.max_count < 1:
            raise ValueError('positive integer duration/count required')
        if not np.isfinite(self.power) or self.power < 0:
            raise ValueError('invalid power')
        if self.count_law not in ('bernoulli', 'binomial'):
            raise ValueError('only Bernoulli or explicitly capped binomial arrivals')
        if self.count_law == 'bernoulli' and self.max_count != 1:
            raise ValueError('Bernoulli means at most one burst')
        if self.distribution != 'proper_complex_gaussian':
            raise ValueError('E3 conditional sample distribution is proper Gaussian')
        if self.fixed_start is not None and (type(self.fixed_start) is not int or self.fixed_start < 0):
            raise ValueError('invalid fixed start')
        if type(self.overlap) is not bool or type(self.cross_symbols) is not bool:
            raise ValueError('boolean overlap/boundary flags required')


@dataclass(frozen=True, slots=True)
class Burst:
    start: int
    duration: int
    power: float


def validate_geometry(config, samples, symbol_samples):
    if config.duration > samples or symbol_samples <= 0 or samples % symbol_samples:
        raise ValueError('invalid burst/frame geometry')
    if not config.cross_symbols and config.duration > symbol_samples:
        raise ValueError('duration cannot fit one symbol')
    if config.fixed_start is not None:
        if config.fixed_start+config.duration > samples:
            raise ValueError('burst would wrap past frame end')
        if not config.cross_symbols and config.fixed_start//symbol_samples != (config.fixed_start+config.duration-1)//symbol_samples:
            raise ValueError('fixed burst crosses a prohibited symbol boundary')


def draw_schedule(rng, config, samples, symbol_samples):
    validate_geometry(config, samples, symbol_samples)
    count = int(rng.binomial(config.max_count, config.probability))
    candidates = np.arange(samples-config.duration+1)
    if not config.cross_symbols:
        candidates = candidates[candidates//symbol_samples == (candidates+config.duration-1)//symbol_samples]
    if config.fixed_start is not None:
        candidates = np.asarray([config.fixed_start])
    schedule = []
    for _ in range(count):
        if not len(candidates):
            raise ValueError('requested nonoverlapping burst count cannot be placed; no silent truncation')
        start = int(rng.choice(candidates))
        schedule.append(Burst(start, config.duration, config.power))
        if not config.overlap:
            candidates = candidates[(candidates+config.duration <= start) | (candidates >= start+config.duration)]
    return tuple(schedule)


def sample_schedule(rng, samples, schedule):
    signal = np.zeros(samples, np.complex128)
    variance = np.zeros(samples)
    for b in schedule:
        if b.start < 0 or b.duration < 1 or b.start+b.duration > samples or not np.isfinite(b.power) or b.power < 0:
            raise ValueError('invalid schedule')
        sl = slice(b.start, b.start+b.duration)
        signal[sl] += complex_awgn(rng, b.duration, b.power)
        variance[sl] += b.power
    return signal, variance


def generate(rng, config, samples, symbol_samples):
    schedule = draw_schedule(rng, config, samples, symbol_samples)
    signal, variance = sample_schedule(rng, samples, schedule)
    return signal, variance, schedule
