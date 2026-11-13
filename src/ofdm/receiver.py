import numpy as np
from src.ofdm.information import ReceiverObservation
from src.ofdm.modulation import Constellation, hard_indices, posterior, bit_llrs
from src.utils.cost import count


def mmse_equalize(y, h, estimated_variance, symbol_energy=1.0):
    if not np.isfinite(estimated_variance) or estimated_variance <= 0 or not np.isfinite(symbol_energy) or symbol_energy <= 0:
        raise ValueError('positive finite practical scale and symbol energy required')
    y, h = np.broadcast_arrays(np.asarray(y, np.complex128), np.asarray(h, np.complex128))
    if not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('nonfinite receiver input')
    q = h.conj()/(np.abs(h)**2+estimated_variance/symbol_energy)
    return q*y, q*h, np.abs(q)**2*estimated_variance


def detect(observation, estimate, constellation, estimated_variance, *, soft=False, llrs=False, ledger=None):
    if type(observation) is not ReceiverObservation:
        raise TypeError('ReceiverObservation required; simulator labels forbidden')
    estimate = np.asarray(estimate)
    if estimate.shape != observation.grid.shape:
        raise ValueError('channel estimate grid shape mismatch')
    mask = observation.grid.data_mask
    y, h = observation.y[mask], estimate[mask]
    z, gain, vz = mmse_equalize(y, h, estimated_variance, constellation.energy)
    index = hard_indices(y, h, constellation)
    result = {'bits': constellation.bits_from_indices(index), 'symbols': constellation.points[index],
              'z': z, 'gain': gain, 'equalized_variance': vz}
    count(ledger, 'equalizer_res', len(y))
    count(ledger, 'hard_slicer_res', len(y))
    if soft or llrs:
        p, mean, var, logp = posterior(y, h, estimated_variance, constellation)
        result.update(probabilities=p, posterior_mean=mean, posterior_variance=var)
        count(ledger, 'symbol_likelihoods', len(y)*len(constellation.points))
        count(ledger, 'posterior_moment_res', len(y))
        if llrs:
            result['llrs'] = bit_llrs(logp, constellation)
            count(ledger, 'bit_llrs', len(y)*constellation.bits_per_symbol)
    return result
