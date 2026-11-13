import numpy as np


def complex_awgn(rng, shape, variance):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('finite nonnegative total complex variance required')
    return np.sqrt(variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))


def variance_from_snr(snr_db, symbol_energy=1.0, ensemble_gain=1.0):
    """Nominal active-RE Es*ensemble_gain/N0; never realized channel energy."""
    if not all(np.isfinite(v) for v in (snr_db, symbol_energy, ensemble_gain)) or min(symbol_energy, ensemble_gain) <= 0:
        raise ValueError('finite SNR and positive energies required')
    variance = symbol_energy*ensemble_gain*10.0**(-snr_db/10)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('SNR outside representable range')
    return variance


def esn0_to_payload_ebn0(esn0_db, bits_per_symbol):
    """Data-symbol-only Eb/N0; excludes all OFDM/pilot overhead."""
    if bits_per_symbol not in (2, 4):
        raise ValueError('supported uncoded bit count required')
    return esn0_db-10*np.log10(bits_per_symbol)


def frame_ebn0_db(transmitted_energy, payload_bits, variance):
    if min(transmitted_energy, payload_bits, variance) <= 0:
        raise ValueError('positive energy, bit count and noise required')
    return float(10*np.log10(transmitted_energy/(payload_bits*variance)))
