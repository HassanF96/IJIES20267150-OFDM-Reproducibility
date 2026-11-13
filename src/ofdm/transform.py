"""Unitary transforms, [...,N,L] grids; serialized symbols with CP."""
import numpy as np


def modulate(grid, cp):
    grid = np.asarray(grid, dtype=np.complex128)
    if grid.ndim < 2 or type(cp) is not int or not 0 <= cp <= grid.shape[-2]:
        raise ValueError('invalid grid or CP')
    blocks = np.fft.ifft(grid, axis=-2, norm='ortho').swapaxes(-1, -2)
    if cp:
        blocks = np.concatenate([blocks[..., -cp:], blocks], axis=-1)
    return blocks.reshape(*blocks.shape[:-2], -1)


def remove_cp(samples, nfft, symbols, cp):
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.shape[-1] != symbols*(nfft+cp) or not 0 <= cp <= nfft:
        raise ValueError('receive-buffer length/CP mismatch')
    return samples.reshape(*samples.shape[:-1], symbols, nfft+cp)[..., cp:]


def demodulate(samples, nfft, symbols, cp):
    useful = remove_cp(samples, nfft, symbols, cp)
    return np.fft.fft(useful, axis=-1, norm='ortho').swapaxes(-1, -2)


def energy_accounting(grid, samples, cp, data_bits):
    if data_bits <= 0:
        raise ValueError('positive payload bit count required')
    useful = float(np.sum(np.abs(grid)**2))
    total = float(np.sum(np.abs(samples)**2))
    return {'useful_energy': useful, 'cp_energy': total-useful,
            'transmitted_energy': total, 'transmitted_energy_per_payload_bit': total/data_bits,
            'sample_energy': total/np.asarray(samples).size, 'cp_samples': cp}
