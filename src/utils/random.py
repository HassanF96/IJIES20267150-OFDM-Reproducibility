"""Order-independent, namespaced PCG64 streams; no global NumPy RNG."""
import hashlib
import json
import numpy as np

STREAMS = ('channel', 'bits', 'awgn', 'interference', 'model', 'split', 'holdout')


def seed_for(master: int, namespace: str, realization: str, stream: str) -> int:
    if type(master) is not int or master < 0 or stream not in STREAMS:
        raise ValueError('nonnegative integer master and registered stream required')
    payload = json.dumps([master, namespace, realization, stream], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], 'big')


def rng_for(master: int, namespace: str, realization: str, stream: str):
    return np.random.Generator(np.random.PCG64(seed_for(master, namespace, realization, stream)))
