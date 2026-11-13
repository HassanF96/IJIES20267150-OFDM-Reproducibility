from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class CostLedger:
    """Actual calls/work only; future G1/G2/G3 owners supply boundary timers."""
    counts: Counter = field(default_factory=Counter)
    seconds: dict = field(default_factory=dict)

    def add(self, name, value=1):
        self.counts[name] += value

    @contextmanager
    def time(self, name):
        start = perf_counter()
        yield
        self.seconds.setdefault(name, []).append(perf_counter()-start)

    def export(self):
        return {'counts': dict(self.counts), 'seconds': self.seconds}


def count(ledger, name, value=1):
    if ledger is not None:
        ledger.add(name, value)
