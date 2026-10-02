import numpy as np
import pytest

from naaa.adversary.strategies import ByzParty
from naaa.protocol.params import PROPOSED
from naaa.protocol.process import Ctx, HonestParty
from naaa.sim.engine import Engine


class Recorder(HonestParty):
    """Honest party that stops after Gather instance 0 and records its view."""
    view_ids = None

    def _gather_done(self, k, view):
        if k == 0:
            self.view_ids = frozenset(view)


@pytest.mark.parametrize("mode,delay", [("sync", "uniform"), ("async", "exp"), ("async", "split"), ("async", "pareto")])
@pytest.mark.parametrize("seed", range(8))
def test_gather_views_and_common_core(mode, delay, seed):
    n, ts, ta = 10, 2, 1
    f = ts if mode == "sync" else ta
    h = n - f
    eng = Engine(seed, n, mode, delay)
    X = np.random.default_rng(seed).uniform(0, 1, size=(h, 1))
    ctx = Ctx(eng, n, ts, ta, 1, 1e-3, 1e3, PROPOSED, range(h), X)
    honest = [Recorder(i, ctx, X[i]) for i in range(h)]
    byz = [ByzParty(h + j, ctx, ["A1", "A2", "A3", "A4"][seed % 4], j) for j in range(f)]
    for p in honest + byz:
        eng.parties[p.id] = p
    for p in honest + byz:
        p.start()
    eng.run()
    views = [p.view_ids for p in honest]
    assert all(v is not None for v in views), "Gather must terminate for every honest party"
    assert all(len(v) >= n - ts for v in views)
    core = frozenset.intersection(*views)
    assert len(core) >= n - ts, f"common core {len(core)} < n-ts={n - ts}"
