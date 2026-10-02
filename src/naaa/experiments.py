"""Experiment definitions E1-E4 and E6 (E5 lives in modelcheck.py) and the parallel runner.

Every run is an independent, seeded simulation, so batches are embarrassingly parallel and
the results are bit-for-bit reproducible for a given NAA_SEED/profile.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from .inputs.intel_lab import load_epochs
from .protocol.run import run_instance, ts_for

log = logging.getLogger("naaa.exp")

# Defaults for the d = 1 experiments: n=10, ts=2 (n > 3*ts with slack), ta=1.
N1, TS1, TA1 = 10, 2, 1
# d = 2: n=9, ts=1 (exact LP safe area is cheap)
N2, TS2, TA2 = 9, 1, 1

ASYNC_DELAYS = ["exp", "lognormal", "pareto", "split"]


def _seed(base: int, exp: int, cell: int, rep: int) -> int:
    return (base * 1_000_003 + exp * 100_003 + cell * 1_009 + rep) & 0x7FFFFFFF


def build_e1(plan: dict, base: int) -> list[dict]:
    """Adaptivity scaling: rounds vs log2(D_H/eps); network sync, adversary A2 (outliers at M)."""
    specs = []
    methods = ["proposed", "B1_sync_fixed", "B2_fixed", "B4_classic_adaptive"]
    for ci, lv in enumerate(plan["e1_levels"]):
        for rep in range(plan["seeds"]):
            for m in methods:   # same seed for all methods => paired comparison
                specs.append(dict(exp="E1", method=m, mode="sync", adversary="A2", gen="G1", n=N1, ts=TS1,
                                  ta=TA1, d=1, DH_exp=lv, seed=_seed(base, 1, ci, rep)))
    # d = 2 (exact safe-area LP), fewer seeds
    for ci, lv in enumerate(plan["e1_levels"]):
        for rep in range(max(2, plan["seeds"] // 2)):
            for m in ["proposed", "B2_fixed"]:
                specs.append(dict(exp="E1_d2", method=m, mode="sync", adversary="A2", gen="G2", n=N2, ts=TS2,
                                  ta=TA2, d=2, DH_exp=lv, seed=_seed(base, 11, ci, rep)))
    # same sweep under asynchrony (lognormal delays; only ta parties corrupted) for the H2 slope ratio
    for ci, lv in enumerate(plan["e1_levels"]):
        for rep in range(max(2, plan["seeds"] // 2)):
            for m in ["proposed", "B2_fixed"]:
                specs.append(dict(exp="E1_async", method=m, mode="async", delay="lognormal", adversary="A2",
                                  gen="G1", n=N1, ts=TS1, ta=TA1, d=1, DH_exp=lv, seed=_seed(base, 12, ci, rep)))
    return specs


def build_e2(plan: dict, base: int) -> list[dict]:
    """Network-agnosticism: same protocol, unknown network (sync or one of several async schedulers)."""
    specs = []
    nets = [("sync", "uniform")] + [("async", d) for d in ASYNC_DELAYS]
    for ci, (mode, delay) in enumerate(nets):
        for rep in range(plan["seeds"]):
            for m in ["proposed", "B1_sync_fixed", "B2_fixed"]:
                specs.append(dict(exp="E2", method=m, mode=mode, delay=delay, adversary="A2", gen="G1", n=N1,
                                  ts=TS1, ta=TA1, d=1, DH_exp=10, seed=_seed(base, 2, ci, rep)))
    return specs


def build_e3(plan: dict, base: int) -> list[dict]:
    """Adversarial robustness + ablations AB1-AB3."""
    specs = []
    methods = ["proposed", "AB1_raw_dhat", "AB2_no_substitution", "AB3_plain_collect"]
    ci = 0
    for adv in ["A1", "A2", "A3", "A4", "A5"]:
        for mode, delay in [("sync", "uniform"), ("async", "lognormal")]:
            ci += 1
            gen = "G3" if adv == "A4" else "G1"
            for rep in range(plan["e3_seeds"]):
                for m in methods:
                    specs.append(dict(exp="E3", method=m, mode=mode, delay=delay, adversary=adv, gen=gen, n=N1,
                                      ts=TS1, ta=TA1, d=1, DH_exp=10, seed=_seed(base, 3, ci, rep)))
    return specs


def build_e4(plan: dict, base: int) -> list[dict]:
    """Scalability in n and d: messages, bytes, rounds."""
    specs = []
    reps = max(2, plan["seeds"] // 3)
    cfgs = [(n, 1) for n in plan["e4_ns"]] + [(7, 2), (9, 2), (13, 2)]
    for ci, (n, d) in enumerate(cfgs):
        ts = ts_for(n, d)
        for rep in range(reps):
            for m in ["proposed", "B2_fixed"]:
                specs.append(dict(exp="E4", method=m, mode="sync", adversary="A2", gen="G1" if d == 1 else "G2",
                                  n=n, ts=ts, ta=min(1, ts), d=d, DH_exp=10, seed=_seed(base, 4, ci, rep),
                                  max_wall=300.0))
    return specs


def build_e6(plan: dict, base: int, intel_path: str):
    """Realism: per-epoch agreement on (temperature, humidity) readings, d = 2."""
    n, ts, ta, d = N2, TS2, TA2, 2
    h = n - ts
    epochs, source = load_epochs(intel_path, h, plan["e6_epochs"], seed=base)
    specs = []
    for ei, X in enumerate(epochs):
        for m in ["proposed", "B2_fixed"]:
            specs.append(dict(exp="E6", method=m, mode="sync", adversary="A2", n=n, ts=ts, ta=ta, d=d, eps=1e-4,
                              M_exp=14, honest_inputs=X, epoch=ei, seed=_seed(base, 6, ei, 0)))
    return specs, source


def run_batch(specs: list[dict], workers: int, shutdown: threading.Event, label: str) -> list[dict]:
    """Run specs in a process pool; stop submitting on shutdown and return what finished."""
    results: list[dict] = []
    if not specs:
        return results
    log.info("%s: %d runs on %d workers", label, len(specs), workers)
    if workers <= 1:
        for sp in specs:
            if shutdown.is_set():
                log.warning("%s: shutdown requested, %d/%d runs done", label, len(results), len(specs))
                break
            results.append(_run(sp))
        return results
    done = 0
    step = max(1, len(specs) // 5)
    slots: dict[int, dict] = {}          # spec index -> record, so output order == spec order (deterministic files)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, sp): i for i, sp in enumerate(specs)}
        try:
            for fut in as_completed(futs):
                slots[futs[fut]] = fut.result()
                done += 1
                if done % step == 0:
                    log.info("%s: %d/%d", label, done, len(specs))
                if shutdown.is_set():
                    log.warning("%s: shutdown requested; cancelling pending runs", label)
                    for f in futs:
                        f.cancel()
                    break
        except KeyboardInterrupt:
            shutdown.set()
            for f in futs:
                f.cancel()
    return [slots[i] for i in sorted(slots)]


def _run(sp: dict) -> dict:
    extra = {k: sp[k] for k in ("exp", "epoch") if k in sp}
    sp2 = {k: v for k, v in sp.items() if k not in ("exp", "epoch")}
    try:
        rec = run_instance(sp2)
    except Exception as exc:  # keep the batch alive; the failure is recorded, not hidden
        rec = dict(method=sp.get("method"), error=repr(exc), terminated=False, seed=sp.get("seed"),
                   mode=sp.get("mode"), adversary=sp.get("adversary"))
    rec.update(extra)
    return rec
