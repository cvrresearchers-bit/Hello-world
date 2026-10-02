"""Build and execute one simulated agreement instance and extract its metrics."""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np

from .. import geometry as geo
from ..adversary.strategies import ByzParty
from ..inputs.generators import GENERATORS
from ..sim.engine import Engine
from .params import METHODS
from .process import Ctx, HonestParty, SyncRoundParty

log = logging.getLogger("naaa.run")

M_EXP_DEFAULT = 24  # a-priori range bound M = eps * 2**24 for fixed-iteration baselines


def ts_for(n: int, d: int) -> int:
    """Largest-ish trim parameter with n > (d+2)*ts and slack so the certified diameter is useful."""
    return max(1, (n - 1) // (d + 3))


def check_thresholds(n: int, ts: int, ta: int, d: int) -> None:
    if not (0 <= ta <= ts):
        raise ValueError("need 0 <= ta <= ts")
    if not n > (d + 2) * ts:
        raise ValueError(f"implementation requires n > (d+2)*ts; got n={n}, d={d}, ts={ts}")
    if not n > 3 * ts:
        raise ValueError("Gather needs n > 3*ts")


def oracle_rounds(ctx, honest, eps: float) -> float:
    """First iteration k at which the honest values (halted parties: their output) are within eps.

    A simulator-only (omniscient) measure of how many iterations were *actually* needed;
    the protocol itself cannot compute it. Compared with R_iter it exposes the slack of the
    worst-case contraction factor rho = 1/2 used to set R.
    """
    kmax = max(p.iters for p in honest)
    for k in range(1, kmax + 1):
        vals = []
        for p in honest:
            if p.id in ctx.hist.get(k, {}):
                vals.append(ctx.hist[k][p.id])
            elif p.output is not None and p.iters < k:
                vals.append(p.output)
        if len(vals) < len(honest):
            continue
        Y = np.stack(vals)
        if (Y.max(axis=0) - Y.min(axis=0)).max() <= eps:
            return float(k)
    return float("nan")


def run_instance(spec: dict) -> dict:
    """Run one instance. ``spec`` keys (defaults in brackets):

    method, mode (sync|async), delay (lognormal), adversary (A2), gen (G1), n, ts, ta, d (1),
    eps (1.0), M_exp (24), DH_exp (log2(D_H/eps)), seed, honest_inputs (optional ndarray),
    exact (True), max_wall (120)
    """
    sp = dict(spec)
    method = sp["method"]
    params = METHODS[method]
    if "exact" in sp:
        params = type(params)(**{**params.__dict__, "exact": sp["exact"]})
    mode = sp.get("mode", "sync")
    n, ts, ta, d = sp["n"], sp["ts"], sp["ta"], sp.get("d", 1)
    eps = float(sp.get("eps", 1.0))
    M = eps * 2.0 ** sp.get("M_exp", M_EXP_DEFAULT)
    adv = sp.get("adversary", "A2")
    seed = int(sp["seed"])
    check_thresholds(n, ts, ta, d)
    f = ts if mode == "sync" else ta                   # actual corruptions under each network
    h = n - f
    rng = np.random.default_rng(seed)
    if "honest_inputs" in sp and sp["honest_inputs"] is not None:
        X0 = np.asarray(sp["honest_inputs"], dtype=float)[:h]
        DH = geo.raw_diameter(X0)
    else:
        DH = eps * 2.0 ** sp["DH_exp"]
        X0 = GENERATORS[sp.get("gen", "G1")](h, d, DH, eps, rng)
        DH = geo.raw_diameter(X0)
    slow = frozenset({0}) if adv == "A4" else frozenset()
    eng = Engine(seed=seed ^ 0x9E3779B9, n=n, mode=mode, delay=sp.get("delay", "lognormal"), slow=slow)
    ctx = Ctx(eng, n, ts, ta, d, eps, M, params, range(h), X0)
    cls = SyncRoundParty if params.termination == "sync_rounds" else HonestParty
    honest = [cls(i, ctx, X0[i]) for i in range(h)]
    byz = [ByzParty(h + j, ctx, adv, j) for j in range(f)]
    for p in honest + byz:
        eng.parties[p.id] = p
    for p in honest + byz:
        p.start()
    reason = eng.run(max_wall=float(sp.get("max_wall", 120.0)))

    done = [p for p in honest if p.output is not None]
    terminated = len(done) == h
    rec = dict(
        method=method, mode=mode, delay=eng.delay, adversary=adv, gen=sp.get("gen", "G1"), n=n, ts=ts, ta=ta,
        f=f, d=d, eps=eps, M=M, DH=DH, DH_eps=DH / eps, log2_DH_eps=math.log2(max(DH / eps, 1e-12)),
        seed=seed, stop_reason=reason, terminated=terminated, n_output=len(done),
        events=eng.events, msgs=eng.msgs, bytes=eng.bytes,
    )
    if done:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices (baselines without D-hat)
            Y = np.stack([p.output for p in done])
            spread = max((np.abs(Y[i] - Y[j]).max() for i in range(len(Y)) for j in range(i + 1, len(Y))), default=0.0)
            hull = [geo.linf_dist_to_hull(y, X0) for y in Y]
            boxv = [geo.box_violation(y, X0) for y in Y]
            dh0 = np.array([p.dhat0 for p in done], dtype=float)
            rec.update(
                R_iter=max(p.iters for p in done), R_iter_min=min(p.iters for p in done),
                causal_rounds=max(p.out_depth for p in done), T_term=max(p.out_time for p in done),
                T_stop=max((getattr(p, "quorum_time", float("nan")) for p in done), default=float("nan")),
                spread=float(spread), spread_over_eps=float(spread / eps),
                hull_violation=float(np.nanmax(hull)), box_violation=float(max(boxv)),
                dhat0_ratio_mean=float(np.nanmean(dh0) / DH) if DH > 0 else float("nan"),
                dhat0_ratio_min=float(np.nanmin(dh0) / DH) if DH > 0 else float("nan"),
                dhat0_ratio_max=float(np.nanmax(dh0) / DH) if DH > 0 else float("nan"),
                raw0_ratio_mean=float(np.nanmean([p.raw0 for p in done]) / DH) if DH > 0 else float("nan"),
                restarts=max(getattr(p, "restarts", 0) for p in done),
                capped=any(getattr(p, "capped", False) for p in done),
                R_target_max=max(getattr(p, "R", 0) or 0 for p in done),
                R_oracle=oracle_rounds(ctx, honest, eps) if terminated and cls is HonestParty else float("nan"),
            )
    return rec
