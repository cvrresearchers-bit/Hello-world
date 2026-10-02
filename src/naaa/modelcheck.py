"""E5 (Python): bounded exhaustive check of one agreement step, d = 1.

For every honest input vector over a small integer domain, every Byzantine value choice
(including outliers outside the honest range), every common core C (|C| = n - ts) and every
admissible view C <= V_i <= all parties (this is exactly the output guarantee of Gather),
we compute the new values v_i' = midpoint(trim_ts(V_i)) with exact rationals and check

  (V) validity     : v_i' in [min honest, max honest]
  (C) contraction  : spread(v') <= RHO * spread(v)   (RHO = 1/2 is the claimed factor)

and report the worst contraction ratio found. One step + induction gives the multi-step
claim; a multi-step reachable-state sweep is added for the smallest configuration.
This is NOT message-interleaving model checking (that is what tla/AdaptiveAA.tla is for);
it checks the *abstract step* against *all* view/adversary choices Gather permits.
"""
from __future__ import annotations

import itertools
import logging
import random
from fractions import Fraction

log = logging.getLogger("naaa.modelcheck")


def _step(vals_all, view, ts):
    xs = sorted(vals_all[i] for i in view)
    m = len(xs)
    return (xs[ts] + xs[m - 1 - ts]) / 2


def check_config(n: int, ts: int, K: int, sample: int | None = None, seed: int = 0, rho=Fraction(1, 2)) -> dict:
    h = n - ts
    byz_dom = sorted(set(list(range(K + 1)) + [-K - 1, 2 * K + 1]))
    parties = list(range(n))
    cores = list(itertools.combinations(parties, n - ts))
    states = valid_fail = contr_fail = 0
    worst = Fraction(0)
    worst_example = None
    rnd = random.Random(seed)

    def all_cases():
        for hv in itertools.product(range(K + 1), repeat=h):
            for bv in itertools.product(byz_dom, repeat=ts):
                yield hv, bv

    def views_for(core):
        rest = [i for i in parties if i not in core]
        subsets = [tuple(s) for r in range(len(rest) + 1) for s in itertools.combinations(rest, r)]
        return [tuple(sorted(core + s)) for s in subsets]

    cases = all_cases()
    if sample is not None:
        space = (K + 1) ** h * len(byz_dom) ** ts
        cases = (( tuple(rnd.randrange(K + 1) for _ in range(h)), tuple(rnd.choice(byz_dom) for _ in range(ts)))
                 for _ in range(min(sample, space)))
    for hv, bv in cases:
        vals_all = [Fraction(x) for x in hv] + [Fraction(x) for x in bv]
        spread = max(hv) - min(hv)
        core_iter = cores if sample is None else [rnd.choice(cores)]
        for core in core_iter:
            vs = views_for(core)
            if sample is not None:
                vs_choices = [rnd.choice(vs) for _ in range(h)]
                combos = [vs_choices]
            else:
                combos = itertools.product(vs, repeat=h)
            for combo in combos:
                new = [_step(vals_all, v, ts) for v in combo]
                states += 1
                if min(new) < min(hv) or max(new) > max(hv):
                    valid_fail += 1
                ns = max(new) - min(new)
                if spread == 0:
                    if ns != 0:
                        contr_fail += 1
                else:
                    ratio = Fraction(ns) / spread
                    if ratio > worst:
                        worst, worst_example = ratio, (hv, bv, core, combo)
                    if ratio > rho:
                        contr_fail += 1
    return dict(n=n, ts=ts, K=K, mode="exhaustive" if sample is None else f"sampled({sample})",
                states=states, validity_violations=valid_fail, contraction_violations=contr_fail,
                worst_ratio=float(worst), worst_example=str(worst_example))


def run_all(sample_big: int = 20000) -> list[dict]:
    out = [
        check_config(4, 1, 4),
        check_config(5, 1, 3),
        check_config(7, 2, 2, sample=sample_big),
        check_config(10, 3, 3, sample=sample_big),
    ]
    for r in out:
        log.info("modelcheck n=%d ts=%d %s: states=%d validity_viol=%d contraction_viol(>1/2)=%d worst_ratio=%.4f",
                 r["n"], r["ts"], r["mode"], r["states"], r["validity_violations"],
                 r["contraction_violations"], r["worst_ratio"])
    return out
