#!/usr/bin/env python3
"""Re-check the frozen protocol constants on seeds DISJOINT from the reported ones.

For c in {0,1,2,3} (extra iterations on top of ceil(log2(Dhat/eps))) and the halting check
on/off, count eps-agreement failures and restarts over a grid of adversaries / networks.
Usage:  python3 scripts/pilot.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naaa.protocol import params as P  # noqa: E402
from naaa.protocol.run import run_instance  # noqa: E402

PILOT_SEED_BASE = 9_000_000   # reported experiments use seeds derived from NAA_SEED, never this range


def main():
    print(f"{'c':>2} {'check':>5} {'runs':>5} {'agree_fail':>10} {'restarts>0':>10} {'median R':>8}")
    for c in (0, 1, 2, 3):
        for check in (True, False):
            P.METHODS["_pilot"] = replace(P.PROPOSED, name="_pilot", c=c, check=check)
            runs = fails = rest = 0
            rs = []
            for adv in ("A1", "A2", "A3", "A4"):
                for mode, delay in (("sync", "uniform"), ("async", "split"), ("async", "pareto")):
                    for gen in ("G1", "G2", "G3"):
                        for seed in range(10):
                            r = run_instance(dict(method="_pilot", mode=mode, delay=delay, adversary=adv, gen=gen,
                                                  n=10, ts=2, ta=1, d=1, DH_exp=12, seed=PILOT_SEED_BASE + seed))
                            runs += 1
                            fails += not (r["terminated"] and r["spread_over_eps"] <= 1.0)
                            rest += r["restarts"] > 0
                            rs.append(r["R_iter"])
            rs.sort()
            print(f"{c:>2} {str(check):>5} {runs:>5} {fails:>10} {rest:>10} {rs[len(rs) // 2]:>8}")


if __name__ == "__main__":
    main()
