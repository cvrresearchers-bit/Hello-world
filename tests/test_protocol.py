from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from naaa.protocol import params as P
from naaa.protocol.run import run_instance, ts_for


def ok(r):
    return r["terminated"] and r["spread_over_eps"] <= 1.0 and r["hull_violation"] < 1e-6 * max(1.0, r["DH"])


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**20),
    net=st.sampled_from([("sync", "uniform"), ("async", "exp"), ("async", "lognormal"), ("async", "pareto"), ("async", "split")]),
    adv=st.sampled_from(["A1", "A2", "A3", "A4", "A5"]),
    gen=st.sampled_from(["G1", "G2", "G3"]),
    dh=st.integers(1, 20),
    n=st.sampled_from([7, 10, 13]),
)
def test_proposed_is_safe_and_live(seed, net, adv, gen, dh, n):
    """Agreement, validity and termination for random schedules, adversaries and inputs (d = 1)."""
    ts = ts_for(n, 1)
    r = run_instance(dict(method="proposed", mode=net[0], delay=net[1], adversary=adv, gen=gen, n=n, ts=ts,
                          ta=min(1, ts), d=1, DH_exp=dh, seed=seed))
    assert ok(r), r


@pytest.mark.parametrize("mode,delay", [("sync", "uniform"), ("async", "lognormal")])
def test_d2_exact_safe_area_hull_validity(mode, delay):
    for seed in range(3):
        r = run_instance(dict(method="proposed", mode=mode, delay=delay, adversary="A2", gen="G2", n=9, ts=1, ta=1,
                              d=2, DH_exp=10, seed=seed))
        assert ok(r), r


def test_adapts_to_honest_diameter():
    small = run_instance(dict(method="proposed", n=10, ts=2, ta=1, d=1, DH_exp=2, seed=1))
    large = run_instance(dict(method="proposed", n=10, ts=2, ta=1, d=1, DH_exp=18, seed=1))
    fixed = run_instance(dict(method="B2_fixed", n=10, ts=2, ta=1, d=1, DH_exp=2, seed=1))
    assert small["R_iter"] + 8 <= large["R_iter"]
    assert small["R_iter"] < fixed["R_iter"] // 2


def test_hidden_honest_outlier_does_not_break_agreement():
    for mode, delay in [("sync", "uniform"), ("async", "lognormal")]:
        for seed in range(10):
            r = run_instance(dict(method="proposed", mode=mode, delay=delay, adversary="A4", gen="G3", n=10, ts=2,
                                  ta=1, d=1, DH_exp=16, seed=seed))
            assert ok(r), r


def test_halting_check_restart_path_is_correct():
    """Force a far-too-small first R (c very negative): the check must grow R and still agree."""
    P.METHODS["_tiny_R"] = replace(P.PROPOSED, name="_tiny_R", c=-100)
    try:
        restarts = 0
        for seed in range(12):
            r = run_instance(dict(mode="sync", n=10, ts=2, ta=1, d=1, DH_exp=14, seed=seed, adversary="A3",
                                  method="_tiny_R"))
            assert ok(r), r
            restarts += r["restarts"]
        # NOTE: we do not assert that *without* the check agreement breaks: in all simulated
        # schedules the real contraction is much faster than the worst-case 1/2, so the check
        # is a safety net that is exercised here but not observed to be necessary.
        assert restarts > 0, "the check never fired"
    finally:
        P.METHODS.pop("_tiny_R", None)


def test_sync_only_baseline_breaks_under_adversarial_scheduling():
    bad = 0
    for seed in range(20):
        r = run_instance(dict(method="B1_sync_fixed", mode="async", delay="split", n=10, ts=2, ta=1, d=1, DH_exp=10,
                              seed=seed, adversary="A2"))
        bad += not ok(r)
    assert bad > 0


def test_threshold_validation():
    with pytest.raises(ValueError):
        run_instance(dict(method="proposed", n=7, ts=2, ta=1, d=2, DH_exp=3, seed=1))   # n <= (d+2)*ts
