import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from naaa import geometry as geo


def test_d1_trimmed_interval():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    r = geo.analyze(X, 1)
    assert r.diam == 2.0 and r.center[0] == 2.0           # [1, 3]


def test_exact_safe_area_inside_every_subset_hull_2d():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(7, 2))
    tau = 1
    r = geo.analyze(X, tau)
    assert r.mode == "exact"
    for sub in itertools.combinations(range(len(X)), len(X) - tau):
        assert geo.linf_dist_to_hull(r.center, X[list(sub)]) < 1e-7


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 10_000), st.sampled_from([1, 2]))
def test_center_inside_honest_hull_despite_outliers(seed, tau):
    """With <= tau Byzantine points of arbitrary value, the centre stays in the honest hull."""
    rng = np.random.default_rng(seed)
    d = 2
    m_honest = (d + 1) * tau + 3
    H = rng.uniform(0, 5, size=(m_honest, d))
    byz = rng.choice([-1e6, 1e6, 3.0], size=(tau, d))
    X = np.vstack([H, byz])
    r = geo.analyze(X, tau)
    assert geo.linf_dist_to_hull(r.center, H) < 1e-6
    assert r.diam <= geo.raw_diameter(H) + 1e-9           # certified diameter <= D_H


def test_hull_distance_basic():
    H = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert geo.linf_dist_to_hull(np.array([0.2, 0.2]), H) < 1e-9
    assert abs(geo.linf_dist_to_hull(np.array([2.0, 0.0]), H) - 1.0) < 1e-7
