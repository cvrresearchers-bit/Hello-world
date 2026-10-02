"""Geometry: safe area, certified diameter, centre and hull-distance (L-infinity).

Safe area of a multiset V of m points with trim parameter tau:
    S(V) = intersection over all (m - tau)-subsets Q of conv(Q).
It is non-empty when m > (d+1)*tau (Tverberg/Helly) and S(V) is contained in the convex
hull of the honest points whenever at most tau points of V are Byzantine.

Modes
-----
* d == 1: S(V) = [x_(tau+1), x_(m-tau)] (closed form, exact).
* d >= 2, exact: an LP over the intersection of hulls (used while the subset count is
  <= EXACT_SUBSET_CAP).
* d >= 2, box: coordinate-wise trimmed box. Only guarantees *box validity* (each output
  coordinate lies in the honest coordinate range). This is the weaker notion discussed
  in the design document and is reported separately from hull validity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

log = logging.getLogger("naaa.geometry")

EXACT_SUBSET_CAP = 120
_LP_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
_CACHE: dict = {}
_CACHE_MAX = 200_000
_warned_fallback = False


@dataclass(frozen=True)
class SafeResult:
    center: np.ndarray   # a point inside the safe area (exact modes) or the box centre
    diam: float          # L-infinity diameter of the safe area (certified diameter D-hat)
    mode: str            # "exact" | "box"


def raw_diameter(X: np.ndarray) -> float:
    """L-infinity diameter of a point set (max coordinate range)."""
    return float((X.max(axis=0) - X.min(axis=0)).max())


def _trim_box(X: np.ndarray, tau: int):
    s = np.sort(X, axis=0)
    return s[tau], s[X.shape[0] - 1 - tau]


def _exact_extremes(X: np.ndarray, tau: int, lo: np.ndarray, hi: np.ndarray):
    """Return 2d points of S(V) minimising/maximising every coordinate, or None on failure.

    The set is normalised into the trimmed box (which contains S) so LP tolerances are
    relative to the *certified* scale, not to Byzantine outliers.
    """
    m, d = X.shape
    scale = float((hi - lo).max())
    Z = (X - lo) / scale
    s = m - tau
    subsets = list(combinations(range(m), s))
    Q = len(subsets)
    nv = d + Q * s
    rows, cols, vals = [], [], []
    for qi, q in enumerate(subsets):
        base = qi * (d + 1)
        off = d + qi * s
        for c in range(d):
            for jj, j in enumerate(q):
                rows.append(base + c); cols.append(off + jj); vals.append(Z[j, c])
            rows.append(base + c); cols.append(c); vals.append(-1.0)
        for jj in range(s):
            rows.append(base + d); cols.append(off + jj); vals.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(Q * (d + 1), nv)).tocsr()
    b = np.zeros(Q * (d + 1))
    b[d::d + 1] = 1.0
    bounds = [(None, None)] * d + [(0, None)] * (Q * s)
    pts = []
    for c in range(d):
        for sign in (1.0, -1.0):
            cost = np.zeros(nv)
            cost[c] = sign
            res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs", options=_LP_OPTS)
            if res.status != 0:
                return None
            pts.append(res.x[:d])
    return lo + scale * np.array(pts)


def analyze(X: np.ndarray, tau: int, allow_exact: bool = True) -> SafeResult:
    """Safe area summary (centre + certified diameter) of view X (m x d) with trim tau."""
    global _warned_fallback
    X = np.ascontiguousarray(X, dtype=float)
    m, d = X.shape
    if m - tau < 1:
        raise ValueError("view too small for trimming")
    key = (X.tobytes(), tau, allow_exact)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    lo, hi = _trim_box(X, tau)
    box_diam = float((hi - lo).max())
    if d == 1:
        res = SafeResult((lo + hi) / 2.0, box_diam, "exact")
    elif box_diam == 0.0:
        res = SafeResult(lo.copy(), 0.0, "exact")
    elif allow_exact and tau > 0 and comb(m, tau) <= EXACT_SUBSET_CAP and m > (d + 1) * tau:
        pts = _exact_extremes(X, tau, lo, hi)
        if pts is None:
            if not _warned_fallback:
                log.warning("exact safe-area LP failed; falling back to trimmed box")
                _warned_fallback = True
            res = SafeResult((lo + hi) / 2.0, box_diam, "box")
        else:
            # pts[2c] minimises coordinate c, pts[2c+1] maximises it
            diam = max(pts[2 * c + 1][c] - pts[2 * c][c] for c in range(d))
            res = SafeResult(pts.mean(axis=0), float(max(diam, 0.0)), "exact")
    else:
        res = SafeResult((lo + hi) / 2.0, box_diam, "box")
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = res
    return res


def linf_dist_to_hull(y: np.ndarray, H: np.ndarray) -> float:
    """L-infinity distance from point y to conv(H); 0 if inside (up to LP tolerance)."""
    y = np.asarray(y, dtype=float)
    H = np.atleast_2d(np.asarray(H, dtype=float))
    h, d = H.shape
    if d == 1:
        return float(max(0.0, H.min() - y[0], y[0] - H.max()))
    lo = H.min(axis=0)
    scale = float(max((H.max(axis=0) - lo).max(), 1e-300))
    Z = (H - lo) / scale
    t = (y - lo) / scale
    # variables: lambda (h), r ; minimise r s.t. |Z^T lambda - t|_inf <= r, sum lambda = 1
    cost = np.zeros(h + 1)
    cost[-1] = 1.0
    A_ub = np.zeros((2 * d, h + 1))
    b_ub = np.zeros(2 * d)
    for c in range(d):
        A_ub[2 * c, :h] = Z[:, c]; A_ub[2 * c, -1] = -1.0; b_ub[2 * c] = t[c]
        A_ub[2 * c + 1, :h] = -Z[:, c]; A_ub[2 * c + 1, -1] = -1.0; b_ub[2 * c + 1] = -t[c]
    A_eq = np.zeros((1, h + 1))
    A_eq[0, :h] = 1.0
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (h + 1), method="highs", options=_LP_OPTS)
    if res.status != 0:
        return float("nan")
    return float(res.x[-1] * scale)


def box_violation(y: np.ndarray, H: np.ndarray) -> float:
    """Largest amount by which any coordinate of y leaves the honest coordinate range."""
    H = np.atleast_2d(H)
    return float(max(0.0, (H.min(axis=0) - y).max(), (y - H.max(axis=0)).max()))
