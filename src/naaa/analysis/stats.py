"""Statistics helpers: bootstrap CIs, paired Wilcoxon + Holm, Cliff's delta, OLS slope, rule of three."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def bootstrap_ci(x, stat=np.mean, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1 or np.all(x == x[0]):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def cliffs_delta(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon(a, b) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value; 1.0 when all differences are zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    try:
        return float(st.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def ols_slope(x, y, alpha: float = 0.05) -> dict:
    """OLS slope with 95% CI and p-value for H0: slope = 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(slope=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"),
                    intercept=float("nan"), r2=float("nan"), n=len(x))
    r = st.linregress(x, y)
    t = st.t.ppf(1 - alpha / 2, len(x) - 2)
    return dict(slope=float(r.slope), lo=float(r.slope - t * r.stderr), hi=float(r.slope + t * r.stderr),
                p=float(r.pvalue), intercept=float(r.intercept), r2=float(r.rvalue ** 2), n=len(x))


def rule_of_three(n_runs: int) -> float:
    """95% upper bound on a violation rate after observing 0 violations in n_runs runs."""
    return 3.0 / n_runs if n_runs > 0 else float("nan")


def loglog_bound(x):
    """Fekete-style reference curve log2(x)/log2(log2(x)) (constants omitted)."""
    x = np.asarray(x, dtype=float)
    lg = np.log2(np.maximum(x, 4.0))
    return lg / np.log2(lg)


def fmt_ci(med, lo, hi, nd: int = 1) -> str:
    if med != med:
        return "n/a"
    return f"{med:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
