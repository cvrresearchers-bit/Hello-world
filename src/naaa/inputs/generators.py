"""Synthetic honest-input generators with an exactly controlled L-infinity diameter D_H.

G1 uniform in a d-cube of side D_H (two pinned parties realise the diameter exactly)
G2 Gaussian, affinely rescaled so the empirical diameter equals D_H
G3 tight cluster (radius ~ eps/4) plus ONE honest outlier (party 0) at distance D_H:
   the direct test of the "hidden honest outlier" concern
G4 sensor-like readings: see inputs/intel_lab.py
"""
from __future__ import annotations

import numpy as np

from ..geometry import raw_diameter


def g1_uniform(h: int, d: int, DH: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    X = rng.uniform(0.0, DH, size=(h, d))
    X[0, 0] = 0.0
    X[min(1, h - 1), 0] = DH
    return X


def g2_gaussian(h: int, d: int, DH: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((h, d))
    X -= X.min(axis=0)
    diam = raw_diameter(X)
    X *= DH / diam if diam > 0 else 1.0
    return X


def g3_cluster_outlier(h: int, d: int, DH: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    r = min(eps / 4.0, DH / 4.0)
    X = rng.uniform(0.0, r, size=(h, d))
    X[0, 0] = DH          # the honest outlier is party 0 (slowed by the A4 scheduler)
    X[1:, 0] = np.minimum(X[1:, 0], r)
    return X


GENERATORS = {"G1": g1_uniform, "G2": g2_gaussian, "G3": g3_cluster_outlier}
