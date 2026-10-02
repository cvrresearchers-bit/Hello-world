"""G4: sensor readings (temperature, humidity), d = 2, one epoch = one agreement instance.

Real data: the Intel Berkeley Research Lab dataset (data.txt: date time epoch moteid
temperature humidity light voltage). If the file is present at NAA_INTEL_LAB_PATH it is
used: rows with missing fields are dropped per epoch (no imputation) and outliers are KEPT.
Otherwise a clearly-labelled *synthetic surrogate* with the same schema is generated
(smooth diurnal signal + per-mote offsets + occasional faulty-sensor outliers). Results
produced on the surrogate must not be reported as real-data results; the run report states
which source was used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("naaa.inputs")


def _load_real(path: Path, h: int, n_epochs: int):
    import pandas as pd
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", on_bad_lines="skip",
                     names=["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"],
                     usecols=range(8))
    df = df.dropna(subset=["epoch", "mote", "temp", "hum"])
    for c in ("epoch", "mote", "temp", "hum"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["epoch", "mote", "temp", "hum"])
    lo = df[["temp", "hum"]].min().to_numpy()
    rng_ = (df[["temp", "hum"]].max() - df[["temp", "hum"]].min()).to_numpy()
    out = []
    for ep, grp in df.groupby("epoch"):
        grp = grp.drop_duplicates("mote")
        if len(grp) >= h:
            X = grp.sort_values("mote")[["temp", "hum"]].to_numpy()[:h]
            out.append((X - lo) / rng_)
        if len(out) >= n_epochs:
            break
    return out


def _surrogate(h: int, n_epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.015, size=(h, 2))
    out = []
    for e in range(n_epochs):
        phase = 2 * np.pi * e / 48.0
        base = np.array([0.5 + 0.2 * np.sin(phase), 0.5 - 0.15 * np.sin(phase)])
        X = base + offsets + rng.normal(0.0, 0.004, size=(h, 2))
        if rng.random() < 0.25:                         # faulty sensor: kept on purpose
            X[rng.integers(h)] += rng.uniform(-0.4, 0.4, size=2)
        out.append(np.clip(X, 0.0, 1.0))
    return out


def load_epochs(path: str, h: int, n_epochs: int, seed: int):
    """Return (list of h x 2 arrays normalised to [0,1], source label)."""
    p = Path(path)
    if p.is_file():
        try:
            ep = _load_real(p, h, n_epochs)
            if ep:
                return ep, f"real:{p}"
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not parse %s (%s); using surrogate", p, exc)
    return _surrogate(h, n_epochs, seed), "synthetic-surrogate"
