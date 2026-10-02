"""Structured logging: console + file in the run directory."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str, log_file: Path | None = None) -> logging.Logger:
    root = logging.getLogger("naaa")
    root.handlers.clear()
    root.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    return root
