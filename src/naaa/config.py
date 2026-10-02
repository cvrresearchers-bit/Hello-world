"""Configuration: .env / environment variables and experiment profiles."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Profiles control how much work Main.py does. Constants of the *protocol*
# (c, check threshold) are frozen in protocol/params.py, not tuned per profile.
PROFILES = {
    "smoke": dict(seeds=2, e1_levels=[1, 8, 16], e6_epochs=4, e4_ns=[4, 7], e3_seeds=2),
    "quick": dict(seeds=16, e1_levels=[1, 4, 7, 10, 13, 16, 19], e6_epochs=24,
                  e4_ns=[4, 7, 10, 19, 31], e3_seeds=16),
    "full": dict(seeds=30, e1_levels=[1, 4, 7, 10, 13, 16, 19], e6_epochs=100,
                 e4_ns=[4, 7, 10, 19, 31, 61], e3_seeds=50),
}


def load_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Tiny .env parser (KEY=VALUE, '#' comments). Real env vars take precedence."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    profile: str = "quick"
    seed: int = 12345
    workers: int = 0
    output_dir: str = "output"
    log_level: str = "INFO"
    run_tests: bool = False
    tlc_jar: str = ""
    intel_lab_path: str = "data/intel_lab/data.txt"
    plan: dict = field(default_factory=dict)

    @classmethod
    def load(cls, dotenv: str = ".env") -> "Settings":
        env = {**load_dotenv(dotenv), **{k: v for k, v in os.environ.items() if k.startswith("NAA_")}}
        s = cls(
            profile=env.get("NAA_PROFILE", "quick").lower(),
            seed=int(env.get("NAA_SEED", 12345)),
            workers=int(env.get("NAA_WORKERS", 0)),
            output_dir=env.get("NAA_OUTPUT_DIR", "output"),
            log_level=env.get("NAA_LOG_LEVEL", "INFO").upper(),
            run_tests=_bool(env.get("NAA_RUN_TESTS", "false")),
            tlc_jar=env.get("NAA_TLC_JAR", ""),
            intel_lab_path=env.get("NAA_INTEL_LAB_PATH", "data/intel_lab/data.txt"),
        )
        if s.profile not in PROFILES:
            raise ValueError(f"NAA_PROFILE must be one of {sorted(PROFILES)}, got {s.profile!r}")
        s.plan = dict(PROFILES[s.profile])
        if s.workers <= 0:
            s.workers = os.cpu_count() or 1
        return s


def next_run_dir(base: str) -> Path:
    """Create and return output/test<N> with N = 1 + max existing N (atomic mkdir)."""
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        nums = [int(m.group(1)) for p in root.iterdir() if (m := re.fullmatch(r"test(\d+)", p.name))]
        cand = root / f"test{(max(nums) + 1) if nums else 1}"
        try:
            cand.mkdir()
            return cand
        except FileExistsError:  # concurrent invocation grabbed it; retry
            continue
