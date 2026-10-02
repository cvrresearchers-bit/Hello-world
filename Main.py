#!/usr/bin/env python3
"""Single entry point:  python3 Main.py

Runs the whole study end to end in one workflow and stores everything in a NEW folder
output/test<N> (N increments on every invocation):

  1. E5  model check of the abstract agreement step (+ TLC if NAA_TLC_JAR is set)
  2. E1  adaptivity scaling      3. E2  network-agnosticism (sync / 4 async schedulers)
  4. E3  adversaries + ablations 5. E4  scalability (n, d)   6. E6  sensor-data realism
  7. statistics, tables, figures and report.md

Configuration comes from .env / environment (see .env.example). Ctrl-C / SIGTERM stops
gracefully and still writes the partial results and report.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _check_deps() -> None:
    missing = []
    for mod in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.stderr.write(f"Missing Python packages: {', '.join(missing)}\n"
                         "Install them with:  python3 -m pip install -r requirements.txt\n")
        raise SystemExit(2)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _run_pytest(out: Path, log) -> bool:
    log.info("running pytest suite ...")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(ROOT / "tests")], cwd=ROOT,
                          capture_output=True, text=True)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    log.info("pytest %s", "PASSED" if proc.returncode == 0 else "FAILED (see pytest.log)")
    return proc.returncode == 0


def _run_tlc(jar: str, out: Path, log) -> dict:
    """Optional: model-check tla/AdaptiveAA.tla with TLC. Returns a table row (or {})."""
    if not jar:
        return {}
    if not Path(jar).is_file():
        log.warning("NAA_TLC_JAR=%s not found; skipping TLC", jar)
        return {}
    try:
        proc = subprocess.run(["java", "-cp", jar, "tlc2.TLC", "-metadir", str(out / "tlc_states"), "-config", str(ROOT / "tla" / "AdaptiveAA.cfg"),
                               "-workers", "auto", str(ROOT / "tla" / "AdaptiveAA.tla")],
                              cwd=ROOT / "tla", capture_output=True, text=True, timeout=900)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.warning("TLC could not run: %s", exc)
        return {}
    txt = proc.stdout + proc.stderr
    (out / "tlc.log").write_text(txt)
    ok = "No error has been found" in txt
    import re
    found = re.findall(r"([\d,]+) states generated", txt)      # last progress / summary line
    states = int(found[-1].replace(",", "")) if found else float("nan")
    log.info("TLC: %s (%s states generated)", "no error found" if ok else "ERROR (see tlc.log)", states)
    return dict(n="TLA+", mode="TLC", states=states,
                validity_violations=0 if ok else 1, contraction_violations=0 if ok else 1)


def main() -> int:
    _check_deps()
    import pandas as pd

    from naaa import __version__, experiments as E, modelcheck
    from naaa.analysis import report
    from naaa.config import Settings, next_run_dir
    from naaa.logging_setup import setup_logging

    os.chdir(ROOT)
    cfg = Settings.load(str(ROOT / ".env"))
    out = next_run_dir(cfg.output_dir if Path(cfg.output_dir).is_absolute() else str(ROOT / cfg.output_dir))
    (out / "raw").mkdir()
    log = setup_logging(cfg.log_level, out / "run.log")
    t0 = time.time()

    import matplotlib, numpy, scipy
    log.info("naaa %s | git %s | python %s | numpy %s | scipy %s | pandas %s | matplotlib %s | %s",
             __version__, _git_hash(), platform.python_version(), numpy.__version__, scipy.__version__,
             pd.__version__, matplotlib.__version__, platform.platform())
    log.info("run folder: %s | profile=%s seed=%d workers=%d", out, cfg.profile, cfg.seed, cfg.workers)
    (out / "config.json").write_text(json.dumps(dict(settings=cfg.__dict__, git=_git_hash(),
                                                      python=platform.python_version()), indent=2, default=str))

    shutdown = threading.Event()

    def _stop(signum, _frame):
        log.warning("signal %s received: finishing gracefully (partial results will be saved)", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    tests_ok = _run_pytest(out, log) if cfg.run_tests else None

    # ---- E5: model check ---------------------------------------------------------------
    log.info("E5 model check ...")
    mc = modelcheck.run_all()
    (out / "modelcheck.json").write_text(json.dumps(mc, indent=2))
    tlc = _run_tlc(cfg.tlc_jar, out, log)

    # ---- simulations ---------------------------------------------------------------------
    plan = cfg.plan
    frames: dict[str, "pd.DataFrame"] = {}
    total = 0

    def batch(label: str, specs: list[dict]):
        nonlocal total
        if shutdown.is_set():
            return
        recs = E.run_batch(specs, cfg.workers, shutdown, label)
        total += len(recs)
        df = pd.DataFrame(recs)
        if df.empty:
            return
        errs = int(df["error"].notna().sum()) if "error" in df else 0
        if errs:
            log.error("%s: %d runs raised exceptions (see raw CSV 'error' column)", label, errs)
        for exp, g in df.groupby("exp"):
            frames[exp] = g.reset_index(drop=True)
            g.to_csv(out / "raw" / f"{exp}.csv", index=False)

    batch("E1 adaptivity", E.build_e1(plan, cfg.seed))
    batch("E2 network modes", E.build_e2(plan, cfg.seed))
    batch("E3 adversaries+ablations", E.build_e3(plan, cfg.seed))
    batch("E4 scalability", E.build_e4(plan, cfg.seed))
    e6_specs, g4_source = E.build_e6(plan, cfg.seed, cfg.intel_lab_path)
    log.info("E6 data source: %s", g4_source)
    batch("E6 sensor epochs", e6_specs)

    # ---- analysis --------------------------------------------------------------------------
    log.info("analysis: statistics, tables, figures ...")
    meta = dict(profile=cfg.profile, seed=cfg.seed, workers=cfg.workers, g4_source=g4_source, total_runs=total,
                wall_s=time.time() - t0, interrupted=shutdown.is_set(), pytest_passed=tests_ok, git=_git_hash())
    try:
        summary = report.analyze(frames, mc, tlc, out, meta)
    except Exception:
        log.exception("analysis failed; raw results are still in %s", out / "raw")
        return 1

    log.info("=" * 78)
    for k, v in summary["hypotheses"].items():
        flag = "" if "supported" not in v else ("[supported] " if v["supported"] else "[NOT supported] ")
        log.info("%s: %s%s", k, flag, v["text"])
    log.info("done in %.1fs. Everything is in: %s", time.time() - t0, out)
    log.info("start with: %s", out / "report.md")
    return 130 if shutdown.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
