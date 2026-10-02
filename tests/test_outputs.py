import os
import subprocess
import sys
from pathlib import Path

from naaa.config import next_run_dir

ROOT = Path(__file__).resolve().parents[1]


def test_next_run_dir_increments(tmp_path):
    a, b, c = (next_run_dir(str(tmp_path)) for _ in range(3))
    assert [p.name for p in (a, b, c)] == ["test1", "test2", "test3"]


def test_main_creates_a_new_folder_each_run(tmp_path):
    env = {**os.environ, "NAA_PROFILE": "smoke", "NAA_OUTPUT_DIR": str(tmp_path), "NAA_WORKERS": "2"}
    for expected in ("test1", "test2"):
        proc = subprocess.run([sys.executable, str(ROOT / "Main.py")], env=env, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        d = tmp_path / expected
        for f in ("report.md", "run.log", "config.json", "summary.json", "raw/E1.csv", "figures/fig1_adaptivity.png"):
            assert (d / f).is_file(), f"{f} missing in {expected}"
