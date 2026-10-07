"""Run-folder terminal log tee."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from srcs.utils.run_logging import close_run_log


def test_start_run_log_captures_print_and_logging(tmp_path: Path) -> None:
  close_run_log()
  run_dir = tmp_path / "trial_a"
  script = f"""
import logging
from pathlib import Path
from srcs.utils.run_logging import start_run_log

log_path = start_run_log(Path({repr(str(run_dir))}))
print("hello stdout")
logging.basicConfig(
  level=logging.INFO,
  format="%(levelname)s %(message)s",
  force=True,
)
logging.getLogger("subproc").info("epoch line")
assert log_path.is_file()
"""
  subprocess.run(
    [sys.executable, "-c", script],
    check=True,
    cwd=Path(__file__).resolve().parents[1],
  )
  text = (run_dir / "train.log").read_text(encoding="utf-8")
  assert "hello stdout" in text
  assert "epoch line" in text
