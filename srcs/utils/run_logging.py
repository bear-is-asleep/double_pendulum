"""Mirror terminal stdout/stderr into a run-folder log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_LOG_NAME = "train.log"
_active_run_dir: Path | None = None
_log_file: TextIO | None = None
_real_stdout: TextIO | None = None
_real_stderr: TextIO | None = None


class _StreamTee:
  """Write-through to terminal and run log; keeps fileno/isatty for tqdm etc."""

  def __init__(self, stream: TextIO, log: TextIO) -> None:
    self._stream = stream
    self._log = log

  def write(self, data: str) -> int:
    self._stream.write(data)
    self._log.write(data)
    return len(data)

  def flush(self) -> None:
    self._stream.flush()
    self._log.flush()

  def fileno(self) -> int:
    return self._stream.fileno()

  def isatty(self) -> bool:
    return self._stream.isatty()

  def __getattr__(self, name: str):
    return getattr(self._stream, name)


def _rebind_logging_to_stderr() -> None:
  """Point existing stream handlers at teed ``sys.stderr`` (post-``basicConfig``)."""
  root = logging.getLogger()
  for handler in root.handlers:
    if isinstance(handler, logging.StreamHandler):
      handler.setStream(sys.stderr)


def attach_training_terminal_log(run_dir: Path | str) -> Path:
  """Tee terminal output into the run folder and refresh logging handlers."""
  log_path = start_run_log(run_dir)
  logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
  )
  return log_path


def start_run_log(run_dir: Path | str, *, filename: str = _LOG_NAME) -> Path:
  """
  Tee ``sys.stdout`` and ``sys.stderr`` into ``run_dir/filename``.

  Safe to call again for the same ``run_dir`` (no truncate). A new directory
  switches the tee to a fresh log file.
  """
  global _active_run_dir, _log_file, _real_stdout, _real_stderr

  root = Path(run_dir).resolve()
  if _active_run_dir == root and _log_file is not None:
    return root / filename

  root.mkdir(parents=True, exist_ok=True)
  log_path = root / filename

  if _real_stdout is None:
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr

  if _log_file is not None:
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    _log_file.close()

  _log_file = log_path.open("w", encoding="utf-8", buffering=1)
  sys.stdout = _StreamTee(_real_stdout, _log_file)
  sys.stderr = _StreamTee(_real_stderr, _log_file)
  _rebind_logging_to_stderr()
  _active_run_dir = root
  return log_path


def close_run_log() -> None:
  """Restore real streams and close the log file (mostly for tests)."""
  global _active_run_dir, _log_file, _real_stdout, _real_stderr

  if _real_stderr is not None:
    sys.stderr = _real_stderr
    for handler in logging.getLogger().handlers:
      if isinstance(handler, logging.StreamHandler):
        handler.setStream(_real_stderr)
  if _real_stdout is not None:
    sys.stdout = _real_stdout
  if _log_file is not None:
    _log_file.close()
  _active_run_dir = None
  _log_file = None
  _real_stdout = None
  _real_stderr = None
