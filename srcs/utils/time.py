"""Time formatting helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
  """UTC timestamp without subseconds, ISO 8601."""
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
