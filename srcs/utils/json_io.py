"""JSON and JSONL read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from srcs.utils.paths import ensure_parent_dir


def read_json(path: Path | str) -> Any:
  """Parse UTF-8 JSON file."""
  return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(
  path: Path | str,
  obj: Any,
  *,
  indent: int | None = 2,
  sort_keys: bool = False,
) -> Path:
  """Write JSON with trailing newline."""
  p = ensure_parent_dir(path)
  text = json.dumps(obj, indent=indent, sort_keys=sort_keys) + "\n"
  p.write_text(text, encoding="utf-8")
  return p


def load_jsonl(path: Path | str) -> list[dict]:
  """Parse newline-delimited JSON; missing or empty file returns []."""
  rows: list[dict] = []
  p = Path(path)
  if not p.is_file() or p.stat().st_size == 0:
    return rows
  with p.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def append_jsonl_line(
  path: Path | str,
  record: dict[str, Any],
  *,
  sort_keys: bool = False,
) -> None:
  """Append one JSON object as a single line."""
  p = Path(path)
  with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=sort_keys) + "\n")


def dir_from_json_field(json_path: Path | str, field: str) -> Path | None:
  """Existing directory named in a JSON field, or None."""
  path = Path(json_path)
  if not path.is_file():
    return None
  raw = read_json(path).get(field)
  if not raw:
    return None
  candidate = Path(raw)
  if candidate.is_dir():
    return candidate
  return None
