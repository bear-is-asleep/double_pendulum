"""Load YAML under ``configs/{data,train,models,vis}/``. No hardcoded stage bounds in train code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from srcs.utils.yaml_io import read_mapping

_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
_KIND_DIRS: dict[str, Path] = {
  "data": _CONFIG_ROOT / "data",
  "train": _CONFIG_ROOT / "train",
  "vis": _CONFIG_ROOT / "vis",
}
_MODELS_DIR = _CONFIG_ROOT / "models"

# Default stem when CLI omits --data-config (matches configs/data/full.yaml).
DEFAULT_CONFIG_STEM = "full"

_TRAIN_JOB_KEYS = frozenset(
  {"data_root", "runs_root", "run_id", "stages", "train"}
)


def list_configs(kind: str) -> list[str]:
  """YAML stems under ``configs/<kind>/`` (``data``, ``train``)."""
  root = _kind_dir(kind)
  return sorted(p.stem for p in root.glob("*.yaml"))


def config_yaml_path(kind: str, ref: str) -> Path:
  """Resolve an on-disk path or ``configs/<kind>/<stem>.yaml``."""
  raw = Path(ref)
  if raw.is_file():
    return raw.resolve()
  stem = raw.stem if raw.suffix == ".yaml" else ref
  path = _kind_dir(kind) / f"{stem}.yaml"
  if not path.is_file():
    known = ", ".join(list_configs(kind)) or "(none)"
    raise FileNotFoundError(f"unknown {kind} config {ref!r}; known: {known}")
  return path


def load_config(kind: str, name: str) -> dict[str, Any]:
  """Load ``configs/<kind>/<name>.yaml``."""
  path = _kind_dir(kind) / f"{name}.yaml"
  if not path.is_file():
    known = ", ".join(list_configs(kind)) or "(none)"
    raise FileNotFoundError(f"unknown {kind} config {name!r}; known: {known}")
  cfg = read_mapping(path)
  declared = cfg.get("name")
  if declared is not None and declared != name:
    raise KeyError(f"{path}: 'name' must be {name!r}, got {declared!r}")
  return cfg


def paired_data_config(
  train_stem: str | None,
  override: str | None,
) -> str:
  """
  Pick sampler YAML stem for a train job.

  Uses ``override`` when set; else the train stem when ``configs/data/<stem>.yaml``
  exists; else ``DEFAULT_CONFIG_STEM``.
  """
  if override is not None:
    return config_yaml_path("data", override).stem
  if train_stem is not None and train_stem in list_configs("data"):
    return train_stem
  return DEFAULT_CONFIG_STEM


def load_vis_eval_config(name: str) -> dict[str, Any]:
  """``configs/vis/<name>.yaml``: single-run eval figures (``metrics`` or ``run_dir``)."""
  cfg = load_config("vis", name)
  if "models" in cfg:
    raise KeyError(
      f"vis/{name}.yaml looks like a compare config (has 'models'); "
      "use a separate stem for single-run eval figures",
    )
  if "metrics" not in cfg and "run_dir" not in cfg:
    raise KeyError(f"vis/{name}.yaml must set 'metrics' or 'run_dir'")
  return cfg


def load_vis_compare_config(name: str) -> dict[str, Any]:
  """``configs/vis/<name>.yaml``: multi-run figure compare (``models`` list required)."""
  cfg = load_config("vis", name)
  if "metrics" in cfg or "run_dir" in cfg:
    raise KeyError(
      f"vis/{name}.yaml looks like a single-run eval config; "
      "use load_vis_eval_config or a separate stem for compare",
    )
  models = cfg.get("models")
  if not isinstance(models, list) or len(models) < 2:
    raise KeyError(f"vis/{name}.yaml must list at least 2 entries under 'models'")
  for i, entry in enumerate(models):
    if not isinstance(entry, dict):
      raise TypeError(f"vis/{name}.yaml models[{i}] must be a mapping")
    if "run_dir" not in entry:
      raise KeyError(f"vis/{name}.yaml models[{i}] missing 'run_dir'")
    if "label" not in entry:
      raise KeyError(f"vis/{name}.yaml models[{i}] missing 'label'")
  return cfg


def load_train_config(name: str) -> dict[str, Any]:
  """``configs/train/<name>.yaml``: paths, stages, train overrides (pools live on disk)."""
  path = config_yaml_path("train", name)
  cfg = read_mapping(path)
  stem = path.stem
  declared = cfg.get("name")
  if declared is not None and declared != stem:
    raise KeyError(f"{path}: 'name' must be {stem!r}, got {declared!r}")
  missing = _TRAIN_JOB_KEYS - cfg.keys()
  if missing:
    raise KeyError(f"{path} missing keys: {sorted(missing)}")
  return cfg


def sampler_val_fraction(sampler_cfg: dict[str, Any]) -> float:
  """Holdout fraction carved from each stage train pool at generation time."""
  vf = float(sampler_cfg.get("val_fraction", 0.1))
  if not 0.0 <= vf < 1.0:
    raise ValueError(f"val_fraction must be in [0, 1), got {vf}")
  return vf


def load_sampler_config(
  path: Path | None = None,
  *,
  data_config: str | None = None,
) -> dict[str, Any]:
  """
  Stage bounds, safety, IC boxes, default pool sizes.

  Default file: ``configs/data/full.yaml``.
  """
  if path is not None and data_config is not None:
    raise ValueError("pass only one of path or data_config")
  if path is not None:
    return read_mapping(path)
  return load_config("data", data_config or DEFAULT_CONFIG_STEM)


def merge_model_train_cfg(
  train_job: dict[str, Any],
  *,
  model_name: str,
  seed: int | None = None,
) -> tuple[dict[str, Any], str | None]:
  """Merge ``train_job['train']`` onto ``configs/models/<model_name>.yaml``."""
  train = dict(train_job["train"])
  train.pop("model", None)
  train.pop("resume_checkpoint", None)
  device = train.pop("device", None)
  if device is not None and not isinstance(device, str):
    device = None
  cfg = load_model_config(model_name)
  cfg.update(train)
  if seed is not None:
    cfg["seed"] = int(seed)
  return cfg, device


def load_model_config(
  model_name: str,
  models_dir: Path | None = None,
) -> dict[str, Any]:
  """
  Load model config: ``baseline`` is ``configs/models/base.yaml`` alone.

  Other stems merge base with ``configs/models/<model_name>.yaml``.
  model_name: baseline | curriculum | active | progressive (no .yaml).
  """
  root = models_dir or _MODELS_DIR
  base = read_mapping(root / "base.yaml")
  if model_name == "baseline":
    merged = dict(base)
  elif model_name == "progressive":
    # Inlet scheduler knobs live in curriculum.yaml; progressive adds PNN-only keys.
    curriculum = read_mapping(root / "curriculum.yaml")
    overlay = read_mapping(root / "progressive.yaml")
    merged = {**base, **curriculum, **overlay}
  else:
    overlay = read_mapping(root / f"{model_name}.yaml")
    merged = {**base, **overlay}
  if "name" not in merged:
    raise KeyError(f"Model config {model_name} missing 'name' after merge")
  return merged


def list_model_configs(models_dir: Path | None = None) -> list[str]:
  """Strategy overlay stems (excludes base)."""
  root = models_dir or _MODELS_DIR
  return sorted(
    p.stem for p in root.glob("*.yaml") if p.stem != "base"
  )


def load_baseline_lock(path: Path | None = None) -> dict[str, Any]:
  """Step 5 locked width/depth/k/LR/batch (experiments 1-4 must reuse)."""
  lock_path = path or (_CONFIG_ROOT / "baseline_lock.yaml")
  return read_mapping(lock_path)


def _kind_dir(kind: str) -> Path:
  if kind not in _KIND_DIRS:
    known = ", ".join(sorted(_KIND_DIRS))
    raise KeyError(f"unknown config kind {kind!r}; known: {known}")
  return _KIND_DIRS[kind]
