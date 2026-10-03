"""Config loading and project paths."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
M04A_DIR = PROCESSED / "m04a"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"

DATASET_TRAIN = PROCESSED / "dataset_train.parquet"
DATASET_TEST = PROCESSED / "dataset_test.parquet"
WEATHER_PARQUET = PROCESSED / "weather_hourly.parquet"
CALENDAR_PARQUET = PROCESSED / "calendar.parquet"


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["period"].items():
        if k != "lag_warmup_days":
            cfg["period"][k] = pd.Timestamp(v)
    return cfg


def ensure_dirs() -> None:
    for d in (RAW / "m04a", RAW / "weather", RAW / "calendar", RAW / "incidents",
              M04A_DIR, RESULTS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    """Deterministic JSON writer (sorted keys, no timestamps) — results/ is the single source of truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=_default)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o)}")
