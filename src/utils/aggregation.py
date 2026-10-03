"""Traffic-weighted aggregation of 5-min M04A records into grid-aligned 15-min windows (spec §3)."""
from __future__ import annotations

import pandas as pd

from .config import M04A_DIR


def load_raw_m04a() -> pd.DataFrame:
    """All filtered 5-min records: columns time (bin START, naive Asia/Taipei), segment, travel_time [s], traffic."""
    files = sorted(M04A_DIR.glob("M04A_*.parquet"))
    if not files:
        raise FileNotFoundError("no M04A parquet in data/processed/m04a — run src/01_download.py")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def valid_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Spec §3: records with TravelTime ≤ 0, Traffic ≤ 0 or missing are excluded."""
    return raw[(raw.travel_time > 0) & (raw.traffic > 0)]


def window_series(raw: pd.DataFrame, segment: str, index: pd.DatetimeIndex, window_min: int = 15) -> pd.DataFrame:
    """Aggregate one segment to windows [s, s + window_min), indexed by window START s.

    tt_min = Σ(TravelTime·Traffic) / Σ(Traffic) / 60 over the valid 5-min bins in the window (minutes)
    flow   = Σ(Traffic) over the valid bins (vehicles / window)
    n_bins = number of valid bins; windows with none are NaN.
    """
    v = valid_records(raw[raw.segment == segment])
    start = v.time.dt.floor(f"{window_min}min")
    g = pd.DataFrame({
        "wt": (v.travel_time.astype("float64") * v.traffic).values,
        "w": v.traffic.astype("float64").values,
        "n": 1,
    }).groupby(start.values).sum()
    out = pd.DataFrame({
        "tt_min": g.wt / g.w / 60.0,
        "flow": g.w,
        "n_bins": g.n,
    }).reindex(index)
    out["n_bins"] = out.n_bins.fillna(0).astype(int)
    return out
