"""Feature and target construction (spec §3–§4).

Everything here is a pure function of (raw 5-min records, calendar, hourly rain, config),
so the leakage tests can rebuild rows from perturbed inputs.

Leakage rule (§3.1): a 5-min record is usable at prediction time t only if its bin ends ≤ t.
With TimeInterval = bin START and grid-aligned windows, the newest usable window is
[t−15, t) whose last bin [t−5, t) ends exactly at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregation import window_series
from .config import load_config

DOW = ["dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat"]  # Sunday = reference



def _hhmm(s: str) -> pd.Timedelta:
    h, m = s.split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m))


def _slot_columns(cfg: dict) -> dict[tuple[int, int], str]:
    """(is_workday, slot of T) → dummy column name, e.g. (1, 31) → 'wd_0745'."""
    g = cfg["grid"]
    step = g["step_min"]
    first = int((_hhmm(g["t_start"]).total_seconds() // 60 + g["horizon_min"]) // step)
    last = int((_hhmm(g["t_end"]).total_seconds() // 60 + g["horizon_min"]) // step)
    return {(w, s): f"{'wd' if w else 'nwd'}_{s * step // 60:02d}{s * step % 60:02d}"
            for w in (1, 0) for s in range(first, last)}


# workday × 15-min-slot interaction dummies (interaction expansion of is_workday and time of day).
# Motivated by EDA: the workday and non-workday daily profiles differ in shape, and the per-slot
# historical profile beat the sin/cos model in the morning peak.
SLOT_COLUMNS = _slot_columns(load_config())
SLOT = list(SLOT_COLUMNS.values())
SLOT_REFERENCE = SLOT_COLUMNS[min(k for k in SLOT_COLUMNS if k[0] == 0)]  # non-workday, first slot

GROUPS = {
    "base": ["tt_now", "flow_now"],
    "lags": ["tt_lag15", "tt_lag60", "tt_trend", "tt_lastweek"],
    "upstream": ["up_tt_now", "up_flow_now"],
    "calendar": DOW + ["is_workday", "pre_long_holiday", "post_long_holiday", "school_break",
                       "sin_hour", "cos_hour"],
    "slot": SLOT,
    "weather": ["is_peak", "rain_1h", "rain_x_peak"],
}
# leave-one-group-out ablations; "caltime" = calendar + slot (everything H3 is about)
ABLATIONS = {
    "lags": GROUPS["lags"], "upstream": GROUPS["upstream"], "calendar": GROUPS["calendar"],
    "slot": GROUPS["slot"], "caltime": GROUPS["calendar"] + GROUPS["slot"], "weather": GROUPS["weather"],
}
HYPOTHESIS = {"H1": "lags", "H2": "upstream", "H3": "caltime", "H4": "weather"}

SET_A = ["tt_now", "flow_now", "hour", "weekday", "rain_1h"]
SET_C = [f for g in GROUPS.values() for f in g]
# window features that may be forward-filled (§4.6)
WINDOW_FEATURES = ["tt_now", "flow_now", "tt_lag15", "tt_lag60", "tt_lastweek", "up_tt_now", "up_flow_now"]
# Exact linear dependencies in Set C (harmless for Ridge, fatal for OLS/VIF). OLS uses Set C
# without these columns — same column space, identical fit:
#   tt_lag15            = tt_now − tt_trend
#   is_workday, is_peak = sums of workday-slot dummies;  sin_hour, cos_hour = functions of the slot
#   SLOT_REFERENCE      = reference cell of the 120 slot dummies (they sum to 1)
OLS_DROP = ["tt_lag15", "is_workday", "is_peak", "sin_hour", "cos_hour", SLOT_REFERENCE]
TARGET = "y"


def feature_sets(cfg: dict) -> dict[str, list[str]]:
    c = list(SET_C)
    if cfg["features"].get("include_workday_x_peak"):
        c.append("workday_x_peak")
    return {"A": list(SET_A), "C": c}


# --------------------------------------------------------------------------- calendar / weather

def read_calendar_csv(paths) -> pd.DataFrame:
    """DGPA office calendar CSVs (西元日期, 星期, 是否放假, 備註); 是否放假: 0 = workday, 2 = day off."""
    frames = []
    for p in sorted(paths):
        for enc in ("utf-8-sig", "cp950"):
            try:
                d = pd.read_csv(p, encoding=enc, dtype=str)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"cannot decode {p}")
        d = d.iloc[:, :4]
        d.columns = ["date", "weekday_zh", "off", "note"]
        frames.append(d)
    cal = pd.concat(frames, ignore_index=True)
    cal["date"] = pd.to_datetime(cal.date.str.strip(), format="%Y%m%d")
    cal["is_workday"] = (cal.off.str.strip() == "0").astype(int)
    cal["note"] = cal.note.fillna("")
    cal = cal.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    full = pd.date_range(cal.date.min(), cal.date.max(), freq="D")
    if len(full) != len(cal):
        raise ValueError("calendar has missing dates")
    return cal[["date", "is_workday", "note"]]


def calendar_features(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-date flags: is_workday, pre/post_long_holiday, school_break."""
    f = cfg["features"]
    cal = cal.sort_values("date").reset_index(drop=True).copy()
    off = (cal.is_workday == 0).values
    pre = np.zeros(len(cal), dtype=int)
    post = np.zeros(len(cal), dtype=int)
    i = 0
    while i < len(cal):
        if off[i]:
            j = i
            while j + 1 < len(cal) and off[j + 1]:
                j += 1
            if j - i + 1 >= f["long_holiday_min_days"]:
                if i - 1 >= 0:
                    pre[i - 1] = 1   # last working day before the stretch
                if j + 1 < len(cal):
                    post[j + 1] = 1  # first working day after the stretch
            i = j + 1
        else:
            i += 1
    cal["pre_long_holiday"] = pre
    cal["post_long_holiday"] = post
    brk = np.zeros(len(cal), dtype=int)
    for a, b in f["school_breaks"]:
        brk |= ((cal.date >= pd.Timestamp(a)) & (cal.date <= pd.Timestamp(b))).values.astype(int)
    cal["school_break"] = brk
    return cal[["date", "is_workday", "pre_long_holiday", "post_long_holiday", "school_break"]]


def read_weather_csv(paths) -> pd.Series:
    """Hourly rainfall (mm) indexed by hour END."""
    w = pd.concat([pd.read_csv(p, parse_dates=["hour_end"]) for p in sorted(paths)], ignore_index=True)
    w = w.drop_duplicates("hour_end").sort_values("hour_end")
    return w.set_index("hour_end").rain_mm.astype(float)


# --------------------------------------------------------------------------- dataset

def build_dataset(raw: pd.DataFrame, cal: pd.DataFrame, rain: pd.Series, cfg: dict,
                  start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                  ) -> tuple[pd.DataFrame, dict]:
    """One row per prediction time t on the 15-min grid, t ∈ [t_start, t_end), dates start..end.

    Returns (rows before NaN-dropping, counts). Use finalize_dataset() to drop incomplete rows.
    """
    g = cfg["grid"]
    p = cfg["period"]
    step = g["step_min"]
    win = g["window_min"]
    assert win == step, "windows are grid-aligned: window_min must equal step_min"
    start = start if start is not None else p["raw_start"]
    end = end if end is not None else p["raw_end"]
    per_day = 24 * 60 // step
    k_h = g["horizon_min"] // step

    # full regular grid (all hours) so that shifts are exact time offsets
    grid = pd.date_range(start - pd.Timedelta(days=8), end + pd.Timedelta(days=2), freq=f"{step}min",
                         inclusive="left")
    tgt = window_series(raw, "target", grid, win)
    up = window_series(raw, "upstream", grid, win)

    df = pd.DataFrame(index=grid)
    df.index.name = "t"
    # value at index t of series.shift(k) is the window starting at t − k·step
    df["y"] = tgt.tt_min.shift(-k_h)                         # [T, T+15),  T = t + horizon
    df["tt_now"] = tgt.tt_min.shift(1)                       # [t−15, t)
    df["flow_now"] = tgt.flow.shift(1)
    df["tt_lag15"] = tgt.tt_min.shift(2)                     # [t−30, t−15)
    df["tt_lag60"] = tgt.tt_min.shift(5)                     # [t−75, t−60)
    df["tt_lastweek"] = tgt.tt_min.shift(7 * per_day - k_h)  # [T−7d, T−7d+15)
    df["up_tt_now"] = up.tt_min.shift(1)
    df["up_flow_now"] = up.flow.shift(1)

    # §4.6: forward-fill window features from the previous 15-min value, at most 2 steps
    n_filled = {}
    for c in WINDOW_FEATURES:
        before = df[c].isna()
        df[c] = df[c].ffill(limit=g["ffill_max_steps"])
        n_filled[c] = before & df[c].notna()
    df["tt_trend"] = df.tt_now - df.tt_lag15

    df = df.reset_index()
    df["T"] = df.t + pd.Timedelta(minutes=g["horizon_min"])
    tod = df.t - df.t.dt.normalize()
    keep = ((tod >= _hhmm(g["t_start"])) & (tod < _hhmm(g["t_end"]))
            & (df.t.dt.normalize() >= start) & (df.t.dt.normalize() <= end)).values
    filled_counts = {c: int(m.values[keep].sum()) for c, m in n_filled.items()}
    df = df[keep].reset_index(drop=True)

    # calendar features describe T's date (known in advance)
    df["date"] = df["T"].dt.normalize()
    calf = calendar_features(cal, cfg)
    df = df.merge(calf, on="date", how="left", validate="many_to_one")
    if df.is_workday.isna().any():
        raise ValueError("calendar does not cover all dataset dates")

    dow = df["T"].dt.dayofweek  # Monday = 0
    for i, name in enumerate(DOW):
        df[name] = (dow == i).astype(int)
    df["weekday"] = dow.astype(int)
    df["hour"] = df["T"].dt.hour.astype(int)
    frac_h = df["T"].dt.hour + df["T"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * frac_h / 24)
    df["cos_hour"] = np.cos(2 * np.pi * frac_h / 24)
    df["slot"] = (df["T"].dt.hour * 60 + df["T"].dt.minute) // step
    cell = pd.Series(list(zip(df.is_workday.astype(int), df.slot.astype(int))), index=df.index).map(SLOT_COLUMNS)
    if cell.isna().any():
        raise ValueError("row outside the configured slot grid")
    dummies = pd.get_dummies(cell).reindex(columns=SLOT, fill_value=False).astype(int)
    df = pd.concat([df, dummies], axis=1)

    f = cfg["features"]
    tod_T = df["T"] - df["T"].dt.normalize()
    df["is_peak"] = ((df.is_workday == 1) & (tod_T >= _hhmm(f["peak_start"]))
                     & (tod_T < _hhmm(f["peak_end"]))).astype(int)

    # weather: rainfall of the last full hour ending ≤ t
    df["rain_hour_end"] = df.t.dt.floor("h")
    df["rain_1h"] = rain.reindex(df.rain_hour_end).values
    df["rain_x_peak"] = df.rain_1h * df.is_peak
    df["workday_x_peak"] = df.is_workday * df.is_peak

    counts = {"grid_rows": int(len(df)), "ffilled_values": filled_counts}
    return df, counts


def finalize_dataset(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Drop rows without a target, then rows with any NaN Set C feature (identical rows for all methods)."""
    feats = feature_sets(cfg)["C"]
    y_nan = df[TARGET].isna()
    feat_nan = df[feats].isna().any(axis=1) & ~y_nan
    out = df[~y_nan & ~feat_nan].sort_values("t").reset_index(drop=True)
    counts = {
        "dropped_y_nan": int(y_nan.sum()),
        "dropped_feature_nan": int(feat_nan.sum()),
        "feature_nan_by_column": {c: int(df.loc[~y_nan, c].isna().sum()) for c in feats
                                  if df.loc[~y_nan, c].isna().any()},
        "final_rows": int(len(out)),
    }
    return out, counts
