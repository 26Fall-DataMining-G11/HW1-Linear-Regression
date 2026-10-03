"""Leakage, alignment and protocol tests (spec §11). Run after 02_build_dataset.py (and 04_cv.py
for the fold tests):  pytest tests            # fast tests
                      pytest tests -m slow    # determinism: re-runs 04_cv.py and compares JSON
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.aggregation import valid_records
from utils.config import DATASET_TRAIN, RESULTS, ROOT, read_json
from utils.data import FOLDS_JSON, HeldOutLeakError, assert_no_test_rows, get_folds, load_test
from utils.features import TARGET, build_dataset, calendar_features, feature_sets

MIN = pd.Timedelta(minutes=1)


def _sample(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return train.iloc[np.random.default_rng(seed).choice(len(train), size=n, replace=False)]


def _weighted_tt(raw: pd.DataFrame, segment: str, a: pd.Timestamp, b: pd.Timestamp):
    """Traffic-weighted mean travel time (min) and flow over 5-min bins starting in [a, b)."""
    v = valid_records(raw[(raw.segment == segment) & (raw.time >= a) & (raw.time < b)])
    if v.empty:
        return np.nan, np.nan
    return float((v.travel_time * v.traffic).sum() / v.traffic.sum() / 60), float(v.traffic.sum())


# --------------------------------------------------------------------------- leakage

def test_no_feature_uses_information_after_t(cfg, raw, cal, rain, train):
    """Corrupt every 5-min record whose bin ends after t and every rainfall hour ending after t:
    all features of the row at t must be unchanged (and the target must change)."""
    feats = feature_sets(cfg)["C"] + ["hour", "weekday"]
    bin_len = cfg["m04a"]["bin_minutes"] * MIN
    for _, row in _sample(train, 12, seed=1).iterrows():
        t, day = row.t, row.t.normalize()
        sub = raw[(raw.time >= day - pd.Timedelta(days=9)) & (raw.time < day + pd.Timedelta(days=2))]
        base, _ = build_dataset(sub, cal, rain, cfg, start=day, end=day)

        bad = sub.copy()
        future = bad.time + bin_len > t  # bin END after t
        bad.loc[future, "travel_time"] = bad.loc[future, "travel_time"] * 7 + 100
        bad.loc[future, "traffic"] = bad.loc[future, "traffic"] * 3 + 1
        bad_rain = rain.copy()
        bad_rain[bad_rain.index > t] += 50.0
        pert, _ = build_dataset(bad, cal, bad_rain, cfg, start=day, end=day)

        b, p = base[base.t == t].iloc[0], pert[pert.t == t].iloc[0]
        np.testing.assert_allclose(b[feats].astype(float).values, row[feats].astype(float).values, rtol=1e-12)
        np.testing.assert_allclose(p[feats].astype(float).values, b[feats].astype(float).values, rtol=1e-12,
                                   err_msg=f"feature changed when post-t data was corrupted (t={t})")
        assert not np.isclose(p[TARGET], b[TARGET]), "sanity: the target must depend on post-t data"


def test_source_windows_end_at_or_before_t(cfg, train):
    """Every source 5-min bin of every window feature ends ≤ t; the weather hour ends ≤ t."""
    g = cfg["grid"]
    w = g["window_min"] * MIN
    for _, row in _sample(train, 200, seed=2).iterrows():
        t, T = row.t, row["T"]
        assert T == t + g["horizon_min"] * MIN
        window_starts = {
            "tt_now": t - 15 * MIN, "flow_now": t - 15 * MIN, "up_tt_now": t - 15 * MIN, "up_flow_now": t - 15 * MIN,
            "tt_lag15": t - 30 * MIN, "tt_lag60": t - 75 * MIN, "tt_lastweek": T - pd.Timedelta(days=7),
        }
        for name, s in window_starts.items():
            assert s + w <= t, f"{name} window ends after t"
        assert row.rain_hour_end <= t and t - row.rain_hour_end < pd.Timedelta(hours=1)
        assert T >= t + w  # the target window starts after every feature window has closed


# --------------------------------------------------------------------------- alignment

def test_target_and_features_match_raw(cfg, raw, train):
    """Recompute y, tt_now, flow_now, tt_lag60, up_tt_now, tt_lastweek for 20 random rows from raw parquet."""
    n_checked = 0
    for _, row in _sample(train, 20, seed=3).iterrows():
        t, T = row.t, row["T"]
        y, _ = _weighted_tt(raw, "target", T, T + 15 * MIN)
        np.testing.assert_allclose(row[TARGET], y, rtol=1e-9)
        checks = {
            "tt_now": ("target", t - 15 * MIN, 0), "flow_now": ("target", t - 15 * MIN, 1),
            "tt_lag60": ("target", t - 75 * MIN, 0), "up_tt_now": ("upstream", t - 15 * MIN, 0),
            "tt_lastweek": ("target", T - pd.Timedelta(days=7), 0),
        }
        for name, (seg, s, k) in checks.items():
            direct = _weighted_tt(raw, seg, s, s + 15 * MIN)[k]
            if np.isnan(direct):  # value was forward-filled (§4.6): must equal one of the 2 previous windows
                prev = [_weighted_tt(raw, seg, s - j * 15 * MIN, s - (j - 1) * 15 * MIN)[k] for j in (1, 2)]
                assert any(np.isclose(row[name], v) for v in prev if not np.isnan(v))
            else:
                np.testing.assert_allclose(row[name], direct, rtol=1e-9, err_msg=name)
                n_checked += 1
        np.testing.assert_allclose(row.tt_trend, row.tt_now - row.tt_lag15, atol=1e-12)
    assert n_checked >= 80


def test_calendar_flags(cfg, cal):
    f = calendar_features(cal, cfg).set_index("date")
    assert f.loc["2025-01-01", "is_workday"] == 0            # New Year's Day
    assert f.loc["2025-01-24", "pre_long_holiday"] == 1      # last workday before Lunar New Year 2025
    assert f.loc["2025-02-03", "post_long_holiday"] == 1     # first workday after it
    assert f.loc["2025-01-28", ["pre_long_holiday", "post_long_holiday"]].sum() == 0
    assert (f.loc[f.is_workday == 0, ["pre_long_holiday", "post_long_holiday"]].values == 0).all()


def test_row_grid(cfg, train):
    tod = train.t - train.t.dt.normalize()
    assert tod.min() >= pd.Timedelta(hours=5) and tod.max() < pd.Timedelta(hours=20)
    assert (train.t.dt.minute % cfg["grid"]["step_min"] == 0).all()
    assert train.t.is_monotonic_increasing and train.t.is_unique
    assert not train[feature_sets(cfg)["C"] + [TARGET]].isna().any().any()
    assert ((train.is_peak == 1) <= (train.is_workday == 1)).all()


# --------------------------------------------------------------------------- protocol

def test_test_set_isolation(cfg, train):
    assert train.date.max() <= cfg["period"]["train_end"]
    assert pd.read_parquet(DATASET_TRAIN, columns=["date"]).date.max() < cfg["period"]["test_start"]
    leaked = train.head(3).assign(date=cfg["period"]["test_start"])
    with pytest.raises(HeldOutLeakError):
        assert_no_test_rows(leaked, cfg)
    with pytest.raises(HeldOutLeakError):
        load_test(cfg, caller="03_eda")
    for script in ("03_eda.py", "04_cv.py", "05_analysis_train.py"):
        src = (ROOT / "src" / script).read_text()
        assert "load_train(" in src and "load_test" not in src and "DATASET_TEST" not in src


@pytest.mark.skipif(not FOLDS_JSON.exists(), reason="run 04_cv.py first")
def test_folds_are_time_ordered_with_gap_and_reused(cfg, train):
    folds = get_folds(train, cfg)
    again = get_folds(train, cfg)
    assert len(folds) == cfg["cv"]["n_folds"]
    prev_val_end = -1
    for (tr, va), (tr2, va2) in zip(folds, again):
        assert np.array_equal(tr, tr2) and np.array_equal(va, va2)   # identical indices for every method
        gap_days = (train.date.iloc[va[0]] - train.date.iloc[tr[-1]]).days
        assert gap_days >= cfg["cv"]["gap_days"] + 1                 # ≥ 1 full day between train and validation
        assert tr[0] == 0 and va[0] > prev_val_end                   # expanding window, disjoint validation blocks
        prev_val_end = va[-1]


@pytest.mark.skipif(not (RESULTS / "cv_ridge.json").exists(), reason="run 04_cv.py first")
def test_persistence_scored_on_same_rows_as_ridge():
    base, ridge = read_json(RESULTS / "cv_baselines.json"), read_json(RESULTS / "cv_ridge.json")
    spec = read_json(FOLDS_JSON)
    n_val = [f["n_val"] for f in spec["folds"]]
    for res in (base["persistence"], base["lgbm_A"], ridge["C"]["best"], ridge["A"]["best"]):
        assert [f["n"] for f in res["folds"]] == n_val


@pytest.mark.slow
def test_cv_is_deterministic():
    files = sorted(RESULTS.glob("cv_*.json")) + [RESULTS / "selected.json", FOLDS_JSON]

    def digest():
        return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}

    before = digest()
    subprocess.run([sys.executable, str(Path(ROOT) / "src" / "04_cv.py")], check=True, cwd=ROOT,
                   capture_output=True)
    assert digest() == before
