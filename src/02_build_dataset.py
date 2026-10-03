"""Build the modelling dataset (spec §3–§4): 15-min aggregation, target, features,
calendar and weather joins.

Outputs
  data/processed/dataset_train.parquet   rows with date ≤ train_end
  data/processed/dataset_test.parquet    rows with date ≥ test_start (read ONLY by 06_final_test.py)
  data/processed/calendar.parquet, weather_hourly.parquet
  results/dataset_report.json            coverage, dropped-row counts, forward-fill counts

The spec names a single dataset.parquet; it is split into two files so that training-only
scripts physically cannot load test rows.
"""
from __future__ import annotations

import pandas as pd

from utils.aggregation import load_raw_m04a, valid_records
from utils.config import (CALENDAR_PARQUET, DATASET_TEST, DATASET_TRAIN, RAW, RESULTS, WEATHER_PARQUET,
                          ensure_dirs, load_config, write_json)
from utils.features import build_dataset, finalize_dataset, read_calendar_csv, read_weather_csv


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    p = cfg["period"]

    raw = load_raw_m04a()
    cal_files = list((RAW / "calendar").glob("*.csv"))
    wx_files = list((RAW / "weather").glob("*.csv"))
    if not cal_files or not wx_files:
        raise FileNotFoundError("calendar / weather CSVs missing — run src/01b_fetch_aux.py "
                                "(or place them manually, see README)")
    cal = read_calendar_csv(cal_files)
    rain = read_weather_csv(wx_files)
    cal.to_parquet(CALENDAR_PARQUET, index=False)
    rain.reset_index().to_parquet(WEATHER_PARQUET, index=False)

    full, c_build = build_dataset(raw, cal, rain, cfg)
    df, c_final = finalize_dataset(full, cfg)

    train = df[df.date <= p["train_end"]].reset_index(drop=True)
    test = df[df.date >= p["test_start"]].reset_index(drop=True)
    assert len(train) + len(test) == len(df)
    train.to_parquet(DATASET_TRAIN, index=False)
    test.to_parquet(DATASET_TEST, index=False)

    # raw 5-min coverage per segment over the raw period (reported in paper §3)
    in_period = raw[(raw.time >= p["raw_start"]) & (raw.time < p["raw_end"] + pd.Timedelta(days=1))]
    expected = int((p["raw_end"] - p["raw_start"]).days + 1) * 288
    coverage = {}
    for seg, g in in_period.groupby("segment"):
        coverage[seg] = {"expected_bins": expected, "records": int(len(g)),
                         "valid_records": int(len(valid_records(g))),
                         "valid_frac": float(len(valid_records(g)) / expected)}

    grid_train = int((full.date <= p["train_end"]).sum())
    report = {
        "period": {k: str(v.date()) for k, v in p.items() if k != "lag_warmup_days"},
        "segments": cfg["m04a"]["segments"],
        "raw_coverage": coverage,
        "grid_rows": c_build["grid_rows"],
        "grid_rows_train": grid_train,
        "grid_rows_test": c_build["grid_rows"] - grid_train,
        "ffilled_values": c_build["ffilled_values"],
        **c_final,
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "days_train": int(train.date.nunique()),
        "days_test": int(test.date.nunique()),
        "calendar": {"workdays": int(cal[(cal.date >= p["raw_start"]) & (cal.date <= p["raw_end"])].is_workday.sum()),
                     "days": int(((cal.date >= p["raw_start"]) & (cal.date <= p["raw_end"])).sum())},
        "weather": {"hours": int(len(rain)), "missing_hours": int(rain.isna().sum())},
    }
    write_json(RESULTS / "dataset_report.json", report)

    print(f"grid rows {report['grid_rows']}: dropped {report['dropped_y_nan']} (no target) + "
          f"{report['dropped_feature_nan']} (NaN feature) → {report['final_rows']}")
    print(f"train {len(train)} rows / {report['days_train']} days   "
          f"test {len(test)} rows / {report['days_test']} days (not inspected here)")
    print("forward-filled values:", report["ffilled_values"])
    print("NaN by feature (after target drop):", report["feature_nan_by_column"])
    print("train y [min]:", train.y.describe(percentiles=[.5, .95]).round(2).to_dict())


if __name__ == "__main__":
    main()
