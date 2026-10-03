"""First pipeline task (spec §2.2): download one week of M04A and list the candidate
southbound Freeway-1 gantry pairs between mileage 0700 and 1000.

Output: printed table + results/gantry_pairs.csv. Use it to fix the segment IDs in config.yaml.
"""
from __future__ import annotations

import re

import pandas as pd

from utils.config import RESULTS, ensure_dirs, load_config
from utils.m04a import fetch_day


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    lg = cfg["m04a"]["list_gantries"]
    lo, hi = lg["mileage_min"], lg["mileage_max"]
    pat = re.compile(rb"^01F(\d{4})S$")
    vt = str(cfg["m04a"]["vehicle_type"]).encode()

    def in_range(g: bytes) -> bool:
        m = pat.match(g)
        return bool(m) and lo <= int(m.group(1)) <= hi

    def keep(gf: bytes, gt: bytes, v: bytes) -> bool:
        # either end on southbound Freeway 1 inside the mileage window
        return v == vt and (in_range(gf) or in_range(gt))

    start = pd.Timestamp(lg["week_start"])
    frames = []
    for day in pd.date_range(start, periods=7):
        df, gaps = fetch_day(day, cfg, keep)
        print(f"{day:%Y-%m-%d}: {len(df):6d} rows, {len(gaps)} missing files")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    valid = df[(df.travel_time > 0) & (df.traffic > 0)]
    summary = (
        df.groupby(["gantry_from", "gantry_to"]).size().rename("n_records").to_frame()
        .join(valid.groupby(["gantry_from", "gantry_to"]).agg(
            n_valid=("traffic", "size"),
            median_tt_sec=("travel_time", "median"),
            p05_tt_sec=("travel_time", lambda s: s.quantile(0.05)),
            p95_tt_sec=("travel_time", lambda s: s.quantile(0.95)),
            mean_traffic_5min=("traffic", "mean"),
        ))
        .reset_index()
    )

    def km(a: str, b: str) -> float:
        ma, mb = pat.match(a.encode()), pat.match(b.encode())
        return (int(mb.group(1)) - int(ma.group(1))) / 10 if ma and mb else float("nan")

    summary["length_km"] = [km(a, b) for a, b in zip(summary.gantry_from, summary.gantry_to)]
    summary["free_flow_kmh"] = summary.length_km / (summary.p05_tt_sec / 3600)
    summary = summary.sort_values(["gantry_from", "gantry_to"]).round(1)
    summary.to_csv(RESULTS / "gantry_pairs.csv", index=False)

    print(f"\nSouthbound Freeway-1 pairs, mileage {lo}–{hi}, vehicle type {vt.decode()}, "
          f"week of {start:%Y-%m-%d}:\n")
    print(summary.to_string(index=False))
    print(f"\nTimeInterval range per day: {df.time.dt.strftime('%H:%M').min()} … "
          f"{df.time.dt.strftime('%H:%M').max()}  (→ bin START; last bin of the day is 23:55)")
    print("Configured segments:", cfg["m04a"]["segments"])
    pairs = set(zip(summary.gantry_from, summary.gantry_to))
    for role, (a, b) in cfg["m04a"]["segments"].items():
        print(f"  {role:10s} {a} → {b}: {'OK' if (a, b) in pairs else 'NOT FOUND'}")


if __name__ == "__main__":
    main()
