"""Download M04A for the raw period (+ lag warm-up days), filter while downloading
(vehicle type 31, configured segments only) and save one small parquet per month.

- Daily tarball when it exists, hourly-folder CSVs otherwise (spec §2.1).
- Retry with backoff, sleep between requests, resume (months already saved are skipped).
- Missing files are logged to results/download_gaps.csv.

Usage: python src/01_download.py [--force]
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from utils.config import M04A_DIR, RESULTS, ensure_dirs, load_config
from utils.m04a import fetch_day, make_session

GAP_COLS = ["date", "time", "kind", "source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download months that already exist")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    m = cfg["m04a"]
    vt = str(m["vehicle_type"]).encode()
    pairs = {(a.encode(), b.encode()): role for role, (a, b) in m["segments"].items()}
    role_of = {(a, b): role for role, (a, b) in m["segments"].items()}

    def keep(gf: bytes, gt: bytes, v: bytes) -> bool:
        return v == vt and (gf, gt) in pairs

    p = cfg["period"]
    start = p["raw_start"] - pd.Timedelta(days=p["lag_warmup_days"])
    days = pd.date_range(start, p["raw_end"], freq="D")
    local = threading.local()

    def job(day: pd.Timestamp):
        if not hasattr(local, "session"):
            local.session = make_session()
        return fetch_day(day, cfg, keep, local.session)

    for month, month_days in days.to_series().groupby(days.to_period("M")):
        out = M04A_DIR / f"M04A_{month.strftime('%Y%m')}.parquet"
        gap_out = M04A_DIR / f"gaps_{month.strftime('%Y%m')}.csv"
        if out.exists() and gap_out.exists() and not args.force:
            print(f"{month}: exists, skip")
            continue
        with ThreadPoolExecutor(max_workers=m["download"]["workers"]) as ex:
            results = list(ex.map(job, month_days))
        df = pd.concat([r[0] for r in results], ignore_index=True)
        gaps = [g for r in results for g in r[1]]
        df["segment"] = [role_of[(a, b)] for a, b in zip(df.gantry_from, df.gantry_to)]
        df = (df[["time", "segment", "travel_time", "traffic"]]
              .astype({"travel_time": "int32", "traffic": "int32"})
              .sort_values(["segment", "time"]).reset_index(drop=True))
        df.to_parquet(out, index=False)
        pd.DataFrame(gaps, columns=GAP_COLS).to_csv(gap_out, index=False)
        print(f"{month}: {len(month_days)} days, {len(df)} rows, {len(gaps)} missing files → {out.name}",
              flush=True)

    gap_files = sorted(M04A_DIR.glob("gaps_*.csv"))
    all_gaps = pd.concat([pd.read_csv(f, dtype=str) for f in gap_files], ignore_index=True)
    all_gaps.to_csv(RESULTS / "download_gaps.csv", index=False)
    print(f"download_gaps.csv: {len(all_gaps)} missing files "
          f"({(all_gaps.kind == 'day_missing').sum()} whole days)")


if __name__ == "__main__":
    main()
