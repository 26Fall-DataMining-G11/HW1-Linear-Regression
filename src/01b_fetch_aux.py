"""Fetch the auxiliary data that spec §2.4 lists as manual steps (both are scriptable):

- D4 government office calendar (data.gov.tw dataset 14718, published by DGPA)
    → data/raw/calendar/calendar_<year>.csv   (official CSV, unchanged)
- D3 hourly rainfall, CWA CODiS station in config (default 467571 新竹)
    → data/raw/weather/codis_<station>_<YYYYMM>.csv   (columns: hour_end, rain_mm)

Files that already exist are skipped, so manually downloaded files with the same
names/columns are also accepted.
"""
from __future__ import annotations

import io
import time

import pandas as pd

from utils.config import RAW, ensure_dirs, load_config
from utils.m04a import make_session

DATASET_API = "https://data.gov.tw/api/v2/rest/dataset/14718"
CODIS_API = "https://codis.cwa.gov.tw/api/station"
TRACE = -9.8  # CODiS code for trace precipitation (< 0.1 mm) → 0.0


def fetch_calendar(session, years: list[int]) -> None:
    out_dir = RAW / "calendar"
    todo = [y for y in years if not (out_dir / f"calendar_{y}.csv").exists()]
    if not todo:
        print("calendar: all years present, skip")
        return
    dist = session.get(DATASET_API, timeout=60).json()["result"]["distribution"]
    for year in todo:
        roc = f"{year - 1911}年"
        cands = [d for d in dist if d["resourceDescription"].startswith(roc)
                 and d["resourceFormat"].upper() == "CSV" and "Google" not in d["resourceDescription"]]
        if not cands:
            raise RuntimeError(f"no calendar CSV found for {year} in data.gov.tw dataset 14718")
        # prefer the revised edition (e.g. "(1141020更新)") when one exists
        cands.sort(key=lambda d: ("更新" in d["resourceDescription"] or "修正" in d["resourceDescription"]))
        pick = cands[-1]
        r = session.get(pick["resourceDownloadUrl"], timeout=60)
        r.raise_for_status()
        (out_dir / f"calendar_{year}.csv").write_bytes(r.content)
        print(f"calendar {year}: {pick['resourceDescription']} ({len(r.content)} bytes)")
        time.sleep(1)


def fetch_weather(session, cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> None:
    w = cfg["weather"]
    out_dir = RAW / "weather"
    for month in pd.period_range(start, end, freq="M"):
        out = out_dir / f"codis_{w['station_id']}_{month.strftime('%Y%m')}.csv"
        if out.exists():
            continue
        a = max(month.start_time, start).normalize()
        b = min(month.end_time, end).normalize()
        payload = {
            "date": f"{a:%Y-%m-%d}T00:00:00.000+08:00", "type": "report_date",
            "stn_ID": w["station_id"], "stn_type": w["station_type"], "more": "",
            "start": f"{a:%Y-%m-%d}T00:00:00", "end": f"{b:%Y-%m-%d}T23:59:59", "item": "",
        }
        for attempt in range(5):
            try:
                r = session.post(CODIS_API, data=payload, timeout=120)
                r.raise_for_status()
                body = r.json()
                dts = body["data"][0]["dts"]
                break
            except Exception as e:  # noqa: BLE001 — retry any transient failure
                if attempt == 4:
                    raise RuntimeError(f"CODiS failed for {month}: {e}") from e
                time.sleep(2 * 2**attempt)
        rows = []
        for rec in dts:
            ts = pd.Timestamp(rec["DataTime"])
            # CODiS labels each record by the END of its hour; hour 24 is written as 23:59
            hour_end = ts.floor("h") + pd.Timedelta(hours=1) if ts.minute == 59 else ts
            v = rec["Precipitation"]["Accumulation"]
            rows.append((hour_end, 0.0 if v == TRACE else v))
        df = pd.DataFrame(rows, columns=["hour_end", "rain_mm"])
        df.to_csv(out, index=False)
        print(f"weather {month}: {len(df)} hours, {df.rain_mm.isna().sum()} missing, "
              f"total {df.rain_mm.sum():.1f} mm", flush=True)
        time.sleep(1)


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    p = cfg["period"]
    session = make_session()
    fetch_calendar(session, list(range(p["raw_start"].year, p["raw_end"].year + 1)))
    fetch_weather(session, cfg, p["raw_start"] - pd.Timedelta(days=1), p["raw_end"])


if __name__ == "__main__":
    main()
