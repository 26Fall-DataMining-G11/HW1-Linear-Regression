"""M04A download helpers: fetch one day (daily tarball or hourly-folder CSVs) and filter in memory.

The national file is never written to disk — only the matched lines are kept.
"""
from __future__ import annotations

import io
import re
import ssl
import tarfile
import time
from typing import Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

COLUMNS = ["time", "gantry_from", "gantry_to", "vehicle_type", "travel_time", "traffic"]
_LINE = re.compile(
    rb"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}),(\w+),(\w+),(\d+),(-?\d+),(-?\d+)\s*$", re.MULTILINE
)


class _RelaxedStrictAdapter(HTTPAdapter):
    """Certificates are still fully verified against the CA bundle; only the RFC 5280
    strictness flag (default since Python 3.13) is cleared, because the Freeway Bureau
    certificate chain lacks a Subject Key Identifier extension."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=requests.certs.where())
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def make_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _RelaxedStrictAdapter())
    return s


def expected_times(day: pd.Timestamp, bin_minutes: int = 5) -> list[pd.Timestamp]:
    return list(pd.date_range(day, day + pd.Timedelta(days=1), freq=f"{bin_minutes}min", inclusive="left"))


def _get(session: requests.Session, url: str, dl: dict) -> requests.Response | None:
    """GET with retry + backoff. Returns None on 404."""
    last = None
    for attempt in range(dl["retries"]):
        try:
            r = session.get(url, timeout=dl["timeout_sec"])
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(dl["sleep_sec"])
            return r
        except requests.RequestException as e:  # network error / 5xx
            last = e
            time.sleep(dl["backoff_sec"] * 2**attempt)
    raise RuntimeError(f"failed after {dl['retries']} retries: {url} ({last})")


def parse_csv_bytes(raw: bytes, keep: Callable[[bytes, bytes, bytes], bool]) -> list[tuple]:
    """Parse a headerless M04A CSV, keeping rows where keep(gantry_from, gantry_to, vehicle_type)."""
    out = []
    for m in _LINE.finditer(raw):
        ts, gf, gt, vt, tt, n = m.groups()
        if keep(gf, gt, vt):
            out.append((ts.decode(), gf.decode(), gt.decode(), int(vt), int(tt), int(n)))
    return out


def fetch_day(day: pd.Timestamp, cfg: dict, keep: Callable[[bytes, bytes, bytes], bool],
              session: requests.Session | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Download one day. Tries the daily tarball first, then the hourly-folder CSVs.

    Returns (filtered rows, gaps) where gaps lists missing files for that day.
    """
    m = cfg["m04a"]
    dl = m["download"]
    session = session or make_session()
    ymd = day.strftime("%Y%m%d")
    times = expected_times(day, m["bin_minutes"])
    rows: list[tuple] = []
    gaps: list[dict] = []
    seen: set[str] = set()

    r = _get(session, f"{m['base_url']}/M04A_{ymd}.tar.gz", dl)
    if r is not None:
        source = "tarball"
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                seen.add(member.name.rsplit("/", 1)[-1])
                rows.extend(parse_csv_bytes(tf.extractfile(member).read(), keep))
    else:
        source = "hourly"
        for ts in times:
            name = f"TDCS_M04A_{ymd}_{ts:%H%M}00.csv"
            r = _get(session, f"{m['base_url']}/{ymd}/{ts:%H}/{name}", dl)
            if r is not None:
                seen.add(name)
                rows.extend(parse_csv_bytes(r.content, keep))

    if not seen:
        gaps.append({"date": f"{day:%Y-%m-%d}", "time": "", "kind": "day_missing", "source": source})
    else:
        for ts in times:
            if f"TDCS_M04A_{ymd}_{ts:%H%M}00.csv" not in seen:
                gaps.append({"date": f"{day:%Y-%m-%d}", "time": f"{ts:%H:%M}",
                             "kind": "file_missing", "source": source})

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M")
    return df, gaps
