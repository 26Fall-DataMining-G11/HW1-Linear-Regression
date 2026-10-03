"""Dataset loading, test-set isolation guard, and the shared CV folds (spec §6, §11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATASET_TEST, DATASET_TRAIN, RESULTS, read_json, write_json

FOLDS_JSON = RESULTS / "folds.json"


class HeldOutLeakError(RuntimeError):
    pass


def assert_no_test_rows(df: pd.DataFrame, cfg: dict) -> None:
    """Scripts 03–05 must never see a row dated on/after the test start (§11)."""
    test_start = cfg["period"]["test_start"]
    n = int((df["date"] >= test_start).sum())
    if n:
        raise HeldOutLeakError(f"{n} rows dated ≥ {test_start:%Y-%m-%d} present in training-only code")


def load_train(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(DATASET_TRAIN).sort_values("t").reset_index(drop=True)
    assert_no_test_rows(df, cfg)
    return df


def load_test(cfg: dict, *, caller: str) -> pd.DataFrame:
    """Only src/06_final_test.py may load the held-out test set."""
    if caller != "06_final_test":
        raise HeldOutLeakError("the test set may only be loaded by src/06_final_test.py")
    df = pd.read_parquet(DATASET_TEST).sort_values("t").reset_index(drop=True)
    p = cfg["period"]
    assert ((df["date"] >= p["test_start"]) & (df["date"] <= p["test_end"])).all()
    return df


# --------------------------------------------------------------------------- folds

def make_folds(df: pd.DataFrame, cfg: dict) -> dict:
    """Expanding-window folds over training DATES with a gap of `gap_days` full days.

    Rows are sorted by t, so each fold is stored as contiguous row ranges [start, stop).
    """
    n_folds, gap = cfg["cv"]["n_folds"], cfg["cv"]["gap_days"]
    dates = np.sort(df["date"].unique())
    size = len(dates) // (n_folds + 1)
    row_date = df["date"].values
    folds = []
    for k in range(n_folds):
        v0 = len(dates) - (n_folds - k) * size
        v1 = v0 + size
        val_dates = dates[v0:v1]
        # leave `gap` full calendar days between the last training date and the first validation date
        last_train = pd.Timestamp(val_dates[0]) - pd.Timedelta(days=gap + 1)
        train_stop = int(np.searchsorted(row_date, np.datetime64(last_train), side="right"))
        val_start = int(np.searchsorted(row_date, val_dates[0], side="left"))
        val_stop = int(np.searchsorted(row_date, val_dates[-1], side="right"))
        folds.append({
            "fold": k,
            "train_rows": [0, train_stop],
            "val_rows": [val_start, val_stop],
            "train_first_date": str(pd.Timestamp(row_date[0]).date()),
            "train_last_date": str(pd.Timestamp(row_date[train_stop - 1]).date()),
            "val_first_date": str(pd.Timestamp(val_dates[0]).date()),
            "val_last_date": str(pd.Timestamp(val_dates[-1]).date()),
            "n_train": train_stop,
            "n_val": val_stop - val_start,
        })
    return {"n_rows": int(len(df)), "first_t": str(df.t.iloc[0]), "last_t": str(df.t.iloc[-1]),
            "gap_days": gap, "folds": folds}


def get_folds(df: pd.DataFrame, cfg: dict, create: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read results/folds.json (create it once in 04_cv.py) and return (train_idx, val_idx) arrays.

    Every method goes through this function, so all methods use identical indices; the stored
    row count and boundary dates are checked against the dataframe on every call.
    """
    if create and not FOLDS_JSON.exists():
        write_json(FOLDS_JSON, make_folds(df, cfg))
    spec = read_json(FOLDS_JSON)
    if spec != make_folds(df, cfg):
        raise RuntimeError("results/folds.json does not match the training dataset — "
                           "delete it only if the dataset was deliberately rebuilt")
    out = []
    for f in spec["folds"]:
        tr = np.arange(*f["train_rows"])
        va = np.arange(*f["val_rows"])
        gap = (pd.Timestamp(f["val_first_date"]) - pd.Timestamp(f["train_last_date"])).days
        assert gap >= spec["gap_days"] + 1 and tr.max() < va.min()
        out.append((tr, va))
    return out
