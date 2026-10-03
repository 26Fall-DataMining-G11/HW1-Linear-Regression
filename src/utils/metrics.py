"""Metrics (spec §7) and resampling tests (spec §8.1). All errors are in minutes."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(y, p, is_peak) -> dict:
    y, p, pk = np.asarray(y), np.asarray(p), np.asarray(is_peak).astype(bool)
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p),
            "peak_mae": mae(y[pk], p[pk]) if pk.any() else float("nan"), "n": int(len(y)),
            "n_peak": int(pk.sum())}


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """mean ± std (sample std, ddof = 1) over folds, plus the per-fold values."""
    out = {"folds": fold_metrics}
    for k in ("mae", "rmse", "r2", "peak_mae"):
        v = np.array([m[k] for m in fold_metrics])
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1))
    return out


def paired_ttest(a: list[float], b: list[float]) -> dict:
    """Paired t-test over fold MAEs; diff = a − b (positive ⇒ b is better)."""
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_rel(a, b)
    d = a - b
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)), "t": float(t),
            "p": float(p), "df": int(len(a) - 1), "n_folds_b_better": int((d > 0).sum())}


def day_block_bootstrap(abs_err_base, abs_err_ours, days, B: int, seed: int) -> dict:
    """95% CI of MAE_baseline − MAE_ours, resampling whole days with replacement."""
    days = np.asarray(days)
    base, ours = np.asarray(abs_err_base), np.asarray(abs_err_ours)
    uniq, inv = np.unique(days, return_inverse=True)
    nd = len(uniq)
    cnt = np.bincount(inv, minlength=nd).astype(float)
    sb = np.bincount(inv, weights=base, minlength=nd)
    so = np.bincount(inv, weights=ours, minlength=nd)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nd, size=(B, nd))
    n = cnt[pick].sum(axis=1)
    diff = sb[pick].sum(axis=1) / n - so[pick].sum(axis=1) / n
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"diff": float(base.mean() - ours.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "B": int(B), "n_days": int(nd), "frac_positive": float((diff > 0).mean())}
