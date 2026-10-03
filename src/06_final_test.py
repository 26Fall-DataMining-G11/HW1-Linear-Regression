"""Final evaluation (spec §6–§8): refit the frozen models on ALL training data and evaluate
ONCE on the held-out test set. Slices, day-block bootstrap, top error cases, residual plots.

  python src/06_final_test.py --dry-run   rehearse the whole script on the last CV fold
                                          (never touches the test set; writes to results/dryrun/)
  python src/06_final_test.py             the real, one-time test evaluation

The real run writes results/final_test.lock and refuses to run again (use --force only to
regenerate outputs after a pure formatting change — never after changing a design decision).
"""
from __future__ import annotations

import argparse
import datetime as dt

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from utils.config import FIGURES, RAW, RESULTS, TABLES, ensure_dirs, load_config, read_json, write_json
from utils.data import get_folds, load_test, load_train
from utils.features import TARGET, feature_sets
from utils.latex import write_table
from utils.metrics import day_block_bootstrap, evaluate, mae
from utils.models import fit_lgbm, fit_predict_linear, fp_mean, fp_persistence, fp_profile
from utils.plotting import COLORS, MUTED, new_fig, save

LOCK = RESULTS / "final_test.lock"
LABELS = {"mean": "Training mean", "persistence": "Persistence", "profile": "Historical profile",
          "ols_tt_now": "OLS on tt_now", "lgbm_A": "LightGBM (Set A)", "ridge_A": "Ridge (Set A)",
          "ridge_B": "Ridge (Set B)", "ridge_C_logy": "Ridge on log y (Set C)",
          "ridge_C": "Ridge (Set C, ours)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    ev = cfg["evaluation"]
    sel = read_json(RESULTS / "selected.json")
    sets = feature_sets(cfg)
    A, C = sets["A"], sets["C"]

    train = load_train(cfg)
    if args.dry_run:
        tr, va = get_folds(train, cfg)[-1]
        train, test = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
        res_dir = fig_dir = tab_dir = RESULTS / "dryrun"
        print(f"DRY RUN on the last CV fold ({len(train)} train / {len(test)} pseudo-test rows)")
    else:
        if LOCK.exists() and not args.force:
            raise SystemExit(f"{LOCK} exists: the test set has already been used once. "
                             "Re-running is only legitimate for formatting changes (--force).")
        test = load_test(cfg, caller="06_final_test")
        res_dir, fig_dir, tab_dir = RESULTS, FIGURES, TABLES
        LOCK.write_text(f"test set evaluated at {dt.datetime.now().isoformat(timespec='seconds')}\n")
    res_dir.mkdir(parents=True, exist_ok=True)

    y = test[TARGET].values
    alpha = sel["ridge_alpha"]
    kB = f"B_k{sel['topk']}"
    cvv = read_json(RESULTS / "cv_variants.json")
    logy_alpha = cvv["ridge_logy"]["best"]["alpha"]
    lgbm = fit_lgbm(train, A, sel["lgbm"], cfg)
    preds = {
        "mean": fp_mean(train, test),
        "persistence": fp_persistence(train, test),
        "profile": fp_profile(train, test),
        "ols_tt_now": fit_predict_linear(["tt_now"], kind="ols")(train, test),
        "lgbm_A": lgbm.predict(test[A]),
        "ridge_A": fit_predict_linear(A, kind="ridge", alpha=alpha["A"])(train, test),
        "ridge_B": fit_predict_linear(C, kind="ridge", alpha=alpha[kB], topk=sel["topk"])(train, test),
        # design alternative, reported for transparency; the pre-declared final model is sel["final_model"]
        "ridge_C_logy": fit_predict_linear(C, kind="ridge", alpha=logy_alpha, log_target=True)(train, test),
        "ridge_C": fit_predict_linear(C, kind="ridge", alpha=alpha["C"])(train, test),
    }
    ours = preds[sel["final_model"]]
    assert all(len(p) == len(y) for p in preds.values())  # identical rows for every method
    metrics = {k: evaluate(y, p, test.is_peak.values) for k, p in preds.items()}

    # ---- slices (§7.1)
    masks = {
        "peak": test.is_peak.values == 1, "off_peak": test.is_peak.values == 0,
        "workday": test.is_workday.values == 1, "non_workday": test.is_workday.values == 0,
        "rain": test.rain_1h.values > 0, "dry": test.rain_1h.values <= 0,
        "onset": (y - test.tt_now.values) >= ev["onset_delta_min"],
        "no_onset": (y - test.tt_now.values) < ev["onset_delta_min"],
    }
    slice_methods = ["ridge_C", "ridge_C_logy", "ridge_A", "persistence", "profile", "lgbm_A"]
    slices = {name: {"n": int(m.sum()), **{k: (mae(y[m], preds[k][m]) if m.any() else None) for k in slice_methods}}
              for name, m in masks.items()}

    # ---- day-block bootstrap (§8.1): CI of MAE_other − MAE_ours
    days = test.date.values
    ae = {k: np.abs(y - p) for k, p in preds.items()}
    pk = masks["peak"]
    boot = {}
    for other in ("lgbm_A", "persistence", "ridge_A", "ridge_B", "profile", "ridge_C_logy"):
        boot[f"{other}_minus_ridge_C"] = day_block_bootstrap(ae[other], ae["ridge_C"], days, ev["bootstrap_B"], cfg["seed"])
        boot[f"{other}_minus_ridge_C_peak"] = day_block_bootstrap(ae[other][pk], ae["ridge_C"][pk], days[pk],
                                                                  ev["bootstrap_B"], cfg["seed"])
    # CI of the headline number itself
    rng = np.random.default_rng(cfg["seed"])
    pk_days = np.unique(days[pk])
    by_day = pd.DataFrame({"d": days[pk], "ae": ae["ridge_C"][pk]}).groupby("d").ae.agg(["sum", "count"])
    idx = rng.integers(0, len(pk_days), size=(ev["bootstrap_B"], len(pk_days)))
    bs = by_day["sum"].values[idx].sum(1) / by_day["count"].values[idx].sum(1)
    peak_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    # ---- error analysis (§8.3)
    resid = y - ours
    cols = ["t", "T", "tt_now", "up_tt_now", "rain_1h", "is_workday", "is_peak", "pre_long_holiday",
            "post_long_holiday", "school_break"]
    top = test.assign(actual=y, predicted=ours, error=resid, abs_error=np.abs(resid)) \
              .nlargest(ev["top_errors"], "abs_error")[cols + ["actual", "predicted", "error"]]
    top_records = [{k: (str(v) if isinstance(v, pd.Timestamp) else (int(v) if isinstance(v, (np.integer,)) else float(v)))
                    for k, v in r.items()} for r in top.to_dict("records")]
    daily = pd.DataFrame({"date": test.date, "ae": np.abs(resid), "res": resid}).groupby("date").agg(
        mae=("ae", "mean"), bias=("res", "mean"))
    worst_days = daily.nlargest(5, "mae")
    q = pd.qcut(test.tt_now, 5, duplicates="drop")
    by_bin = pd.DataFrame({"bin": q, "res": resid, "ae": np.abs(resid)}).groupby("bin", observed=True).agg(
        bias=("res", "mean"), mae=("ae", "mean"), n=("res", "size"))
    month = test.date.dt.strftime("%Y-%m")
    by_month = pd.DataFrame({"m": month, "res": resid, "ae": np.abs(resid)}).groupby("m").agg(
        bias=("res", "mean"), mae=("ae", "mean"), n=("res", "size"))
    incidents = sorted(p.name for p in (RAW / "incidents").glob("*") if p.is_file())

    thr = ev["peak_mae_threshold_min"]
    out = {
        "dry_run": bool(args.dry_run),
        "n_train": int(len(train)), "n_test": int(len(test)), "n_test_days": int(test.date.nunique()),
        "test_first_t": str(test.t.iloc[0]), "test_last_t": str(test.t.iloc[-1]),
        "final_model": sel["final_model"], "hyperparameters": {"ridge_alpha": alpha, "ridge_logy_alpha": logy_alpha, "lgbm": sel["lgbm"],
                                                               "lgbm_best_iteration": int(lgbm.best_iteration_)},
        "metrics": metrics,
        "threshold": {"peak_mae_threshold_min": thr, "peak_mae": metrics["ridge_C"]["peak_mae"],
                      "peak_mae_ci95": peak_ci, "meets_threshold": bool(metrics["ridge_C"]["peak_mae"] <= thr)},
        "slices": slices, "bootstrap": boot,
        "top_errors": top_records,
        "share_of_sq_error_top5_days": float(
            pd.Series(resid ** 2).groupby(test.date.values).sum().nlargest(5).sum() / (resid ** 2).sum()),
        "worst_days": [{"date": str(d.date()), "mae": float(r.mae), "bias": float(r.bias)} for d, r in worst_days.iterrows()],
        "residual_by_tt_now_quintile": [{"bin": str(b), "bias": float(r.bias), "mae": float(r.mae), "n": int(r.n)}
                                        for b, r in by_bin.iterrows()],
        "residual_by_month": [{"month": m, "bias": float(r.bias), "mae": float(r.mae), "n": int(r.n)}
                              for m, r in by_month.iterrows()],
        "residual_mean": float(resid.mean()),
        "incident_files": incidents,
        "incident_join": "not performed (no files in data/raw/incidents/)" if not incidents
                         else "files present — join manually on the top-error timestamps (±60 min)",
    }
    write_json(res_dir / "test_results.json", out)

    # ---- tables
    cvb, cvr = read_json(RESULTS / "cv_baselines.json"), read_json(RESULTS / "cv_ridge.json")
    cv = {"mean": cvb["mean"], "persistence": cvb["persistence"], "profile": cvb["profile"],
          "ols_tt_now": cvb["ols_tt_now"], "lgbm_A": cvb["lgbm_A"], "ridge_A": cvr["A"]["best"],
          "ridge_B": cvr[kB]["best"], "ridge_C_logy": cvv["ridge_logy"]["best"], "ridge_C": cvr["C"]["best"]}
    order = list(LABELS)
    rows = [[LABELS[k], cv[k]["mae_mean"], cv[k]["rmse_mean"], metrics[k]["mae"], metrics[k]["rmse"],
             metrics[k]["r2"], metrics[k]["peak_mae"]] for k in order]
    write_table("tab_main", ["Method", "CV MAE", "CV RMSE", "Test MAE", "Test RMSE", "Test $R^2$", "Test peak MAE"],
                rows, std={1: [cv[k]["mae_std"] for k in order], 2: [cv[k]["rmse_std"] for k in order]},
                best={1: "min", 2: "min", 3: "min", 4: "min", 5: "max", 6: "min"},
                nd={1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}, midrule_after={3, 4, 7}, out_dir=tab_dir,
                caption_note=f"minutes; CV = mean ± std over 5 folds; decision threshold: peak MAE <= {thr} min")

    s_names = [("peak", "Peak"), ("off_peak", "Off-peak"), ("workday", "Workday"), ("non_workday", "Non-workday"),
               ("rain", "Rain"), ("dry", "Dry"), ("onset", "Congestion onset"), ("no_onset", "No onset")]
    rows = [[lab, slices[k]["n"]] + [slices[k][m] if slices[k][m] is not None else float("nan") for m in slice_methods]
            for k, lab in s_names]
    # bold the best method per ROW: transpose trick — write one tiny table per row is overkill, so mark manually
    for r in rows:
        vals = [v for v in r[2:] if not np.isnan(v)]
        if vals:
            b = min(vals)
            r[2:] = [(rf"\textbf{{{v:.3f}}}" if np.isclose(v, b) else f"{v:.3f}") if not np.isnan(v) else "--"
                     for v in r[2:]]
    write_table("tab_slices", ["Slice", "$n$"] + [LABELS[m].replace(" (Set C, ours)", " C").replace(" (Set A)", " A").replace("Ridge on log y (Set C)", "Ridge log$y$")
                                                  for m in slice_methods], rows,
                raw_cols=set(range(2, 2 + len(slice_methods))), midrule_after={1, 3, 5}, out_dir=tab_dir,
                caption_note="test MAE (minutes) by slice; best method per row in bold")

    rows = [[str(r["T"])[:16], r["actual"], r["predicted"], r["error"], r["tt_now"], r["up_tt_now"], r["rain_1h"],
             "".join(f for f, k in (("W", "is_workday"), ("P", "is_peak"), ("b", "pre_long_holiday"),
                                    ("a", "post_long_holiday"), ("S", "school_break")) if r[k]) or "-"]
            for r in top_records]
    write_table("tab_errors", ["Target time $T$", "Actual", "Pred.", "Error", "tt\\_now", "up\\_tt\\_now", "Rain", "Flags"],
                rows, nd={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, out_dir=tab_dir,
                caption_note="top absolute errors of the final model; minutes / mm; flags: W workday, P peak, "
                             "b/a = day before/after a long holiday, S school break")

    # ---- residual figure: over time (daily) and vs tt_now
    fig, axes = new_fig(3.3, nrows=2)
    ax = axes[0]
    ax.bar(daily.index, daily.mae, width=0.9, color=COLORS["ours"], linewidth=0)
    for d, r in worst_days.head(2).iterrows():
        ax.annotate(f"{d:%m-%d}", (d, r.mae), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5.5)
    ax.set_ylabel("Daily MAE (min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("Date (month-day)")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for pkv, key, label, mk in ((0, "offpeak", "Off-peak", "o"), (1, "peak", "Peak", "^")):
        m = test.is_peak.values == pkv
        ax.scatter(test.tt_now.values[m], resid[m], s=3, marker=mk, color=COLORS[key], alpha=0.4, linewidths=0,
                   label=label, rasterized=True)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("Current travel time tt_now (min)")
    ax.set_ylabel("Actual $-$ predicted (min)")
    ax.legend(markerscale=2.5, handletextpad=0.2)
    fig.subplots_adjust(hspace=0.5)
    save(fig, "fig_residuals", fig_dir)

    tag = "DRY-RUN (last CV fold)" if args.dry_run else "TEST"
    print(f"\n{tag}: MAE | RMSE | R² | peak MAE")
    for k in order:
        m = metrics[k]
        print(f"  {LABELS[k]:24s} {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} | {m['peak_mae']:.3f}")
    t = out["threshold"]
    print(f"peak MAE {t['peak_mae']:.3f} (95% CI {peak_ci[0]:.3f}–{peak_ci[1]:.3f}) vs threshold {thr}: "
          f"{'MEETS' if t['meets_threshold'] else 'DOES NOT MEET'}")
    for k, b in boot.items():
        print(f"  {k:34s} {b['diff']:+.4f}  [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
