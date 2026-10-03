"""Cross-validated experiments on TRAINING rows only (spec §5–§8.1).

Creates results/folds.json once and reuses it for every method. Writes
  results/cv_baselines.json   trivial / simple / LightGBM baselines
  results/cv_ridge.json       Ridge tuning curves for Set A, B (k = 3, 5, 8), C and leave-one-group-out
  results/cv_variants.json    Lasso / Elastic Net / log(y) design alternatives on Set C
  results/cv_tests.json       paired t-tests over the fold MAEs
  results/selected.json       hyperparameters frozen for 06_final_test.py
plus LaTeX tables and figures. Deterministic: two runs give identical JSON.
"""
from __future__ import annotations

import itertools

import numpy as np

from utils.config import RESULTS, ensure_dirs, load_config, write_json
from utils.data import get_folds, load_train
from utils.features import ABLATIONS, TARGET, feature_sets
from utils.latex import write_table
from utils.metrics import evaluate, paired_ttest, summarize_folds
from utils.models import (fit_predict_lgbm, fit_predict_linear, fp_mean, fp_persistence, fp_profile,
                          linear_pipeline)
from utils.plotting import COLORS, MUTED, new_fig, save


def cv_run(fp, df, folds) -> dict:
    fm = []
    for tr, va in folds:
        train, val = df.iloc[tr], df.iloc[va]
        fm.append(evaluate(val[TARGET].values, fp(train, val), val["is_peak"].values))
    return summarize_folds(fm)


def tune(df, folds, features, grid: list[dict], **fixed) -> dict:
    """Grid search by mean CV MAE (ties → first, i.e. the smaller alpha)."""
    curve = []
    for params in grid:
        res = cv_run(fit_predict_linear(features, **fixed, **params), df, folds)
        curve.append({**params, **{k: v for k, v in res.items() if k != "folds"}, "folds": res["folds"]})
    best = min(curve, key=lambda c: c["mae_mean"])
    return {"features": features, "fixed": {k: v for k, v in fixed.items()},
            "curve": [{k: v for k, v in c.items() if k != "folds"} for c in curve], "best": best}


def fold_maes(res: dict) -> list[float]:
    return [f["mae"] for f in res["folds"]]


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    np.random.seed(cfg["seed"])
    df = load_train(cfg)
    folds = get_folds(df, cfg, create=True)
    sets = feature_sets(cfg)
    A, C = sets["A"], sets["C"]
    cvc = cfg["cv"]
    thr = cfg["evaluation"]["peak_mae_threshold_min"]

    # ------------------------------------------------------------------ baselines
    base = {
        "mean": cv_run(fp_mean, df, folds),
        "persistence": cv_run(fp_persistence, df, folds),
        "profile": cv_run(fp_profile, df, folds),
        "ols_tt_now": cv_run(fit_predict_linear(["tt_now"], kind="ols"), df, folds),
    }
    lg = cvc["lgbm"]
    lgbm_grid = []
    for nl, lr in itertools.product(lg["num_leaves"], lg["learning_rate"]):
        iters: list[int] = []
        res = cv_run(fit_predict_lgbm(A, {"num_leaves": nl, "learning_rate": lr}, cfg, iters), df, folds)
        lgbm_grid.append({"num_leaves": nl, "learning_rate": lr, "best_iterations": iters, **res})
        print(f"lgbm leaves={nl:2d} lr={lr}: MAE {res['mae_mean']:.4f} ± {res['mae_std']:.4f}", flush=True)
    best_lgbm = min(lgbm_grid, key=lambda r: r["mae_mean"])
    base["lgbm_A"] = best_lgbm
    base["lgbm_grid"] = [{k: v for k, v in r.items() if k != "folds"} for r in lgbm_grid]
    write_json(RESULTS / "cv_baselines.json", base)

    # ------------------------------------------------------------------ Ridge: sets A / B / C + leave-one-group-out
    a = cvc["ridge_alphas"]
    alphas = [{"alpha": float(x)} for x in np.logspace(a["start"], a["stop"], a["num"])]
    ridge = {"A": tune(df, folds, A, alphas, kind="ridge"),
             "C": tune(df, folds, C, alphas, kind="ridge")}
    for k in [cvc["topk"]] + list(cvc["topk_sensitivity"]):
        r = tune(df, folds, C, alphas, kind="ridge", topk=k)
        sel = []  # features picked inside each training fold at the selected alpha
        for tr, _ in folds:
            m = linear_pipeline(kind="ridge", alpha=r["best"]["alpha"], topk=k).fit(
                df.iloc[tr][C].values, df.iloc[tr][TARGET].values)
            sel.append([c for c, s in zip(C, m.named_steps["topk"].get_support()) if s])
        r["selected_features_per_fold"] = sel
        ridge[f"B_k{k}"] = r
    for group, cols in ABLATIONS.items():
        feats = [c for c in C if c not in cols]
        ridge[f"C_minus_{group}"] = tune(df, folds, feats, alphas, kind="ridge")
    write_json(RESULTS / "cv_ridge.json", ridge)
    for name, r in ridge.items():
        print(f"ridge {name:20s} alpha={r['best']['alpha']:<8g} MAE {r['best']['mae_mean']:.4f} ± "
              f"{r['best']['mae_std']:.4f}  peak {r['best']['peak_mae_mean']:.4f}", flush=True)

    # ------------------------------------------------------------------ design alternatives on Set C
    la = cvc["lasso_alphas"]
    l_alphas = [float(x) for x in np.logspace(la["start"], la["stop"], la["num"])]
    variants = {
        "lasso": tune(df, folds, C, [{"alpha": x} for x in l_alphas], kind="lasso"),
        "enet": tune(df, folds, C, [{"alpha": x, "l1_ratio": r} for r in cvc["enet_l1_ratios"] for x in l_alphas],
                     kind="enet"),
        "ridge_logy": tune(df, folds, C, alphas, kind="ridge", log_target=True),
        "ols_C": {"best": cv_run(fit_predict_linear(C, kind="ols"), df, folds)},
    }
    write_json(RESULTS / "cv_variants.json", variants)

    # ------------------------------------------------------------------ paired tests (diff = other − ours)
    ours = fold_maes(ridge["C"]["best"])
    kB = f"B_k{cvc['topk']}"
    tests = {
        "lgbm_A_vs_ridge_C": paired_ttest(fold_maes(best_lgbm), ours),
        "persistence_vs_ridge_C": paired_ttest(fold_maes(base["persistence"]), ours),
        "ridge_A_vs_ridge_C": paired_ttest(fold_maes(ridge["A"]["best"]), ours),
        f"ridge_{kB}_vs_ridge_C": paired_ttest(fold_maes(ridge[kB]["best"]), ours),
    }
    for group in ABLATIONS:
        tests[f"ridge_C_minus_{group}_vs_ridge_C"] = paired_ttest(
            fold_maes(ridge[f"C_minus_{group}"]["best"]), ours)
    for name in ("lasso", "enet", "ridge_logy"):
        tests[f"{name}_vs_ridge_C"] = paired_ttest(fold_maes(variants[name]["best"]), ours)
    write_json(RESULTS / "cv_tests.json", tests)

    # every method was scored on exactly the same validation rows
    n_ref = [f["n"] for f in base["persistence"]["folds"]]
    for res in [*(base[k] for k in ("mean", "profile", "ols_tt_now", "lgbm_A")),
                *(r["best"] for r in ridge.values()), *(v["best"] for v in variants.values())]:
        assert [f["n"] for f in res["folds"]] == n_ref

    # ------------------------------------------------------------------ frozen choices for the final test
    selected = {
        "final_model": "ridge_C",
        "ridge_alpha": {k: v["best"]["alpha"] for k, v in ridge.items()},
        "topk": cvc["topk"],
        "lgbm": {"num_leaves": best_lgbm["num_leaves"], "learning_rate": best_lgbm["learning_rate"]},
        "peak_mae_threshold_min": thr,
        "cv_peak_mae_ridge_C": ridge["C"]["best"]["peak_mae_mean"],
        "cv_meets_threshold": bool(ridge["C"]["best"]["peak_mae_mean"] <= thr),
    }
    write_json(RESULTS / "selected.json", selected)

    # ------------------------------------------------------------------ tables
    def row(label, r):
        return [label, r["mae_mean"], r["rmse_mean"], r["r2_mean"], r["peak_mae_mean"]]

    def stds(rs):
        return {1: [r["mae_std"] for r in rs], 2: [r["rmse_std"] for r in rs], 4: [r["peak_mae_std"] for r in rs]}

    main_rows = [("Training mean", base["mean"]), ("Persistence", base["persistence"]),
                 ("Historical profile", base["profile"]), ("OLS on tt_now", base["ols_tt_now"]),
                 ("LightGBM (Set A)", best_lgbm), ("Ridge (Set A)", ridge["A"]["best"]),
                 (f"Ridge (Set B, k={cvc['topk']})", ridge[kB]["best"]), ("Ridge (Set C, ours)", ridge["C"]["best"])]
    write_table("tab_cv_main", ["Method", "MAE", "RMSE", "$R^2$", "Peak MAE"],
                [row(l, r) for l, r in main_rows], std=stds([r for _, r in main_rows]),
                best={1: "min", 2: "min", 3: "max", 4: "min"}, nd={1: 3, 2: 3, 3: 3, 4: 3},
                midrule_after={3, 4}, caption_note="5-fold time-ordered CV on training data, mean ± std (minutes)")

    abl = [("Set A (raw)", "A"), (f"Set B (top-{cvc['topk']})", kB)]
    abl += [(f"Set B (top-{k})", f"B_k{k}") for k in cvc["topk_sensitivity"]]
    abl += [("Set C (ours)", "C")]
    abl += [(f"C $-$ {g}".replace("caltime", "calendar $-$ slot"), f"C_minus_{g}") for g in ABLATIONS]
    rows = []
    for label, key in abl:
        r = ridge[key]["best"]
        t = None if key == "C" else paired_ttest(fold_maes(r), ours)
        rows.append([label, len(ridge[key]["features"]) if "B_k" not in key else int(key.split("k")[1]),
                     r["mae_mean"], r["peak_mae_mean"],
                     float("nan") if t is None else t["mean_diff"],
                     "--" if t is None else ("$<$0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")])
    write_table("tab_ablation", ["Feature set", "\\#feat.", "MAE", "Peak MAE", "$\\Delta$MAE vs C", "$p$"], rows,
                std={2: [ridge[k]["best"]["mae_std"] for _, k in abl],
                     3: [ridge[k]["best"]["peak_mae_std"] for _, k in abl]},
                best={2: "min", 3: "min"}, nd={2: 3, 3: 3, 4: 3}, raw_cols={0, 5}, midrule_after={3, 4},
                caption_note="Ridge, alpha tuned per set on the same folds; ΔMAE = set − C (positive: C better)")

    var_rows = [("Ridge (ours)", ridge["C"]["best"]), ("OLS", variants["ols_C"]["best"]),
                ("Lasso", variants["lasso"]["best"]), ("Elastic Net", variants["enet"]["best"]),
                ("Ridge on $\\log y$", variants["ridge_logy"]["best"])]
    write_table("tab_variants", ["Set C model", "MAE", "RMSE", "$R^2$", "Peak MAE"],
                [row(l, r) for l, r in var_rows], std=stds([r for _, r in var_rows]),
                best={1: "min", 2: "min", 3: "max", 4: "min"}, nd={1: 3, 2: 3, 3: 3, 4: 3}, raw_cols={0})

    # ------------------------------------------------------------------ figures
    fig, ax = new_fig(2.1)
    for key, label, color, ls, mk in (("C", "Set C (ours)", COLORS["ours"], "-", "o"),
                                      (kB, f"Set B (top-{cvc['topk']})", COLORS["baseline"], "--", "s"),
                                      ("A", "Set A (raw)", MUTED, ":", "^")):
        cur = ridge[key]["curve"]
        x = np.array([c["alpha"] for c in cur])
        m = np.array([c["mae_mean"] for c in cur])
        s = np.array([c["mae_std"] for c in cur])
        ax.plot(x, m, color=color, ls=ls, marker=mk, label=label)
        if key == "C":
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
            b = ridge["C"]["best"]
            ax.annotate(f"selected $\\alpha$={b['alpha']:.4g}", (b["alpha"], b["mae_mean"]),
                        xytext=(-22, 12), textcoords="offset points", ha="center", fontsize=6,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.set_xscale("log")
    ax.set_xlabel("Ridge penalty $\\alpha$ (log scale)")
    ax.set_ylabel("CV MAE (min), mean $\\pm$ std")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, columnspacing=0.8, handlelength=1.8)
    save(fig, "fig_alpha_curve")

    fig, ax = new_fig(2.3)
    shown = [g for g in ABLATIONS if g != "caltime"]
    labels = ["A", f"B\n(k={cvc['topk']})", "C\n(ours)"] + [f"C−\n{g[:5]}" for g in shown]
    keys = ["A", kB, "C"] + [f"C_minus_{g}" for g in shown]
    x = np.arange(len(keys))
    for off, metric, label, color, hatch in ((-0.2, "mae", "MAE (all rows)", COLORS["ours"], None),
                                             (0.2, "peak_mae", "Peak MAE", COLORS["peak"], "////")):
        ax.bar(x + off, [ridge[k]["best"][f"{metric}_mean"] for k in keys], width=0.38, color=color,
               hatch=hatch, edgecolor="white", linewidth=0, label=label,
               yerr=[ridge[k]["best"][f"{metric}_std"] for k in keys],
               error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"))
    ax.axhline(thr, color="#52514e", lw=0.7, ls="--")
    ax.text(len(keys) - 0.5, thr, f"decision threshold {thr:g} min", ha="right", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("CV error (min), mean $\\pm$ std")
    ax.set_xlabel("Feature set (Ridge)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig_ablation")

    print("\nCV summary (MAE ± std | peak MAE):")
    for l, r in main_rows:
        print(f"  {l:24s} {r['mae_mean']:.3f} ± {r['mae_std']:.3f} | {r['peak_mae_mean']:.3f}")
    for k, t in tests.items():
        print(f"  {k:40s} Δ={t['mean_diff']:+.4f}  t={t['t']:+.2f}  p={t['p']:.4f}")


if __name__ == "__main__":
    main()
