"""Coefficient analysis on TRAINING rows only (spec §8.2).

OLS (statsmodels) on standardized Set C features with HAC / Newey–West standard errors, VIF,
Ridge-vs-OLS coefficient gap, forest plot, workday×slot effect curves, and the numbers behind
the H1–H4 verdicts.

Design matrix: Set C contains exact linear dependencies (see features.OLS_DROP), which Ridge
tolerates but OLS cannot. OLS and VIF use Set C without those columns — the column space and
the fitted values are identical. The Ridge model (full Set C, selected alpha) is re-expressed
in the same parameterization by projecting its predictions onto the OLS design (exact).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.config import RESULTS, ensure_dirs, load_config, read_json, write_json
from utils.data import load_train
from utils.features import (GROUPS, HYPOTHESIS, OLS_DROP, SLOT, SLOT_COLUMNS, SLOT_REFERENCE, TARGET,
                            _hhmm, feature_sets)
from utils.latex import write_table
from utils.models import linear_pipeline
from utils.plotting import COLORS, INK2, MUTED, new_fig, save

KEY_FEATURES = {  # coefficients quoted for each hypothesis
    "H1": ["tt_lag60", "tt_trend", "tt_lastweek"],
    "H2": ["up_tt_now", "up_flow_now"],
    "H3": ["pre_long_holiday", "post_long_holiday", "school_break"],
    "H4": ["rain_x_peak", "rain_1h"],
}
# H4 claims a PEAK-SPECIFIC rain effect, so only the interaction counts as evidence for it;
# rain_1h (the all-day effect) is reported as context.
CONTEXT_ONLY = {"H4": ["rain_1h"]}


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    df = load_train(cfg)
    ev = cfg["evaluation"]
    C = feature_sets(cfg)["C"]
    feats = [c for c in C if c not in OLS_DROP]

    X = df[feats].astype(float)
    mu, sd = X.mean(), X.std(ddof=0)
    Z = sm.add_constant((X - mu) / sd)
    y = df[TARGET].values
    assert np.linalg.matrix_rank(Z.values) == Z.shape[1], "OLS design is rank-deficient"

    ols = sm.OLS(y, Z).fit()
    hac = {L: ols.get_robustcov_results(cov_type="HAC", maxlags=L) for L in [ev["hac_maxlags"], *ev["hac_sensitivity"]]}
    main_fit = hac[ev["hac_maxlags"]]
    names = list(Z.columns)
    pos = {c: i for i, c in enumerate(names)}
    ci = main_fit.conf_int(alpha=0.05)
    vif = {c: float(variance_inflation_factor(Z.values, i)) for i, c in enumerate(names) if c != "const"}

    # Ridge (full Set C, selected alpha) in the OLS parameterization: exact projection
    alpha = read_json(RESULTS / "selected.json")["ridge_alpha"]["C"]
    pipe = linear_pipeline(kind="ridge", alpha=alpha).fit(df[C].values, y)
    ridge_pred = pipe.predict(df[C].values)
    ridge_params, *_ = np.linalg.lstsq(Z.values, ridge_pred, rcond=None)
    assert np.abs(Z.values @ ridge_params - ridge_pred).max() < 1e-6, "Set C is not spanned by the OLS design"

    coefs = []
    for i, c in enumerate(names):
        if c == "const":
            continue
        coefs.append({
            "feature": c,
            "group": next(g for g, fs in GROUPS.items() if c in fs),
            "beta_std": float(main_fit.params[i]),          # minutes per 1 SD
            "beta_raw": float(main_fit.params[i] / sd[c]),  # minutes per raw unit
            "sd": float(sd[c]),
            "se_hac": float(main_fit.bse[i]),
            "ci_low": float(ci[i][0]), "ci_high": float(ci[i][1]),
            "p_hac": float(main_fit.pvalues[i]),
            "se_ols": float(ols.bse.iloc[i]),
            **{f"se_hac{L}": float(hac[L].bse[i]) for L in ev["hac_sensitivity"]},
            **{f"p_hac{L}": float(hac[L].pvalues[i]) for L in ev["hac_sensitivity"]},
            "vif": vif[c], "vif_flag": bool(vif[c] >= ev["vif_flag"]),
            "ridge_beta_std": float(ridge_params[i]),
        })
    gap = max(abs(d["beta_std"] - d["ridge_beta_std"]) for d in coefs)
    gap_ok = max(abs(d["beta_std"] - d["ridge_beta_std"]) for d in coefs
                 if not d["vif_flag"] and d["feature"] not in SLOT)
    worst = max(coefs, key=lambda d: abs(d["beta_std"] - d["ridge_beta_std"]))["feature"]
    by = {d["feature"]: d for d in coefs}

    # ---- workday × slot effects in minutes, relative to the reference cell (raw coefficient = β / sd)
    step = cfg["grid"]["step_min"]
    slot_effects = []
    for (w, s), col in SLOT_COLUMNS.items():
        d = by.get(col)
        slot_effects.append({
            "is_workday": w, "slot": s, "time": f"{s * step // 60:02d}:{s * step % 60:02d}", "feature": col,
            "effect_min": 0.0 if d is None else d["beta_raw"],
            "ci_low": 0.0 if d is None else d["ci_low"] / d["sd"],
            "ci_high": 0.0 if d is None else d["ci_high"] / d["sd"],
        })
    # Shape contrast (HAC inference): how much the peak slots stand out from the day's own average
    # slot effect, workday minus non-workday. Level differences between day types are absorbed by
    # the day-of-week dummies, so only this difference-in-differences is identified.
    f = cfg["features"]
    p0, p1 = (int(_hhmm(f[k]).total_seconds() // 60 // step) for k in ("peak_start", "peak_end"))
    peak_slots = list(range(p0, p1))
    all_slots = sorted({s for _, s in SLOT_COLUMNS})
    r = np.zeros(len(names))
    for w, sign in ((1, 1.0), (0, -1.0)):
        for s in all_slots:
            col = SLOT_COLUMNS[(w, s)]
            if col in pos:  # the reference cell has effect 0 by construction
                weight = (1 / len(peak_slots) if s in peak_slots else 0.0) - 1 / len(all_slots)
                r[pos[col]] += sign * weight / sd[col]
    tt = main_fit.t_test(r)
    cc = np.asarray(tt.conf_int()).ravel()
    contrast = {"estimate_min": float(np.ravel(tt.effect)[0]), "ci_low": float(cc[0]), "ci_high": float(cc[1]),
                "p_hac": float(np.ravel(tt.pvalue)[0]), "n_slots": len(peak_slots)}

    # ---- hypothesis evidence: key coefficients + leave-one-group-out ablation (from 04_cv.py)
    tests = read_json(RESULTS / "cv_tests.json")
    hyp = {}
    for h, group in HYPOTHESIS.items():
        t = tests[f"ridge_C_minus_{group}_vs_ridge_C"]
        keys = [by[k] for k in KEY_FEATURES[h]]
        sig = [k["feature"] for k in keys if k["p_hac"] < 0.05 and k["feature"] not in CONTEXT_ONLY.get(h, [])]
        if h == "H3" and contrast["p_hac"] < 0.05 and contrast["estimate_min"] > 0:
            sig.append("workday_peak_contrast")
        abl_ok = t["mean_diff"] > 0 and t["p"] < 0.05
        verdict = "supported" if (abl_ok and sig) else ("not supported" if (not abl_ok and not sig) else "mixed")
        hyp[h] = {
            "group": group,
            "ablation_delta_mae_min": t["mean_diff"], "ablation_p": t["p"],
            "ablation_folds_improved": t["n_folds_b_better"],
            "key_coefficients": {k["feature"]: {x: k[x] for x in ("beta_std", "beta_raw", "ci_low", "ci_high", "p_hac")}
                                 for k in keys},
            "significant_key_features": sig,
            "provisional_verdict": verdict,
            "rule": "supported = removing the group raises CV MAE (paired t, p < 0.05) AND ≥ 1 key coefficient "
                    "(excluding context-only ones) has HAC p < 0.05; not supported = neither; otherwise mixed. "
                    "Practical size = ΔMAE in minutes.",
        }
    hyp["H3"]["workday_peak_contrast"] = contrast
    for g in ("calendar", "slot"):
        t = tests[f"ridge_C_minus_{g}_vs_ridge_C"]
        hyp["H3"][f"ablation_{g}_only"] = {"delta_mae_min": t["mean_diff"], "p": t["p"]}

    out = {
        "n_obs": int(ols.nobs), "n_features": len(feats), "r2": float(ols.rsquared), "r2_adj": float(ols.rsquared_adj),
        "dropped_for_exact_collinearity": OLS_DROP, "slot_reference": SLOT_REFERENCE,
        "hac_maxlags": ev["hac_maxlags"], "hac_sensitivity": ev["hac_sensitivity"],
        "intercept": float(main_fit.params[0]),
        "ridge_alpha": alpha, "ridge_ols_max_abs_coef_diff": float(gap), "ridge_ols_max_diff_feature": worst,
        "ridge_ols_max_abs_coef_diff_nonslot_lowvif": float(gap_ok),
        "n_vif_flagged": int(sum(d["vif_flag"] for d in coefs)),
        "vif_flagged": [d["feature"] for d in coefs if d["vif_flag"]],
        "max_se_ratio_hac_vs_ols": float(max(d["se_hac"] / d["se_ols"] for d in coefs)),
        "coefficients": coefs, "slot_effects": slot_effects, "hypotheses": hyp,
    }
    write_json(RESULTS / "coefficients.json", out)

    # ---- table + forest plot: the non-slot coefficients (slot dummies are shown as curves)
    order = sorted((d for d in coefs if d["feature"] not in SLOT), key=lambda d: -abs(d["beta_std"]))
    rows = [[d["feature"] + ("*" if d["vif_flag"] else ""), d["beta_std"],
             f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]",
             "$<$0.001" if d["p_hac"] < 0.001 else f"{d['p_hac']:.3f}", d["vif"]] for d in order]
    write_table("tab_coefficients", ["Feature", r"$\beta$ (min/SD)", "95\\% CI (HAC)", "$p$", "VIF"], rows,
                nd={1: 3, 4: 1}, align="lrcrr", raw_cols={3},
                caption_note=f"OLS on standardized Set C (reduced design, slot dummies omitted from the table); "
                             f"HAC maxlags={ev['hac_maxlags']}; * = VIF >= {ev['vif_flag']}")

    fig, ax = new_fig(2.9)
    yy = np.arange(len(order))[::-1]
    for d, yv in zip(order, yy):
        ax.plot([d["ci_low"], d["ci_high"]], [yv, yv], color=COLORS["ours"], lw=1.0, solid_capstyle="round")
        ax.plot(d["beta_std"], yv, marker="o", ms=3.5, color=COLORS["ours"],
                markerfacecolor="white" if d["vif_flag"] else COLORS["ours"], markeredgewidth=0.9)
        ax.plot(d["ridge_beta_std"], yv, marker="|", ms=5, color=COLORS["baseline"], markeredgewidth=1.1)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([d["feature"] + ("*" if d["vif_flag"] else "") for d in order], fontsize=5.8)
    ax.set_xlabel("Std. coefficient (min per SD), 95% HAC CI")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", ls="-", color=COLORS["ours"], label="OLS (hollow, *: VIF $\\geq$ %g)" % ev["vif_flag"])
    ax.plot([], [], marker="|", ls="", color=COLORS["baseline"], markeredgewidth=1.1,
            label=f"Ridge ($\\alpha$={alpha:.4g})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncols=2, fontsize=5.8, columnspacing=1.0)
    ax.tick_params(axis="y", colors=INK2)
    save(fig, "fig_forest")

    # ---- slot-effect curves
    fig, ax = new_fig(2.2)
    se = pd.DataFrame(slot_effects)
    for w, key, label, ls in ((1, "workday", "Workday", "-"), (0, "nonworkday", "Non-workday", "--")):
        g = se[se.is_workday == w].sort_values("slot")
        x = g.slot * step / 60
        ax.fill_between(x, g.ci_low, g.ci_high, color=COLORS[key], alpha=0.18, linewidth=0)
        ax.plot(x, g.effect_min, color=COLORS[key], ls=ls, label=f"{label} (95% HAC CI)")
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xticks(range(6, 22, 3))
    ax.set_xlabel("Target time $T$ (hour of day)")
    ax.set_ylabel("Slot effect on $y$ (min)\nvs. non-workday 06:00")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, columnspacing=1.0, handlelength=1.6)
    save(fig, "fig_slot_effects")

    print(f"OLS R²={out['r2']:.4f}, n={out['n_obs']}, k={len(feats)}; "
          f"max |Ridge−OLS| std coef = {gap:.4f} ({worst}); VIF≥{ev['vif_flag']}: {out['vif_flagged']}")
    for d in order[:10]:
        print(f"  {d['feature']:18s} β={d['beta_std']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}] "
              f"p={d['p_hac']:.3g} VIF={d['vif']:.1f}")
    print(f"  max |Ridge−OLS| among non-slot, VIF<{ev['vif_flag']} features: {gap_ok:.4f}")
    print(f"  peak-slot excess, workday − non-workday: {contrast['estimate_min']:+.3f} min "
          f"[{contrast['ci_low']:+.3f}, {contrast['ci_high']:+.3f}] p={contrast['p_hac']:.3g}")
    for h, v in hyp.items():
        print(f"  {h} ({v['group']}): ΔMAE={v['ablation_delta_mae_min']:+.4f} (p={v['ablation_p']:.3f}), "
              f"sig={v['significant_key_features']} → {v['provisional_verdict']}")


if __name__ == "__main__":
    main()
