"""EDA on TRAINING rows only (spec §9) → paper/figures/fig_eda_*.pdf, paper/tables/tab_*.tex,
results/eda.json."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.config import RESULTS, ensure_dirs, load_config, write_json
from utils.data import load_train
from utils.features import SLOT, TARGET, feature_sets
from utils.latex import write_table
from utils.plotting import COLORS, INK2, MUTED, new_fig, plt, save

COLLINEAR = ["tt_now", "tt_lag15", "tt_lag60", "tt_lastweek", "up_tt_now"]


def slot_label(slot: int, step: int) -> str:
    m = slot * step
    return f"{m // 60:02d}:{m % 60:02d}"


def vif_table(X: pd.DataFrame) -> dict[str, float]:
    Z = ((X - X.mean()) / X.std()).values
    Z = np.column_stack([np.ones(len(Z)), Z])
    return {c: float(variance_inflation_factor(Z, i + 1)) for i, c in enumerate(X.columns)}


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    df = load_train(cfg)  # raises if any test-period row is present
    feats = feature_sets(cfg)["C"]
    step = cfg["grid"]["step_min"]
    out: dict = {"n_rows": int(len(df))}

    # ---- fig_eda_profile: median y (IQR band) by 15-min slot of T, workday vs non-workday
    fig, ax = new_fig(2.2)
    prof = {}
    for flag, key, label, ls in ((1, "workday", "Workday", "-"), (0, "nonworkday", "Non-workday", "--")):
        q = df[df.is_workday == flag].groupby("slot")[TARGET].quantile([0.25, 0.5, 0.75]).unstack()
        x = q.index * step / 60
        ax.fill_between(x, q[0.25], q[0.75], color=COLORS[key], alpha=0.18, linewidth=0)
        ax.plot(x, q[0.5], color=COLORS[key], ls=ls, label=f"{label} (median, IQR)")
        prof[key] = {slot_label(s, step): {"q25": float(r[0.25]), "median": float(r[0.5]), "q75": float(r[0.75])}
                     for s, r in q.iterrows()}
    f = cfg["features"]
    h0, h1 = (int(s[:2]) + int(s[3:]) / 60 for s in (f["peak_start"], f["peak_end"]))
    ax.axvspan(h0, h1, color=MUTED, alpha=0.12, linewidth=0)
    ax.text(h0 + 0.1, 0.97, "peak window", transform=ax.get_xaxis_transform(), rotation=90, ha="left",
            va="top", fontsize=6, color=INK2)
    ax.set_xlabel("Target time $T$ (hour of day)")
    ax.set_ylabel("Travel time $y$ (min)")
    ax.set_xticks(range(6, 22, 3))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, columnspacing=1.0, handlelength=1.6)
    save(fig, "fig_eda_profile")
    out["profile"] = prof
    wd = df[df.is_workday == 1].groupby("slot")[TARGET].median()
    nwd = df[df.is_workday == 0].groupby("slot")[TARGET].median()
    excess = (wd - nwd)
    out["peak_window_check"] = {
        "configured": [f["peak_start"], f["peak_end"]],
        "workday_median_max_slot": slot_label(int(wd.idxmax()), step),
        "slots_workday_median_exceeds_nonworkday_by_0.5min": [slot_label(int(s), step)
                                                             for s in excess[excess >= 0.5].index],
    }

    # ---- fig_eda_scatter: y vs tt_now and y vs up_tt_now, colored by is_peak
    fig, axes = new_fig(2.2, ncols=2, sharey=True)
    rng = np.random.default_rng(cfg["seed"])
    sub = df.iloc[rng.choice(len(df), size=min(6000, len(df)), replace=False)]
    for ax, col, xl in ((axes[0], "tt_now", "Current travel time\ntt_now (min)"),
                        (axes[1], "up_tt_now", "Upstream travel time\nup_tt_now (min)")):
        for pk, key, label, mk in ((0, "offpeak", "Off-peak", "o"), (1, "peak", "Peak", "^")):
            s = sub[sub.is_peak == pk]
            ax.scatter(s[col], s[TARGET], s=3, marker=mk, color=COLORS[key], alpha=0.35, linewidths=0,
                       label=label, rasterized=True)
        ax.set_xlabel(xl)
    lim = [df.tt_now.min(), df[TARGET].quantile(0.999)]
    axes[0].plot(lim, lim, color=MUTED, lw=0.7, ls=":")
    axes[0].set_ylabel("Travel time at $t$+60, $y$ (min)")
    axes[0].legend(loc="upper left", markerscale=2.5, handletextpad=0.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_eda_scatter")

    # ---- fig_eda_rank: |Pearson r| and Spearman rho of every Set C feature with y
    rank = []
    for c in feats:
        r = stats.pearsonr(df[c], df[TARGET])[0]
        rho = stats.spearmanr(df[c], df[TARGET])[0]
        rank.append({"feature": c, "pearson": float(r), "abs_pearson": float(abs(r)), "spearman": float(rho)})
    rank.sort(key=lambda d: -d["abs_pearson"])
    out["rank"] = rank
    full_rank, rank = rank, [d for d in rank if d["feature"] not in SLOT]  # plot without the 120 slot dummies
    fig, ax = new_fig(3.4)
    yy = np.arange(len(rank))[::-1]
    ax.barh(yy + 0.2, [d["abs_pearson"] for d in rank], height=0.38, color=COLORS["pearson"],
            label="|Pearson $r$|")
    ax.barh(yy - 0.2, [abs(d["spearman"]) for d in rank], height=0.38, color=COLORS["spearman"],
            hatch="////", edgecolor="white", linewidth=0, label="|Spearman $\\rho$|")
    ax.set_yticks(yy)
    ax.set_yticklabels([d["feature"] for d in rank], fontsize=5.8)
    ax.set_xlabel("Absolute correlation with $y$ (unitless)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig_eda_rank")
    out["rank_best_slot_dummy"] = next(d for d in full_rank if d["feature"] in SLOT)

    # ---- fig_eda_rain: y by rain (yes/no) × peak (yes/no)
    fig, ax = new_fig(2.1)
    groups, labels, cols, rain_stats = [], [], [], {}
    for pk, pk_name in ((0, "Off-peak"), (1, "Peak")):
        for rn, rn_name in ((0, "dry"), (1, "rain")):
            s = df[(df.is_peak == pk) & ((df.rain_1h > 0).astype(int) == rn)][TARGET]
            groups.append(s.values)
            labels.append(f"{pk_name}\n{rn_name}\n(n={len(s):,})")
            cols.append(COLORS["peak" if pk else "offpeak"])
            rain_stats[f"{pk_name.lower()}_{rn_name}"] = {"n": int(len(s)), "mean": float(s.mean()),
                                                          "median": float(s.median())}
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker=".", markersize=1.5, markerfacecolor=MUTED, markeredgewidth=0, alpha=0.4),
                    medianprops=dict(color="#0b0b0b", linewidth=1.1), boxprops=dict(linewidth=0.6),
                    whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))
    for patch, c, lab in zip(bp["boxes"], cols, labels):
        patch.set_facecolor(c)
        patch.set_alpha(0.55 if "dry" in lab else 0.95)
    ax.set_ylabel("Travel time $y$ (min)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_eda_rain")
    rain_stats["rain_effect_offpeak_min"] = rain_stats["off-peak_rain"]["mean"] - rain_stats["off-peak_dry"]["mean"]
    rain_stats["rain_effect_peak_min"] = rain_stats["peak_rain"]["mean"] - rain_stats["peak_dry"]["mean"]
    out["rain"] = rain_stats

    # ---- tab_eda_corr: correlation among lag/upstream features + VIF
    corr = df[COLLINEAR].corr()
    vif = vif_table(df[COLLINEAR])
    out["corr"] = {a: {b: float(corr.loc[a, b]) for b in COLLINEAR} for a in COLLINEAR}
    out["vif_collinear_block"] = vif
    flag = cfg["evaluation"]["vif_flag"]
    rows = [[c] + [float(corr.loc[c, d]) for d in COLLINEAR]
            + [f"{vif[c]:.1f}" + (r"$^{\ast}$" if vif[c] >= flag else "")] for c in COLLINEAR]
    write_table("tab_eda_corr", ["Feature"] + [rf"\texttt{{{c}}}".replace("_", r"\_") for c in COLLINEAR] + ["VIF"],
                rows, raw_cols={len(COLLINEAR) + 1},
                caption_note=f"Pearson correlation among lag/upstream features; * = VIF >= {flag}")

    # ---- tab_data_summary
    key = [TARGET, "tt_now", "tt_lag60", "tt_lastweek", "up_tt_now", "flow_now", "up_flow_now", "rain_1h"]
    units = {TARGET: "min", "tt_now": "min", "tt_lag60": "min", "tt_lastweek": "min", "up_tt_now": "min",
             "flow_now": "veh/15 min", "up_flow_now": "veh/15 min", "rain_1h": "mm"}
    rows, summ = [], {}
    for c in key:
        s = df[c]
        st = {"count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
              "median": float(s.median()), "p95": float(s.quantile(0.95)), "max": float(s.max())}
        summ[c] = st
        rows.append([f"{c} ({units[c]})", st["count"], st["mean"], st["std"], st["min"], st["median"],
                     st["p95"], st["max"]])
    out["summary"] = summ
    out["shares"] = {"workday_rows": float(df.is_workday.mean()), "peak_rows": float(df.is_peak.mean()),
                     "rain_rows": float((df.rain_1h > 0).mean()),
                     "onset_rows": float(((df[TARGET] - df.tt_now) >= cfg["evaluation"]["onset_delta_min"]).mean())}
    write_table("tab_data_summary", ["Variable", "Count", "Mean", "Std", "Min", "Median", "P95", "Max"], rows)

    write_json(RESULTS / "eda.json", out)
    print("top-8 |r|:", [(d["feature"], round(d["abs_pearson"], 3)) for d in rank[:8]])
    print("VIF (lag/upstream block):", {k: round(v, 1) for k, v in vif.items()})
    print("rain effect (min): off-peak %.2f, peak %.2f" % (rain_stats["rain_effect_offpeak_min"],
                                                          rain_stats["rain_effect_peak_min"]))
    print("peak window check:", out["peak_window_check"])
    plt.close("all")


if __name__ == "__main__":
    main()
