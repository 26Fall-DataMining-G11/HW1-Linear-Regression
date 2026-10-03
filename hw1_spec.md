# HW1 Spec — Next-Hour Freeway Travel Time into Hsinchu Science Park

Implementation spec for NYCU Data Mining HW1 (linear regression, ACL-style paper).
Items marked **[VERIFY]** must be checked against real data before building on them.
Items marked **[TEAM]** are decisions the team must make and justify in the paper.

---

## 1. Problem

- **Application:** Forecast southbound morning congestion on National Freeway 1 (國道1號) approaching the Hsinchu Science Park exits (竹北 → 新竹 interchange area).
- **User:** Commuters driving south to Hsinchu Science Park.
- **Decision supported:** "Leave now, or leave 30–60 minutes later?"
- **Research question:** Using only information available at prediction time, can a linear model with upstream-speed and calendar features forecast the segment travel time 60 minutes ahead accurately enough (peak MAE below the decision threshold) to change a commuter's departure decision?
- **Hard constraints (from assignment):**
  - Final predictor must be a linear model (OLS / Ridge / Lasso / Elastic Net / polynomial or interaction expansion). Nonlinear models only as baselines.
  - Target is continuous with a unit.
  - All methods share the same split, folds, and metrics.
  - Test set is used exactly once, after all design decisions are fixed.
  - No feature may use information after prediction time `t`.

---

## 2. Data Sources

| ID | Dataset | Source | Use |
|---|---|---|---|
| D1 | **M04A** inter-gantry median travel time by vehicle type, 5-min | Freeway Bureau traffic DB (tisvcloud) `https://tisvcloud.freeway.gov.tw/history/TDCS/M04A/` | target + main features |
| D2 | M03A gantry traffic volume by vehicle type, 5-min (optional) | same DB, `/history/TDCS/M03A/` | volume features if M04A `Traffic` is insufficient |
| D3 | Hourly rainfall, Hsinchu station | CWA CODiS (氣候觀測資料查詢服務) | weather features |
| D4 | Government office calendar (政府行政機關辦公日曆表), 2025 + 2026 | data.gov.tw | workday / holiday / make-up workday |
| D5 | Freeway incident records (optional) | Freeway Bureau | error analysis only, **never a feature** |

### 2.1 M04A format

- Path pattern (hourly folders): `history/TDCS/M04A/YYYYMMDD/hh/TDCS_M04A_YYYYMMDD_hhmmss.csv`
- Older dates may also be available as daily archives: `history/TDCS/M04A/M04A_YYYYMMDD.tar.gz` **[VERIFY which form exists per date; implement both]**
- CSV has **no header**. Columns: `TimeInterval, GantryFrom, GantryTo, VehicleType, TravelTime, Traffic`
- `VehicleType`: 31 car, 32 light truck, 41 bus, 42 heavy truck, 5 trailer. **Use 31 only.**
- `TravelTime` unit: seconds **[VERIFY]**
- `TimeInterval` semantics: start vs end of the 5-min bin **[VERIFY by inspecting values]**
- Gantry ID format: `01F0928S` = Freeway 01, F (at-grade; H = elevated), mileage 092.8 km, S = southbound.

### 2.2 Segments

| Role | GantryFrom → GantryTo | Notes |
|---|---|---|
| **Target segment** | `01F0880S → 01F0928S` (88.0K → 92.8K, crosses 竹北 interchange, ~4.8 km) | `01F0928S` confirmed (竹北–新竹). `01F0880S` inferred from naming (`01F0880N` confirmed) **[VERIFY]** |
| **Upstream segment** | `<prev gantry>S → 01F0880S` | Pick the nearest upstream southbound gantry from the actual `GantryFrom` list (likely 07xx/08xx) **[VERIFY]** |
| Downstream segment (optional) | `01F0928S → <next gantry>S` | For EDA only |

**First task of the pipeline:** download one week, list all distinct `(GantryFrom, GantryTo)` pairs whose IDs start with `01F` and end with `S` between mileage 0700 and 1000, and print them. Fix the segment IDs in `config.yaml` from that list.

### 2.3 Period and split

- Raw period: **2025-01-01 to 2026-08-31** (≈ 20 months).
- **Training:** 2025-01-01 to 2026-05-31.
- **Test (held out, used once):** 2026-06-01 to 2026-08-31.
- Rows kept: prediction target time `T = t + 60 min` in **[06:00, 21:00)** local time (Asia/Taipei). Justification for the paper: late-night traffic is free-flow, would inflate accuracy, and is outside the user's decision window.
- Raw data for all hours is still used to compute lag features.

### 2.4 Manual steps (agent cannot do)

- D3 (CODiS): download hourly rainfall CSVs for the Hsinchu station covering the full period; place in `data/raw/weather/`.
- D4: download 2025 and 2026 calendar CSVs; place in `data/raw/calendar/`.
- D5 (optional): place incident files in `data/raw/incidents/`.

---

## 3. Target Definition

- **Grid:** prediction times `t` every 15 minutes, `t ∈ [05:00, 20:00)` daily, so `T = t + 60 ∈ [06:00, 21:00)`.
- **One row = one prediction time `t`.**
- **y (minutes):** traffic-weighted mean of 5-min `TravelTime` (vehicle type 31) on the target segment over the window **[T, T+15 min)**, converted to minutes:

  `y = Σ(TravelTime_i × Traffic_i) / Σ(Traffic_i) / 60`, over the three 5-min bins in the window.
- Expected range: ≈ 2.5 min (free flow ~100 km/h) to ≈ 20 min (heavy congestion). Report actual min / median / p95 / max in the paper.
- Invalid 5-min records: `TravelTime ≤ 0`, `Traffic ≤ 0`, or missing → excluded from the weighted mean. If all three bins are invalid → `y = NaN` → drop row (count and report dropped rows).

### 3.1 Prediction-time rule (leakage guard)

- A 5-min record may be used as a feature at time `t` only if its bin **ends at or before `t`**.
- Note for the paper: M04A travel time is recorded when vehicles reach the downstream gantry, so the latest value already describes conditions a few minutes in the past.
- Weather: only the hourly rainfall for the hour **ending at or before `t`**.
- Calendar features may describe `T` (calendar is known in advance).

---

## 4. Features

All "window" features below use the same traffic-weighted aggregation as `y`, over the stated window, for vehicle type 31.

### 4.1 Base (raw)

| Feature | Definition |
|---|---|
| `tt_now` | target-segment travel time over `[t−15, t)`, minutes |
| `flow_now` | target-segment `Traffic` summed over `[t−15, t)`, vehicles / 15 min |

### 4.2 Temporal lags — tests **H1** (congestion persists; lags are the strongest signal)

| Feature | Definition |
|---|---|
| `tt_lag15` | travel time over `[t−30, t−15)` |
| `tt_lag60` | travel time over `[t−75, t−60)` |
| `tt_trend` | `tt_now − tt_lag15` |
| `tt_lastweek` | target-segment travel time over `[T−7d, T−7d+15)` (same target window one week earlier) |

### 4.3 Upstream spatial — tests **H2** (congestion shockwaves propagate upstream; upstream state adds information beyond own lags)

| Feature | Definition |
|---|---|
| `up_tt_now` | upstream-segment travel time over `[t−15, t)` |
| `up_flow_now` | upstream-segment `Traffic` over `[t−15, t)` |

### 4.4 Calendar — tests **H3** (morning peak exists only on workdays; long holidays change the pattern)

| Feature | Definition |
|---|---|
| `dow_Mon` … `dow_Sat` | one-hot day of week of `T`, Sunday as reference |
| `is_workday` | 1 if `T`'s date is a working day per D4 (includes make-up Saturdays; national holidays = 0) |
| `pre_long_holiday` | 1 if `T`'s date is the last working day before a non-working stretch of ≥ 3 days |
| `post_long_holiday` | 1 if `T`'s date is the first working day after a non-working stretch of ≥ 3 days |
| `school_break` | 1 if `T`'s date falls in winter/summer school break **[TEAM: define dates, cite source]** |
| `sin_hour`, `cos_hour` | `sin/cos(2π · h / 24)`, `h` = fractional hour of `T` |

### 4.5 Weather and interactions — tests **H4** (rain lengthens travel time mainly during the peak)

| Feature | Definition |
|---|---|
| `is_peak` | 1 if `is_workday == 1` and `T ∈ [07:00, 09:30)` **[TEAM: confirm window from EDA]** |
| `rain_1h` | rainfall (mm) in the last full hour ending ≤ `t` |
| `rain_x_peak` | `rain_1h × is_peak` |
| `workday_x_peak` | `is_workday × is_peak` (keep only if EDA supports it) |

Rule: every interaction term must be motivated by an EDA figure or domain argument in §4 of the paper.

### 4.6 Missing feature values

- Forward-fill a window feature from the previous 15-min value at most 2 steps (30 min); otherwise leave NaN.
- Drop rows with any remaining NaN in features of the full set (Set C) so that **all feature sets and all models use identical rows**. Report the count.

### 4.7 Feature sets for ablation (same model, same folds, same tuning procedure)

| Set | Contents |
|---|---|
| **A · Raw** | `tt_now`, `flow_now`, `hour` (integer 0–23 of `T`), `weekday` (integer 0–6 of `T`), `rain_1h` |
| **B · Top-k** | top `k = 5` features from the Set C pool by \|Pearson r\| with `y`, selected **inside each training fold**; also report k ∈ {3, 8} as sensitivity |
| **C · Ours** | all features in §4.1–4.5 |

Optional: leave-one-group-out from Set C (drop lags / upstream / calendar / weather+interactions one at a time).

---

## 5. Models

### 5.1 Main model

- **Ridge regression**, features standardized inside a scikit-learn `Pipeline` (`StandardScaler` → `Ridge`).
- Justification (to be confirmed by EDA): `tt_now`, lags, `tt_lastweek`, and `up_tt_now` are strongly collinear (report VIF); Ridge keeps all and shrinks unstable coefficients.
- Hyperparameter: `alpha ∈ logspace(-3, 3, 13)`, selected by CV MAE on training only. Output the full tuning curve (alpha vs CV MAE mean ± std).
- Optional comparison (same folds): Lasso / Elastic Net (`l1_ratio ∈ {0.1, 0.5, 0.9}`), and `log(y)` target variant (predictions back-transformed to minutes before scoring). These are design alternatives; the paper must state which was rejected and why.

### 5.2 Baselines (same rows, folds, metrics)

| Level | Baseline | Definition |
|---|---|---|
| Trivial | Training mean | mean of `y` in the training fold |
| Trivial | Persistence | `ŷ = tt_now` |
| Trivial | Historical profile | mean `y` in training fold grouped by (`is_workday`, 15-min slot of `T`) |
| Simple | Single-feature OLS | OLS on `tt_now` |
| Strong | LightGBM on Set A | small grid on the same folds: `num_leaves ∈ {15, 31, 63}`, `learning_rate ∈ {0.03, 0.1}`, `n_estimators` via early stopping inside each training fold (last 10% of fold as early-stopping set), `random_state` fixed |

---

## 6. Validation Protocol

- **Outer split:** fixed by date (§2.3). Test rows are not loaded by any script except `06_final_test.py`.
- **Inner CV:** 5 time-ordered folds over training **dates** (expanding window, like `TimeSeriesSplit`), with a **gap of 1 full day** between each training fold's last date and its validation fold's first date. Implement with date-based indices so the gap is exact; save fold indices to `results/folds.json` and reuse them for every method.
  - Reason for gap: `y` looks 60 min ahead and `tt_lastweek` reaches back 7 days; adjacent rows share information.
- **Tuning:** inside the CV folds only. Final model refit on all training data with the selected hyperparameters, then evaluated once on test.
- **Seeds:** fix all random seeds (`numpy`, LightGBM, bootstrap).

---

## 7. Metrics and Threshold

| Metric | Unit | Role |
|---|---|---|
| MAE | minutes | primary; typical error a commuter feels |
| RMSE | minutes | penalizes large misses (sudden jams) |
| R² (test) | — | variance explained, context only |
| **Peak MAE** | minutes | MAE on rows with `is_peak == 1`; the decision-relevant slice |

- **Decision threshold:** peak MAE ≤ **1.5 min** **[TEAM: justify, e.g., delaying departure by 15 min is not worth it if the predicted saving is under 1.5 min]**. Every result in §7 of the paper is compared against it.
- Report CV metrics as mean ± std over 5 folds.

### 7.1 Slices (test set, final model + key baselines)

- peak vs off-peak
- workday vs non-workday
- rain (`rain_1h > 0`) vs dry
- **congestion onset:** `y − tt_now ≥ 2 min` (travel time rising sharply in the next hour); expected to be where H2 features help most

---

## 8. Statistical Tests and Analysis

### 8.1 Model comparison

- Paired t-test over the 5 fold MAEs: Ridge (Set C) vs LightGBM, and Ridge (Set C) vs persistence.
- Day-block bootstrap on the test set (resample whole days, B = 2000): 95% CI of `MAE_baseline − MAE_ours` for the same two comparisons.
- Ablation: Set C vs A and C vs B with the same paired test and fold std.

### 8.2 Coefficients

- Fit OLS (`statsmodels`) on standardized Set C features over all training rows.
- Standard errors: HAC / Newey–West, `maxlags = 8` (2 hours of 15-min rows); report sensitivity with 4 and 16.
- Output a table and a forest plot: feature, standardized β (minutes per 1 SD), 95% CI, p-value, VIF. Mark VIF ≥ 10.
- Report the max absolute difference between Ridge (selected alpha) and OLS coefficients, to justify interpreting OLS estimates.
- For each hypothesis H1–H4: state the verdict (supported / not supported / mixed) with the specific coefficient or ablation number. Distinguish statistical from practical significance (effect in minutes).

### 8.3 Error analysis

- Top 5 largest absolute test errors from the final model: timestamp, actual, predicted, error (actual − predicted, minutes), plus `tt_now`, `up_tt_now`, `rain_1h`, calendar flags.
- If D5 is available, join incidents within ±60 min on or near the segment.
- Residuals over time and vs `tt_now`; describe the shared pattern and the first fix to try.

---

## 9. EDA (training rows only)

| Output | Content |
|---|---|
| `fig_eda_profile` | median `y` (with IQR band) by 15-min slot of `T`, workday vs non-workday |
| `fig_eda_scatter` | `y` vs `tt_now`, and `y` vs `up_tt_now` (colored by `is_peak`) |
| `fig_eda_rank` | \|Pearson r\| and Spearman ρ of every Set C feature with `y`, sorted |
| `fig_eda_rain` | boxplot of `y` by rain (yes/no) × peak (yes/no) |
| `tab_eda_corr` / VIF | correlation matrix among lag/upstream features + VIF |
| `tab_data_summary` | summary statistics (count, mean, std, min, median, p95, max) of `y` and key features |

Each figure must have labeled axes with units. EDA outputs feed the hypotheses H1–H4 and the Set B ranking.

---

## 10. Repository Layout

```
HW1/
├── README.md               # project summary, data download steps (incl. manual steps), how to reproduce
├── config.yaml             # segment gantry IDs, periods, grid, thresholds, seeds
├── requirements.txt        # python 3.11; pandas, numpy, pyarrow, scikit-learn, statsmodels, lightgbm, matplotlib, requests, pyyaml
├── Makefile                # make data / eda / experiments / test / analysis / paper
├── data/
│   ├── raw/                # gitignored; m04a/, weather/, calendar/, incidents/
│   └── processed/          # filtered parquet (small, committed)
├── src/
│   ├── 00_list_gantries.py # one-week download, print candidate S-bound gantry pairs 0700–1000
│   ├── 01_download.py      # download M04A (daily tarball or hourly CSVs), filter immediately, save monthly parquet
│   ├── 02_build_dataset.py # 15-min aggregation, target, features, calendar/weather joins → data/processed/dataset.parquet
│   ├── 03_eda.py           # training rows only → figures + tables
│   ├── 04_cv.py            # folds, baselines, Ridge tuning, ablation, Lasso/EN/log-y variants → results/cv_*.json
│   ├── 05_analysis_train.py# OLS + HAC coefficients, VIF, forest plot (training only)
│   ├── 06_final_test.py    # refit selected models on all training data, evaluate ONCE on test, slices, bootstrap, error cases
│   └── utils/              # aggregation, features, metrics, plotting, latex table writers
├── results/                # json with every number reported in the paper (single source of truth)
├── paper/
│   ├── figures/            # PDF figures written by scripts
│   ├── tables/             # LaTeX tables (booktabs) written by scripts
│   └── sections/           # one .tex per paper section
└── tests/                  # leakage and alignment tests
```

### 10.1 Download requirements (`01_download.py`)

- Filter while downloading: keep only `VehicleType == 31` and rows whose `(GantryFrom, GantryTo)` is in the configured segment list. Never store the full national file.
- Retry with backoff; sleep between requests; resume (skip months already saved).
- Log missing files/hours to `results/download_gaps.csv` (report in paper §3).

### 10.2 Output conventions

- Every number in the paper comes from `results/*.json` or a generated `paper/tables/*.tex`. No hand-typed results.
- Figures: PDF, readable at single-column ACL width, axes labeled with units, consistent colors across figures.
- Best value in each results-table column bolded by the table writer.

---

## 11. Tests and Acceptance Checks

- **Leakage test:** for random rows, assert every source 5-min bin used by any feature ends ≤ `t`; assert weather hour ends ≤ `t`.
- **Alignment test:** recompute `y` and `tt_now` for 20 random rows directly from raw parquet and compare.
- **Persistence sanity:** persistence MAE must be computed on exactly the same rows as Ridge.
- **Test-set isolation:** `03`–`05` raise an error if any row with date ≥ 2026-06-01 is present.
- **Fold reuse:** all methods read `results/folds.json`; assert identical indices.
- **Determinism:** two runs of `04_cv.py` produce identical JSON.

---

## 12. Paper Mapping

| Paper section | Produced by |
|---|---|
| §3 Task & Data | §2–3 of this spec, `tab_data_summary`, `download_gaps.csv`, dropped-row counts |
| §4 Exploratory Feature Analysis | §9 outputs, H1–H4 |
| §5 Method | pipeline figure (raw → preprocessing → features → Ridge → prediction), §4–5, alpha tuning curve |
| §6 Experimental Setup | §6–7, threshold justification |
| §7 Results | main table (CV MAE/RMSE ± std, test MAE/R², peak MAE), paired tests, bootstrap CI, ablation figure + slice table |
| §8 Analysis | coefficient forest plot/table, H1–H4 verdicts, top-5 error cases, residual plot |

---

## 13. Known Risks

- **Gantry IDs / segment geometry:** resolve in `00_list_gantries.py` before anything else.
- **Data gaps:** report coverage; do not silently interpolate long gaps.
- **Distribution shift in test (summer):** school break and possible construction/lane closures; check Freeway Bureau construction notices and discuss as regime change in §8 if errors cluster.
- **Strong persistence baseline:** at a 60-min horizon persistence may be competitive; the value of Set C should be shown via the onset slice and ablation, not only the overall MAE.
- **Target saturation:** extreme jams are rare; RMSE will be dominated by a few days — show them in error analysis rather than removing them.
