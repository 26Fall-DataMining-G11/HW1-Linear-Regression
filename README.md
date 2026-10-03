# HW1 — Next-hour freeway travel time into Hsinchu Science Park

NYCU Data Mining HW1 (linear regression, ACL-style paper). Forecast the car travel time on
National Freeway 1 southbound `01F0880S → 01F0928S` (88.0K → 92.8K, across the 竹北 interchange)
**60 minutes ahead**, every 15 minutes, with a Ridge regression. Full specification:
[hw1_spec.md](hw1_spec.md).

## Status

| Step | State |
|---|---|
| Data download (M04A 2024-12-25 … 2026-08-31, calendar, rainfall) | done — 0 missing files |
| Dataset (30,960 training rows / 5,520 test rows, no dropped rows; 143 Set C features) | done |
| EDA, CV experiments, coefficient analysis (training data only) | done |
| Tests (leakage, alignment, isolation, folds, determinism) | 9 passing (incl. determinism re-run with the final feature set) |
| **Final test evaluation** (`make final-test`) | **done once on 2026-10-02** (`results/final_test.lock`) — Ridge Set C: test MAE 0.720, peak MAE 0.748 min (threshold 1.5). Do not re-run after changing any design decision. |

## Reproduce

```bash
make setup         # venv on the system Python (developed on Python 3.14) + requirements.txt
make gantries      # 00: list candidate gantry pairs (segment IDs are already fixed in config.yaml)
make data          # 01 + 01b + 02: download (resumable), calendar/rainfall, build dataset
make eda           # 03: figures + tables, training rows only
make experiments   # 04: folds, baselines, Ridge tuning, ablation, variants
make analysis      # 05: OLS + HAC coefficients, VIF, forest plot, H1–H4 evidence
make test          # leakage / alignment / protocol tests   (make test-slow: determinism)
make dryrun        # rehearse 06 on the last CV fold — does NOT touch the test set
make final-test    # 06: ONE-TIME evaluation on 2026-06-01 … 2026-08-31
make paper         # 07: results/*.json → paper/tables/numbers.tex (+ latexmk if installed)
```

`data/processed/` (3 MB of filtered parquet) is meant to be committed, so everything from
`make eda` onward runs without re-downloading.

## Data

| ID | Source | How it is obtained |
|---|---|---|
| D1 M04A travel times | `https://tisvcloud.freeway.gov.tw/history/TDCS/M04A/` | `src/01_download.py` — daily `M04A_YYYYMMDD.tar.gz` when it exists (all dates up to 2026-08-28), hourly-folder CSVs otherwise (2026-08-29 …). Filtered in memory to vehicle type 31 and the three configured segments; saved per month in `data/processed/m04a/`. Missing files → `results/download_gaps.csv`. |
| D3 hourly rainfall | CWA CODiS, station 467571 新竹 | `src/01b_fetch_aux.py` → `data/raw/weather/codis_467571_YYYYMM.csv` (`hour_end,rain_mm`; trace = 0) |
| D4 office calendar | data.gov.tw dataset 14718 (DGPA), 2025 (revised 2025-10-20 edition) + 2026 | `src/01b_fetch_aux.py` → `data/raw/calendar/calendar_YYYY.csv` (official CSV, unchanged) |
| D5 incidents (optional) | Freeway Bureau | manual: put files in `data/raw/incidents/`; used only for error analysis, never as a feature (no automatic join is implemented) |

The spec lists D3/D4 as manual steps; both turned out to be scriptable. To supply them by hand
instead, place files with the same names/columns in `data/raw/weather/` and `data/raw/calendar/`
— existing files are never re-downloaded.

### [VERIFY] items resolved against real data

- **File form:** daily tarballs exist for every date up to 2026-08-28; later dates exist only as hourly folders. Both are implemented.
- **`TravelTime` unit:** seconds (free flow on the 4.8 km target ≈ 165 s ≈ 105 km/h).
- **`TimeInterval`:** start of the 5-min bin (files run 00:00 … 23:55). A record is usable at `t` only if `TimeInterval + 5 min ≤ t`.
- **Segments** (`results/gantry_pairs.csv`): target `01F0880S → 01F0928S` exists; nearest upstream pair is `01F0750S → 01F0880S` (13.0 km); downstream `01F0928S → 01F0950S`.

## [TEAM] decisions (frozen at their defaults before the final test; justify them in the paper)

1. **Peak window** `07:00–09:30`: check against `paper/figures/fig_eda_profile.pdf` (workday median peaks at 07:45; it exceeds the non-workday median by ≥ 0.5 min from 06:45 to 10:00).
2. **School-break dates**: defaults follow the MOE K-12 calendar (AY113 winter 1/21–2/10, summer 7/1–8/31; AY114 winter 1/24–2/22, summer 7/1–8/30). Verify with the Hsinchu education bureau calendar and cite it.
3. **Decision threshold** peak MAE ≤ 1.5 min: write the justification.
4. **`log(y)` variant**: lower overall MAE than plain Ridge (CV 0.75 vs 0.80, test 0.66 vs 0.72) with the same peak MAE. Plain Ridge on Set C was pre-declared as the final model (`results/selected.json`); the `log(y)` variant is reported next to it in every test table.

The test set has now been used, so these can no longer be changed without invalidating the
held-out evaluation.

## Deviations from the spec (deliberate)

- **Python 3.14** (the machine's existing Anaconda) instead of 3.11.
- **Two dataset files** (`dataset_train.parquet`, `dataset_test.parquet`) instead of one `dataset.parquet`, so scripts 03–05 physically cannot load test rows. Only `06_final_test.py` may call `load_test()`.
- **7 warm-up days** (2024-12-25 …) are downloaded so `tt_lastweek` exists from the first training day.
- **Workday × slot features added** (not in the spec): 120 indicators, one per (workday, 15-min slot of `T`) cell, as an interaction expansion motivated by the EDA profile. They cut CV MAE from 0.90 to 0.80 and peak MAE from 1.11 to 0.80. The pre-change CV results are kept in `results/_before_slot/`.
- **`workday_x_peak` excluded**: `is_peak` already requires a workday, so the product equals `is_peak`.
- **OLS / VIF use a reduced Set C** (without `tt_lag15`, `is_workday`, `is_peak`, `sin_hour`, `cos_hour` and one reference slot cell): these are exact linear combinations of other columns. Ridge still uses all of Set C; it is projected onto the same parameterization for the Ridge-vs-OLS comparison.
- **Lasso / Elastic Net grid** starts at 1e-3 instead of 1e-4 (smaller values take minutes per fit with 143 features and were never selected).
- **`log(y)` Ridge is also evaluated on the test set** as a reported alternative; the final model was fixed beforehand.
- **LightGBM** predicts with the early-stopped model (fit on the first 90 % of the training fold); it is not refit on the full fold.
- **Extra script** `07_paper_numbers.py` turns `results/*.json` into `\res{key}` LaTeX macros so no number in the paper is typed by hand.

## Layout

```
config.yaml            segments, periods, grid, thresholds, seeds
src/00 … 07_*.py       pipeline steps (see Makefile)
src/utils/             aggregation, features, data/folds, metrics, models, plotting, latex
data/processed/        filtered M04A parquet, calendar, weather, dataset_{train,test}.parquet
results/               JSON with every reported number; folds.json; download_gaps.csv
paper/                 main.tex, sections/, figures/ (PDF + PNG preview), tables/ (generated)
tests/                 leakage, alignment, isolation, fold-reuse, determinism
```

`paper/references.bib` holds the 10 cited references (details checked against publisher/index
pages; re-verify before submission). The paper draft (`paper/main.tex`) needs `acl.sty` from the ACL style files for the ACL layout
(a plain two-column fallback is used otherwise). Red `[TEAM: …]` marks are text the team must
write; `??` marks numbers that appear once `make final-test` and `make paper` have run.
