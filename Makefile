PY ?= .venv/bin/python

.PHONY: setup gantries data dataset eda experiments analysis dryrun final-test test test-slow paper all

setup:            ## venv on the system Python + requirements
	python3 -m venv --system-site-packages .venv
	$(PY) -m pip install -q -r requirements.txt

gantries:         ## spec §2.2 first task: list candidate gantry pairs
	$(PY) src/00_list_gantries.py

data:             ## download M04A (resumable) + calendar + rainfall, then build the dataset
	$(PY) src/01_download.py
	$(PY) src/01b_fetch_aux.py
	$(PY) src/02_build_dataset.py

dataset:
	$(PY) src/02_build_dataset.py

eda:
	$(PY) src/03_eda.py

experiments:      ## CV: baselines, Ridge tuning, ablation, variants (training data only)
	$(PY) src/04_cv.py

analysis:         ## OLS + HAC coefficients, VIF, forest plot (training data only)
	$(PY) src/05_analysis_train.py

dryrun:           ## rehearse the final evaluation on the last CV fold (does not touch the test set)
	$(PY) src/06_final_test.py --dry-run

final-test:       ## ONE-TIME evaluation on the held-out test set — only after all [TEAM] decisions are fixed
	$(PY) src/06_final_test.py

test:
	$(PY) -m pytest tests -q -m "not slow"

test-slow:        ## determinism: re-runs 04_cv.py and compares JSON
	$(PY) -m pytest tests -q -m slow

paper:            ## regenerate number macros, then compile if a LaTeX toolchain is installed
	$(PY) src/07_paper_numbers.py
	@if command -v latexmk >/dev/null; then cd paper && latexmk -pdf -interaction=nonstopmode main.tex; \
	else echo "latexmk not found: upload paper/ to Overleaf (main.tex) to compile"; fi

all: data eda experiments analysis test dryrun paper
