# Reproducing the results

Every number, table and figure in the paper is produced by the four scripts in `src/`. This document gives the exact sequence.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The reported results were produced with scikit-learn 1.6.1, XGBoost 3.3.0 and pandas 2.2.2 on two CPU cores with 12 GB of RAM. Tree-based models in scikit-learn are deterministic given a fixed seed, so the numbers below should reproduce exactly on the same library versions. Later versions may differ in the last decimal places if a default splitting rule changes.

## 1. Data preparation

Download both datasets (links in the README) into `data/`.

**CIC-UNSW-NB15.** One file with 447,915 records, 76 features and a multi-class `Label` column where 0 is benign and 1–9 are attack categories. The pipeline derives the binary label by mapping all non-zero values to the positive class. Some distributions split the features and labels into separate files; if so, concatenate them column-wise, checking first that the row counts match and that no shared identifier column implies a different row order.

**UNSW-NB15.** Concatenate the distributed training and testing partitions:

```python
import pandas as pd
a = pd.read_csv('data/UNSW_NB15_training-set.csv', low_memory=False)
b = pd.read_csv('data/UNSW_NB15_testing-set.csv', low_memory=False)
assert list(a.columns) == list(b.columns)
pd.concat([a, b], ignore_index=True).to_csv('data/unsw_all.csv', index=False)
# 175,341 + 82,332 = 257,673
```

**Why concatenate, and what it costs.** The pipeline performs its own repeated stratified splitting, so it needs a single pool. But the distributed partition of UNSW-NB15 is deliberately constructed to be difficult, and pooling it lets similar records fall on either side of a fold boundary. This raises accuracy relative to the prescribed split — Section 4.9 of the paper quantifies the effect using a published model that scores 95.08% under cross-validation and 91.31% under the prescribed hold-out on this dataset. The absolute figures here are therefore **not comparable** with prescribed-partition studies. They are internally consistent, because both datasets are treated identically, and the matched-condition analysis depends on differences rather than absolute values.

To evaluate on the prescribed split instead, pass `--nrows` with the training file alone, or modify `cross_validate` to accept a fixed partition.

## 2. Inspect before running

```python
import pandas as pd, re
SUSPECT = re.compile(r'ip|port|flow.?id|timestamp|stime|ltime|^id$|index', re.I)
for f in ['data/CIC_NB15.csv', 'data/unsw_all.csv']:
    df = pd.read_csv(f, nrows=200_000, low_memory=False)
    print(f, df.shape)
    print('  identifier-like:', [c for c in df.columns if SUSPECT.search(c)])
    print('  columns:', list(df.columns))
```

Add anything identifier-like to `extra_leakage_columns` in `config/config.json`. Then confirm that both sides of every pair in `common_feature_map` appear in the printed column lists; the pipeline prints how many pairs it matched and E2 is meaningless below about five.

## 3. Smoke test

```bash
python src/ids_evaluation.py \
    --cic data/CIC_NB15.csv --unsw data/unsw_all.csv \
    --label-cic Label --label-unsw label \
    --config config/config.json --outdir results/smoke \
    --sample 20000 --models RF,DT --n-repeats 1 --skip-shap
```

Check three things in the output:

- `dropped_identifier_columns` contains everything you listed
- `malicious_pct` is near 20–24 for CIC-UNSW-NB15 and 48–57 for UNSW-NB15, depending on sample size
- accuracy is around 0.97–0.98 and 0.89–0.92 respectively. A Decision Tree scoring above 0.99 on 20,000 records indicates a remaining identifier column, not success.

## 4. Main run

```bash
python src/ids_evaluation.py \
    --cic data/CIC_NB15.csv --unsw data/unsw_all.csv \
    --label-cic Label --label-unsw label \
    --config config/config.json --outdir results/main \
    --sample 100000 --models RF,XGB,AdaBoost,DT,KNN \
    --n-splits 5 --n-repeats 3
```

60–90 minutes. The computation runs three times — baseline, E1, E2 — so the total is roughly triple a single evaluation. KNN dominates: its inference cost grows with training-set size and dimensionality, and it accounts for most of the wall-clock time in the 67-dimensional CICFlowMeter space. Drop it from `--models` for a run of about 25 minutes.

`--sample` is a uniform random subsample and is safe for reported experiments. `--nrows` is a non-random head slice and is for smoke tests only.

Expected console output:

```
  target attack ratio=0.2384  matched n=44,823 per dataset
  CIC:  44,823 rows, 23.84% attack
  UNSW: 44,823 rows, 23.84% attack
```

## 5. Ablation run

```bash
python src/ids_evaluation.py \
    --cic data/CIC_NB15.csv --unsw data/unsw_all.csv \
    --label-cic Label --label-unsw label \
    --config config/config_ablation.json --outdir results/ablation \
    --sample 100000 --models RF,XGB,DT --n-splits 5 --n-repeats 3
```

The ablation config additionally drops `Bwd Packet Length Min` and `Fwd Seg Size Min` from CIC-UNSW-NB15 and `sttl`, `ct_state_ttl` and `service` from UNSW-NB15. Because the column names do not collide between the two files, one list serves both.

Expect no meaningful change. That is the finding.

Note that dropping columns makes more rows exact duplicates, so the deduplication count rises slightly (34,142 to 34,154 for UNSW-NB15) and the record set differs by a few rows from the main run. This is why E2 results are bit-identical between the two runs for CIC-UNSW-NB15 but differ in the third decimal place for UNSW-NB15.

## 5b. E2 sensitivity analysis (optional, ~25 min)

The twelve-feature alignment of Table 4 in the paper involves judgement, most obviously in the two TCP-window pairs and in the forward/backward direction convention. One command reruns E2 under both variants and reports how far the gap moves:

```bash
python src/run_e2_sensitivity.py \
    --cic data/CIC_NB15.csv --unsw data/unsw_all.csv \
    --baseline results/main --outdir results/e2_sensitivity \
    --sample 100000
```

A note on what to expect. In the full 67-feature model, `Bwd Init Win Bytes` and `FWD Init Win Bytes` rank 5th and 7th by gain on CIC-UNSW-NB15, whereas on UNSW-NB15 `swin` ranks 18th of 54 and `dwin` has gain of exactly zero — XGBoost never split on it. Dropping the window pairs should therefore cost CIC-UNSW-NB15 more than UNSW-NB15 and narrow the gap. If it does, that is a genuine result and should be reported: report the ten-pair figure as the primary E2 result and keep the twelve-pair figure alongside it.

## 6. Statistics and figures

```bash
python src/significance_tests.py --results results/main --model XGB
python src/make_figures.py --results results/main --outdir figures
python src/make_methodology_figure.py
```

## 7. Mapping outputs to the paper

| Paper | File |
|---|---|
| Table 2 (dataset characteristics) | `results/main/dataset_summary.csv` |
| Table 4 (E2 feature alignment) | `config/config.json`, `results/main/E2_feature_mapping.csv` |
| Tables 5 and 6 (per-dataset results) | `results/main/main_results_CIC.csv`, `main_results_UNSW.csv` |
| Table 7 (McNemar) | `results/main/mcnemar_CIC.csv`, `mcnemar_UNSW.csv` |
| Table 8 (matched conditions) | `results/main/matched_E1.csv`, `matched_E2.csv` |
| Table 9 (between-dataset significance) | `results/main/dataset_significance.csv` |
| Table 10 (MCC retained) | derived from `main_results_*` and `matched_E2.csv` |
| Table 11 (ablation) | `results/ablation/main_results_*.csv` |
| Table 12 (cost) | `results/main/cost_CIC.csv`, `cost_UNSW.csv` |
| Table 13 (prior work) | not generated; compiled from the literature |
| Figures 2–8 | `figures/`, numbered in order of appearance |
| Confusion matrices | `results/main/confusion_*.csv` |
| SHAP attribution | `results/main/shap_*.csv` |
| Per-fold raw scores | `results/main/folds_*.csv` |

The `folds_*.csv` files hold the fifteen individual fold scores behind every mean and standard deviation, and are the source for the error bars and for all of `significance_tests.py`.

## Known limitations of this reproduction

- **Hyper-parameters are fixed, not tuned.** `classifiers()` in `ids_evaluation.py` uses library defaults except for the values listed in Table 3 of the paper. No search was performed. Between-dataset comparisons are unaffected, since both datasets are treated identically, but the ranking of classifiers against one another is weaker than it would be after tuning.
- **Experiment E3 is not reproducible from public data.** `build_paired_flows.py` joins the two representations on the five-tuple, which both public releases have removed. It requires raw captures.
- **Fold scores are not independent.** Three repetitions over the same records. The confidence intervals in Table 12 are consequently narrower than intervals from fully independent samples, and the p-values are descriptive.
- **Results are for a 100,000-record subsample.** Standard deviations do not exceed 0.013 on any metric, so the comparisons hold, but absolute values on the full files may differ slightly.
