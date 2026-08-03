# CIC-UNSW-NB15 vs. UNSW-NB15: a matched-condition comparison

Code, results and figures for the paper *Enhancing Intrusion Detection Performance: A Comparative Study of Machine Learning on CIC-UNSW-NB15 vs. UNSW-NB15*.

CIC-UNSW-NB15 re-extracts features from the same raw traffic capture that produced UNSW-NB15, using CICFlowMeter in place of Argus and Bro-IDS. Comparing the two is not as simple as comparing accuracy figures, because they differ simultaneously in feature representation, class prior and volume. This repository implements a protocol that separates those factors.

Repository: <https://github.com/moh1984/cic-unsw-nb15-matched-comparison>

## Results not printed in the paper

Three sets of figures were moved out of the manuscript to meet the journal page limit and are held here in full:

| Reported in the paper as | Full figures |
|---|---|
| "descriptive bootstrap intervals are given in the accompanying data release" | `results/main/dataset_significance.csv` (`ci_low`, `ci_high`) |
| E2 sensitivity: "the gap changes by less than one percentage point on every metric" | `results/e2_sensitivity/e2_sensitivity.csv` |
| Ablation: "accuracy and MCC unchanged to four decimal places" | `results/ablation/` |

## Headline result

Five classifiers evaluated under three repetitions of stratified five-fold cross-validation. XGBoost, the best model on both datasets:

| Condition | Accuracy gap | Attack-recall gap | Balanced-accuracy gap |
|---|---|---|---|
| As published | +5.93 pp | +7.85 pp | +6.21 pp |
| E1 — class prior and size matched | **+3.49 pp** | **+15.45 pp** | +7.60 pp |
| E2 — restricted to 12 common features | +5.42 pp | **+20.96 pp** | **+10.76 pp** |

Equalising the class prior *narrows* the accuracy gap and *widens* every gap in the metrics that are robust to class imbalance. Accuracy and detection capability move in opposite directions, so ranking intrusion-detection datasets by accuracy can misstate both the size and the direction of a difference.

Two further findings:

- The apparent threefold advantage of CIC-UNSW-NB15 in false alarm rate (2.32% against 6.89%) is entirely an artefact of the class prior. Under matched priors it becomes 2.34% against 2.08%.
- Removing the attributes that carry 90.4% (CIC-UNSW-NB15) and roughly 76% (UNSW-NB15) of gain-based feature importance changes accuracy and MCC by less than one standard deviation. Gain-based importance in these datasets does not indicate necessity.

## Repository layout

```
src/
  ids_evaluation.py               main pipeline: preprocessing, CV, metrics, McNemar, E1, E2, SHAP
  significance_tests.py           between-dataset tests: Mann-Whitney, Cliff's delta, bootstrap CI
  run_e2_sensitivity.py           E2 sensitivity: reruns E2 without the TCP-window pairs and with the
                                  direction convention reversed, then reports the change in the gap
  aggregate_independent_runs.py   confidence intervals across disjoint blocks
  make_figures.py                 Figures 2-8 from the results CSVs
  make_methodology_figure.py      Figure 1
  build_paired_flows.py           experiment E3; requires releases that retain the five-tuple
config/
  config.json                  identifier columns to drop, common-feature map for E2
  config_ablation.json         same, with the highest-attribution features also dropped
  config_E2_no_window.json     E2 sensitivity: drops the two TCP-window pairs
  config_E2_swapped.json       E2 sensitivity: reverses the forward/backward convention
notebooks/
  IDS_Colab_Runbook.ipynb      end-to-end run on Google Colab
results/
  main/                        every number in the paper
  ablation/                    the ablation run
  e2_sensitivity/              E2 sensitivity. no_window/ drops the two TCP-window pairs;
                               swapped/ is a null test, see the note below
figures/                       Figures 1-8 at 300 dpi
docs/reproduce.md              step-by-step reproduction
```

## Data

The datasets are **not redistributed here**. Download them and place the CSVs in `data/`:

- **UNSW-NB15** — https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CIC-UNSW-NB15** — https://www.unb.ca/cic/datasets/

The experiments reported in the paper use the concatenation of the distributed UNSW-NB15 training and testing partitions (257,673 records), because the pipeline performs its own repeated stratified splitting. See `docs/reproduce.md` for how this affects comparability with prior work.

## Quick start

```bash
pip install -r requirements.txt

python src/ids_evaluation.py \
    --cic  data/CIC_NB15.csv \
    --unsw data/unsw_all.csv \
    --label-cic Label --label-unsw label \
    --config config/config.json \
    --outdir results/main \
    --sample 100000 --models RF,XGB,AdaBoost,DT,KNN \
    --n-splits 5 --n-repeats 3

python src/significance_tests.py --results results/main --model XGB
python src/make_figures.py --results results/main --outdir figures
```

Roughly 60–90 minutes on two CPU cores. K-Nearest Neighbours dominates the runtime; drop it from `--models` for a faster run.

## Two things to check before trusting any output

**1. Identifier columns.** `LEAKAGE_COLUMNS` at the top of `src/ids_evaluation.py` lists the attributes dropped before training. Any IP address, port, flow ID or timestamp left in the feature matrix lets a model memorise which hosts generated the attacks instead of learning traffic behaviour, which is the usual cause of implausible scores above 99%. The releases used here have already had these removed; check your own copies and extend the list through `config.json` if needed.

**2. The E2 feature map.** `common_feature_map` in `config.json` aligns twelve attributes across the two feature spaces. CICFlowMeter column names vary between tool versions, so verify the mapping against your actual headers. The script prints how many pairs it matched.

## Notes on the analysis

`ids_evaluation.py` fits the Min–Max scaler inside a scikit-learn `Pipeline`, so it is refitted on the training fold of every split rather than on the full dataset.

Deduplication runs *after* sampling, so the resulting class balance depends on the sample size. With `--sample 100000` the malicious proportion is 23.84% for CIC-UNSW-NB15 and 48.07% for UNSW-NB15. E1 targets the observed proportion and computes the matched size as the largest value at which *both* datasets can attain it, which for these data is 44,823 records each.

McNemar's test compares classifiers on identical held-out instances. It cannot compare the two datasets, whose records differ, so `significance_tests.py` uses Mann–Whitney U, Cliff's delta and a bootstrap interval instead. The fifteen fold scores are not mutually independent — three repetitions over the same records — so the p-values are descriptive; the assumption-free result is whether the two fold-score distributions overlap at all. They do not, in 13 of the 15 comparisons reported in the paper; the two exceptions are the false alarm rate under E1 (Cliff's delta 0.73) and under E2 (−0.98). `significance_tests.py` also reports macro F1, giving 16 of 18 non-overlapping comparisons in its output.

To obtain intervals that do not rest on that assumption, run the pipeline once per disjoint block and combine the results:

```bash
for K in 0 1 2 3 4; do
  python src/ids_evaluation.py ... --disjoint-blocks 5 --block $K --n-repeats 1
done
python src/aggregate_independent_runs.py --runs results/block{0,1,2,3,4} --model XGB
```

Blocks share no records, so a t interval across them is valid. Expect it to be wider than the bootstrap interval; that is the correct behaviour.

`results/e2_sensitivity/swapped/` is retained for documentation only and **must not be read as evidence**. Reversing the forward/backward convention is a null test by construction: each classifier trains on the twelve columns of its own dataset, so exchanging which column of one dataset is deemed to correspond to which column of the other leaves both feature sets identical. The residual differences in that directory (fourth decimal place) come from column ordering affecting tie-breaking, not from the convention. Only `no_window/` supports the sensitivity result reported in the paper.

Experiment E3 would compare the same physical flows in both representations by joining on the five-tuple. It is not reproducible from the public releases, which have had those columns removed. `build_paired_flows.py` is included for use with raw captures.

## Citation

```bibtex
@article{alkhazaleh2026cicunsw,
  title   = {Enhancing Intrusion Detection Performance: A Comparative Study of
             Machine Learning on CIC-UNSW-NB15 vs. UNSW-NB15},
  author  = {Alkhazaleh, Mohammad and Baklizi, Mahmoud and Mjlae, Salameh A. and
             Al-Zghoul, Musab},
  year    = {2026},
  note    = {Under review}
}
```

## Licence

MIT — see `LICENSE`. The datasets carry their own terms; consult the sources linked above.
