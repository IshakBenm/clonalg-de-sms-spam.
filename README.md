# CLONALG + DE-weighted ensemble for SMS spam detection

 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23335141.svg)](https://doi.org/10.5281/zenodo.23335141) 

Code and results for the article:

> **Hybrid Bio-Inspired Feature Selection Under Feature Budgets: A Clonal Selection Wrapper with DE-Weighted Ensembles for SMS Spam Detection.** Ishak Benmohammed and Hacene Belhadef. *Algorithms* (submitted), 2026. DOI: to be added after publication.

The method selects a fixed number K of word and character n-gram features with an adapted
CLONALG algorithm (clonal selection), then combines four classifiers (Naive Bayes, logistic
regression, calibrated linear SVM, random forest) with voting weights and a decision threshold
optimised by Differential Evolution on out-of-fold predictions.

## Repository contents

| File / folder | Content |
|---|---|
| `main_clonalg_de.py` | Proposed method and the three main experiments (`budget`, `salman`, `senol`) |
| `revision_experiments.py` | Re-run baselines and paired tests, deployable-model footprint, parameter sensitivity |
| `results/` | All result files reported in the article (CSV / JSON / figures) |
| `data/` | Place the two datasets here (not redistributed, see below) |

## Installation

Python 3.11 was used for the article.

```bash
git clone https://github.com/<your-username>/clonalg-de-sms-spam.git
cd clonalg-de-sms-spam
pip install -r requirements.txt
```

## Data

The datasets are public but are not redistributed here; download them into `data/`:

* **UCI SMS Spam Collection** (Almeida et al., 2011): <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
  → file `data/SMSSpamCollection`
* **Super SMS Dataset** (Salman et al., 2024): see the link given in their article
  → file `data/super_sms_dataset.csv` (columns `SMSes` and `Labels`)

## Reproducing the results

```bash
# quick test (a few minutes)
python main_clonalg_de.py --quick --data data/super_sms_dataset.csv --uci data/SMSSpamCollection

# main experiments
python main_clonalg_de.py --experiment budget --data data/super_sms_dataset.csv   # Tables 2, Fig. 2
python ablation_sweep.py --experiment sweep --data data/super_sms_dataset.csv --out results_final   # Tables 3-4, chi2 curve of Fig. 2
python main_clonalg_de.py --experiment salman --data data/super_sms_dataset.csv   # Table 7, Table 11
python main_clonalg_de.py --experiment senol  --uci  data/SMSSpamCollection       # Tables 8, 11, 12

# experiments that reuse the results above
python revision_experiments.py --task baselines   --data data/super_sms_dataset.csv --uci data/SMSSpamCollection  # Tables 5, 6
python revision_experiments.py --task footprint   --data data/super_sms_dataset.csv                               # Table 9, Fig. 3
python revision_experiments.py --task sensitivity --data data/super_sms_dataset.csv                               # Table 10
```

Outputs are written to `results_clonalg_de/` and `results_revision/`. All seeds are fixed;
re-running the code reproduces every accuracy figure of the article exactly. Running times
depend on the machine (Intel Core i7, 10th generation, 16 GB RAM, CPU only in the article).

### Where each table comes from

| Article | Result file |
|---|---|
| Table 2, Fig. 2 | `results/results_clonalg_de/budget_super/results_per_run.csv` |
| Tables 3–4 (ablations), χ² curve of Fig. 2 | `results/results_final/sweep/summary_table.csv`, `statistical_tests.csv` (Holm-adjusted columns), `results_per_run.csv` |
| Table 5 | `results/results_revision/baselines/baselines_*.csv` |
| Table 6 | `results/results_revision/baselines/paired_vs_baselines_*.csv` |
| Table 7 | `results/results_clonalg_de/salman_super/` |
| Table 8 | `results/results_clonalg_de/senol_uci/` |
| Table 9, Fig. 3 | `results/results_revision/footprint/footprint.csv` |
| Table 10 | `results/results_revision/sensitivity/sensitivity.csv` |
| Table 11 | `weights` and `threshold` columns of the `salman_super` and `senol_uci` result files |
| Table 12 | `results/results_clonalg_de/senol_uci/selected_features.json` |

### Notes

* The UCI file is read with Latin-1 encoding, as in the experiments of the article; the pound
  sign therefore appears as `â£` in the feature names of `selected_features.json`.
* The UCI split with `random_state = 111` contains 1,673 test messages (1,672 in Şenol et al.,
  who removed messages emptied by their preprocessing before splitting).

## Authors

* **Ishak Benmohammed** (corresponding author, ishak.benmohammed@univ-constantine2.dz) — MISC Laboratory, University of Constantine 2-Abdelhamid Mehri, Constantine 25016, Algeria; NTIC Faculty, University of Constantine 2-Abdelhamid Mehri, Constantine 25016, Algeria
* **Hacene Belhadef** (hacene.belhadef@univ-constantine2.dz) — NTIC Faculty, University of Constantine 2-Abdelhamid Mehri, Constantine 25016, Algeria

## Citation

The archived code is available at https://doi.org/10.5281/zenodo.23335141  

If you use this code, please cite the article (see `CITATION.cff`):

```bibtex
@article{benmohammed2026clonalg,
  author  = {Benmohammed, Ishak and Belhadef, Hacene},
  title   = {Hybrid Bio-Inspired Feature Selection Under Feature Budgets: A Clonal Selection Wrapper with {DE}-Weighted Ensembles for {SMS} Spam Detection},
  journal = {Algorithms},
  year    = {2026},
  note    = {Submitted}
}
```

## Licence

MIT, see `LICENSE`.

### Note on `ablation_sweep.py`

`ablation_sweep.py` runs the 5×2cv sweep on the Super SMS Dataset for CLONALG + DE, χ² top-K + DE,
equal-weight averaging and the four base classifiers alone, with Wilcoxon and 5×2cv F-tests
(Holm-corrected across K). Its CLONALG + DE runs agree with `main_clonalg_de.py --experiment budget`
to within 0.0008 F1 on each of the 60 runs.
