# LC fuzzy clustering: a reproducible benchmark on 21 UCI datasets

This repository holds the **LC** (local-mean) fuzzy clustering methods and a complete,
self-checking benchmark. It compares them with seven established clustering methods on 21 public
UCI datasets. One command creates the environment, runs the tests, fits every method, writes every
table and figure, and independently audits the results.

- **10 methods:** three LC variants (LC-FixedHalfT, LC-FixedT, LC-SoftFixedT) and seven comparison
  methods: FCM, PCM, PFCM, DPC, eFCM, MC-FCM and k-means++.
- **21 datasets**, all bundled in `data/uci/` with SHA-256 checksums.
- **30 seeds** per dataset and method: **6,300 fits** in total.
- **Metrics:** ARI, NMI and Purity (agreement with the true classes; these decide wins and ranks),
  plus Silhouette, PC and PE as descriptive measures.

## Run the full benchmark

You need **Python 3.11, 3.12 or 3.13** and an internet connection the first time (to install the
pinned packages). The commands are the same in Windows **cmd**, **PowerShell** and a Linux/macOS
terminal. Download the repository and run it:

```bash
git clone https://github.com/mark-0polo/LC-fuzzy-clustering.git
cd LC-fuzzy-clustering
python reproduce.py
```

Without git, use **Code → Download ZIP** on GitHub, unzip it, open a terminal in the unzipped
folder and run `python reproduce.py`.

On Linux/macOS, use `python3` if `python` is not found. This command:

1. creates a private environment in `.venv/` and installs the exact versions in `requirements.txt`
   (your global Python is not changed);
2. runs the unit tests;
3. fits all 6,300 runs (10 methods × 21 datasets × 30 seeds, seeds 42–71);
4. writes the results to `results/benchmark/`: `RESULTS.md`, CSV tables and 22 PNG figures;
5. recomputes every number from the saved predictions and writes the audit to `verification.json`.

It ends with `independent verification PASS`. On a recent desktop CPU the full run took about 26
minutes: 22 for fitting (the fits run one after another on one core) and 4 for the report, figures
and audit. It needs about 500 MB of disk space, mostly for the saved predictions.

### Options

| Command | What it does |
| --- | --- |
| `python reproduce.py --background` | Runs without keeping the terminal open; progress is in `results/benchmark/workflow_state.json` and `run.log`. |
| `python reproduce.py` (again, after an interruption) | Resumes: completed fits are kept, only the missing ones run. |
| `python reproduce.py --n-init 1 --output results/smoke` | Quick check: one seed (210 fits, a few minutes). |
| `python reproduce.py --figures-only` | Rebuilds the tables, figures and audit from saved fits, without fitting. |
| `python reproduce.py --output results/another_run` | A separate full run in a new folder. |
| `python reproduce.py --skip-bootstrap` | Uses the current Python instead of `.venv` (the packages in `requirements.txt` must already be installed). |

Changed code or data cannot resume an old output folder (the run is refused); use a new `--output`.

## Results

Open **`results/benchmark/RESULTS.md`** after the run. It contains:

1. the overall ranking (mean ARI, NMI and Purity, wins, mean rank) on all 21 datasets, on the 8
   development datasets and on the 13 datasets added later;
2. dataset × method tables for ARI, NMI and Purity, with the winner of each dataset;
3. significance tests: each LC variant against each other method, Wilcoxon signed-rank over
   datasets with Holm correction;
4. Silhouette, run health (non-convergence and empty clusters), run time, and mean ± SD for every
   cell.

| File in `results/benchmark/` | Content |
| --- | --- |
| `RESULTS.md` | All results as readable tables |
| `all_runs.csv` | Every fit: one row per dataset, method and seed |
| `summary.csv`, `ranking.csv` | Mean and SD per cell; wins and ranks |
| `lc_significance.csv`, `wilcoxon_holm.csv`, `friedman.csv`, `nemenyi.csv` | Statistical tests |
| `Figures/` | 22 PNG figures at 600 DPI |
| `Predictions/` | Saved labels and memberships of every fit |
| `verification.json`, `experiment_manifest.json` | The audit; code, data and package versions |

## The LC method

LC first smooths the data locally: each point moves halfway towards the mean of itself and its
k = ⌈log₂ M⌉ nearest neighbours, where M is the number of distinct points. It then runs
entropy-regularized fuzzy c-means, `u ∝ exp(−d² / t)`, at a temperature calibrated once from the data:
T = (mean squared radius) / (rank + 2). The three variants differ only in the temperature: **T/2**
(LC-FixedHalfT), **T** (LC-FixedT) and **2T** (LC-SoftFixedT). Nothing is tuned per dataset.


## Datasets

| Group | Datasets | Why they are there |
| --- | --- | --- |
| development (8) | Iris, Wine, Breast Cancer, Seeds, Glass, Ecoli, Yeast, Image Segmentation | Used while the methods were developed |
| regime (5) | optdigits, pendigits, satimage, mfeat-kar, new-thyroid | Chosen to test where LC should win (the first four) or lose (new-thyroid) |
| held-out (8) | banknote, wall-following, wifi, vehicle, parkinsons, ionosphere, sonar, libras | Not looked at before their first full run |

`RESULTS.md` reports the groups separately, so the results do not rest on the datasets the

## Run the tests only

`reproduce.py` runs the tests before every benchmark. To run them on their own, inside the
environment:

```bash
python -m unittest discover -s tests -t .
```

This needs `src` on the Python path. In PowerShell first run `$env:PYTHONPATH="src"`, in cmd
`set PYTHONPATH=src`, and in a Linux/macOS terminal `export PYTHONPATH=src`. With pytest
installed, `python -m pytest` works without that step.

## Use one algorithm in your own code

```bash
python -m pip install -e .
```

```python
from algo.lc_soft_fixed_t import LCSoftFixedT

# X: standardized feature rows (NumPy array); K: number of clusters.
fit = LCSoftFixedT(n_clusters=K, random_state=42).fit(X)
labels = fit.U.argmax(axis=1)      # fit.U holds the fuzzy memberships
```

Every method in `src/algo/` works the same way: construct it with K, call `fit(X)`, and read the
result.


## Licence

The code is under the MIT licence ([LICENSE](LICENSE)). The datasets are from the UCI Machine
Learning Repository under CC BY 4.0.
