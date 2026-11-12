# Cross-validation: event-level, month-grouped, 4 folds

## Why we split by month, not by sliding window

The supervised catalog contains **569 annotated MPB timestamps** between 2014 and 2020. MAG has timestamp coverage for all 569, and SWEA for 559 (10 unavailable events from the recorded BigQuery timing QC).

The final models use **600-second sliding windows**, shifted by **one native measurement** in each instrument. Adjacent windows share nearly all their samples, so randomly splitting individual windows would introduce severe train/validation leakage.

We therefore assign each **UTC calendar month** as one indivisible temporal group. All events in a month, all their extracted observation neighborhoods, all resulting sliding windows, and all negative candidates from those neighborhoods must remain in the same fold.

We have **25 distinct UTC months**, with four original catalog source groups. Group 3 appears in only **four** monthly blocks, so **4 folds** is the largest number that guarantees each source group can contribute at least one monthly block to each validation fold. The algorithm processes source groups with the fewest months first, and distributes larger month blocks to folds with fewer source-group examples and then fewer examples overall. The assignment is deterministic, without random splitting.

Source-group balancing is **not** physical-class stratification: these source groups identify annotation provenance, not MPB classes. The source groups are partly confounded with observation years, so this evaluation measures unseen-month generalization, **not** extrapolation to entirely unseen mission years.

## Fixed split summary

| Validation fold | MAG train | MAG validation | SWEA train | SWEA validation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 418 | 151 | 413 | 146 |
| 2 | 426 | 143 | 420 | 139 |
| 3 | 433 | 136 | 424 | 135 |
| 4 | 430 | 139 | 420 | 139 |

Every event belongs to exactly one validation fold. For fold `k`, **train on events with `fold != k`** and evaluate on **`fold == k`**. MAG and SWEA share the same fold assignments to allow fair comparisons, with `swea_available = 0` excluded only from SWEA/joint processing.

The 10 SWEA-unavailable labels are documented in `data/splits/swea_missing_event_timestamps.csv`, derived from the earlier timestamp coverage query. These are metadata flags, **not** physical measurements.

## Temporal non-overlap audit

The initial intended use is to generate window centers within **±30 minutes** of each event's `t_MPB`. Each input window itself is 10 minutes, so the complete measurement support of one event is within **±35 minutes**. Two events placed in different folds must therefore be separated by **more than 70 minutes**, even if their raw signal snippets lie on either side of midnight or month boundaries.

The closest crossing timestamps assigned to different folds are separated by approximately **278.55 minutes** (4 h 38 min 33 s), passing this initial audit.

**Important:** this does not certify arbitrary future windows. If window centers extend more than ±30 minutes, if unlabeled negative candidates are sampled elsewhere, or if additional orbits are downloaded, redo the overlap test for the **actual physical support** of every training and validation window. Annotations from the same month must never be split between folds.

## Versioned artifacts and reproducibility

- `data/splits/cv4_monthly_manifest.csv`: all 569 labeled events with columns `month_block`, `fold`, `mag_available`, `swea_available`.
- `data/splits/cv4_summary.csv`: train/validation counts per fold and source group.
- `data/splits/cv4_config.json`: split algorithm context and temporal audit.
- `data/splits/fold_01_train.csv` and `fold_01_val.csv` (similarly for folds 02–04): direct inputs for the eventual MAG/SWEA data loaders.
- `scripts/04_make_cv_splits.py`: regenerates all these small files from the versioned annotations. It does **not** connect to BigQuery or download MAVEN measurements.

On your personal PC, after `git pull origin main`:

```bash
# Already committed, so this rerun is optional:
python scripts/04_make_cv_splits.py
```

To load the event lists in Python:

```python
import pandas as pd

train = pd.read_csv("data/splits/fold_01_train.csv")
val = pd.read_csv("data/splits/fold_01_val.csv")

mag_train = train[train["mag_available"] == 1]
swea_train = train[train["swea_available"] == 1]
swea_val = val[val["swea_available"] == 1]
```

These rows contain **timestamps and labels**, not MAG vectors or SWEA flux arrays. Our loaders will use those timestamps to retrieve small BigQuery blocks into RAM as needed. SWEA features remain restricted to **30–100 eV**; the native SWEA cadence can vary between windows.

## Model-selection and reporting precautions

- Split labels **before** building overlapping windows. Assign every window the fold of its event/month. Never call a random `train_test_split` on the expanded window tensor.
- Calculate training-only normalization statistics inside each fold; do not fit scalers with validation observations.
- For example, train MAG on its available train events, SWEA on its available train events, then report matched-event metrics on the common SWEA-available validation subset.
- Report event-level precision, recall, F1 and timing error, not just per-window accuracy. Events, not overlapping windows, are the independent evaluation units.
- If hyperparameters (such as `sigma`, window rejection thresholds, or late-fusion weights) are selected using validation folds, those same folds no longer provide an unbiased final performance estimate. Use an inner split of the training months or an additional untouched test set for an unbiased result.
- Label omissions are possible, so do not assume every unannotated time interval is a genuine negative. Establish a cautious negative-window sampling protocol separately.

This step constructs the **data split only**, not the sliding-window training dataset itself.
