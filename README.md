# MAVEN-MPB-DeepLearning

Automatic detection and temporal localization of Martian Magnetic Pileup Boundary (MPB) crossings using Deep Learning and MAVEN time series.

Final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Scientific objective

The MPB is a characteristic plasma boundary in the induced magnetosphere of Mars. This project aims to identify MPB crossings in MAVEN measurements and estimate the **single annotated crossing timestamp** for each event.

**Scope revision:** The available training tables provide one `MPB time` per event, not four physically defined entry/exit times. Accordingly, the original two-stage architecture (binary classifier plus autoencoder and four-time localization head) is **not** the current supervised training objective. No estimate of the full boundary duration can be validated using these labels alone.

## Proposed model: 1D CNN for time-resolved detection

- **Input:** windows of MAVEN MAG `Bx`, `By`, `Bz` and `|B|`, sampled on a common time axis.
- **Output:** a one-dimensional temporal score `p_MPB(t)` at each sample or output time step.
- **Targets:** smooth Gaussian peaks centered on the timestamps recorded in `MPB time`. The width controls training tolerance; it does **not** represent measured MPB thickness or duration.
- **Inference:** identify local maxima above a calibrated threshold and apply a minimum-separation rule to reduce duplicate detections.
- **Evaluation:** event-level precision/recall/F1 within stated temporal tolerances and timing error of matched predictions; avoid splitting overlapping windows from the same event across training and test data.

A simple binary window classifier can serve as a baseline. An autoencoder is optional for later representation learning, not required for the first training runs.

The physical definition of the original `MPB time` annotations (onset, center, or another marker) still needs verification.

## Available annotations

Three unique uploaded tables have been retained in `data/labels/`:

| Group | Number of rows |
| --- | ---: |
| Group 1 | 108 |
| Group 2 | 219 |
| Group 4 | 172 |
| **Unique annotated timestamps** | **499** |

`data/labels/mpb_crossings.csv` contains the standardized, chronological annotation table. See `data/labels/README.md` for provenance, column definitions, duplicates and quality-control caveats.

## Data sources and preparation plan

- **BigQuery:** MAG time series and spacecraft positions, with SWEA measurements available for potential multimodal extensions.
- Start with MAG as the supervised baseline; incorporate SWEA after establishing an end-to-end evaluation.
- Retrieve and cache selected time intervals rather than repeatedly querying entire BigQuery tables.
- Handle cadence, gaps, synchronization and scaling; select well-defined training candidates and use care when choosing negatives from incompletely labeled periods.
- Separate training, validation and test periods by event/orbit or contiguous temporal blocks to reduce information leakage.

## Project layout

```text
data/
  raw/            # Full downloaded measurements (not tracked)
  processed/      # Prepared time series and windows (not tracked)
  labels/         # Versioned small annotation tables
notebooks/        # Exploration and diagnostics
src/
  preprocessing/  # BigQuery extraction, QC, windows and targets
  detection/      # Primary 1D CNN event-localization model
  localization/   # Reserved for alternative localization experiments
  evaluation/     # Event matching, metrics and diagnostics
scripts/          # Reproducible entrypoints
models/           # Trained weights (not tracked)
results/
  figures/        # Generated plots (not tracked)
  metrics/        # Generated metrics (not tracked)
```

## Environment

The current development setup uses Python **3.11** and PyTorch on CPU (AMD Picasso integrated graphics has no NVIDIA CUDA device).

```bash
git clone https://github.com/CamilaMastronardi/MAVEN-MPB-DeepLearning.git
cd MAVEN-MPB-DeepLearning
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The dependencies in `requirements.txt` are provisional and should be pinned after a reproducible run.

## Status

Dataset annotations are committed. The training-dataset builder, negative sampling, window duration, Gaussian width, model architecture, and event-matching tolerance will be selected and tested iteratively. No model-training result is claimed yet.
