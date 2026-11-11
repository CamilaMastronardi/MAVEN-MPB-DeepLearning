# MAVEN-MPB-DeepLearning

Deep learning for automatic detection and temporal localization of crossings of the Martian Magnetic Pileup Boundary (MPB) using MAVEN observations.

## Scientific objective

The Magnetic Pileup Boundary (MPB) is a characteristic plasma boundary in the induced magnetosphere of Mars, arising from the interaction of the solar wind with the Martian plasma environment. This project aims to detect candidate MPB crossings automatically and estimate their temporal extent using time series from the MAVEN spacecraft.

This repository is developed as the final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Proposed two-stage approach

### Stage 1 — Detection

A one-dimensional convolutional neural network (1D CNN) processes sliding windows of magnetic-field measurements and produces a binary classification score for possible MPB crossings. Candidate detections will be used to select windows for the second stage.

### Stage 2 — Temporal localization

A convolutional autoencoder with an additional temporal-localization head processes candidate crossing windows:

- The reconstruction branch learns a latent representation by reconstructing the input signals.
- The localization branch predicts Gaussian distributions for four characteristic times that delimit the crossing.
- Temporal-order constraints enforce `t1 < t2 < t3 < t4`.

The contribution of the reconstruction objective will be assessed against a direct localization baseline.

## Initial input variables

- MAVEN MAG: `Bx`, `By`, `Bz`, and `|B|`.
- Possible future extensions: ion and electron observations from SWIA and SWEA.

## Project layout

```text
data/
  raw/            # Original data (not tracked by Git)
  processed/      # Prepared time series (not tracked by Git)
  labels/         # Crossing annotations (not tracked by Git)
notebooks/        # Exploratory analyses
src/
  preprocessing/  # Data preparation and windowing
  detection/      # Stage 1: binary 1D CNN
  localization/   # Stage 2: autoencoder and temporal head
  evaluation/     # Detection and timing metrics
scripts/          # Executable workflows
models/           # Trained weights (not tracked by Git)
results/
  figures/        # Generated plots (not tracked by Git)
  metrics/        # Generated metrics (not tracked by Git)
```

## Status

**Initial repository scaffold.** The network architectures, event-label definitions, window durations, losses, and evaluation protocol will be specified and tested during the project. No trained model or experimental result is claimed at this stage.

## Local setup

```bash
git clone https://github.com/CamilaMastronardi/MAVEN-MPB-DeepLearning.git
cd MAVEN-MPB-DeepLearning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is provisional; exact versions will be pinned after the development environment is validated.

## Data policy

Raw MAVEN observations, processed datasets, annotations, and model weights are excluded from ordinary Git tracking. Store large scientific datasets outside the repository or use a dedicated data artifact workflow.
