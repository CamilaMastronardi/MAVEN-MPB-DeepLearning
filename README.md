# MAVEN-MPB-DeepLearning

Detection and temporal localization of Martian Magnetic Pileup Boundary (MPB) crossings from MAVEN observations with Deep Learning.

Final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Scientific objective

The MPB is a characteristic plasma boundary in the induced magnetosphere of Mars. The available curated labels provide **one annotated time** (`t_MPB`) for each event, not the four entry/exit times originally contemplated. The supervised goal is therefore **to score how close the annotated MPB time is to the center of a sliding input window**, then recover candidate crossing times from peaks of this score along the orbit.

The physical interpretation of each annotation (`onset`, `center`, or another marker) must be checked against the catalog definition. The width or thickness of the physical MPB cannot be measured from these single-time annotations alone.

## Proposed architecture: convolutional autoencoder + center detection

**Inputs:** synchronized windows of MAVEN measurements, with separate encoders for magnetic-field time series (MAG: `Bx`, `By`, `Bz`, `|B|`) and electron energy-time information (SWEA). Spacecraft position may be used as metadata and later as an additional input. The final SWEA representation and sampling strategy are to be determined from the BigQuery schema.

1. **Encoder:** a convolutional representation of each input modality, optionally fused into a shared latent vector `z`.
2. **Reconstruction branch:** decode the latent representation to approximate the original MAG and/or SWEA window.
3. **Supervised detection branch:** predict one scalar `s(X_t) in [0, 1]` per window, intended to peak when its center `t` coincides with an annotated `t_MPB`.

A proposed smooth training target for a window centered at `t_c` is

```text
y(t_c) = exp[- (t_c - t_MPB)^2 / (2 sigma^2)]
```

where `sigma` is a tunable label-smoothing width **not a physical MPB thickness**. When several annotated times are near a window, the target-generation rule must be defined consistently. The output is a learned center-proximity **score**, not automatically a calibrated probability.

During inference, apply the same fixed-width window across the time series with a specified stride, plot the score against the center times, then identify peaks above a chosen threshold. Evaluate event-level precision, recall, F1 and temporal errors at defined matching tolerances. Split contiguous orbit/event blocks rather than overlapping windows to avoid leakage.

A reconstruction objective can help representation learning, but it is not guaranteed to improve detection; compare with a supervised-only CNN baseline.

## Latent-space analysis

Represent detected and labeled crossing windows with latent vectors, then investigate:

- PCA/UMAP projections and cluster stability across runs.
- K-means or density-based clustering, without presupposing distinct physical categories.
- Associations with physical context such as inbound/outbound geometry, position, field properties and SWEA energy distributions.

The primary planned comparisons are **MAG-only**, **SWEA-only** and **MAG + SWEA** using matched train/test partitions and reported coverage.

## Available crossing annotations

| Group | Annotated timestamps |
| --- | ---: |
| Group 1 | 108 |
| Group 2 | 219 |
| Group 3 | 70 |
| Group 4 | 172 |
| **Total** | **569** |

The small curated CSV sources and the unified table are tracked in `data/labels/`. The source for Group 3 also contains 16 blank rows that are omitted from the unified table. See `data/labels/README.md` for source details and data-quality notes.

## Data sources and preparation

- **BigQuery:** MAG, SWEA and MAVEN position tables.
- Cache limited date/time selections locally to avoid repeated full-table queries.
- Handle cadence differences, measurement gaps, timing alignment and train-only scaling.
- Sample negative or offset windows carefully; absence of a catalog annotation does not prove that no MPB crossing occurred.
- Assess sensitivity to window duration, stride, Gaussian width and input modalities.

## Project layout

```text
data/
  raw/            # Downloaded measurements (ignored by Git)
  processed/      # Aligned series and windows (ignored)
  labels/         # Small, versioned crossing tables
notebooks/        # Exploration
src/
  preprocessing/  # BigQuery extraction, quality checks, window labels
  detection/      # Encoder, decoder and sliding-window detection
  localization/   # Reserved for further localization experiments
  evaluation/     # Timing metrics and latent-space analysis
scripts/          # Reproducible command-line entry points
models/           # Model weights (ignored)
results/
  figures/        # Plots (ignored)
  metrics/        # Metrics (ignored)
```

## Environment

Python 3.11 and PyTorch, initially using CPU:

```bash
cd MAVEN-MPB-DeepLearning
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Dependencies are provisional and will be pinned after the environment and first experiment are validated.

## Status

Annotations integrated; no training or evaluation result is yet claimed. Next: inspect BigQuery table schemas, build synchronized MAG/SWEA windows, define targets, train/test partitions and reconstruction/detection losses.
