# MAVEN-MPB-DeepLearning

Detection and temporal localization of Martian Magnetic Pileup Boundary (MPB) crossings from MAVEN observations with Deep Learning.

Final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Scientific objective

The MPB is a characteristic plasma boundary in the induced magnetosphere of Mars. The available curated labels provide **one annotated time** (`t_MPB`) for each event, not the four entry/exit times originally contemplated. The supervised goal is therefore **to score how close the annotated MPB time is to the center of a sliding input window**, then recover candidate crossing times from peaks of this score along the orbit.

The physical interpretation of each annotation (`onset`, `center`, or another marker) must be checked against the catalog definition. The width or thickness of the physical MPB cannot be measured from these single-time annotations alone.

## Proposed architecture: convolutional autoencoder + center detection

**Inputs:** independent MAG and SWEA windows, each using its instrument's own sampling cadence; raw MAG and SWEA measurements do not need to share one time grid. MAG uses `Bx`, `By`, `Bz`, `|B|`; SWEA uses electron energy-time measurements. Spacecraft position can be retained as metadata.

1. **MAG:** independent 1D convolutional autoencoder, reconstruction decoder, and supervised detection head.
2. **SWEA:** independent convolutional autoencoder, reconstruction decoder, and supervised detection head.
3. **Output:** each model predicts a separate scalar score `s(X_t) in [0, 1]` per window. Compare separate time-score curves and consider optional late fusion.

A proposed smooth training target for a window centered at `t_c` is

```text
y(t_c) = exp[- (t_c - t_MPB)^2 / (2 sigma^2)]
```

where `sigma` is a tunable label-smoothing width **not a physical MPB thickness**. When several annotated times are near a window, the target-generation rule must be defined consistently. The output is a learned center-proximity **score**, not automatically a calibrated probability.

**Initial configuration:** center a **10-minute (600-second)** window at each measurement timestamp and move it by **one sample of the relevant instrument**. This produces separate native-timestamp detection-score curves for MAG and SWEA. At nominal 1 Hz, MAG has approximately 601 samples if both endpoints of `[t_c-300 s, t_c+300 s]` are included; SWEA's count depends on its cadence. Gaps and irregular sampling must be handled within each instrument to obtain suitable CNN inputs. During inference, plot each score against its corresponding window-center timestamp and identify peaks above a chosen threshold. Evaluate event-level precision, recall, F1 and temporal errors at defined matching tolerances. Split contiguous orbit/event blocks rather than overlapping windows to avoid leakage.

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

## BigQuery coverage diagnostic

A first diagnostic script is available at `scripts/01_bigquery_crossing_coverage.py`. It evaluates how many of the 569 labeled crossings have timestamp coverage for a **10-minute window** centered at the MPB label, treating MAG and SWEA independently with a **one-native-sample window stride** planned for training and inference.

It only scans `datetime_utc` for relevant date partitions, defaults to **dry-run estimates**, and does not download complete magnetic-field or electron-flux time series.

```bash
# Check actual schemas and partitioning (metadata only).
python scripts/01_bigquery_crossing_coverage.py --metadata-only

# Pilot: estimate cost for five events, then query those events.
python scripts/01_bigquery_crossing_coverage.py --limit-events 5
python scripts/01_bigquery_crossing_coverage.py --limit-events 5 --execute

# All 569 labels: estimate first, execute second.
python scripts/01_bigquery_crossing_coverage.py
python scripts/01_bigquery_crossing_coverage.py --execute
```

See [coverage workflow](docs/bigquery_coverage.md) for safety limits, outputs, interpretation and next steps.

## Coverage diagnostic result (local BigQuery run, 2026-11-11)

The 569-event timestamp coverage diagnostic completed successfully with an estimated total of **0.2552 GiB processed** across MAG and SWEA:

| Instrument | Windows with any time samples | Preliminary timestamp-QC pass |
| --- | ---: | ---: |
| MAG | 569/569 | 569/569 |
| SWEA | 559/569 | 559/569 |
| Both | 559/569 | 559/569 |

These figures refer **only to temporal coverage within ±300 s** around the labeled time, not to the validity of MAG vector values or SWEA flux/quality/energy channels.

A new **offline** script summarizes the two saved CSVs without charging BigQuery:

```bash
python scripts/01b_analyze_crossing_coverage.py
```

It reports the missing SWEA events and per-instrument cadence distributions; results are saved under `results/metrics/coverage_analysis/`. The output will guide targeted, dry-run-costed scientific-data extraction.

## Next step: pilot scientific-data extraction

The offline analysis of saved timestamp-coverage results found median intersample intervals of **1.0 s (MAG)** and **16.0 s (SWEA)**. SWEA is absent from ten labeled windows (one Group 3 event and nine Group 2 events, mostly in September–October 2018).

Before constructing sliding-window training samples, extract and inspect **one label-centered 10-minute window** of real field data and SWEA distributions:

```bash
# Estimate scanned bytes without querying the data.
python scripts/02_bigquery_event_pilot.py

# Run the bounded one-event pilot, after reviewing the estimate.
python scripts/02_bigquery_event_pilot.py --execute
```

The source arrays and metadata are cached to `data/raw/bigquery_pilot/` and are ignored by Git. This stage uses each instrument's own timestamps, validates SWEA energy/angle axes, and reports observed quality-flag values without assuming their interpretation. See [pilot extraction workflow](docs/bigquery_event_pilot.md).

## Status

All four annotation groups are integrated. BigQuery table-coverage code is committed but has **not yet been executed against the private BigQuery dataset**; no coverage counts or model-training results are claimed. Next: review extracted real MAG/SWEA arrays and quality flags from the one-event pilot, then design instrument-specific window builders.
