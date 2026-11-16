# MAVEN-MPB-DeepLearning

Detection and temporal localization of Martian Magnetic Pileup Boundary (MPB) crossings from MAVEN observations with Deep Learning.

Final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Scientific objective

The MPB is a characteristic plasma boundary in the induced magnetosphere of Mars. The available curated labels provide **one annotated time** (`t_MPB`) for each event, not the four entry/exit times originally contemplated. The supervised goal is therefore **to score how close the annotated MPB time is to the center of a sliding input window**, then recover candidate crossing times from peaks of this score along the orbit.

The physical interpretation of each annotation (`onset`, `center`, or another marker) must be checked against the catalog definition. The width or thickness of the physical MPB cannot be measured from these single-time annotations alone.

## Proposed architecture: convolutional autoencoder + center detection

**Inputs:** independent MAG and SWEA windows, each using its instrument's own sampling cadence; raw MAG and SWEA measurements do not need to share one time grid. MAG uses `Bx`, `By`, `Bz`, `|B|`; SWEA uses electron energy-time measurements **restricted to 30–100 eV inclusive, using the physical energy axis rather than channel numbers**. Spacecraft position can be retained as metadata.

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
- Keep MAG and SWEA measurements in BigQuery; stream bounded, date-partition-filtered intervals to transient RAM only. Avoid persistent local raw/processed datasets and avoid one SQL query per sliding-window shift.
- Handle cadence differences, measurement gaps, timing alignment and train-only scaling.
- Sample negative or offset windows carefully; absence of a catalog annotation does not prove that no MPB crossing occurred.
- Assess sensitivity to window duration, stride, Gaussian width and input modalities.

## Project layout

```text
data/
  raw/            # Unused for scientific observations: measurements stay in BigQuery
  processed/      # Unused for persisted windows: construct windows in RAM
  labels/         # Small, versioned crossing tables
notebooks/        # Exploration
src/
  preprocessing/  # BigQuery extraction, quality checks, window labels
  detection/      # Encoder, decoder and sliding-window detection
  localization/   # Reserved for further localization experiments
  evaluation/     # Timing metrics and latent-space analysis
scripts/
   1_bigquery/         # 1a–1c: BigQuery coverage and event inspection
   2_cross_validation/ # 2a: source-aware temporal folds
   3_windowing/        # 3a–3b: native sliding windows and pilot
   4_preprocessing/    # 4a–4b: normalization, batches and pilot
tests/             # Offline tests (test_3a_*, test_4a_*)
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

A first diagnostic script is available at `scripts/1_bigquery/1a_bigquery_crossing_coverage.py`. It evaluates how many of the 569 labeled crossings have timestamp coverage for a **10-minute window** centered at the MPB label, treating MAG and SWEA independently with a **one-native-sample window stride** planned for training and inference.

It only scans `datetime_utc` for relevant date partitions, defaults to **dry-run estimates**, and does not download complete magnetic-field or electron-flux time series.

```bash
# Check actual schemas and partitioning (metadata only).
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --metadata-only

# Pilot: estimate cost for five events, then query those events.
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --limit-events 5
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --limit-events 5 --execute

# All 569 labels: estimate first, execute second.
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --execute
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
python scripts/1_bigquery/1b_analyze_crossing_coverage.py
```

It reports the missing SWEA events and per-instrument cadence distributions; results are saved under `results/metrics/coverage_analysis/`. The output will guide targeted, dry-run-costed scientific-data extraction.

## Next step: BigQuery-only scientific data pilot

The offline timestamp diagnostic found median per-event cadences of **1 s (MAG)** and **16 s (SWEA)**. To avoid filling the local computer with MAVEN observations, scientific measurements remain **only in BigQuery**.

The updated `scripts/1_bigquery/1c_bigquery_event_pilot.py` queries one label-centered **10-minute** interval in both instruments. It computes diagnostics in **RAM** and does **not write `.npz`, raw flux files, or other scientific-data copies locally**. Before each query it estimates BigQuery bytes; `--execute` is required to run it.

For SWEA, BigQuery discards electron-energy bins outside **30–100 eV** and computes the exploratory angular mean **inside BigQuery** for the retained channels only. It uses the stored `energy_ev` axis, not assumed energy-channel indices. No `quality`-flag filtering is applied, and this mean should not be called a calibrated omnidirectional flux. The 30–100 eV interval is our chosen modeling input band.

```bash
# Dry-run only: estimated BigQuery query cost.
python scripts/1_bigquery/1c_bigquery_event_pilot.py

# If the estimate is acceptable: print one-event MAG/SWEA diagnostics.
python scripts/1_bigquery/1c_bigquery_event_pilot.py --execute
```

See [BigQuery-only pilot workflow](docs/bigquery_event_pilot.md). In training, query temporal blocks once and form all overlapping one-native-sample-stride windows **within RAM**, rather than running BigQuery for every window.

## First scientific-data pilot result (2026-11-12)

A 600-second window centered at **2014-11-12 13:46:33 UTC** was examined **directly in BigQuery** and analyzed in RAM, without writing any scientific measurements locally.

- **MAG:** 600 samples (4 channels), event median cadence 1.0 s, max gap approximately 1.001 s, zero invalid field values.
- **SWEA svy3d (pilot before band restriction):** 18 timestamps, **median cadence 32 s** (compared with median 16 s across the previously measured 559 covered labeled events); original cube 6 elevation × 16 azimuth × 64 energy, provisional full-range energy-time summary of shape 18 × 64, energy range 3–4630 eV, no missing valid angular flux for the 1,152 energy-time cells. The subsequent **30–100 eV-only** query was executed and selected **11** physical energy channels in the tested event.
- **SWEA `quality` (historical pilot):** all 18 samples have value `-1`. We are not using that flag as an exclusion criterion; model inputs will be selected by the agreed energy range.
- **Scientific caution:** do not assume that 600 s always contains 38 SWEA measurements. The native SWEA cadence can vary between events; build variable-time-length inputs or apply an explicitly documented *within-SWEA* padding/masking strategy, preserving instrument-specific cadence and window movement by one native measurement.

The 30–100 eV SWEA pilot and native-cadence window generators were subsequently implemented and verified on real events; see the windowing and preprocessing sections below.

## Cross-validation splits (four folds)

The complete **569 labeled MPB crossings** have been assigned to **four deterministic, month-grouped folds**, balanced across the four original annotation groups. Events in a given UTC calendar month stay in the same fold; **overlapping MAG/SWEA sliding windows must inherit that event's fold**.

| Validation fold | MAG train | MAG validation | SWEA train | SWEA validation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 418 | 151 | 413 | 146 |
| 2 | 426 | 143 | 420 | 139 |
| 3 | 433 | 136 | 424 | 135 |
| 4 | 430 | 139 | 420 | 139 |

The 10 SWEA-unavailable timestamps identified in BigQuery coverage QC remain included for MAG only. The split metadata and per-fold train/val CSV files live in `data/splits/`. The nearest timestamps assigned to different folds are **278.55 minutes apart**, versus the initial required non-overlap of 70 minutes for 10-minute windows centered within ±30 minutes of the labeled event.

```bash
# These splits are already committed; running this regenerates the small CSV files:
python scripts/2_cross_validation/2a_make_cv_splits.py
```

See [cross-validation design and leakage caveats](docs/cross_validation.md). **No scientific data are downloaded**, and no model has yet been trained. The splitting is by labeled event/month, not by individual overlapping window. Future data loaders will create native-cadence MAG and 30–100 eV SWEA windows in RAM after fold assignment.

## Native-cadence sliding-window pilot (MAG and SWEA)

The split-aware window generator is in `scripts/3_windowing/3a_windowing.py`, with a BigQuery-to-RAM pilot in `scripts/3_windowing/3b_bigquery_window_pilot.py`. For a selected CV fold/event, it retrieves each instrument's **70-minute** block once (window centers within ±30 min of the labeled MPB, with ±5 min support) and yields **600-second windows, one native sample apart**. It rejects missing-edge, large-gap, and nonfinite windows; retains native MAG/SWEA cadences and SWEA energy channels in **30–100 eV**; creates a Gaussian target from the nearest *known* MPB annotation in the current training or validation subset.

```bash
python -m unittest discover -s tests -v
python scripts/3_windowing/3b_bigquery_window_pilot.py --fold 1 --subset val
python scripts/3_windowing/3b_bigquery_window_pilot.py --fold 1 --subset val --execute
```

By default the pilot performs **dry-run estimates only**. `--execute` queries at most one selected event by default, checks byte/row caps, generates window statistics in RAM, and **does not save scientific measurement arrays locally**. See [window generation details](docs/window_generation.md). This stage does not assume that low-target windows are verified physical negatives, and does not fit normalization using held-out data.

## Fold-aware normalization and masked minibatches

The first CV fold event produced **3,600 MAG windows (4×601)** and **112 SWEA windows (1×11×19)**, with **zero rejected windows**; the selected SWEA energy axis contains eleven channels within 30–100 eV. The Gaussian target `y>=0.5` occurs in 142 MAG and 4 SWEA windows. These heavily overlapping windows are *not independent events*.

`scripts/4_preprocessing/4a_native_batches.py` provides train-only per-channel streaming normalization (MAG: z-score; SWEA: `log10(1+flux)` then per-energy z-score), shape-preserving native-time padded batches with time masks, and provisional masked reconstruction / weighted proximity regression losses. **Normalization must be fitted separately for each CV fold on unique training epochs only**, never on held-out measurements. Model convolutions and readout must also handle padded time indices.

A limited, dry-run-first integrated pilot checks two real BigQuery event blocks (one training, one held-out validation) without saving measurements:

```bash
python -m unittest discover -s tests -v

# Estimate only:
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument mag

# Optional bounded execution:
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument mag --execute
```

See [normalization and batching workflow](docs/preprocessing_and_batches.md). The two-event integration pilot's normalizer is **not** the final fold-level normalizer, and no CNN has been trained.

## Full-fold training normalization (step 4c)

MAG and SWEA train/validation normalization **pilots have now both succeeded on WSL**. The SWEA validation event used 11 physical energy channels and padded batches of different native lengths (37 train vs 19 validation time positions), with statistics fit exclusively from the training event. These pilot statistics are **not yet fold-wide**.

`scripts/4_preprocessing/4c_fit_fold_normalizers.py` is the next step. It groups the ±35-minute training-event supports by UTC date, merges overlapping intervals, and processes unique observation epochs from BigQuery one date at a time. Train-only mean and variance are updated in memory; tiny JSON checkpoints allow safe resumption. The instrument's full-fold normalizer is written **only when every planned training day is processed**.

```bash
python -m unittest discover -s tests -v
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument mag
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument mag --execute
```

By default only **3 pending UTC dates** are selected per run, with dry-run estimates and byte caps. Repeat the same `--execute` command to continue from the small checkpoint. See [full-fold normalization](docs/full_fold_normalization.md).

## Jupyter notebooks for supervised autoencoders (real data only)

The educational notebooks [MAG](notebooks/01_mag_autoencoder.ipynb) and
[SWEA](notebooks/02_swea_autoencoder.ipynb) have been created with **no
synthetic measurements and no execution outputs**. They are real-MAVEN-only
scaffolds: load a finished training-fold normalization JSON, select a TRAIN
event, make a bounded BigQuery request and visualize one native-cadence window.
An explicit notebook flag, `EXECUTE_REAL_BIGQUERY = False`, blocks any
science-data request by default. Model architecture and training will be built
incrementally after inspecting real windows. See [notebook instructions](notebooks/README.md).

## Status

All four annotation groups are integrated. Timestamp coverage has been measured for 569 MAG and 559 SWEA events. The first BigQuery-only pilot ran successfully for MAG and SWEA without persisting scientific data. The **30–100 eV** SWEA band has been verified in real data (11 selected channels for the pilot event). Cross-validation event splits (4 monthly-grouped folds) have been created and versioned; no model training results are claimed. Both single-event train/validation normalization pilots have run successfully for MAG and SWEA. Next: execute the resumable full-fold normalization step 4c, then construct mask-aware CNN architectures and cautious label sampling.

## Primer entrenamiento real de MAG con 4 folds

Se añadió `src/detection/mag_autoencoder.py` como definición compartida
entre el notebook MAG y `scripts/5_training/5a_train_mag_cv.py`.
El script ofrece validación cruzada mensual, normalizadores globales TRAIN-only,
uso de datos MAVEN reales en RAM, estimación de costos BigQuery previa a
consultar, checkpoints por época y métricas de error temporal por cruce.
El entrenamiento **no se ejecuta automáticamente**: requiere `--execute`.

Comenzar con un **piloto restringido** en los cuatro folds y no interpretar
estos resultados como validación completa. Ver
[guía de entrenamiento MAG](docs/mag_cross_validation_training.md).

## Figuras reproducibles del entrenamiento

Los scripts de `scripts/6_figures/` generan figuras a partir de los archivos
con métricas ya guardados. Para el primer entrenamiento completo MAG:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py
```

Entrada `results/metrics/mag_cv_full/cv_summary.json`; salida
`results/metrics/mag_cv_full/figures/mag_cv_first_training.png`.

### Diagnóstico de reconstrucciones MAG en dos cruces reales

`scripts/6_figures/6b_plot_mag_reconstruction_cases.py` genera, a partir
 de los checkpoints finales y dos consultas MAG de 70 minutos, una figura
 comparativa del caso de localización correcta e incorrecta. Incluye la
 reconstrucción por ventanas solapadas de 10 min y la salida supervisada
 del detector (puntaje no calibrado). Ejecutar con `--execute` para recuperar
 solo los dos eventos seleccionados y guardar la figura en
 `results/metrics/mag_cv_full/figures/`.
