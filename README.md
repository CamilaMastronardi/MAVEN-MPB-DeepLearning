# MAVEN-MPB-DeepLearning

Detection and temporal localization of Martian Magnetic Pileup Boundary (MPB) crossings from MAVEN observations with Deep Learning.

Final project for **Machine Learning in Geofluids** (DCAO, Universidad de Buenos Aires).

## Current deliverable candidate: V14 (new experiment)

The original V10 localizer is the established four-fold reference. The new
**V14** retains V10's frozen V7b candidate search, but trains **two new
eight-layer convolutional autoencoders**, one for MAG and one for SWEA,
with a **three-scalar latent bottleneck per instrument**. Both bottlenecks
are used by the temporal refiner and have reconstruction supervision.
Their 3D embeddings are exported for physical interpretation.

V14 has **not yet been trained or validated**; do not substitute its
metrics for the completed V10 results until the four folds finish.
Different folds have different latent axes and must be visualized
separately. This is conditional localization in labeled MPB neighborhoods,
not a demonstrated autonomous whole-orbit detector.

- [V14 architecture, training and limitations](docs/v14_dual_3d_refiner.md)
- [Original V10 reference](docs/mag_swea_refiner_v10.md)
- [V13 latent-physics exploratory analysis](docs/v13_latent_physics_trajectories.md)

The later sections preserve the historical design and development trail.


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

## Segundo experimento MAG (600 centros, mayor peso de detección)

El entrenamiento completo actual conserva 4 folds y todos los eventos,
pero genera **hasta 600 etiquetas gaussianas por cruce** (una por centro
de rolling window), recorre las etiquetas durante al menos 25 épocas y
cambia el objetivo a `L = L_rec + 10 L_det`. El ancho de la gaussiana
permanece en `sigma=60 s`; 60 era el ancho en segundos, no la cantidad
de etiquetas. Resultados nuevos: `results/metrics/mag_cv_600windows_det10/`.
El primer experimento, `mag_cv_full/`, se conserva intacto.

## Tercer entrenamiento MAG: pérdida física del módulo

La versión actual de `scripts/5_training/run_mag_cv_full.sh` agrega una
pérdida de reconstrucción física para `|B|`: compara el módulo medido con
el cuarto canal reconstruido y con la norma de las componentes reconstruidas,
desnormalizadas a nT con el normalizador TRAIN de cada fold.

La pérdida completa es `L = L_rec + 10 L_det + L_mag`. Se mantienen los
cuatro folds, las 600 etiquetas por evento y la rotación de 24 ventanas por
época. El máximo es **1000 épocas por fold**, con **paciencia 20** y un
mínimo de 25 épocas. La selección del mejor checkpoint continúa basada en
la pérdida de detección del holdout interno; `train_mag` y `val_mag`
se registran para diagnóstico.

Nuevo experimento independiente:
`results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/`.

```bash
bash scripts/5_training/run_mag_cv_full.sh
```

Detalles en `docs/mag_cross_validation_training.md`.

## Cuarto experimento MAG — detector temporal antes/después (arquitectura 1/3)

Se añadió `src/detection/mag_transition_autoencoder.py`: usa el mismo encoder
CNN de dos canales latentes y el mismo decoder del tercer experimento, pero
cambia las entradas del detector a cuatro resúmenes `[centro, antes, después,
después-antes]`. Los promedios temporales laterales se calculan alrededor
de ±120 s, excluyendo ±15 s del centro y el padding, siempre en segundos
físicos de MAVEN. Se mantienen la pérdida `L_rec + 10 L_det + L_mag`, la
validación cruzada de cuatro folds, máximo 1000 épocas y paciencia 20.

**No interrumpir el tercer entrenamiento para probar este cuarto**. El
launcher original `run_mag_cv_full.sh` mantiene la arquitectura anterior.
Cuando termine, lanzar por separado:

```bash
bash scripts/5_training/run_mag_cv_transition.sh
```

Salida: `results/metrics/mag_cv_transition_v1/`. Se preservan los
checkpoints del tercer experimento y los anteriores. Ver
`docs/mag_cross_validation_training.md`.

## Figuras del tercer entrenamiento MAG (sin reentrenar)

Los scripts existentes de `scripts/6_figures/` utilizan ahora por defecto
`results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/`.

```bash
# Sin BigQuery: resumen CV por fold, train/monitor losses (rec/det/mag),
# histograma y CDF de los 569 errores, y comparación con el segundo modelo:
python scripts/6_figures/6a_plot_mag_cv_summary.py

# Estima dos consultas BigQuery de 70 min, sin recuperar mediciones:
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py

# Ejecuta solo esas dos consultas y crea dos reconstrucciones + score MPB:
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py --execute
```

En el panel del módulo de 6b también se dibuja la norma de las componentes
MAG reconstruidas, y en el JSON acompañante se guardan tres RMSE en nT.
Todas las figuras quedan en `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/`.
Los modelos y resultados de los entrenamientos previos y del cuarto
entrenamiento en curso permanecen independientes. Ver
`scripts/6_figures/README.md`.

## Quinto entrenamiento MAG — encoder residual y caché local de BigQuery

El quinto modelo reemplaza el encoder CNN del cuarto por una CNN temporal
residual con **32 canales ocultos, 8 latentes y 7 convoluciones dilatadas**
(campo receptivo teórico de 515 muestras). Mantiene el detector asimétrico
antes/después, las pérdidas y los cuatro folds. El entrenamiento previo no
se ve afectado: launcher y resultados independientes.

La quinta versión activa una **caché local persistente**, en
`data/cache/mag_events/` (ignorada por Git), con MAG original sin normalizar,
checksums y procedencia de BigQuery. En la primera ejecución consulta los
bloques faltantes; después los reutiliza, incluso tras apagar la PC.

```bash
# Cuando haya terminado el cuarto modelo:
git pull origin main
bash scripts/5_training/run_mag_cv_residual.sh
```

Resultados: `results/metrics/mag_cv_residual_transition_v1/`.
Manual: `docs/mag_fifth_training.md`.

## Sexto entrenamiento MAG — detección temporal densa

Nueva arquitectura: reutiliza el encoder residual del quinto modelo pero
produce **un puntaje por muestra MAG**, fusionado en timestamps nativos en las
ventanas superpuestas. Pérdida supervisada gaussiana por muestra.

**Early stopping desde época 15, paciencia 10**. Se entrenan 40 centros
rotativos por evento y época, cubriendo los 600 centros al cumplir 15 épocas.
Se mantienen los 569 cruces, la validación de cuatro folds y la caché MAG
local del quinto entrenamiento.

```bash
git pull origin main
bash scripts/5_training/run_mag_cv_dense_temporal.sh
```

Salida independiente: `results/metrics/mag_cv_dense_temporal_v1/`.
Manual: `docs/mag_sixth_training.md`.

## Séptimo entrenamiento — MAG+SWEA y clasificación física de regiones

El modelo v7 cambia el objetivo: en lugar de predecir una gaussiana de
proximidad a la MPB usando solo MAG, aprende **regiones magnetosheath / magnetosfera
inducida y transiciones MPB** con dos encoders independientes MAG y SWEA,
ambos a su cadencia nativa. Dos decoders conservan la reconstrucción como
aprendizaje auxiliar. Las etiquetas débiles se construyen con tiempos MPB y BS
asociados, dejando fuera intervalos ambiguos y posibles cruces adicionales.

Se admiten hasta 559 eventos MAG+SWEA del manifiesto; se reutiliza la caché
MAG y se crea una caché SWEA con energías físicas 30–300 eV, sin remuestreo
temporal. Se mantienen cuatro folds externos y monitor por meses internos.

```bash
git pull origin main
bash scripts/5_training/run_mag_swea_regions_v7.sh
```

Salida nueva: `results/metrics/mag_swea_regions_v1/`.
Manual: `docs/mag_swea_seventh_training.md`.

## Séptimo experimento v7b — ranking por evento y negativos difíciles

Sobre el clasificador físico MAG+SWEA del v7 se incorporó una pérdida
pairwise de ranking: los candidatos MPB deben obtener logits superiores a
los controles físicamente seguros del **mismo evento**. Dos negativos
difíciles registrados durante TRAIN se combinan con dos controles aleatorios
por época; la selección del checkpoint incluye ranking en el monitor interno.

La revisión usa la **misma arquitectura dual de encoders** que v7 y tiene
resultados aislados, sin sobrescribir v7:

```bash
PILOT_EVENTS_PER_FOLD=8 OUTPUT_DIR=results/metrics/mag_swea_regions_ranking_pilot \
  bash scripts/5_training/run_mag_swea_regions_ranking_v7b.sh
```

Guía metodológica: `docs/mag_swea_seventh_ranking.md`.
