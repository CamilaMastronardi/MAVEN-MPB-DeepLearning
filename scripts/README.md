# Scripts — orden por etapa

Los scripts se agrupan según su función. Dentro de cada etapa se usa la numeración
`1a_`, `1b_`, `2a_`, etc. Los módulos auxiliares mantienen el mismo prefijo
que sus pilotos.

## 1. BigQuery — `1_bigquery/`

- `1a_bigquery_crossing_coverage.py`: verifica cobertura temporal de MAG/SWEA para los cruces etiquetados.
- `1b_analyze_crossing_coverage.py`: analiza offline los CSV pequeños del diagnóstico anterior.
- `1c_bigquery_event_pilot.py`: inspecciona un cruce con mediciones MAG y SWEA (SWEA restringido a 30–100 eV).

## 2. Cross-validation — `2_cross_validation/`

- `2a_make_cv_splits.py`: crea cuatro folds agrupados por mes, con conjuntos train/val.

## 3. Ventanas — `3_windowing/`

- `3a_windowing.py`: funciones reutilizables que construyen ventanas de 600 s por muestra nativa y etiquetas gaussianas.
- `3b_bigquery_window_pilot.py`: prueba ventanas sobre un evento/fold consultado a BigQuery.

## 4. Preprocesamiento — `4_preprocessing/`

- `4a_native_batches.py`: normalización train-only, logaritmo de SWEA, padding y máscaras.
- `4b_fold_preprocessing_pilot.py`: prueba integrada con un evento train y uno validation.
- `4c_fit_fold_normalizers.py`: ajusta estadísticas del fold completo por fechas UTC, sin muestras repetidas y con checkpoints para continuar tandas.

Las pruebas offline correspondientes están en `tests/test_3a_windowing.py` y
`tests/test_4a_native_batches.py`. Conservan `test_` al principio para permitir
la detección automática con `unittest`.

### Comandos

Ejecutar desde la raíz del repositorio, con `.venv` activado:

```bash
python -m unittest discover -s tests -v

# Diagnóstico inicial: dry-run (sin extraer mediciones).
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --limit-events 5

# Reconstruir los folds, sin consultar BigQuery.
python scripts/2_cross_validation/2a_make_cv_splits.py

# Piloto de ventanas: estimación y ejecución opcional.
python scripts/3_windowing/3b_bigquery_window_pilot.py --fold 1 --subset val
python scripts/3_windowing/3b_bigquery_window_pilot.py --fold 1 --subset val --execute

# Piloto de normalización: MAG y SWEA.
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument mag
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument swea

# Normalización de un fold entero, 3 fechas por tanda (primero dry-run):
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument mag
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument swea
```

Los comandos con BigQuery hacen estimaciones por defecto. Sólo `--execute`
realiza consultas de mediciones. Los pilotos trabajan temporalmente en RAM:
los datos científicos permanecen almacenados en BigQuery.

## 5. Entrenamiento MAG — `5_training/`

- `5a_train_mag_cv.py`: entrenamiento **real MAG**, independiente de Jupyter, con encoder–decoder–detector supervisado, validación cruzada de 4 folds agrupados por mes, estadísticos de normalización train-only, descarga de eventos una sola vez en RAM y checkpoints por época.

```bash
# Pruebas locales, sin BigQuery:
python -m unittest discover -s tests -v

# Estimación para un piloto de los 4 folds, sin consultas de mediciones:
python scripts/5_training/5a_train_mag_cv.py --folds all --epochs 3 --max-train-events 30 --max-val-events 10 --max-total-gib 35

# Ejecutar el piloto después de revisar la estimación:
python -u scripts/5_training/5a_train_mag_cv.py --folds all --epochs 3 --max-train-events 30 --max-val-events 10 --max-total-gib 35 --resume --execute
```

El piloto **no es la validación completa**. Para metodología, gastos estimados,
reanudación y métricas de localización, ver `docs/mag_cross_validation_training.md`.

**Inicio nocturno en tmux con un único comando:**
`bash scripts/5_training/run_mag_cv_pilot.sh`. Este runner ejecuta las
pruebas offline antes de consultar BigQuery y conserva logs/checkpoints.

## 6. Figuras — `6_figures/`

Los dos scripts de figuras existentes apuntan por defecto al **tercer
entrenamiento** MAG, sin tocar los modelos del cuarto entrenamiento en curso.

- `6a_plot_mag_cv_summary.py`: sin BigQuery, genera resumen de los cuatro
  folds, las tres parejas de pérdidas TRAIN/monitor interno
  (`rec`, `det`, `mag`), histograma y CDF de los errores de localización,
  y comparación con el segundo entrenamiento a la misma resolución.
  Además imprime la mediana global exacta usando los CSV por evento.
- `6b_plot_mag_reconstruction_cases.py`: reconstrucción de 70 minutos en
  cuatro canales MAG, casos acertado y fallido, y puntajes de MPB. En
  `|B|` también muestra el módulo de las componentes reconstruidas y
  guarda métricas RMSE físicas en nT. Utiliza solo dos consultas acotadas
  a BigQuery para los eventos seleccionados.

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py            # dry-run
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py --execute  # figura
```

Salida: `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/`.
Para elegir otro experimento, usar `--summary` (6a) o `--results-dir` (6b).
Ver `scripts/6_figures/README.md`.

### Segundo entrenamiento MAG: 600 ventanas + detección priorizada

La versión actual de `run_mag_cv_full.sh` calcula **hasta 600 etiquetas
por evento** (una por centro de ventana de 600 segundos), utiliza 24 de
esas ventanas en cada época con una rotación reproducible y garantiza al
menos 25 épocas para cubrir todos los objetivos. La pérdida conjunta es
`L_rec + 10*L_det`, manteniendo `sigma=60 s`. Se evalúan hasta 600 centros
por MPB. Guarda la ejecución en `results/metrics/mag_cv_600windows_det10/`
para no alterar `mag_cv_full/` del primer entrenamiento.

Ver `docs/mag_cross_validation_training.md` para detalles y límites.

### Tercer entrenamiento MAG: regularización física de |B|

El launcher `run_mag_cv_full.sh` usa hasta **1000 épocas**, paciencia de
**20 épocas** y la pérdida
`L = L_rec + 10 L_det + L_mag`. `L_mag` penaliza tanto el error del
canal reconstruido de módulo como el error de la norma de las tres
componentes reconstruidas respecto del módulo observado. El cálculo
se realiza en nT y usa la escala de |B| del normalizador de TRAIN.

Resultados en `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/`,
sin modificar los experimentos 1 y 2. Ver
`docs/mag_cross_validation_training.md`.

### Arquitectura MAG 1/3 — detector de transición antes/después

`src/detection/mag_transition_autoencoder.py` reutiliza el encoder y decoder
actuales, sustituyendo solo la cabeza supervisada por una que consume
`[z_centro, z_antes, z_después, z_después-z_antes]`. El nuevo runner es
`scripts/5_training/run_mag_cv_transition.sh` y guarda exclusivamente en
`results/metrics/mag_cv_transition_v1/`. El tercer entrenamiento continúa
usando `run_mag_cv_full.sh` con su arquitectura original.

```bash
# CUARTO experimento completo, cuando haya terminado el tercero:
bash scripts/5_training/run_mag_cv_transition.sh
```

### Quinto experimento MAG — encoder residual profundo y caché persistente

`src/detection/mag_residual_transition_autoencoder.py` conserva el detector
de transiciones antes/después del cuarto entrenamiento, pero utiliza un encoder
residual multiescala de 32 canales ocultos, 8 canales latentes y 7 dilataciones
(515 muestras de campo receptivo teórico).

`scripts/5_training/run_mag_cv_residual.sh` ejecuta **cuatro folds completos**,
con pérdida y early stopping iguales a los del cuarto, y almacena datos MAG
medidos originales en `data/cache/mag_events/` con verificaciones de SHA-256
para posteriores reutilizaciones sin BigQuery.

```bash
# No lanzar simultáneamente con el cuarto si compiten por CPU/RAM:
bash scripts/5_training/run_mag_cv_residual.sh
```

Salidas exclusivas: `results/metrics/mag_cv_residual_transition_v1/`.
Detalles en `docs/mag_fifth_training.md`.
