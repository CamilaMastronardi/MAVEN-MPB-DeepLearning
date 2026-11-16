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

- `6a_plot_mag_cv_summary.py`: figura comparativa del primer entrenamiento
  MAG de 4 folds (error mediano en minutos y fracción con error ≤60 s).
  Lee `results/metrics/mag_cv_full/cv_summary.json` y guarda PNG en
  `results/metrics/mag_cv_full/figures/`.

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py
```

No ejecuta BigQuery ni reentrena. Ver `scripts/6_figures/README.md`.

- `6b_plot_mag_reconstruction_cases.py`: figura diagnóstica de dos ejemplos
  reales de MAG (uno bien localizado y otro mal localizado) con cuatro
  canales observados/reconstruidos sobre 70 minutos y curva de puntaje
  MPB en un quinto panel. Usa el modelo guardado de cada fold y dos
  consultas BigQuery de 70 min, con *dry-run* por defecto.

```bash
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py           # estimar consultas
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py --execute # dibujar
```

Salida: `results/metrics/mag_cv_full/figures/mag_cv_reconstruction_good_vs_bad.png`.

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
