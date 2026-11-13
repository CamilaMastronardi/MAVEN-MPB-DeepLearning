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
