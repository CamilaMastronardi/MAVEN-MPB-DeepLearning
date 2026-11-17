# V10-HP1 — nuevo entrenamiento con ajuste de hiperparámetros

## Objetivo

Repetir el refinador temporal V10 original (V7b MAG+SWEA congelado)
con una configuración de entrenamiento más prolongada y mayor peso de
los objetivos de localización, preservando las comparaciones originales.

**Esto NO es V13 ni el experimento de expertos por grupo MAG.**
No se modifica la arquitectura ni el teacher, ni se reagrupan eventos.

## Hipótesis y configuración preespecificada

| Parámetro | V10 original | V10-HP1 |
| --- | ---: | ---: |
| Learning rate | 0.0002 | 0.0001 |
| Máximo de épocas | 45 | 150 |
| Mínimo de épocas | 10 | 20 |
| Paciencia | 9 | 20 |
| Batch size | 16 | 16 |
| Peso de KL temporal | 1.5 | 2.0 |
| Peso de SmoothL1 offset | 0.5 | 1.0 |
| Sigma de gaussiana | 32 s | 32 s |
| Positivos por evento y época | 3 | 4 |
| Negativos por evento y época | 3 | 3 |
| Stride de evaluación | 32 s | 32 s |
| Teacher V7b | congelado | congelado |

Función de pérdida:

```
BCE_presencia + 2.0 * KL_temporal + 1.0 * SmoothL1_offset
```

La distribución temporal del target utiliza sigma=32 s.
Estas elecciones son una **hipótesis experimental**, no valores ya optimizados.
La calibración de alpha (incluido alpha=0, que reproduce V7b) y
de media/máximo sigue realizándose únicamente sobre MONITOR interno.

## Controles metodológicos

- Mismos cuatro folds externos y mismas etiquetas MPB que V10.
- Misma selección de meses MONITOR internos, mismo seed por defecto
  (42), mismas ventanas reales MAG+SWEA y medidas de seguridad
  de separación respecto de BS/otras MPB.
- V7b original permanece congelado y sus checkpoints no cambian.
- El nuevo experimento guarda checkpoints y métricas en un directorio
  exclusivo, sin sobrescribir V7b o V10.
- Verificar resultados con el mismo conjunto de eventos (559 si se
  conserva la cobertura original). Comparar mediana, porcentaje <=60 s,
  <=120 s, <=300 s, p90, errores pareados y alpha calibrado por fold.
- **No seleccionar nuevos hiperparámetros usando el outer holdout**:
  los cuatro folds ya participaron en experimentos anteriores, por lo
  que todas estas comparaciones siguen siendo exploratorias.

## Carga de datos y costo

El lanzador ejecuta primero una comprobación completa de integridad y
procedencia de los archivos MAG/SWEA en caché utilizando los mismos
lectores que V7b. Este preflight se ejecuta **sin cliente ni consultas
BigQuery** y aborta si falta algún bloque. Si la caché está completa,
V10 vuelve a comprobar los bloques y no necesita BigQuery.
No se fuerzan descargas adicionales.

## Ejecutar

Desde WSL2, en la raíz del repositorio:

```bash
git pull origin main
source .venv/bin/activate
tmux new -s mpb_v10_hp1
bash scripts/5_training/run_mag_swea_refiner_v10_hp1.sh
```

Separar la sesión de tmux: Ctrl+B, luego D.
Recuperar: `tmux attach -t mpb_v10_hp1`.

Piloto opcional y separado, **no comparable** con la muestra completa:

```bash
PILOT_EVENTS_PER_FOLD=8 \
bash scripts/5_training/run_mag_swea_refiner_v10_hp1.sh
```

La configuración puede cambiarse mediante variables de entorno:
`MAX_EPOCHS`, `MIN_EPOCHS`, `PATIENCE`, `LEARNING_RATE`,
`LOCALIZATION_WEIGHT`, `OFFSET_WEIGHT`, `GAUSSIAN_SIGMA_S`,
`TRAIN_POSITIVES_PER_EVENT`, `TRAIN_NEGATIVES_PER_EVENT`,
`BATCH_SIZE`, `SEED`, `FOLDS` y `OUTPUT_DIR`.
**Usar un OUTPUT_DIR diferente para cada configuración**:
el chequeo de firma impide reanudar en un directorio con
hiperparámetros incompatibles.

## Salidas

```
results/metrics/mag_swea_refiner_v10_hp1/
    fold_01_metrics.json
    fold_01_event_metrics.csv
    fold_01_best.pt
    fold_01_final.pt
    ...
    cv_summary.json
    training.log
```

Para examinar el resumen final:

```bash
python -m json.tool results/metrics/mag_swea_refiner_v10_hp1/cv_summary.json
```

Si el preflight aborta por caché incompleta, **no se ha
iniciado el entrenamiento**. Revisar las entradas faltantes y
resolverlas por separado antes de reintentar.
