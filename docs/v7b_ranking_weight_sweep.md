# V7b: comparación controlada del peso del ranking

## Pregunta

¿Cambia la capacidad de localizar una MPB cuando se modifica **solo**
`--rank-weight` entre **0.5, 1, 2 y 4**? El modelo de referencia es V7b,
con entradas MAG+SWEA y ranking entre candidatos del mismo evento.

## Parámetros constantes

- **Los cuatro folds originales, agrupados por mes UTC**. El experimento usa
  la misma lista de eventos, etiquetas y exclusiones físicas para cada peso.
- Semilla **42**, AdamW, learning rate **0.0003**, weight decay **0.001**;
  batch **16**, máximo **100 épocas**, mínimo **10**, paciencia **8**.
- BCE de transición con coeficiente **1**; BCE de regiones **0.5**;
  reconstrucción auxiliar MAG+SWEA **0.15**.
- Margen ranking **0.5**, **2 positivos**, **2 negativos difíciles** y
  **2 aleatorios** por evento y época. Negativos seguros cada **120 s**.
- Ventanas MAG/SWEA de **600 s** cada una con cadencia nativa; SWEA usa
  12 energías fijas entre **30 y 300 eV**.
- Para evaluar una MPB conocida: exploración de **±30 minutos** con
  centros espaciados **32 s**.
- Los mismos archivos de caché y manifiesto en los cuatro entrenamientos.
  El lanzador registra los parámetros de cada run. Las salidas incluyen
  el SHA256 del manifiesto y la configuración fija completa.

Los controles negativos difíciles seleccionados a lo largo de las épocas
**pueden variar** con el peso: es una consecuencia legítima del aprendizaje,
no un cambio en el protocolo experimental.

## Selección sin consultar el fold externo

Cada run utiliza un **monitor interno separado por meses** dentro del TRAIN
para seleccionar el checkpoint y el umbral binario. En cada fold, el criterio
de early stopping es

`monitor_BCE_trans + 0.5*monitor_BCE_regiones + lambda*monitor_rank`.

No se puede comparar **ese valor numérico** directamente entre pesos
porque cambia la función objetivo.

Una vez entrenados los cuatro valores, el comparador selecciona
**un peso por fold**, utilizando solamente métricas **independientes de
lambda** obtenidas en el monitor interno:

1. Mayor `monitor_event_top1_fraction`.
2. En caso de empate, mayor `monitor_event_pair_accuracy`.
3. Si persiste el empate, menor `monitor_event_rank_loss`.
4. Finalmente, menor `rank_weight`.

Solo después se informa el resultado sobre el fold externo del peso
seleccionado internamente. Este resumen constituye la comparación anidada
predefinida, sin elegir pesos a partir de errores del fold externo.

Se calculan además las métricas exploratorias de cada uno de los cuatro
pesos sobre los mismos eventos, incluyendo comparación emparejada de aquellos
que resultan evaluables con **todos** los pesos. No utilizar esta tabla
exploratoria para cambiar el peso y luego reportar el mismo fold como test
independiente. El experimento sigue siendo **localización condicionada**
de cruces catalogados, no detección autónoma sobre órbitas completas.

## Ejecutar

Desde la raíz del repositorio, activar `.venv`, y ejecutar:

```bash
bash scripts/5_training/run_v7b_ranking_weight_sweep.sh
```

Por defecto ejecuta los **4 folds completos** de manera **secuencial**,
reutilizando las cachés MAG/SWEA existentes. Si faltan bloques, el script
original realiza dry-runs con límites antes de consultar BigQuery. El
lanzador ejecuta antes las pruebas unitarias; si fallan, no comienza el
entrenamiento.

**Piloto más económico (8 eventos por fold):**

```bash
PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_ROOT=results/metrics/v7b_ranking_weight_sweep_pilot \
bash scripts/5_training/run_v7b_ranking_weight_sweep.sh
```

Una sola partición para depuración:

```bash
FOLDS=1 PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_ROOT=results/metrics/v7b_ranking_weight_sweep_fold1_pilot \
bash scripts/5_training/run_v7b_ranking_weight_sweep.sh
```

No usar la misma `OUTPUT_ROOT` para combinaciones diferentes de piloto,
folds o hiperparámetros fijos. `--resume` revisa la firma de cada modelo
antes de reutilizar checkpoints finalizados.

## Resultados

La salida por defecto está en
`results/metrics/v7b_ranking_weight_sweep/`, con carpetas independientes:

- `rank_0p5/`
- `rank_1/`
- `rank_2/`
- `rank_4/`

Cada carpeta contiene checkpoints `fold_XX_best.pt`, `fold_XX_last.pt`,
`fold_XX_final.pt`, métricas de fold, CSV por evento y `cv_summary.json`.

El comparador genera en el directorio superior:

- `comparison_by_fold_and_weight.csv`: métricas por fold y por peso,
  porcentaje de MPB dentro de 60 y **120 s** (sobre todos / evaluables),
  mediana del error, top-1, ranking y accuracy débil de regiones.
- `paired_outer_events.csv`: errores de las cuatro versiones para cada
  MPB y bandera de pertenencia al subconjunto evaluable común.
- `chosen_weights_by_inner_monitor.csv`: peso seleccionado con el monitor
  interno de **cada** fold y su evaluación externa.
- `nested_inner_selected_summary.json`: resultados combinados de la
  selección anidada y número de eventos evaluables comunes.

Puede regenerarse la comparación **sin reentrenar ni consultar BigQuery**:

```bash
python scripts/5_training/compare_v7b_rank_weight_sweep.py \
  --results-root results/metrics/v7b_ranking_weight_sweep \
  --folds all
```

El comparador se detiene si faltan corridas o difieren la configuración,
los identificadores de evento, cobertura, normalizadores o conteos de
candidatos. Para un piloto incompleto no se debe asumir que existen los
cuatro folds completos.