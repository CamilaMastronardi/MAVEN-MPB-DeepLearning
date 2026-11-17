# V7b: experimento controlado del peso de clasificación de regiones

## Objetivo

Estudiar el efecto del peso `region_weight` de la pérdida BCE para las dos regiones plasmáticas, sin cambiar ningún otro hiperparámetro del modelo V7b MAG+SWEA. Los pesos de regiones estudiados son **0.25, 0.5, 1 y 2**.

Se mantienen fijos:

- `rank_weight=2` y `rank_margin=0.25`, seleccionados como candidatos exploratorios en los barridos anteriores;
- peso de reconstrucción 0.15, tasa de aprendizaje 0.0003, batch size 16, seed 42;
- 100 épocas máximas, 10 mínimas, paciencia de 8, paso de evaluación de 32 s;
- 2 positivos, 2 negativos difíciles y 2 negativos aleatorios por evento; margen de controles 120 s;
- los mismos meses para FIT, monitor interno y fold externo, y los mismos datos MAG/SWEA reales.

La función de pérdida varía únicamente en este término:

```text
L = BCE_transición
  + region_weight * BCE_regiones
  + 0.15 * (MSE_MAG + MSE_SWEA)
  + 2.0 * L_ranking(margin=0.25)
```

## Ejecución

En la raíz del repositorio, con el entorno virtual activo:

```bash
bash scripts/5_training/run_v7b_region_weight_sweep.sh
```

El script comprueba primero los datos de entrenamiento y su compatibilidad, corre los tests, y entrena secuencialmente las configuraciones que faltan, con `--resume`. Solo reutiliza archivos completos y con configuración validada.

### Referencia reutilizada

La corrida `rank_weight=2, rank_margin=0.25, region_weight=0.5` ya existe en:

```text
results/metrics/v7b_ranking_margin_sweep/margin_0p25/
```

El lanzador comprueba la huella SHA256 del manifiesto, los hiperparámetros fijos, el estado de los cuatro folds y los archivos de métricas/checkpoints. Si pasan esos controles, crea un **enlace simbólico** `region_0p5` hacia esa carpeta y **no la vuelve a entrenar**. Nunca borra ni sobrescribe esa corrida.

Por lo tanto, en el caso estándar de CV completa quedan **12 entrenamientos nuevos**, no 16.

Si la corrida anterior no existe, el script detiene el lanzamiento de CV completa antes de entrenar, en lugar de suponer que puede reutilizarse. Para realizar una corrida nueva sin reuso puede ejecutarse `REUSE_BASELINE=0 bash scripts/5_training/run_v7b_region_weight_sweep.sh`; en ese caso sí realizará 16 entrenamientos.

## Resultados

Las corridas tienen directorios independientes:

```text
results/metrics/v7b_region_weight_sweep/
  region_0p25/
  region_0p5/          # enlace simbólico a la corrida anterior
  region_1/
  region_2/
```

Después de los entrenamientos, el comparador ejecuta:

```bash
python scripts/5_training/compare_v7b_region_weight_sweep.py \
  --results-root results/metrics/v7b_region_weight_sweep \
  --folds all
```

Genera:

- `comparison_by_fold_and_region_weight.csv`: resultados por fold y globales de cada peso fijo, con error mediano, percentil 90, y porcentaje dentro de 60 y 120 s.
- `paired_outer_events.csv`: errores individuales de cada MPB para comparar eventos idénticos.
- `chosen_region_weights_by_inner_monitor.csv`: selección independiente por fold.
- `nested_inner_selected_summary.json`: resumen de los cuatro folds seleccionados por monitor interno.

El comparador rechaza diferencias en manifiesto, normalizadores, cantidades de eventos, ventanas evaluadas o hiperparámetros distintos de `region_weight`.

**Selección del peso:** se usa exclusivamente la exactitud top-1 de ranking del monitor interno; en empate se usa la exactitud de pares, y luego el menor peso. La pérdida del monitor con el coeficiente variable no se compara directamente entre modelos. Los resultados de los folds externos se usan para informar métricas, no para elegir el coeficiente en el procedimiento anidado.

**Limitación:** el problema es localización condicionada de una MPB catalogada dentro de ±30 minutos, no detección autónoma sobre órbitas completas. La elección fija de `rank_weight=2` y `rank_margin=0.25` tuvo una etapa exploratoria anterior con esos folds; por lo tanto, la estimación de desempeño sigue siendo exploratoria y no una evaluación independiente final.
