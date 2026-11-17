# Séptima prueba, revisión v7b: ranking por evento y negativos difíciles

## Qué corrige este experimento

El piloto MAG+SWEA original (`mag_swea_regions_pilot`, 8 eventos por fold)
dio buena concordancia con **etiquetas débiles de regiones** (81.6%-97.2%),
pero muchos falsos positivos entre candidatos:

- Fold 1: TP 7, FP 48, FN 9, TN 396.
- Fold 2: TP 16, FP 322, FN 0, TN 197.
- Fold 3: TP 10, FP 125, FN 6, TN 373.
- Fold 4: TP 14, FP 142, FN 2, TN 359.

El máximo temporal era incorrecto en numerosos cruces. **Elevar el umbral
de clasificación no cambia el máximo**. Por eso v7b entrena directamente
la preferencia entre candidatos del mismo evento.

## Se conserva de v7

- Encoders/decoders MAG (1 Hz nativo aproximado) y SWEA (~32 s),
  fusión de las representaciones a ambos lados de la transición.
- SWEA log-flux con 12 energías entre 30 y 300 eV, sin remuestrear tiempo.
- Cabeza de regiones magnetosheath / magnetosfera inducida.
- Cabeza de logit binario de transición MPB.
- Pérdidas BCE de transición, BCE de región, y reconstrucción auxiliar.
- Misma partición CV externa por meses, normalizadores solo con fit-TRAIN,
  monitor independiente del fold externo y caché BigQuery existente.
- Mismos controles para rechazar etiquetas físicas dudosas alrededor
  de MPB, BS y otras MPB/BS.

## Cambio 1: ranking de transiciones de un mismo evento

Sea `z_i` el **logit** de transición de un candidato. Para cada MPB
entrenada, se presentan simultáneamente candidatos positivos (cerca de
la anotación) y controles negativos del mismo evento, nunca de otro.
Se añade pérdida logística suave:

```
L_rank(evento) = mean_pos,neg softplus( margen + z_neg - z_pos )
L_rank = promedio_eventos L_rank(evento)
```

`margen = 0.5` en logits por defecto. Se calculan también:

- `pair_accuracy`: fracción de pares (pos, neg) correctamente ordenados;
- `event_top1`: fracción de eventos en los que algún positivo supera
  a **todos los controles elegibles**. En TRAIN solo mide los controles
  del batch; en el MONITOR incluye todo el pool elegible del evento.

Nueva pérdida total:

```
L = BCE_transición + 0.5*BCE_región
    + 0.15*(MSE_MAG + MSE_SWEA)
    + 1.0*L_rank
```

Los coeficientes se configuran con argumentos de línea de comandos.

## Cambio 2: pool de negativos más exigente

Antes, se usaban hasta ocho offsets fijos (ej. ±600, ±900,
±1200, ±1500 segundos). En v7b se genera una malla de candidatos
negativos de 600 a 1800 segundos a ambos lados con salto de 120 s.

**Cada candidato debe seguir pasando las reglas de exclusión física**;
no se transforma automáticamente una región desconocida en negativa.

Se eligen por evento en cada época:

- Hasta **2 positivos** de los tres candidatos ±30 y 0 s;
- **2 negativos difíciles**: entre los que obtuvieron mayores logits
  al pasar por TRAIN en la época previa;
- **2 negativos aleatorios nuevos**: para no sobreajustar el ranking
  únicamente a los controles que ya conoce.

En época 1, los dos lugares de negativos difíciles se rellenan con
controles aleatorios seguros. Es minería online **aproximada y barata**,
sin pases extra de la red sobre todos los candidatos.

Los grupos se mantienen íntegros en cada minibatch; no se calculan
comparaciones entre eventos diferentes. Se persiste `hard_memory`
por fold en `fold_XX_last.pt`, para reanudar sin reiniciar minería.

**IMPORTANTE:** `hard_memory` utiliza únicamente el conjunto fit-TRAIN,
nunca los meses del monitor interno, ni los eventos de validación externa.

## Cambio 3: selección del checkpoint con ranking

El early stopping v7b se basa en

```
val_select = val_BCE_transición
           + 0.5*val_BCE_región
           + 1.0*val_L_rank
```

sobre meses reservados dentro del TRAIN externo.

Se conservan por defecto mínimo 10 épocas, paciencia 8, máximo 100.
El monitor imprime `val_rank` y `val_top1`, además de las BCE, y
el JSON guarda las curvas por época. La métrica top-1 no se usa como
criterio continuo único de early stopping por su alta discretización
en pilotos de 2 a 4 eventos de monitor.

El máximo de puntuación en ±30 min y las métricas de MPB etiquetadas
se calculan **solo después** de elegir el checkpoint, como antes.

## Archivos y aislamiento

Nuevo archivo:
`src/detection/mpb_event_ranking.py`

Extensiones compatibles:
`src/data/mpb_weak_regions.py`,
`scripts/5_training/5b_train_mag_swea_regions.py`,
`scripts/7_figures/7a_plot_mag_swea_results.py`.

Launcher **independiente**:
`scripts/5_training/run_mag_swea_regions_ranking_v7b.sh`.

El launcher viejo mantiene `--rank-weight 0` implícito y los resultados
originales v7 siguen siendo compatibles. La nueva configuración tiene
una firma de checkpoint y ruta diferentes; nunca se debe usar la carpeta
del piloto v7 original como salida de v7b.

## Ejecutar primero el piloto emparejado de 32 MPB

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_swea_regions_ranking_pilot \
bash scripts/5_training/run_mag_swea_regions_ranking_v7b.sh
```

Antes de entrenar se ejecutan `py_compile` y todos los unit tests.
El piloto reutiliza los 32 MAG/SWEA ya cacheados de la ejecución
anterior si la caché local está completa. Las consultas faltantes
están protegidas con dry-runs y límites en GiB.

Los cuatro folds del piloto generan automáticamente figuras, incluyendo
una **comparación por evento con el piloto v7 anterior** si existe
`results/metrics/mag_swea_regions_pilot/cv_summary.json`.

Salida:
`results/metrics/mag_swea_regions_ranking_pilot/`

Revisar:
- `cv_summary.json`;
- `fold_XX_metrics.json` y curvas `train_rank`/`val_rank`;
- `monitor_event_top1_fraction` y número de eventos de monitor;
- `fold_XX_event_metrics.csv` y mediana de localización por fold;
- `figures/mag_swea_v7_vs_v7b_shared.png` (eventos idénticos).

## Si el piloto es razonable, ejecutar la CV completa

```bash
tmux new -s mpb_v7b
bash scripts/5_training/run_mag_swea_regions_ranking_v7b.sh
```

Salida:
`results/metrics/mag_swea_regions_ranking_v1/`

El entrenamiento completo trabajará con eventos MAG+SWEA realmente
cargados (hasta 559 del catálogo); la disponibilidad del manifiesto
no garantiza el contenido de las tablas.

## Qué significa una mejora

Comparar v7 y v7b sobre **los mismos eventos evaluables**, no solo
los porcentajes globales. Nos interesa reducir error mediano y la
cantidad de picos lejanos, además de aumentar top-1 del monitor.

El F1 de candidatos es **F1 respecto de etiquetas débiles**, no
precision/recall de detección autónoma en órbitas completas.

En v7b todavía existe selección condicionada a saber que hay una MPB
catalogada cerca: no hay todavía un benchmark exhaustivo con negativos
orbitales verificados. No atribuir una mejora a MAG o SWEA aisladamente
sin una comparación controlada de entradas.
