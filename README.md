# MAVEN–MPB: localización de la frontera MPB con V7b

Trabajo final de **Machine Learning en Geofluidos** (UBA). La versión activa es **V7b**, un modelo convolucional de MAG+SWEA con supervisión débil de regiones y ranking de candidatos del mismo evento.

La rama `main` está dedicada a V7b. Los prototipos V9/V10/V14, los modelos MAG-only y las pruebas tanh fueron retirados del árbol, pero permanecen recuperables en el historial Git.

## Organización

```text
data/
  labels/                         Catálogos MPB originales y curados
  splits/                         Manifiesto CV de cuatro folds mensuales
src/
  preprocessing/
    bigquery_source.py            Consultas, schema y límites BigQuery
    catalog.py                    Validación de manifiesto y MAG
    cv_splits.py                  Separación FIT/MONITOR por meses
    mag_cache.py, swea_cache.py   Lectura/escritura de caché científica
    weak_regions.py               Ejemplos y etiquetas débiles MPB/BS
    native_windows.py             Ventanas, normalización, máscaras
  model/
    v7b.py                        CNN, decoders y cabezas de clasificación
    ranking.py                    Ranking y minería de negativos difíciles
    score_shape.py                Descriptores de picos temporales con manejo de gaps
scripts/
  01_data/
    01_build_cv_manifest.py       Regeneración explícita de folds
  02_training/
    01_train_v7b.py               Entrenamiento y evaluación CV
  03_experiments/
    01_run_rank_weight.sh         Barrido de peso de ranking
    01_compare_rank_weight.py
    02_run_rank_margin.sh         Barrido del margen
    02_compare_rank_margin.py
    03_run_region_weight.sh       Barrido del peso de regiones
    03_compare_region_weight.py
notebooks/
  01_MPB_V7b.ipynb                Informe científico
tests/
  test_01_preprocessing.py
  test_02_model.py
  test_03_rank_weight.py
  test_04_rank_margin.py
  test_05_region_weight.py
  test_06_score_shape.py
docs/
  01_data_preparation.md
  02_model_v7b.md
  03_training.md
  04_hyperparameter_experiments.md
```

## Diagnóstico de la forma del score temporal

El notebook `notebooks/01_MPB_V7b.ipynb`, sección **7.2–7.3**, incorpora un análisis exploratorio de las curvas de puntuación del modelo en los eventos de validación externa. Calcula prominencia, anchura a media prominencia, diferencia con el mejor candidato separado al menos cinco minutos y número de máximos secundarios. Las anchuras se dejan indefinidas al cruzar huecos de datos o cuando el máximo cae en un extremo.

Los gráficos se guardan en `results/figures/v7b_notebook/15_ejemplos_forma_score.png` y `16_morfologia_score_vs_error.png` (o en la ruta indicada por `MAVEN_V7B_NOTEBOOK_FIGURES`). Las curvas reales se almacenan como NPZ en `results/metrics/v7b_score_shape/curves/` y sus métricas en `score_shape_external_folds.csv`. No se reentrena ni se consulta BigQuery; hacen falta checkpoints de los cuatro folds y cachés MAG/SWEA. De forma predeterminada el notebook procesa hasta 20 cruces evaluables por fold con semilla fija, y se puede ampliar a todos los cruces cacheados.

Los valores de score sigmoid no están calibrados como probabilidades; las correlaciones reportadas son descriptivas de la muestra efectivamente analizada.

## Matrices de confusión y correlación de canales latentes

El notebook V7b incluye tres análisis adicionales, sin necesidad de reentrenar:

- **6.4:** matriz de confusión de la clasificación de ventanas como MPB próxima o control negativo débil, obtenida de los conteos de los cuatro folds externos (figura `07b_confusion_candidatos.png`).
- **6.5–6.6:** matriz de confusión de regiones plasmáticas: magnetovaina (0) frente a magnetosfera inducida (1), por fold, global y separada para los lados anterior/posterior. Si los conteos todavía no están en los resultados antiguos, pueden reconstruirse mediante inferencia con checkpoints y cachés MAG/SWEA (figuras `07c_confusion_regiones.png` y `07d_confusion_regiones_por_lado.png`). El análisis calcula los recalls de cada clase y la exactitud balanceada a partir de los resultados disponibles.
- **8.7:** dos matrices de Pearson de 16 × 16, una para MAG y otra para SWEA. Primero se resume cada ventana promediando los dos lados del embedding y después cada evento promediando sus ventanas. Las correlaciones se calculan por fold y se promedian por índice de canal entre los folds disponibles; se ordenan mediante clustering jerárquico. Se guarda una sola figura (`14_correlacion_promedio_MAG_SWEA.png`) y CSVs de las matrices, el orden de canales y la cantidad de folds que aportan cada coeficiente. **Limitación:** los índices latentes de distintas redes no están alineados, por lo que la media es exploratoria.

Para analizar estas secciones, primero deben ejecutarse las celdas de inicialización y lectura de resultados, y, para correlación, la sección **8.1 de extracción de representaciones latentes**. El límite actual de 24 eventos por fold hace que las correlaciones sean exploratorias; se puede aumentar `MAX_LATENT_EVENTS_PER_FOLD` para analizarlas con una muestra mayor, siempre que existan las cachés correspondientes. No se consultará BigQuery durante estos análisis.

**Limitaciones:** las clases de referencia son etiquetas débiles; las ventanas pueden solaparse, por lo que no son independientes. Las métricas de clasificación no miden detección autónoma de MPB en una órbita completa. El promedio por índice entre folds puede cancelar correlaciones de canales no alineados; tampoco prueba equivalencia funcional ni significado físico específico, y la falta de datos en caché puede producir matrices parciales.

## Piloto independiente: penalización temporal de la localización MPB

El notebook de V7b incluye la sección **6.7**, una celda opcional que ejecuta el script independiente **scripts/03_experiments/04_quick_temporal_loss_pilot.py**. **No modifica** el entrenamiento V7b, sus checkpoints ni las métricas anteriores.

Se parte de un checkpoint V7b con ranking (fold externo elegido) y se comparan tres variantes con los **mismos eventos de evaluación**: checkpoint original, ajuste fino de control con la loss V7b habitual y ajuste fino con una nueva penalización temporal diferenciable. La penalización es la esperanza softmax de la distancia absoluta entre los centros candidatos y la MPB anotada, normalizada por 600 s. La localización en evaluación sigue usando el argmax global habitual; la etiqueta temporal del fold externo no participa en los gradientes.

La configuración actual utiliza **peso temporal 5**, temperatura 1,0 y **4 épocas adicionales**, recorriendo los **cuatro folds** y todos los eventos FIT y de validación externa disponibles en caché. En el script, los argumentos --train-events, --validation-events y --scan-limit aceptan **0 = sin límite**. Los batches se mantienen iguales entre ambas variantes, sin minería dinámica de negativos. Para ejecutar desde el notebook, cambiar RUN_TEMPORAL_PILOT a True en la sección 6.7; también se puede ejecutar el Python por separado por fold con los argumentos de rutas y --help. La ejecución completa puede consumir bastante CPU y memoria.

**No se consulta BigQuery ni se reentrena desde cero**; el script necesita checkpoints y cachés MAG/SWEA existentes y detiene el proceso si son insuficientes. Es una prueba piloto de ajuste fino, no una comparación concluyente ni una nueva validación cruzada completa.

Los resultados con peso 5 se guardan en results/metrics/v7b_temporal_loss_pilot_weight_5/fold_XX/ (XX = 01–04), **sin sobrescribir el piloto anterior con peso 0,5**. Para cada fold se generan summary.json, paired_event_errors.csv, training_history.csv, training_losses.png y temporal_localization_comparison.png. El notebook además muestra una tabla agregada, el conteo de eventos que mejoran/empeoran y la comparación conjunta de errores; se exportan resumen_agregado.csv y errores_por_evento_agregados.csv.

## Uso desde `main`

La reorganización V7b ya está integrada en **`main`**. Cuando no haya entrenamiento activo y los resultados estén guardados:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git status
git fetch origin
git switch main
git pull --ff-only origin main
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Si aparecen modificaciones locales, guardarlas antes del cambio de rama (por ejemplo con `git stash push -u`). La rama temporal de reorganización ya no es necesaria para trabajar.

Ejecutar el entrenamiento V7b con los hiperparámetros actuales:

```bash
python scripts/02_training/01_train_v7b.py \
  --folds all --rank-weight 2 --rank-margin 0.25 --region-weight 0.5 \
  --output-dir results/metrics/v7b_manual --resume --execute
```

Para el barrido en curso, con las mismas carpetas locales y reutilización de la referencia:

```bash
bash scripts/03_experiments/03_run_region_weight.sh
```

Los directorios `data/cache/` y `results/` se conservan localmente y permanecen ignorados por Git. **Actualizar `main` no elimina esos archivos ignorados por Git.**

## Alcance físico

V7b estima el instante más compatible con la MPB dentro de **±30 minutos de una MPB ya catalogada**, explorando ventanas MAG+SWEA de diez minutos. No se ha validado la detección autónoma de todas las MPB presentes en órbitas completas.

Documentación técnica: [datos](docs/01_data_preparation.md) · [modelo](docs/02_model_v7b.md) · [entrenamiento](docs/03_training.md) · [experimentos](docs/04_hyperparameter_experiments.md).
