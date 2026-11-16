## Segundo entrenamiento MAG: 600 etiquetas gaussianas por evento + detección ponderada

**Configuración nueva (sin modificar los resultados del experimento anterior):**

- Entradas: cada ventana sigue teniendo **600 segundos** de MAG a cadencia nativa,
  con cuatro canales; el modelo devuelve **un puntaje por ventana**.
- Para cada evento TRAIN se escogen **hasta 600 centros de ventanas
  deslizantes válidas**, y se calcula **un valor de la etiqueta gaussiana
  para cada centro**. La etiqueta ya existía por ventana; antes reteníamos
  solamente 24 ventanas por evento.
- Importante: `sigma=60` significa **desviación estándar de 60 segundos**,
  no un vector de 60 elementos. Mantenemos el mismo ancho físico de la
  etiqueta para poder atribuir las diferencias al muestreo y al peso de
  la detección.
- Para contener el costo computacional, por época se eligen **24 de las
  600 ventanas por evento**, recorriendo una permutación reproducible y
  distinta en cada evento. En **25 épocas** se recorren las 600
  etiquetas cuando existen todas. Por eso `--min-epochs=25` impide que
  early stopping termine antes.
- Validación externa: hasta **600 centros** por evento (antes 121). Los
  intervalos de observación y etiquetas de MPB siguen siendo los mismos.
- Función objetivo: `L = L_rec + 10 * L_det` (antes `L_rec + 1 * L_det`).
  `L_det` mantiene su ponderación por cercanía a etiquetas y el encoder
  sigue recibiendo gradientes de ambas ramas. Esto **no convierte** el
  score en una probabilidad calibrada, ni pondera directamente cada
  muestra de reconstrucción.
- Se mantienen cuatro folds externos, validación interna por meses, 100
  épocas máximas, patience 10 y normalizadores completos preexistentes.

Todos los resultados del **primer entrenamiento** siguen intactos en
`results/metrics/mag_cv_full/`. El nuevo experimento guarda por defecto
en **`results/metrics/mag_cv_600windows_det10/`**, evitando reanudar
checkpoints incompatibles.

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
source .venv/bin/activate
git pull origin main
tmux new -s mag_cv_600
bash scripts/5_training/run_mag_cv_full.sh
```

El launcher verifica sintaxis y pruebas offline, hace estimaciones BigQuery
para todos los eventos y solo ejecuta consultas con el límite configurado
(`MAX_TOTAL_GIB=150` por defecto). Como la primera ejecución no guardó
las mediciones científicas, **será necesario consultar nuevamente** los
bloques MAG. Este segundo experimento puede requerir más tiempo y RAM,
especialmente al evaluar 600 ventanas por evento.

Las métricas del nuevo experimento no son comparables sin más con las del
primero porque también se aumentó la densidad de los centros externos de
evaluación (121 a 600). Para comparación estricta hay que evaluar ambos
checkpoints sobre los mismos centros.

---

## Entrenamiento COMPLETO MAG con los cuatro folds (todos los eventos)

Cuando se requiere el entrenamiento completo, **NO** usar
`run_mag_cv_pilot.sh` ni seleccionar `--max-train-events 30` /
`--max-val-events 10`. Hay un launcher específico:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
source .venv/bin/activate
git pull origin main
tmux new -s mag_cv_full
bash scripts/5_training/run_mag_cv_full.sh
```

Por defecto utiliza los cuatro folds, **todos los eventos train/val**,
`--max-train-events 0 --max-val-events 0`, un **máximo de 100 épocas por fold**,
**early stopping con paciencia 10** y checkpoints tras cada época, más
un límite total de **150 GiB
estimados** para las consultas BigQuery de todos los eventos distintos.
Realiza las pruebas offline y el preflight de las consultas *antes de*
descargar cualquier medición. Si el costo estimado supera el límite,
**se detiene antes de ejecutar las consultas de mediciones**.

Los 150 GiB son un límite de bytes procesados estimados para esta
ejecución, no una cota de RAM ni un presupuesto monetario garantizado.
Las cuotas personalizadas de BigQuery pueden ser más restrictivas.
Se puede fijar otro límite antes de lanzar con
`MAX_TOTAL_GIB=120 bash scripts/5_training/run_mag_cv_full.sh`;
esto no implica recomendación de subir la cuota.

Para dejar trabajando tmux: `Ctrl+B`, después `D`.
La computadora debe seguir encendida y sin suspenderse.
Volver con `tmux attach -t mag_cv_full`.

Salidas: `results/metrics/mag_cv_full/`: `fold_01_last.pt`,
`fold_01_final.pt`, `fold_01_metrics.json`,
`fold_01_event_metrics.csv` (y análogos para los demás folds),
`cv_summary.json` y `training.log`.
Si se interrumpe, el launcher reanuda los checkpoints, pero debe
**volver a consultar los bloques de MAVEN** que estaban sólo en RAM.

**Advertencia metodológica:** trabajar con todos los eventos
no significa que el modelo esté optimizado. El máximo de 100 épocas,
la paciencia de 10 y el muestreo de 24 ventanas de entrenamiento y 121
centros de validación por evento son elecciones metodológicas iniciales. Los resultados evalúan localización
del pico **condicionada a MPB conocida**, no tasa de falsos positivos
en trayectorias sin cruce. No son métricas finales de detección general.

---

# Entrenamiento inicial de MAG: autoencoder convolucional supervisado y 4 folds

El programa `scripts/5_training/5a_train_mag_cv.py` es un **primer entrenamiento
real con MAG**, independiente de Jupyter, basado en el mismo modelo que el
notebook `01_mag_autoencoder.ipynb`. No se usan mediciones artificiales.

## Diseño y separación de datos

- Cuatro folds de eventos agrupados por mes UTC (manifiesto creado por 2a).
- En el fold k, solo eventos de los otros tres folds entran en TRAIN; el fold
  k queda completamente reservado para VALIDATION.
- Los cuatro normalizadores MAG de 4c deben existir como archivos JSON finales.
  Se verifica la firma del manifiesto y los parámetros `mean/scale`.
- MAG: Bx, By, Bz, |B|. Cada ejemplo es una ventana de 600 s con muestras
  nativas, máscara temporal y etiqueta de proximidad gaussiana (`sigma=60s`).
- Encoder CNN 1D temporal → representación latente → (a) decoder de
  reconstrucción y (b) detector supervisado escalar.
- Pérdida: MSE de reconstrucción en muestras no rellenadas por padding
  más `lambda_detection` veces MSE supervisado ponderado.
- Para TRAIN se toma un conjunto disperso de ventanas por evento, dividido
  entre centros próximos y alejados de la etiqueta, para reducir el peso de
  los miles de ventanas casi idénticas por cruce. Para VALIDATION se toman
  centros repartidos sobre ±30 minutos; se busca el instante de puntuación
  máxima por evento.
- Se conserva la definición de "etiqueta gaussiana": una ventana lejos de un
  cruce registrado **no es un negativo físico confirmado**.

El código consulta cada bloque de evento **una sola vez en la ejecución** y
lo conserva en RAM para los 4 folds; no crea Parquet/CSV/NPZ con mediciones
científicas. Cada fold externo reserva su VALIDATION hasta el final. Dentro de los otros
tres folds se separa aproximadamente el **10% de los meses** como
subconjunto de monitorización interna: no reciben gradientes y sus
mediciones no se solapan con los bloques de entrenamiento usados para
actualizar pesos. El normalizador externo ya calculado sobre los tres
folds de TRAIN se mantiene congelado (incluye los meses internos, pero
**nunca** incluye el fold de validación externo).

Cada época compara la pérdida supervisada en hasta 21 centros por evento
del subconjunto interno. Se guarda `fold_XX_best.pt` cuando mejora al
menos 0,0001 y se detiene cuando transcurren 10 épocas consecutivas sin
mejora. Solo después se evalúa una única vez el fold de validación externo,
utilizando los pesos del mejor checkpoint interno.

## Primera ejecución recomendada: piloto de 4 folds con datos reales

Para hacerlo en un único paso, el repositorio incluye
`scripts/5_training/run_mag_cv_pilot.sh`. Comprueba sintaxis y pruebas offline
**antes de cualquier consulta facturable**; luego inicia los cuatro folds
con el mismo límite de 35 GiB, escribe un log y reanuda checkpoints previos.
Dentro de tmux basta con:

```bash
bash scripts/5_training/run_mag_cv_pilot.sh
```

Si alguna prueba falla, el script se detiene sin ejecutar BigQuery. Para
inspeccionar manualmente el costo antes de ejecutar, usar el comando de
estimación que figura a continuación.

Ejecutar desde la raíz del repositorio con `.venv` activado:

```bash
python -m unittest discover -s tests -v

# SOLO ESTIMACIÓN, sin consultar mediciones:
python scripts/5_training/5a_train_mag_cv.py \
    --folds all --epochs 3 \
    --max-train-events 30 --max-val-events 10 \
    --max-total-gib 35

# Para ejecutar el piloto real (solo si la estimación es menor que el cap):
mkdir -p results/metrics/mag_cv
python -u scripts/5_training/5a_train_mag_cv.py \
    --folds all --epochs 3 \
    --max-train-events 30 --max-val-events 10 \
    --max-total-gib 35 \
    --resume --execute \
    2>&1 | tee -a results/metrics/mag_cv/training.log
```

El parámetro `--execute` es un opt-in explícito. Sin él, se inspeccionan
los normalizadores, las divisiones y los costos estimados pero no se descargan
las mediciones. El programa estima **todos** los eventos que necesita antes
de disparar cualquier consulta facturable. Por defecto, cada consulta está
limitada a 1 GiB y cada evento a 25 000 filas retornadas. El cap de 35 GiB
suma las estimaciones de **todos los eventos distintos seleccionados en esa
ejecución**, no es un cupo global de BigQuery; una cuota diaria custom del
proyecto puede detener el trabajo antes.

Los eventos limitados a 30 TRAIN y 10 VAL por fold son un **piloto**, no una
evaluación completa con los 569 eventos. La selección es determinista y
equilibra los cuatro grupos de origen de manera aproximada. Las etiquetas
de proximidad tienen en cuenta el catálogo completo de cada split aun en
el piloto.

## Mantenerlo ejecutándose en WSL con tmux

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
source .venv/bin/activate
git pull origin main
tmux new -s mag_cv
```

Dentro de tmux, ejecutar el comando de entrenamiento de arriba. Para separar
la terminal sin detenerlo, presionar `Ctrl+B`, luego `D`. Para volver:

```bash
tmux attach -t mag_cv
```

La máquina **debe permanecer encendida y sin suspensión**. Si una excepción
detiene el proceso, el checkpoint conserva el final de la última época
completada, pero como las observaciones se guardan solo en RAM, reiniciar
requiere volver a consultar los bloques de MAVEN.

## Salidas

Se guardan SOLO pesos del modelo, estado de AdamW y estadísticas pequeñas:

- `results/metrics/mag_cv/fold_01_last.pt`: checkpoint de última época.
- `results/metrics/mag_cv/fold_01_best.pt`: mejor modelo según validación interna.
- `results/metrics/mag_cv/fold_01_final.pt`: pesos finales del fold.
- `results/metrics/mag_cv/fold_01_metrics.json`: historial de pérdidas y
  resumen de localización de MPB en VAL.
- `results/metrics/mag_cv/fold_01_event_metrics.csv`: por cada evento
  de VAL: offset del pico, error temporal absoluto, puntuación del pico.
- Análogos para folds 2–4, además de `cv_summary.json`.

Las métricas de localización incluyen error absoluto mediano, percentil 90
y proporción de cruces dentro de 60/120 s del instante catalogado. Estas
cifras **no equivalen a precision, recall, F1 ni a un benchmark de falsos
positivos en órbitas sin MPB**, porque únicamente se consideran vecindades
de eventos conocidos y todavía no contamos con anotaciones completas de
ausencia de MPB. El `sigma` de las etiquetas es un parámetro de supervisión,
no un espesor físico de la MPB.

## Validación cruzada completa (después de comprobar el piloto)

Se necesita elegir TODOS los eventos (ambos límites = 0). Utilizar
**otro directorio de resultados** para no mezclar los checkpoints del
piloto, y un cap explícito suficientemente grande tras leer la estimación:

```bash
# Estima los cuatro folds completos, sin gastar consultas de mediciones:
python scripts/5_training/5a_train_mag_cv.py \
    --folds all --epochs 3 \
    --max-train-events 0 --max-val-events 0 \
    --output-dir results/metrics/mag_cv_full \
    --max-total-gib 150

# Mismo comando con --execute --resume SOLO si la estimación total,
# los límites del proyecto y el tiempo de ejecución son aceptables.
```

La consulta de cada evento se reutiliza en memoria entre folds dentro de
una misma ejecución. El gasto depende de los datos realmente estimados por
BigQuery, y el límite de 150 GiB del ejemplo **no constituye una recomendación
de gasto**: es únicamente un límite configurado que debe revisarse antes de
ejecutar.

## Precauciones para el informe final

1. La red no se ha optimizado ni contrastado con modelos de referencia.
2. Validación por folds mensuales evita contaminación inmediata por ventanas
   solapadas, pero **no** garantiza independencia física completa entre órbitas
   próximas ni generalización a años completamente no observados.
3. Los ejemplos alejados de una etiqueta no son negativos certificados.
4. La elección del número de épocas y demás hiperparámetros se establece
   de antemano para este piloto; no se debe informar VAL como test
   independiente si luego se usa para ajustarlos.
5. El criterio de mayor pico por cada ventana de evento es un diagnóstico
   de **localización condicionada al evento**; no es todavía un detector
   operacional sobre un continuo temporal completo.
