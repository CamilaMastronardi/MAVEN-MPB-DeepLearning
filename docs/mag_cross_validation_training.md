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
científicas. La validación se aplica después de las épocas predefinidas, sin
ajustar el modelo con el fold de validación.

## Primera ejecución recomendada: piloto de 4 folds con datos reales

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
