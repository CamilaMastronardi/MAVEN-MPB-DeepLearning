# Sexto entrenamiento MAG — localización temporal densa

## Hipótesis y cambio arquitectónico

En el quinto experimento, el encoder residual CNN mejoró el error mediano
global sobre 569 cruces, pero el detector seguía produciendo **un puntaje
escalar por ventana de 600 s**. Los errores elevados y los máximos secundarios
mostraron que no siempre se elegía la transición correspondiente a la MPB.

La sexta versión **mantiene exactamente la topología del encoder residual y
del decoder del quinto modelo** (7 bloques dilatados residuales, 32 canales
ocultos, 8 latentes) y cambia la cabeza supervisada:

- El encoder conserva el tiempo nativo: `(B,4,T) -> (B,8,T)`.
- En cada muestra válida `t` construye cuatro grupos de características
  latentes: `[z(t), z_antes(t), z_después(t), z_después(t)-z_antes(t)]`.
- Los bloques laterales se promedian usando **tiempos físicos de MAG**:
  antes, de −210 a −15 s; después, de +15 a +210 s (valores por defecto).
  Los puntos inválidos y el padding no entran a los promedios. Se calculan
  mediante sumas acumuladas y búsqueda en timestamps (sin remuestrear).
- La cabeza `Conv1d 1×1 -> GELU -> Conv1d 1×1 -> sigmoid` produce
  un puntaje `s(t)` **por instante MAG**: `(B,T)`.
- El puntaje es una etiqueta supervisada de **proximidad a la MPB
  catalogada**, NO una probabilidad calibrada ni una identificación física
  independiente de las etiquetas.

No se incrementó nuevamente la profundidad del encoder para mantener la
comparación arquitectónica más interpretable.

## Etiquetas temporales y pérdida

Para cada ventana el tiempo central es `t_c`, el catálogo indica la MPB
anotada `t_MPB`, y cada muestra tiene tiempo nativo `t_i`.
Se entrena con:

`y(t_i) = exp[-0.5 ((t_i - t_MPB) / 60 s)^2]`.

Los valores nativos se calculan con el desplazamiento de la etiqueta más
cercana al **centro** ya definido por el catálogo del fold. No se fabrican
muestras ni se interpolan gaps. Una ventana cercana a múltiples anotaciones
deberá analizarse explícitamente en una investigación posterior.

La pérdida de detección se calcula en **todos los instantes válidos**,
excluyendo padding y normalizando por la suma de los pesos:

`L_dense_det = sum(mask * w(y) * (s-y)^2)/sum(mask*w(y))`

con `w(y)=1+9y` y:

`L_total = L_rec + 10 L_dense_det + L_mag`

Todas las componentes MAG se normalizan con estadísticas congeladas de
TRAIN del fold; las pérdidas de reconstrucción y módulo conservan sus
definiciones anteriores.

## Evaluación y comparación con quinto modelo

Se mantienen los **mismos cuatro folds**, los 569 eventos, las mismas
ventanas de 600 s y los **600 centros de ventana** seleccionados para
cada evento de validación. A diferencia del modelo 5, que selecciona
el máximo de 600 puntajes escalares, en el modelo 6:

1. La red devuelve una serie `s_j(t)` en timestamps nativos para cada
   ventana `j` seleccionada.
2. Los puntajes de todas las ventanas que cubren un mismo timestamp se
   promedian en esa muestra física, sin interpolar ni generar un grid
   artificial.
3. Se elige el máximo sobre los timestamps válidos dentro de **±30 min**
   de la MPB catalogada, como en los anteriores entrenamientos.
4. Se registra el error respecto de la etiqueta, además de la fracción
   de eventos con error ≤60 s y la distribución completa.

La comparación con v5 comparte los **mismos eventos y ventanas de entrada**,
pero NO la misma resolución de puntajes de salida: v5 compite en hasta
600 centros por MPB y v6 en los timestamps nativos cubiertos por esas
ventanas. Además, la pérdida supervisada es temporal y no escalar.
Estos cambios deben mencionarse al interpretar la magnitud de la mejora.
El resultado sigue siendo una evaluación **condicionada a eventos anotados**;
no determina precision/recall de cruces desconocidos en días completos.

## Early stopping: desde la época 15, paciencia 10

La sexta versión reduce:
- Época mínima elegible para checkpoint: **15** (antes 25).
- Paciencia: **10** épocas sin mejora de `val_det` (antes 20).
- Épocas máximas: 1000 (sin cambios).
- Ventanas de entrenamiento por evento y época: **40** (antes 24).
- Pool total de centros por evento: **600** (sin cambios).

La rotación determinista de 40 centros por época garantiza que se hayan
utilizado los 600 centros distintos de TRAIN por evento al completar
**15 épocas** (`15 × 40 = 600`). No se habilita early stopping antes
de esa cobertura mínima, y se conserva la selección mediante el monitor
interno de meses TRAIN, sin mirar validación externa por épocas.

La modificación de ventanas por época también cambia el presupuesto de
optimización, además de la cabeza supervisada y el criterio de parada:
para aislar causalmente cada factor sería necesaria una ablación posterior.

## Lanzamiento con caché MAG

Se reutiliza la caché original medida por BigQuery, que sigue en
`data/cache/mag_events/`, excluida de Git. No se vuelven a descargar
eventos que ya están guardados con timestamps y checksum válidos. No
se sobrescriben ni los checkpoints ni las métricas del quinto modelo.

Desde la raíz del repositorio, una vez instaladas las dependencias:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

# Ejecución en tmux, opcional:
tmux new -s mag_cv_dense
bash scripts/5_training/run_mag_cv_dense_temporal.sh
```

El launcher ejecuta pruebas offline y verificación de sintaxis antes de
leer mediciones. El nuevo directorio de salida exclusivo es:

`results/metrics/mag_cv_dense_temporal_v1/`

Los checkpoints de cada fold son recuperables mediante `--resume`.
Se almacenan la arquitectura, los parámetros de parada, la resolución
temporal de los puntajes y el código identificador de ejecución.

## Gráficos y resultados

Después de completar los cuatro folds:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_dense_temporal_v1/cv_summary.json \
  --compare-to results/metrics/mag_cv_residual_transition_v1/cv_summary.json

python scripts/6_figures/6b_plot_mag_reconstruction_cases.py \
  --results-dir results/metrics/mag_cv_dense_temporal_v1 --execute
```

Se generan CV por fold, pérdidas TRAIN/monitor, errores globales,
comparación v5-v6, cambios emparejados por evento y reconstrucción MAG
con la curva temporal `s(t)` fusionada a tiempos nativos.

Una caída de error no debe interpretarse automáticamente como mejora
del reconocimiento físico de la MPB: hay que revisar ejemplos, el
sobreajuste y la dependencia respecto de las anotaciones disponibles.
