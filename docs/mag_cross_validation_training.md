## Sexto modelo: salida temporal nativa y early stopping anticipado

Se añadió una arquitectura temporal densa manteniendo el encoder residual del
quinto modelo. Cada ventana MAG de 600 s ahora devuelve `s(t)` para cada
muestra válida y utiliza una pérdida supervisada gaussiana en tiempos nativos.
Al evaluar, se promedian puntajes de ventanas superpuestas en los instantes
físicos correspondientes, sin remuestrear.

Se habilita early stopping desde época **15** con paciencia **10** y se usan
**40** centros rotativos por evento/época para asegurar 600 centros cubiertos
antes de la primera época elegible. Se conservan 4 folds, datos reales, la
caché MAG y checkpoints aislados.

**Ver `docs/mag_sixth_training.md` para metodología y comandos.**

---

## Quinto modelo MAG: encoder residual y datos MAG cacheados

Nueva arquitectura experimental en `src/detection/mag_residual_transition_autoencoder.py`
(7 bloques convolucionales residuales, 32 canales ocultos, 8 latentes, el
mismo detector asimétrico y los mismos folds y pérdidas que en el cuarto).
El nuevo launcher `scripts/5_training/run_mag_cv_residual.sh` activa el
almacenamiento persistente de observaciones originales en
`data/cache/mag_events/` (excluidas de Git), con integridad y procedencia
verificadas. No toca el entrenamiento del cuarto modelo.

**Leer el protocolo y comandos completos en `docs/mag_fifth_training.md`.**

---

## Cuarto entrenamiento MAG — cambio de arquitectura 1/3: detector antes/después

El primer cambio arquitectónico se implementa como experimento **separado**,
sin modificar el encoder ni el decoder de `MagSupervisedAutoencoder` ni
interrumpir el tercer entrenamiento (regularización física del módulo).

**Hipótesis:** una MPB debe producir cambios entre las regiones anteriores
y posteriores al cruce, que una única representación latente central o un
promedio simétrico podrían no expresar suficientemente.

Nuevo módulo: `src/detection/mag_transition_autoencoder.py`.
`MagTransitionAutoencoder` reutiliza exactamente el **encoder CNN de 16
canales ocultos y 2 canales latentes**, y el mismo decoder de cuatro canales.
Solo modifica las entradas del detector supervisado:

```text
z_center  = representación latente en el instante central
z_before  = promedio gaussiano de z(t) en t < -15 s, centrado en -120 s
z_after   = promedio gaussiano de z(t) en t > +15 s, centrado en +120 s
delta_z   = z_after - z_before

detector_input = concat(z_center, z_before, z_after, delta_z)
score = MLP(detector_input) -> Sigmoid -> 1 puntaje por ventana
```

Las ponderaciones laterales tienen desviación `90 s` (sobre los tiempos
**físicos**, no índices). Se excluye el padding con `time_mask`. Si una
región carece de muestras válidas, el agregado correspondiente recurre
al vector central; los bloques MAG completos aceptados por QC suelen
tener muestras a ambos lados.

Este cambio NO hace la red más profunda ni la convierte todavía
en detector con una salida por segundo: esas son las modificaciones
**2/3** y **3/3** a estudiar por separado.

**Constantes para una comparación controlada con el tercer entrenamiento:**
los 569 eventos, cuatro folds por mes, 600 etiquetas/centros por MPB,
24 ventanas rotativas/evento/época, `sigma=60 s`, normalizadores,
`L=L_rec+10L_det+L_mag`, hasta 1000 épocas, mínimo 25,
paciencia 20, y selección por `val_det` interna.

Se conservan los cuatro modelos anteriores sin sobreescribirlos.
El launcher **del tercero** sigue siendo
`scripts/5_training/run_mag_cv_full.sh`; el **del cuarto** es
`scripts/5_training/run_mag_cv_transition.sh`, con carpeta nueva:

```text
results/metrics/mag_cv_transition_v1/
```

Ejecutar el cuarto cuando el tercero haya concluido y se hayan
revisado sus resultados:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate
tmux new -s mag_cv_transition
bash scripts/5_training/run_mag_cv_transition.sh
```

El programa prueba el código, estima las consultas y, si están dentro
del límite, vuelve a recuperar datos reales MAG desde BigQuery: no se
guardaron las series originales. La arquitectura se registra en las
métricas y en el checkpoint final; para hacer las figuras posteriores
se puede reutilizar `6a_plot_mag_cv_summary.py` y
`6b_plot_mag_reconstruction_cases.py` apuntando a
`--summary` o `--results-dir` de `mag_cv_transition_v1`.

**Nota:** por ahora no hay resultados de entrenamiento de esta
arquitectura. Su eficacia debe evaluarse empíricamente y sobre centros
de validación idénticos a los del tercer modelo.

---

## Tercer entrenamiento MAG: pérdida física de |B|, paciencia 20, máximo 1000 épocas

**Configuración actual de `scripts/5_training/run_mag_cv_full.sh`** (los apartados
más abajo describen configuraciones históricas del primer y segundo entrenamiento):

Se mantiene el encoder–decoder–detector de MAG, los 569 eventos catalogados,
cuatro folds mensuales, normalizadores TRAIN-only, hasta 600 centros por evento,
24 ventanas rotativas por evento y época, y `sigma=60 s`.

Se agrega una penalización explícita de módulo magnético. Para cada instante
válido se desnormalizan a **nT** las componentes reconstruidas y el módulo:

```text
B̂xyz(t) = [B̂x(t), B̂y(t), B̂z(t)]
|B̂xyz(t)| = sqrt(B̂x(t)^2 + B̂y(t)^2 + B̂z(t)^2)
B̂mag(t) = salida reconstruida del cuarto canal |B|
Bmag(t) = |B| real observado
```

La pérdida de magnitud, promediada sobre los instantes observados
(no sobre padding), es:

```text
L_mag = mean_t [ ((B̂mag - Bmag)^2 + (|B̂xyz| - Bmag)^2) / (2 σ²_mag,TRAIN) ]
```

`σ_mag,TRAIN` es la desviación estándar del módulo en el normalizador
del fold de entrenamiento (no usa estadísticas de validación). Por lo tanto,
`L_mag` carece de unidades y tiene una escala razonable respecto de
`L_rec`, que ya compara los cuatro canales estandarizados.

**Pérdida conjunta nueva:**

```text
L_total = L_rec + 10 * L_det + 1 * L_mag
```

El nuevo término exige (con una penalización, no una igualdad algebraica
exacta) que tanto el canal reconstruido `|B|` como el módulo calculado
a partir de las tres componentes reconstruidas concuerden con el módulo
real. No se cambia la arquitectura ni se imponen pesos infinitos.

- Máximo de épocas por fold: **1000** (antes 100).
- Paciencia de early stopping: **20** (antes 10).
- Mínimo de épocas: **25**, para cubrir al menos una rotación de los
  600 objetivos gaussianos.
- Criterio de selección del mejor modelo: **`val_det`** de los meses
  internos de entrenamiento, igual que antes, para priorizar la localización.
  La pérdida nueva `val_mag` también se calcula y registra para diagnóstico;
  no sustituye al criterio de early stopping.
- El número de épocas realmente ejecutadas puede ser muy inferior a 1000
  si dejan de observarse mejoras en `val_det`.
- Los archivos `fold_XX_metrics.json` incluyen `train_mag` y `val_mag`
  por época, además de las métricas anteriores.

**Los experimentos anteriores quedan intactos.** El tercer entrenamiento
usa por defecto:

```text
results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/
```

Comando desde la raíz del proyecto:

```bash
git pull origin main
source .venv/bin/activate
tmux new -s mag_cv_mag1
bash scripts/5_training/run_mag_cv_full.sh
```

El launcher ejecuta las pruebas sin consultas a BigQuery, luego realiza
la estimación de costo y consulta los mismos datos MAVEN reales si se
superan correctamente las verificaciones. Para modificar el peso físico
sin editar el código: `LAMBDA_MAGNITUDE=2 bash scripts/5_training/run_mag_cv_full.sh`
(recomendado ajustar también `OUTPUT_DIR` para separar experimentos).
Los nuevos checkpoints **no deben mezclarse ni reanudarse** desde una
carpeta de experimentos anteriores.

---

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
  early stopping termine antes y **solo permite elegir el mejor checkpoint
  desde la época 25**: así los pesos seleccionados también han sido
  entrenados con una rotación completa.
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
