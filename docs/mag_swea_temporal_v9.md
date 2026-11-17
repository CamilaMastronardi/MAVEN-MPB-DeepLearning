# V9 — localización temporal densa de la MPB con MAG y SWEA

## Objetivo

**Dado que ya se sabe que hay una MPB en el intervalo de unos 70 min**,
determinar el instante del cruce con mayor precisión. No detecta
presencia/ausencia de una MPB en series arbitrarias. Este modelo sustituye
como siguiente experimento a V8; V7b se conserva intacto como referencia.

Referencias de V7b completa (559 eventos):

- Mediana de error: 72 s.
- Dentro de +/-60 s: 250/559 = 44.7 %.
- Dentro de +/-120 s: 375/559 = 67.1 %.
- Dentro de +/-300 s: 469/559 = 83.9 %.

## Diferencia de arquitectura

V7b extrae vectores latentes promedio de zonas anteriores y posteriores a
cada candidato, excluyendo la región +/-60 s central. Ese mecanismo sirve
para distinguir cambios globales de régimen, pero pierde la posición
temporal local de la transición.

V9 reemplaza la cabeza de clasificación de transición por un **mapa temporal
denso a la resolución de MAG**. Mantiene dos encoders 1D convolucionales,
inicializados con los pesos de V7b en el mismo fold:

- MAG: Bx, By, Bz y |B| a cadencia nativa, aproximadamente 1 s.
- SWEA: 12 canales de flujo angular medio, 30-300 eV, frames a ~32 s.
- Los latentes de SWEA se asocian a los tiempos MAG por **vecino temporal
  más cercano solo en el espacio latente**; NO se crea un espectro SWEA
  observado artificial ni se interpola su flujo en el tiempo.
- Para cada instante MAG, la red fusiona latentes MAG, latentes SWEA
  próximos y contextos MAG de 31 s y 121 s.
- La cabeza convolucional predice un logit por muestra MAG.
- Un segundo valor de confianza por ventana se deriva de la distribución
  de logits (log-average-exp + bias aprendido).
- No se descartan los +/-60 s centrales.

## Supervisión en una ventana de 10 minutos

Para evitar el atajo de que «la MPB siempre está en el centro», los
candidatos positivos se centran a offsets:

    -240, -180, -120, -60, 0, +60, +120, +180, +240 s

respecto del cruce catalogado; por tanto, el instante MPB aparece en
posiciones distintas dentro de la ventana. No se usan todos los positivos
en cada época: se eligen aleatoriamente hasta tres por evento.

Los controles negativos siguen estando lejos de MPB y BS catalogados:

    -1500, -1200, -900, -600, +600, +900, +1200, +1500 s

pero se conservan solo si las reglas físicas de V7b los consideran
válidos. No se reinterpretan intervalos inciertos como negativos. Se
muestrean hasta tres controles por evento y época.

Pérdida de entrenamiento:

    L_total = BCE_ventana + 1.5 * KL(P_objetivo || P_MPB)

donde el objetivo temporal es una gaussiana de sigma=32 s alrededor del
instante catalogado, normalizada sobre muestras MAG efectivamente
medidas. La pérdida de tiempo se aplica SOLO en ventanas positivas.
Las negativas contribuyen a BCE de presencia de frontera **dentro de
10 minutos**, no a clasificación de una serie completa de 70 minutos.

Esta sigmoide de ventana es una variable auxiliar para ponderar las
predicciones, NO una evaluación de detección autónoma.

## Evaluación (los mismos cuatro folds que V7b)

El entrenamiento reserva meses internos del fold TRAIN para early
stopping. El fold externo no interviene en selección de checkpoints ni
normalización.

Para cada evento de prueba, V9 recorre centros a intervalos de 32 s
en +/-30 minutos, extrae ventanas MAG+SWEA completas de 600 s y
produce una distribución sobre todos los tiempos MAG observados.
Combina las distribuciones temporales ponderadas por la confianza de
cada ventana, suaviza el mapa temporal resultante en 9 s y elige su
máximo dentro de +/-30 min. **No se facilita el instante MPB a la red
como entrada**; sí se usa el intervalo catalogado para delimitar la
búsqueda, igual que en V7b.

Se compara cada uno de los 559 cruces con su error V7b original:

- Mediana global y media de error temporal.
- Fracciones con error <=60, <=120 y <=300 segundos.
- Número de eventos que mejoran/empeoran/empatan.
- CSV emparejado por evento y resultados por fold.

El criterio de éxito es superar la referencia V7b en esos mismos
eventos. El método de salida cambió además de la arquitectura, por lo
que atribuir una mejora a una sola modificación requiere una ablación.

## Ejecución en CPU (WSL2)

Detener primero V8 (Ctrl+C dentro de tmux). Luego:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

tmux new -s mpb_v9
bash scripts/5_training/run_mag_swea_temporal_v9.sh
```

El script realiza `py_compile` y `unittest discover -s tests -v`
antes de cargar los datos o entrenar. Si falla alguna comprobación,
se detiene. Reusa las cachés MAG/SWEA, salvo faltantes; cualquier
consulta nueva tiene preflight y límites BigQuery.

Para separar terminal: Ctrl+B, luego D. Reconectar:

```bash
tmux attach -t mpb_v9
```

Opcional: prueba pequeña antes del entrenamiento completo:

```bash
PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_swea_temporal_v9_pilot \
bash scripts/5_training/run_mag_swea_temporal_v9.sh
```

La inicialización de ese piloto procede igualmente de los modelos V7b
completos, pero los eventos de prueba de cada fold siguen sin haber
participado en el entrenamiento del checkpoint que les corresponde.

### Salidas

Directorio por defecto:

    results/metrics/mag_swea_temporal_v9/

- `training.log`: avance de todas las épocas y folds.
- `fold_XX_best.pt`, `fold_XX_last.pt`: selección y reanudación.
- `fold_XX_final.pt`: pesos seleccionados y normalizadores.
- `fold_XX_metrics.json`, `fold_XX_event_metrics.csv`: validación.
- `cv_summary.json`: mediana global real y comparación emparejada V7b.

`--resume` está activado en el launcher. No usar como
`OUTPUT_DIR` ninguna carpeta previa de V7/V7b/V8.

## Limitaciones

- El instante de MPB de catálogo no necesariamente coincide con una
  transición infinitamente fina. Una gaussiana de 32 s evita obligar
  al modelo a reproducir un delta exacto.
- La validación externa es sobre meses/folds ya consultados en el
  desarrollo del proyecto. Se requiere un período verdaderamente
  intacto para afirmar capacidad de generalización sin sesgo por
  selección de modelos.
- La suma de ventanas superpuestas produce una estimación condicionada
  a un intervalo con una MPB conocida. No es una medida de precisión
  o sensibilidad de detección orbital.
- SWEA es un promedio angular de flujo medido, no una estimación
  calibrada independiente de flujo omnidireccional.
