# Quinto entrenamiento MAG: encoder residual profundo + caché local

## Objetivo científico

El cuarto modelo utilizó un encoder CNN dilatado con 16 canales ocultos,
2 canales latentes y un detector que compara la representación del campo
antes y después de cada centro de ventana. El quinto modelo conserva el
detector de transiciones, las etiquetas, las pérdidas y el protocolo de
validación, pero incrementa **únicamente la capacidad del encoder**.

| Parámetro | Modelo 4 | Modelo 5 |
| --- | --- | --- |
| Encoder | CNN dilatada | CNN temporal residual dilatada |
| Canales ocultos | 16 | 32 |
| Canales latentes | 2 | 8 |
| Capas dilatadas | 5 | 7 |
| Dilataciones | 2,4,8,16,32 | 1,2,4,8,16,32,64 |
| Campo receptivo teórico | 255 muestras | 515 muestras |
| Detector | Centro, antes, después, diferencia | **Igual** |
| Salida supervisada | Un puntaje por ventana | **Igual** |
| Decoder | CNN de 2 capas | Misma topología (adaptada a canales) |

El campo receptivo está expresado en **muestras** de MAG, no segundos
exactos; a cadencia ~1 Hz es aproximadamente 515 s. Los siete bloques
convolucionales tienen conexiones residuales para facilitar la propagación
de gradientes. La cantidad de parámetros entrenables se imprime en el log
y se guarda por fold para documentar el aumento de complejidad.

Se mantiene:
- MAG real: Bx, By, Bz, |B|, ventanas de 600 s a cadencia nativa;
- los 569 eventos del catálogo, CV externa de cuatro folds agrupados por mes;
- normalizadores calculados solamente con datos de los folds TRAIN;
- 600 centros por evento, 24 ventanas rotativas/evento/época;
- etiqueta gaussiana de desviación estándar 60 s;
- pérdida `L_rec + 10 L_det + L_mag` (regularización física del módulo);
- máximo de 1000 épocas, mínimo 25, paciencia de early stopping 20;
- selección del mejor checkpoint según pérdida de detección
  del subconjunto de meses TRAIN reservado como monitor;
- outer-validation sin intervención en early stopping.

**No** se implementa todavía una salida por instante temporal: eso
correspondería al sexto cambio/experimento y no debe mezclarse con este.

## Caché local de bloques MAG originales

La quinta versión incorpora `src/data/mag_event_cache.py`. En el primer
entrenamiento que use la caché se consultarán los eventos que falten
en BigQuery; **cada evento se guarda inmediatamente después de su descarga**,
en un archivo NPZ comprimido.

Ubicación por defecto, **excluida de Git**:
`data/cache/mag_events/`.

Cada archivo almacena exclusivamente:
- timestamp UTC original como Unix seconds float64;
- Bx, By, Bz y |B| originales en nT float64;
- metadatos de procedencia: tabla, proyecto, hash SQL, centro y límites;
- cantidad de muestras y SHA-256 de los arrays.

No se almacena normalización, etiquetas, divisiones TRAIN/VAL ni ventanas
derivadas. Así, todos los folds normalizan los mismos datos crudos con
sus propias estadísticas y no hay fuga de información por la caché.

Cuando se reanuda o se lanza otro entrenamiento con los mismos intervalos,
el programa verifica la especificación, las formas, los timestamps y el
checksum antes de reutilizar cada archivo. Si hay un archivo corrupto,
**se detiene** y exige refrescar la caché; no usa silenciosamente datos
inconsistentes. Si falta un evento, se hace dry-run y consulta **solo ese
evento**, dentro de los límites de cuota. Si están los 569 bloques válidos,
**no se realizan consultas ni dry-runs de BigQuery**.

Los archivos sobreviven a apagar o reiniciar la computadora. Es una
**instantánea local**; si se actualiza la tabla de BigQuery, los datos
cacheados no se invalidan automáticamente. Para reconsultar y actualizar
todos los bloques, usar `REFRESH_MAG_CACHE=1` cuando corresponda.

El cuarto entrenamiento ya en curso **no** estaba guardando sus bloques
a disco y no se puede extraer su RAM al quinto proceso. Por lo tanto la
primera ejecución de la quinta versión todavía tendrá que descargar los
datos que falten; el ahorro principal aparece en las ejecuciones futuras.

## Ejecutar

Desde la raíz del repositorio, una vez terminado el cuarto entrenamiento
para no competir por recursos de CPU, memoria y BigQuery:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate
tmux new -s mag_cv_residual
bash scripts/5_training/run_mag_cv_residual.sh
```

El launcher ejecuta **todas las pruebas offline y comprobaciones de sintaxis**
antes de consultar BigQuery, y entrena los cuatro folds completos usando
mediciones reales. Si se interrumpe, guarda checkpoints por época y
mantiene en disco cada evento ya descargado.

Salida exclusiva:
`results/metrics/mag_cv_residual_transition_v1/`.

Verificar actividad y los eventos que provienen de caché:

```bash
tail -f results/metrics/mag_cv_residual_transition_v1/training.log
```

Se imprimirá `MAG cache: X/569 validated events available; Y require BigQuery`
(al comenzar una ejecución completa sin folds terminados).

Opciones opcionales de entorno, sin editar el código:

```bash
# Otra ubicación local persistente:
MAG_CACHE_DIR=/home/cami/maven_cache bash scripts/5_training/run_mag_cv_residual.sh

# Fuerza recarga de los datos desde BigQuery, respetando los límites:
REFRESH_MAG_CACHE=1 bash scripts/5_training/run_mag_cv_residual.sh
```

No es necesario incrementar la cuota de BigQuery para reutilizar archivos
locales ya validados. Los límites del código siguen protegiendo cualquier
consulta aún necesaria.

## Análisis y comparación con el cuarto

Los scripts existentes de figuras aceptan los checkpoints nuevos:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_residual_transition_v1/cv_summary.json \
  --compare-to results/metrics/mag_cv_transition_v1/cv_summary.json

python scripts/6_figures/6b_plot_mag_reconstruction_cases.py \
  --results-dir results/metrics/mag_cv_residual_transition_v1 --execute
```

La comparación tiene sentido porque mantiene las mismas particiones,
etiquetas, 600 centros de evaluación y criterio de localización temporal.
Se registran mediana de error por fold, fracción localizada en ≤60 s,
curvas de detección y reconstrucción física. Las métricas de localización
están condicionadas a eventos MPB conocidos: no constituyen precision,
recall o F1 en órbitas sin cruces.

## Compatibilidad y precauciones

- El script del cuarto experimento sigue siendo
  `scripts/5_training/run_mag_cv_transition.sh`, sin cambios.
- La arquitectura nueva se activa solamente con
  `--architecture residual-transition`.
- Los checkpoints y firma de reanudación de las arquitecturas anteriores
  siguen siendo compatibles porque el uso de caché solo modifica I/O.
- El quinto usa un **directorio nuevo**; no reutilizar checkpoints de otros
  modelos con arquitectura diferente.
- Una red más profunda podría sobreajustar. El aumento de capacidad es
  una hipótesis a evaluar empíricamente, no una garantía de mejora.

## Análisis completo posterior al quinto entrenamiento

Una vez completados los cuatro folds, ejecutar (no reentrena y no usa
BigQuery en el primer comando):

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_residual_transition_v1/cv_summary.json \
  --compare-to results/metrics/mag_cv_transition_v1/cv_summary.json

python scripts/6_figures/6b_plot_mag_reconstruction_cases.py \
  --results-dir results/metrics/mag_cv_residual_transition_v1 --execute
```

La primera línea compara exactamente los mismos eventos y genera un informe
emparejado en PNG, JSON y CSV bajo `mag_cv_residual_transition_v1/figures/`
(con prefijo `mag_cv_paired_fourth_vs_fifth`). El JSON cuantifica mejoras,
empeoramientos, eventos ganados y perdidos al exigir ≤60 s, y frecuencia de
picos cerca de ±30 min. Si faltan eventos o cambia el número de ventanas
por MPB, se aborta en lugar de comparar conjuntos distintos.

El script de reconstrucción reutiliza por defecto el caché raw MAG de
`data/cache/mag_events/` generado al entrenar el quinto, evitando BigQuery
si ambos intervalos de 70 minutos se encuentran completos y validados.
Para cambiar la ubicación, pasar `--mag-cache-dir RUTA`.

Enviar para revisión las figuras resultantes y el JSON emparejado: así se
podrán interpretar las estadísticas completas y no solamente el fold 1.
