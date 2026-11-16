# Séptima prueba: detección de MPB como transición de regiones MAG+SWEA

## Motivación

Los modelos 1--6 obtuvieron buenas reconstrucciones de MAG, pero errores
de localización elevados. La quinta versión (referencia más fuerte) obtuvo
mediana global ~369.5 s y 148/569 cruces dentro de 60 s; la sexta con
salida temporal no mejoró en fold 1. Ya no se intenta aproximar una función
gaussiana de cercanía a la etiqueta como objetivo principal.

V7 reformula el problema como **clasificación multifuente de estados
plasmáticos y de sus transiciones**. La física entra por la estructura del
modelo y por el diseño conservador de las etiquetas, no porque se imponga
una regla sobre las predicciones.

## Arquitectura

- Entrada MAG: [Bx, By, Bz, |B|] a su **cadencia nativa** (~1 s).
- Entrada SWEA: flujo medio angular en energías físicamente seleccionadas,
  con una grilla fija de **12 energías logarítmicas de 30 a 300 eV**.
  Se extraen canales medidos entre 25 y 330 eV y se interpola
  **únicamente en energía logarítmica**, dentro de la cobertura medida;
  los datos sin suficiente cobertura se descartan. No se interpola SWEA
  en tiempo, ni se finge una cadencia 1 Hz.
- Dos encoders CNN 1D independientes, con bloques residuales.
- Dos decoders reconstruyen MAG y el espectro SWEA como auxiliares.
- Las características latentes anteriores (< -60 s) y posteriores
  (> +60 s) se promedian por separado en **cada instrumento**.
- Las características MAG y SWEA se fusionan y alimentan:
  1. un clasificador de **región** por lado (magnetosheath=0,
     magnetosfera inducida=1);
  2. un clasificador **binario de MPB** para el instante candidato,
     basado en regiones izquierda/derecha, sus diferencias y magnitudes.

La red no recibe tiempos MPB/BS, fecha absoluta, grupos del catálogo,
orbita o posición como entrada, para evitar depender de la anotación.

`L = BCE_transicion + 0.5 BCE_region + 0.15 (MSE_MAG + MSE_SWEA)`

Se pueden ajustar los multiplicadores mediante flags, pero cualquier
optimización debe usar **solo meses de TRAIN del fold**.

## Supervisión débil desde MPB + BS

El archivo `data/splits/cv4_monthly_manifest.csv` contiene los 569
cruces MPB, tiempos BS asociados y cuatro folds por meses. 559 filas
tienen marcado SWEA disponible; la disponibilidad declarada no garantiza
que los bloques ya estén cargados en BigQuery. V7 no inventa mediciones.

La posición temporal relativa BS/MPB aporta orientación tentativa:

- **BS anterior a MPB:** inbound, magnetosheath antes de MPB,
  magnetosfera inducida después.
- **MPB anterior a BS:** outbound, magnetosfera inducida antes,
  magnetosheath después.

Este criterio es una **etiqueta débil**, no una verificación física
independiente. Se descartan asociaciones BS con separación <5 min o
>2 h. No se conoce espesor de MPB a partir de una sola etiqueta.

Ejemplos positivos: candidatos a ±30 y 0 s de MPB con contexto físico
seguro de ±5 min y BS suficientemente alejado. Negativos aproximados:
instantes ±10, 15, 20 o 25 min del tiempo MPB, SOLO si el contexto está
lejos de todas las MPB y BS catalogadas; la región magnetosheath debe
permanecer entre BS y MPB, nunca extrapolarse al viento solar.

**No se etiquetan las regiones ambiguas**, cruces cercanos a BS, ventanas
incompletas con gaps, o vecindades de otras MPB. Una etiqueta 0 significa
un control seguro con respecto al catálogo, no ausencia demostrada de
toda frontera en la realidad.

## Preprocesamiento y cache

Se mantienen ventanas candidatas de 600 s, pero cada instrumento conserva
sus timestamps y muestras físicas. Se exige cobertura a ambos lados,
cadencia creciente, ausencia de gaps grandes y valores finitos.

Los datos se obtienen de `maven.mag_fields`,
`maven.swea_svy3d` y `maven.swea_svy3d_axes` en BigQuery.
Se reutiliza `data/cache/mag_events/` del quinto entrenamiento.
Se añade `data/cache/swea_events/` con matrices espectrales reales,
transformadas `log1p`, grilla física y checksums. Esta transformación
fija no depende del fold y la cache no almacena etiquetas ni normalizadores.

**Medias y desvíos** de cada canal MAG/SWEA se estiman únicamente con
épocas únicas de los meses fit-TRAIN; nunca con el monitor interno ni
la validación externa. Se normaliza antes de cada batch.

Se ejecutan dry-runs de BigQuery **solo para los bloques no cacheados**,
con límites por consulta y total ANTES de hacer descargas. Si faltan
observaciones SWEA necesarias, el script no genera señales ficticias:
excluye clips sin datos o detiene el fold cuando no puede evaluarlo.

## Entrenamiento y validación

- Cuatro folds externos idénticos al esquema anterior, pero restringidos
  a filas con MAG y SWEA indicados como disponibles.
- Dentro de cada outer TRAIN, se reservan meses para el monitor interno,
  conservando exclusión de soportes superpuestos.
- En cada época se eligen por evento hasta **2 positivos y 2 controles**,
  con balance aproximado por evento.
- Entrenamiento en CPU (por defecto 2 threads), batch 16.
- **Early stopping** desde época 10, paciencia 8, máximo 100.
  Se elige el menor `BCE_transition_monitor + 0.5 BCE_region_monitor`,
  no el resultado del fold de validación.
- Umbral de clasificación de candidatos elegido únicamente en el monitor
  interno, maximizando F1 de las etiquetas débiles.

Evaluación externa:

1. Para cada MPB se exploran candidatos en ±30 min con salto físico
   de 32 s, con sus clips MAG/SWEA nativos.
2. Se conserva el candidato con mayor score de transición y se mide
   el error respecto del tiempo catalogado (mediana, <=60s, <=120s,
   <=300s).
3. Adicionalmente se evalúa BCE/precisión/recall/F1 **de candidatos con
   etiquetas débiles en los mismos vecindarios** y accuracy de región.

**Importante:** la precisión/recall/F1 de candidatos **NO es precision,
recall o F1 para búsqueda autónoma en órbitas completas**. Los controles
pueden contener fronteras no catalogadas; se necesita un conjunto de
negativos orbitales revisados para medir ese desempeño.

La salida de v7 tiene resolución de 32 s por defecto. Al compararla
con v5, emparejar solo los mismos eventos evaluables: v5 incluye hasta
569 MPB y v7 tiene como máximo 559 por la cobertura SWEA.

## Ejecutar

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

# Para ejecución completa en sesión separada:
tmux new -s mpb_v7
bash scripts/5_training/run_mag_swea_regions_v7.sh
```

El launcher ejecuta `py_compile` y todos los tests offline, luego prepara
caches, preflight BigQuery, entrenamiento y evaluación.
La nueva ruta de resultados es:

`results/metrics/mag_swea_regions_v1/`

Archivos relevantes: `cv_summary.json`,
`fold_XX_metrics.json`, `fold_XX_event_metrics.csv`,
`fold_XX_best.pt`, `fold_XX_final.pt` y `training.log`.

**Piloto opcional** (resultado NO comparable con cuatro folds completos):

```bash
PILOT_EVENTS_PER_FOLD=8 OUTPUT_DIR=results/metrics/mag_swea_regions_pilot \
  bash scripts/5_training/run_mag_swea_regions_v7.sh
```

Si una consulta excede cuotas, NO desactivar los controles sin revisar
primero los bytes estimados de BigQuery. Si los datos SWEA aún no están
cargados para un evento, tampoco se debe modificar el código para
rellenarlos con números artificiales.

Para forzar refresco de todos los SWEA desde BigQuery:
`REFRESH_SWEA_CACHE=1` (solo si hubo cambios reales en la fuente).

## Figuras al finalizar

```bash
python scripts/7_figures/7a_plot_mag_swea_results.py \
  --results-dir results/metrics/mag_swea_regions_v1 \
  --compare-v5 results/metrics/mag_cv_residual_transition_v1
```

Genera figuras de métricas por fold, curvas TRAIN/monitor,
distribución global y comparación emparejada con v5. Los paneles de F1
indican explícitamente que se trata de clasificación de candidatos
con etiquetas débiles, no detección orbital independiente.

## Advertencias y criterios de validación

- La interpretación de inbound/outbound a partir de BS/MPB debe
  revisarse en casos particulares; ya se realizó una revisión visual
  general, pero no se debe asumir que todas las 559 etiquetas son perfectas.
- Una caída de error respecto de v5 es una evidencia útil, no una prueba
  de que la red detecta MPB desconocidas en toda la órbita.
- Se modificaron simultáneamente **instrumentos, etiquetas, arquitectura,
  pérdida y muestreo**. La comparación V5-V7 mide su efecto conjunto;
  ninguna mejora puede atribuirse causalmente a SWEA sin una ablación
  MAG-only del clasificador de regiones.
- Los modelos v1-v6 y sus checkpoints no se modifican.
