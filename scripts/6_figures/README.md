# 6. Figuras científicas

Esta carpeta contiene scripts de figuras reproducibles. La figura 6a usa
exclusivamente las métricas guardadas; la figura 6b también debe consultar
dos intervalos de mediciones reales en BigQuery, ya que las señales MAG
no se guardaron en el entrenamiento. Mantener separado:

- `scripts/5_training/`: entrenamiento, validación y exportación de métricas.
- `scripts/6_figures/`: lectura de resultados y generación de figuras.
- `results/metrics/<experimento>/figures/`: imágenes finales.

## 6a — Resumen, evolución de pérdidas y errores del entrenamiento MAG (tercero por defecto)

Desde la **raíz del repositorio**, con `.venv` activado:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py
```

Entrada por defecto:
`results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/cv_summary.json`

Salida por defecto:
- `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/mag_cv_third_training.png`
- `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/mag_cv_training_losses.png`
- `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/mag_cv_error_distribution.png`
- `results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/mag_cv_second_vs_third.png` (si hay resumen del segundo entrenamiento)

El mismo script reproduce el primer, segundo o cuarto entrenamiento indicando
el `--summary` correspondiente. Todas las imágenes se guardan junto a las
métricas del experimento elegido; no se sobrescriben figuras de otras carpetas.

De forma predeterminada produce **tres figuras**, más una comparación
con el segundo entrenamiento si sus métricas siguen disponibles:
1. Resumen por fold: mediana del error y porcentaje de MPB con error ≤60 s
2. Curvas `train_rec/val_rec`, `train_det/val_det` y `train_mag/val_mag`
   para cada uno de los cuatro folds; las curvas `val_*` representan el
   **holdout interno por meses** de early stopping, NO la evaluación externa
3. Histograma de errores absolutos sobre los **eventos de validación externa**
   y curva acumulada (total y por fold), con mediana global **exacta**
   calculada a partir de los CSV individuales
4. Comparación (si existe el `cv_summary.json` del segundo experimento):
   `mag_cv_second_vs_third.png`, con error mediano y fracción ≤60 s
   por fold en ambos entrenamientos. Los dos evaluaron 600 centros
   temporales por evento, por lo que la resolución es comparable.

Para generar solamente la figura resumen, usar `--summary-only`.
Para desactivar la comparación automática, usar `--no-comparison`;
`--compare-to` permite indicar otro resumen.

La figura contiene dos paneles: error mediano de localización (minutos)
y porcentaje de eventos localizados con error absoluto ≤60 segundos por fold.
El resumen global agrega los recuentos de eventos de los cuatro folds.

El script lee números reales de `cv_summary.json`, los cuatro archivos
`fold_XX_metrics.json` y los cuatro CSV de errores por evento. No
reentrena, no consulta BigQuery ni altera checkpoints. Las métricas de
localización **no son precision/recall/F1 ni probabilidades calibradas**.

Podés personalizar las rutas:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_full/cv_summary.json \
  --output results/metrics/mag_cv_full/figures/mag_cv_first_training.png
```


## 6b — Reconstrucción y puntuación de dos cruces MAG (acierto/error)

Este diagnóstico requiere los **checkpoints finales entrenados** de MAG,
los 4 archivos `fold_XX_event_metrics.csv` y los normalizadores del fold.

Selecciona un cruce correctamente localizado (error ≤60 s) y otro mal
localizado (error ≥600 s). Escoge casos cercanos a la mediana de cada
subgrupo para evitar seleccionar solo los extremos más espectaculares.
Por defecto, ambos casos son del **fold 1**, para comparar la reconstrucción
del mismo modelo y con el mismo normalizador. Opcionalmente se puede cambiar
el fold; con `--fold all`, los casos podrían pertenecer a modelos distintos.

La red MAG se entrenó para ventanas de **600 s**, no de 70 minutos.
El script recupera 70 minutos de datos reales alrededor de cada etiqueta,
reproduce las **600 ventanas de 600 s** de la evaluación externa por evento
en los entrenamientos segundo, tercero y cuarto (121 en el primero),
calcula la reconstrucción en nT en cada una y promedia las porciones
solapadas en los mismos instantes nativos. Los **centros** evaluables
están aproximadamente entre −30 y +30 min porque cada ventana
necesita 5 min a cada lado; los datos observados abarcan ±35 min. Si no hay ventanas válidas,
quedan huecos sin interpolar. **No se vuelve a entrenar.**

La figura tiene dos columnas (caso bueno y malo) y cinco filas:
- 4 paneles de **MAG observado vs reconstruido**: Bx, By, Bz, |B|
  durante los 70 minutos. En el panel de |B| se superpone, con línea
  discontinua, `sqrt(Bx_rec^2 + By_rec^2 + Bz_rec^2)` para comprobar
  consistencia física;
- abajo, **puntaje MPB del detector** frente al tiempo del centro
  de ventana, junto con el objetivo gaussiano de entrenamiento.

Bandas verdes = las ventanas de 10 min centradas en la **MPB del catálogo**;
bandas naranjas = las ventanas de 10 min centradas en el **pico del modelo**.
Las bandas NO indican el espesor físico de la MPB. El puntaje entre 0 y 1
NO es una probabilidad calibrada.

**Por qué requiere BigQuery:** el entrenamiento anterior guardó pesos y
métricas, pero NO las series temporales físicas ni sus reconstrucciones.
Por lo tanto son necesarias dos consultas acotadas a las 70 min elegidas.
Por defecto el script hace solo estimaciones (*dry-run*).

Desde la raíz del repositorio, con `.venv` activo:

```bash
# Estimar cuánto procesarían dos consultas de 70 min (sin recuperar datos):
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py

# Hacer las dos consultas reales, inferir las 600 ventanas por evento y graficar:
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py --execute
```

Salida por defecto:
`results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000/figures/mag_cv_reconstruction_good_vs_bad.png`

Se genera junto con la figura un JSON de trazabilidad con el error
temporal y tres errores RMS en **nT**: del canal reconstruido |B| respecto al
observado, de la norma xyz reconstruida respecto al |B| observado, y de
la diferencia entre ambos módulos reconstruidos. Todos se calculan
solo en muestras observadas cubiertas por reconstrucción.

El script lee automáticamente del `cv_summary.json` la cantidad de
centros de evaluación y `sigma`, para que las predicciones coincidan
con los errores usados al seleccionar los casos. El uso del primer
entrenamiento sigue disponible con:
```bash
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py \
  --results-dir results/metrics/mag_cv_full --execute
```
En ese caso se generan las figuras en `mag_cv_full/figures/`.

Se añade un JSON pequeño con fechas/folds y errores para trazabilidad,
pero **no se guardan archivos con mediciones MAG sin procesar**.

Por defecto se usa `--fold 1`. Si no hay eventos de ambos grupos
(acierto/error) en ese fold, usar `--fold 2`, `--fold 3` o
`--fold all`; esta última opción puede seleccionar modelos diferentes
para cada caso. Los parámetros permiten elegir entre todos
los folds. `--max-total-gib` y `--max-query-gib` ajustan los límites
de las consultas. Si una puntuación recalculada difiere de la registrada
en el CSV por más de 2 s de desplazamiento, el programa se detiene
en lugar de presentar un caso potencialmente mal etiquetado.

## Quinto MAG residual-transition (opcional)

Los dos scripts también aceptan el quinto experimento: `6a` compara sus
métricas con el cuarto sobre la misma CV, y `6b` carga los pesos de su
encoder residual y el detector asimétrico para graficar eventos reales.
Cuando termine el quinto:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_residual_transition_v1/cv_summary.json \
  --compare-to results/metrics/mag_cv_transition_v1/cv_summary.json

python scripts/6_figures/6b_plot_mag_reconstruction_cases.py \
  --results-dir results/metrics/mag_cv_residual_transition_v1 --execute
```
