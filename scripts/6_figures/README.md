# 6. Figuras científicas

Esta carpeta contiene scripts que generan figuras reproducibles a partir de
**resultados ya calculados**, no de consultas a BigQuery. Mantener separado:

- `scripts/5_training/`: entrenamiento, validación y exportación de métricas.
- `scripts/6_figures/`: lectura de resultados y generación de figuras.
- `results/metrics/<experimento>/figures/`: imágenes finales.

## 6a — Resultados del primer entrenamiento MAG

Desde la **raíz del repositorio**, con `.venv` activado:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py
```

Entrada:
`results/metrics/mag_cv_full/cv_summary.json`

Salida:
`results/metrics/mag_cv_full/figures/mag_cv_first_training.png`

La figura contiene dos paneles: error mediano de localización (minutos)
y porcentaje de eventos localizados con error absoluto ≤60 segundos por fold.
El resumen global agrega los recuentos de eventos de los cuatro folds.

El script lee los números reales del JSON, **no tiene cifras fijas**.
No recomputa las redes, no importa PyTorch, no consulta BigQuery ni altera
los checkpoints. La figura no representa precision/recall ni la mediana
global del error (esta última requiere leer los CSV por evento).

Podés personalizar las rutas:

```bash
python scripts/6_figures/6a_plot_mag_cv_summary.py \
  --summary results/metrics/mag_cv_full/cv_summary.json \
  --output results/metrics/mag_cv_full/figures/otro_nombre.png
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
reproduce las **121 ventanas de 600 s** de la evaluación externa por evento,
calcula la reconstrucción en nT en cada una y promedia las porciones
solapadas en los mismos instantes nativos. Los **centros** evaluables
están aproximadamente entre −30 y +30 min porque cada ventana
necesita 5 min a cada lado; los datos observados abarcan ±35 min. Si no hay ventanas válidas,
quedan huecos sin interpolar. **No se vuelve a entrenar.**

La figura tiene dos columnas (caso bueno y malo) y cinco filas:
- 4 paneles de **MAG observado vs reconstruido**: Bx, By, Bz, |B|
  durante los 70 minutos;
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

# Hacer las dos consultas reales, inferir las 121 ventanas por evento y graficar:
python scripts/6_figures/6b_plot_mag_reconstruction_cases.py --execute
```

Salida:
`results/metrics/mag_cv_full/figures/mag_cv_reconstruction_good_vs_bad.png`

Se añade un JSON pequeño con fechas/folds y errores para trazabilidad,
pero **no se guardan archivos con mediciones MAG sin procesar**.

Por defecto se usa `--fold 1`. Opcional: `--fold 2` o `--fold 3`
para otro modelo específico, o `--fold all` para elegir entre todos
los folds. `--max-total-gib` y `--max-query-gib` ajustan los límites
de las consultas. Si una puntuación recalculada difiere de la registrada
en el CSV por más de 2 s de desplazamiento, el programa se detiene
en lugar de presentar un caso potencialmente mal etiquetado.
