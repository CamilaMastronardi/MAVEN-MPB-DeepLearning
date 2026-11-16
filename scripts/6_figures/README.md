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
