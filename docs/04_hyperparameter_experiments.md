# 04 · Barridos controlados de hiperparámetros V7b

Los tres barridos mantienen los mismos folds, seeds y fuentes; modifican un único coeficiente por prueba.

| Etapa | Variable | Valores | Configuración fija |
| --- | --- | --- | --- |
| 01 | Peso de ranking | 0.5, 1, 2, 4 | Margen 0.5, regiones 0.5 |
| 02 | Margen de ranking | 0.25, 0.5, 1, 2 | Peso ranking 2, regiones 0.5 |
| 03 | Peso de regiones | 0.25, 0.5, 1, 2 | Peso ranking 2, margen 0.25 |

Desde la raíz:

```bash
bash scripts/03_experiments/01_run_rank_weight.sh
bash scripts/03_experiments/02_run_rank_margin.sh
bash scripts/03_experiments/03_run_region_weight.sh
```

Los lanzadores entrenan secuencialmente, con `--resume`, y después invocan los comparadores:
`01_compare_rank_weight.py`, `02_compare_rank_margin.py` y `03_compare_region_weight.py`.

Carpetas locales preservadas:
- `results/metrics/v7b_ranking_weight_sweep/`
- `results/metrics/v7b_ranking_margin_sweep/`
- `results/metrics/v7b_region_weight_sweep/`

El barrido de margen reutiliza los cuatro folds de peso ranking 2 y margen 0.5; el barrido de regiones reutiliza los cuatro folds de ranking 2 y margen 0.25. Las reutilizaciones se hacen con enlaces simbólicos, después de validar configuración y archivos; no se eliminan checkpoints.

La selección de un coeficiente dentro de cada fold usa la exactitud top-1 de ranking del **monitor interno** (desempate con exactitud por pares). Las métricas de validación externa se calculan sobre eventos idénticos y no se utilizan dentro del algoritmo de selección por monitor.

**Limitación:** la elección previa de ranking 2 y margen 0.25 fue exploratoria sobre los mismos folds. Esos resultados no deben presentarse como desempeño sobre un test completamente independiente.
