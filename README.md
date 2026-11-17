# MAVEN–MPB: localización y espacio latente (V14-5D)

Proyecto final de **Machine Learning en Geofluidos** (DCAO, Universidad de Buenos Aires). Se estudian cruces de la *Magnetic Pileup Boundary* (MPB) de Marte mediante mediciones de MAVEN MAG y SWEA.

## Estado

**Versión de trabajo:** V14 con dos autoencoders CNN de ocho convoluciones y cuello de botella de **cinco coordenadas escalares por instrumento**. El entrenamiento completo puede estar en curso; **no se atribuyen a V14 resultados globales no calculados**. Para el piloto anterior de 3D, consultar el historial de Git; no forma parte de esta versión.

El modelo combina:
- **Búsqueda gruesa congelada V7b:** puntúa ventanas MAG+SWEA en un intervalo de ±30 min alrededor de una MPB catalogada, con centros separados típicamente 32 s.
- **Nuevo refinador V14:** MAG (Bx, By, Bz, |B|) y SWEA (12 canales espectrales, 30–300 eV) conservan sus cadencias originales. Cada instrumento pasa por ocho convoluciones hasta un vector de cinco componentes; sendos decoders reconstruyen las observaciones.
- **Localización temporal:** las diez coordenadas latentes, el score V7b y dos logits de región entran en la cabeza temporal. La corrección local se limita a ±240 s y su escala se calibra exclusivamente en meses MONITOR internos.
- **Supervisión:** BCE de presencia, KL temporal, pérdida SmoothL1 de desplazamiento y dos MSE de reconstrucción.

La pérdida y la calibración están detalladas en [V14 5D](docs/v14_dual_5d_refiner.md).

## Datos y validación

Las anotaciones curadas están en `data/labels/`; el manifiesto mensual de cuatro folds está en `data/splits/cv4_monthly_manifest.csv`. Se utiliza el mismo esquema de partición temporal que en V7b/V10; la normalización corresponde al TRAIN del fold respectivo. Las ventanas tienen diez minutos. Las series MAG y SWEA se leen de caché local, **sin interpolar SWEA en el tiempo**.

La evaluación reporta el error absoluto de localización frente a la anotación, la mediana y la fracción de eventos con error **≤60 s** y **≤120 s**, entre otras métricas. **Es localización condicionada a una vecindad con MPB conocida, no detección autónoma validada sobre órbitas completas.** Las coordenadas latentes entre folds no son directamente comparables sin alineación.

## Ejecutar en WSL (CPU)

Requiere Python y dependencias de `requirements.txt`; la caché científica local y los checkpoints V7b del fold correspondiente deben existir en las rutas esperadas. Esos archivos de gran tamaño se excluyen de Git.

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_v14_dual_bottleneck.py' -v

# Piloto independiente
PILOT_EVENTS_PER_FOLD=8 FOLDS=1 bash scripts/5_training/run_v14_dual_5d.sh

# Entrenamiento completo de los cuatro folds
tmux new -s mpb_v14_5d
PILOT_EVENTS_PER_FOLD=0 FOLDS=all bash scripts/5_training/run_v14_dual_5d.sh
```

El lanzador hace comprobaciones y aborta si faltan datos en caché, antes de ejecutar consultas a BigQuery. `--resume` permite recuperar checkpoints compatibles. **Si el entrenamiento está corriendo en WSL, no hacer `git pull` ni borrar scripts en esa máquina hasta que finalice.**

Las salidas locales completas quedan en `results/metrics/v14_dual_5d_refiner/`. El piloto se guarda por separado en `results/metrics/v14_dual_5d_refiner_pilot/`.

## Figuras y espacio latente

```bash
python scripts/6_latent_analysis/6c_plot_v14_latents_5d.py
```

Se grafican, **por fold**, proyecciones PCA 2D/3D de los vectores MAG y SWEA de cinco dimensiones, el error temporal y la evolución de las cinco coordenadas originales en ventanas retrospectivamente alineadas con la MPB.

El exportador guarda `mag_z1...mag_z5`, `swea_z1...swea_z5` en `fold_XX_latent_embeddings.csv`. El uso de ventanas centradas en el tiempo etiquetado es solo **análisis retrospectivo**, nunca una entrada del predictor.

## Estructura esencial

| Ruta | Función |
| --- | --- |
| `src/detection/mpb_v14_dual_bottleneck_refiner.py` | Arquitectura V14 |
| `scripts/5_training/5n_train_v14_dual_bottleneck.py` | Entrenamiento, calibración, CV y exportación |
| `scripts/5_training/run_v14_dual_5d.sh` | Lanzador principal |
| `scripts/6_latent_analysis/6c_plot_v14_latents_5d.py` | Figuras latentes |
| `tests/test_v14_dual_bottleneck.py` | Pruebas del modelo |
| `scripts/5_training/5a_*, 5b_*, 5e_*, 5f_*, 5l_*` | **Dependencias todavía importadas por V14** para splits, caché, ventanas, pérdida y evaluación; no son nuevos experimentos a ejecutar |
| `docs/` | Antecedentes metodológicos V7b/V10 y documentación vigente V14 |
| `data/labels/, data/splits/` | Catálogo y partición temporal |

Los notebooks exploratorios anteriores se retiraron. Se creará **un notebook final nuevo, desde cero**, una vez que estén disponibles los resultados de V14. Git conserva el historial de los archivos eliminados, pero no estarán en la rama principal.
