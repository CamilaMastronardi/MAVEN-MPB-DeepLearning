# V14 (revisión 5D) — cuello de botella de cinco dimensiones por instrumento

## Motivo

El piloto inicial V14 con dos latentes 3D registró para fold 1
**8/8 eventos**, mediana **80.4 s** y **37.5 %** con error <=60 s
(alpha=0.80). Es una muestra piloto muy pequeña y no puede
compararse directamente con las métricas del fold completo V10/HP1:
no prueba una degradación global. Se experimenta con **5 coordenadas
escalares** por sensor para aumentar la capacidad de representación.

El piloto 3D anterior queda preservado en
`results/metrics/v14_dual_3d_refiner_pilot/` y NO se reanuda con
la nueva arquitectura.

## Qué cambió

- MAG: 4 canales observados → 8 Conv1d → **5 coordenadas z1–z5**
  → decoder de ocho convoluciones que reconstruye Bx,By,Bz,|B|.
- SWEA: 12 canales energéticos (30–300 eV) → 8 Conv1d
  → **5 coordenadas z1–z5** → decoder de ocho convoluciones.
- Los tensores de entrada siguen con su cadencia temporal propia.
  No se interpola SWEA temporalmente.
- La cabeza de refinamiento usa **13** características fusionadas:
  5 MAG + 5 SWEA + 1 logit de transición V7b + 2 logits de región V7b.
- V7b se mantiene **congelado y sin modificaciones** como
  buscador de candidatos en ±30 min.
- La corrección de tiempo sigue representada sobre MAG dentro
  de ±250 s. Se calibran alpha y método con meses MONITOR internos.
- No se cambian los hiperparámetros de entrenamiento de V14:
  `learning_rate=1e-4`, 150 épocas máximas, paciencia 20,
  sigma temporal 24 s, pesos KL 3.0, offset 1.5,
  reconstrucción 0.2, seis positivos/tres controles por evento.
- Los modelos V13 3D se conservaron **intactos**.

Se mantienen los objetivos de pérdida

```
L = BCE + 3 * KL_temporal(sigma=24s)
       + 1.5 * SmoothL1_offset
       + 0.2 * (MSE_rec_MAG + MSE_rec_SWEA)
```

Esta revisión sigue siendo un **nuevo entrenamiento**, no un
ajuste de pesos de un checkpoint 3D. La ventaja de cinco
dimensiones para localización y física está por comprobarse.

## Ejecutar piloto nuevo en WSL

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

PILOT_EVENTS_PER_FOLD=8 FOLDS=1 \
bash scripts/5_training/run_v14_dual_5d.sh
```

El lanzador comprueba pruebas unitarias, caché MAG+SWEA
y ejecuta el entrenamiento en CPU. Aborta antes de BigQuery
si el preflight de caché no está completo. Salidas del piloto:

```
results/metrics/v14_dual_5d_refiner_pilot/
  fold_01_best.pt
  fold_01_last.pt
  fold_01_final.pt
  fold_01_metrics.json
  fold_01_event_metrics.csv
  fold_01_latent_embeddings.csv
  cv_summary.json
  training.log
```

Una vez verificado el piloto, para los cuatro folds completos:

```bash
tmux new -s mpb_v14_5d
bash scripts/5_training/run_v14_dual_5d.sh
```

Salida del entrenamiento completo:
`results/metrics/v14_dual_5d_refiner/`.

## Espacio latente y figuras

Cada evento exporta nueve vistas del catálogo y otra
centrada en la predicción. CSV por fold con
`mag_z1,...,mag_z5` y `swea_z1,...,swea_z5`.
No interpretar vistas centradas en el tiempo verdadero
como localización autónoma: el catálogo se utiliza para
seleccionar/ordenar esas vistas a posteriori.

Para producir figuras **del piloto**:

```bash
python scripts/6_latent_analysis/6c_plot_v14_latents_5d.py \
  --results-dir results/metrics/v14_dual_5d_refiner_pilot
```

Para los folds completos:

```bash
python scripts/6_latent_analysis/6c_plot_v14_latents_5d.py
```

Figuras: dispersión PCA 2D y 3D (PCA de las cinco coordenadas
ajustada descriptivamente a las vistas predicted_centered del
fold), PCA coloreada por error, y evolución temporal de
**las cinco coordenadas originales** con mediana e IQR.

Los vectores aprendidos por diferentes folds no comparten
coordenadas compatibles. No se deben combinar en un mismo
espacio PCA sin un procedimiento de alineación.

## Advertencias

Se evalúa la **localización condicionada** a que ya sabemos
que existe una MPB etiquetada en la zona explorada,
no el F1 de detección autónoma en toda la órbita.
La elección arquitectónica de 5D debe considerarse
exploratoria y evaluarse con los mismos eventos que las
otras versiones.

El documento [histórico V14 3D](v14_dual_3d_refiner.md)
se conserva para reproducibilidad.
