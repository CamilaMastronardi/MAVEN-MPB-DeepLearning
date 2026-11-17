> **Archivo histórico:** esta documentación corresponde al primer piloto V14 con latente de 3 dimensiones. La revisión vigente utiliza **5 dimensiones por sensor**, un lanzador separado y checkpoints independientes. Ver [V14 5D](v14_dual_5d_refiner.md). El antiguo lanzador 3D se retiró para evitar reanudaciones incompatibles.\n\n# V14 — Localización V10 con dos autoencoders CNN de ocho capas y latentes 3D

## Objetivo para la entrega de ML Geofluidos

Conservar el esquema de localización gruesa + refinamiento temporal de
V10, pero disponer de representaciones **compactas y directamente
analizables en tres dimensiones**, una para MAG y otra para SWEA, como
en el análisis exploratorio de V13.

Este es un entrenamiento **nuevo**: no se pueden sustituir los 16
canales del V7b congelado por tres sin reentrenar y cambiar sus cabezas.
Para preservar la comparación justa, V14 mantiene intacto V7b como
*coarse proposer* y añade un **refinador nuevo** cuyos únicos atributos
aprendidos de las series son los dos bottlenecks 3D.

V14 **no hereda** los checkpoints de autoencoders V13 (se entrenaron con
otra partición de descubrimiento). En V14 se entrenan encoders y
decoders nuevos de forma independiente en cada fold, evitando
incorporar representaciones entrenadas sobre el OUTER de otro fold.

## Arquitectura exacta

1. **Búsqueda gruesa** — mismo V7b por fold, con pesos y normalizadores
   congelados. Escanea candidatos de 10 minutos en ±30 minutos,
   a saltos de 32 segundos. Sigue usando su arquitectura original de
   16 canales temporales; sus pesos NO se modifican.
2. **MAG** — cuatro entradas `Bx,By,Bz,|B|`, eight layers Conv1d con
   reducción temporal progresiva y salida `z_MAG ∈ R³` por ventana.
   Decoder CNN simétrico reconstruye las cuatro componentes.
3. **SWEA** — 12 canales de 30–300 eV, en tiempos nativos sin
   interpolación temporal, ocho capas Conv1d y `z_SWEA ∈ R³`.
   Decoder reconstruye el espectro normalizado.
4. **Refinamiento temporal** — la red fina fusiona
   `[z_MAG(3), z_SWEA(3), score_V7b(1), logit_region_V7b(2)]`,
   total 9 atributos. Una cabeza aprende 21 pesos sobre funciones
   gaussianas temporales superpuestas y devuelve una distribución
   sobre la línea temporal nativa de MAG dentro de ±250 segundos.
   Otra cabeza predice presencia auxiliar.
5. **Predicción** — igual que V10:
   `t_pred = t_V7b + alpha * Δt_fine`, con una corrección fina
   limitada a ±240 segundos. Se eligen `alpha` y el estadístico
   media/máximo exclusivamente en MONITOR interno.

**El cuello de botella es de tres coordenadas escalares por ventana,
NO de tres mapas temporales**. Eso permite graficar los puntos 3D
directamente, sin aplicar PCA.

En el refinador, los logits del teacher se usan como tres atributos
físicos supervisados congelados. Por ello la representación compacta
contiene las observaciones aprendidas por las dos ramas de V14,
aunque el sistema completo retenga la búsqueda gruesa V7b.

## Datos y pérdida

- Ventanas MAG/SWEA reales de 10 minutos, máscaras de padding y tiempos
  originales por sensor.
- Controles seguros respecto de otras MPB y BS.
- Mismos folds por meses, normalizadores TRAIN del V7b del mismo fold.
- Positivos desplazados para evitar que la etiqueta esté siempre
  centrada; negativos sin offset MPB ficticio.
- Peso del teacher congelado.
- Las dos reconstrucciones evitan que el espacio latente solo aprenda
  a maximizar precisión temporal y pierda morfología observacional.

```
L = BCE_presencia
  + 3.0 * KL_tiempo(sigma = 24 s)
  + 1.5 * SmoothL1_offset
  + 0.2 * (MSE_MAG_recon + MSE_SWEA_recon)
```

Configuración inicial: `lr=1e-4`, `batch=16`, `max_epochs=150`,
`min_epochs=20`, `patience=20`, `6` positivos y `3` negativos
por evento/época. Es una hipótesis de arquitectura e hiperparámetros,
**no un resultado validado**. Se permite `alpha=0` para recuperar
V7b exactamente.

## Ejecución en WSL2 (CPU, caché obligatoria)

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_v14_dual_bottleneck.py' -v
```

Un piloto, con ocho eventos por fold y checkpoints completamente
separados de los cuatro folds completos:

```bash
PILOT_EVENTS_PER_FOLD=8 FOLDS=1 \
bash scripts/5_training/run_v14_dual_3d.sh
```

Entrenamiento completo:

```bash
tmux new -s mpb_v14
bash scripts/5_training/run_v14_dual_3d.sh
```

Salir de tmux sin cortar: Ctrl+B, luego D.
Volver: `tmux attach -t mpb_v14`.

El script comprueba sintaxis, pruebas y caché MAG/SWEA de los eventos
seleccionados. Si falta algún bloque, **aborta antes de BigQuery**.
No hay necesidad de descargar datos cuando la caché está completa.

Se puede ejecutar un único fold con `FOLDS=1`; de todos modos sus
positivos FIT vienen de los otros tres folds. Los experimentos piloto
NO se utilizan para evaluar el desempeño definitivo.

## Resultados y espacio latente

```
results/metrics/v14_dual_3d_refiner/
  fold_01_last.pt
  fold_01_best.pt
  fold_01_final.pt
  fold_01_metrics.json
  fold_01_event_metrics.csv
  fold_01_latent_embeddings.csv
  ...
  cv_summary.json
  training.log
```

Cada `fold_XX_latent_embeddings.csv` incluye:

- `mag_z1,mag_z2,mag_z3` y
  `swea_z1,swea_z2,swea_z3`.
- Una vista por ventana desplazada alrededor del catálogo
  (-240,-180,-120,-60,0,+60,+120,+180,+240 s).
- Una vista adicional centrada en la MPB **predicha** por V14.
- `fold`, `source_group`, `view`, desplazamiento temporal y
  error de localización.

**Cuidado metodológico:** las nueve ventanas catalogadas se seleccionan
usando el instante MPB verdadero solamente para análisis retrospectivo.
No se debe afirmar que esas figuras reproducen detección autónoma.
La vista `predicted_centered` no necesita el instante verdadero para
formar la ventana cuando se ha definido un intervalo de búsqueda.
El error sí utiliza ese instante para evaluar.

**Los ejes latentes de diferentes folds NO son comparables directamente**:
cada fold aprendió pesos independientes y puede haber simetrías de
rotación, escala y permutación. No unir z1–z3 de distintos folds
en un mismo scatter sin alineación explícita.

Las primeras figuras se producen a partir de CSV sin BigQuery:

```bash
python scripts/6_latent_analysis/6b_plot_v14_latents.py
```

Genera por fold e instrumento:
- `mag_latent_3d.png` y `swea_latent_3d.png`, puntos en el espacio
  tridimensional de ventanas centradas en el tiempo **predicho**.
- `mag_latent_error.png` y `swea_latent_error.png`,
  color por error absoluto de localización.
- `mag_latent_trajectories.png` y
  `swea_latent_trajectories.png`, medianas e IQR por tiempo
  desplazado respecto de la MPB **del catálogo**.

Las derivadas de estas trayectorias respecto al desplazamiento de la
ventana NO son velocidades del plasma.

## Evaluación y límites

Comparar al final V7b, V10 original, HP1 (solo si terminó todos los
folds) y V14 sobre exactamente los mismos cruces. Reportar mediana,
media, p90 y porcentaje de error ≤60 s. No afirmar que V14 mejora
automáticamente por tener un cuello de botella de tres dimensiones:
podría empeorar al reducir capacidad informativa.

El conjunto OUTER de los cuatro folds ya se consultó durante el
desarrollo: resultados exploratorios, no validación ciega final.
Tampoco se infiere F1 de detección autónoma sobre órbitas completas.

Archivos principales:
- `src/detection/mpb_v14_dual_bottleneck_refiner.py`
- `scripts/5_training/5n_train_v14_dual_bottleneck.py`
- `scripts/5_training/run_v14_dual_3d.sh`
- `scripts/6_latent_analysis/6b_plot_v14_latents.py`
- `tests/test_v14_dual_bottleneck.py`
