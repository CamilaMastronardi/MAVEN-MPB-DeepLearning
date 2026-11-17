# V13 — Interpretación física del espacio latente y trayectorias

## Objetivo

Investigar qué magnitudes medidas pueden estar representando las tres
coordenadas aprendidas por los autoencoders V13 MAG y SWEA, y cómo
cambia la representación al desplazar una ventana temporal de 10 min
alrededor de cada cruce MPB catalogado.

**Esta tarea NO entrena nuevas redes ni promete mejorar la precisión
de V10.** Es un análisis exploratorio de los encoders existentes.

El agrupamiento de V13 fue descriptivo y utiliza vistas de ventanas
definidas con el instante MPB catalogado. Estas mismas marcas se
utilizan aquí para alinear las señales, pero no son entradas a las CNN.

## Modelos y fuentes ya disponibles

- MAG V13: 8 Conv1d encoder, embebido 3D, 8 Conv1d decoder.
  Checkpoint en
  results/metrics/mag_autoencoder_clusters_v13/discovery_final.pt.
- SWEA V13: 8 Conv1d encoder, embebido 3D, 8 Conv1d decoder
  con hasta 256 frames reales y máscara.
  Checkpoint en
  results/metrics/swea_mag_joint_clusters_v13_frames256/discovery_final.pt.
- Ambos CSV de grupos se cargan por MPB timestamp UTC exacto,
  comprobando la coincidencia de folds y splits.
- Se leen los bloques de MAG y SWEA **solo de la caché existente**,
  usando el lector original que valida la procedencia y los checksums.
  **No se consulta BigQuery**; ausencias de caché quedan registradas.
- El grupo MAG/SWEA y sus z1,z2,z3 ya entrenados se conservan.
  No se vuelve a ajustar ni KMeans ni normalizadores.

## 1. Parámetros físicos de MAG

Para cada cruce MPB con tiempo catalogado t0, se calculan:

- Sector anterior: [t0-240 s, t0-60 s].
- Sector posterior: [t0+60 s, t0+240 s].
- Medias de |B| en nT; salto firmado (posterior-anterior);
  magnitud absoluta del salto y razón entre intensidades.
- Desviación típica de |B| en cada sector.
- RMS de incrementos vectoriales B(t+dt)-B(t), con dt<=2.5 s,
  como descriptor de fluctuaciones a la cadencia MAG.
- Ángulo entre los vectores B medios de ambos sectores.
- Medias cartesianas de Bx, By y Bz en cada sector.
- Diferencia entre la cuarta columna |B| y el módulo reconstruido
  algebraicamente desde Bx,By,Bz, como control de consistencia.

No usamos MAG predicho por el decoder para derivar física.
No suponemos que el tramo anterior sea siempre magnetosheath:
la dirección del cruce importa para el signo de los saltos.

## 2. Parámetros físicos/observacionales SWEA

Los 12 canales SWEA de V7/V13 corresponden a una cuadrícula
fija log-espaciada entre **30 y 300 eV**. La caché contiene
log1p(flujo promediado angularmente), interpolado en ENERGÍA
dentro del soporte medido. No se interpola temporalmente.

Se calculan en los mismos sectores respecto de t0:

- Promedio del log1p(flujo) antes y después de la MPB.
- Cambio firmado del nivel medio log1p(flujo).
- Diferencia media absoluta entre espectros medios.
- Desviación temporal media de los espectros.
- Diferencia alto-bajo de intensidad espectral:
  promedio >/=120 eV menos promedio <=80 eV.
- Cambio de esa diferencia y pendiente empírica de
  log1p(flujo) versus log(E).

La pendiente es solamente un descriptor observacional;
**no** se debe interpretar automáticamente como una ley de
potencias ni como una temperatura electrónica.

## 3. Correlaciones con las coordenadas latentes

Se calculan coeficientes de Spearman entre las variables
físicas y cada una de las tres coordenadas latentes del
instrumento correspondiente. Se informan n, rho y p.

Los p-values de **FIT** se corrigen con Benjamini-Hochberg
(FDR) para controlar pruebas múltiples. Los del **fold 4
(outer_holdout)** se informan por separado como comprobación
descriptiva, no para seleccionar las variables con los que
se entrena un nuevo modelo. Las correlaciones no implican
causalidad ni dan significado físico único a una coordenada.

Se generan mapas de calor para ambos splits e instrumentos
y boxplots de características medidas por grupo en FIT.

## 4. Trayectorias de ventanas en el espacio latente

Para cada evento se desplaza el centro t_c de una ventana
de 10 min según:

    t_c - t_MPB = -240, -210, ..., 0, ..., +210, +240 segundos

Solo se utilizan ventanas con mediciones MAG+SWEA válidas
y sin BS u otra MPB que invalide la clasificación segura
utilizada originalmente por V13.

Cada ventana se pasa por las dos CNN V13 ya entrenadas,
sin retropropagación. Se guardan sus tres coordenadas,
probabilidad auxiliar de presencia y error de reconstrucción.

Con esto se dibuja:

    z(t_c) = [z1(t_c), z2(t_c), z3(t_c)]

Las trayectorias también se resumen mediante su longitud
de arco en 3D, máximo desplazamiento latente por segundo
de centro desplazado, y posición de ese máximo respecto
de la MPB. Se excluyen segmentos a través de ventanas
omitidas; no se construyen trayectorias artificiales a
través de gaps.

**La derivada dz/dt_c no es una velocidad física del plasma**.
Mide sensibilidad del encoder al corrimiento de su ventana
de observación. Los espectros SWEA conservan los tiempos
reales y el marco temporal no se interpola.

Como control reproducimos el promedio de las nueve vistas
separadas 60 segundos usado para formar los grupos
originales. La discrepancia con el CSV previo queda en
el resumen para detectar cualquier desajuste de fuentes
o de normalización.

## 5. Ejecución

En WSL:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

python -m unittest discover \
  -s tests -p 'test_latent_physics_trajectories_v13.py' -v
```

Piloto reducido con 8 cruces distribuidos por fold:

```bash
EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/v13_latent_physics_trajectories_smoke \
bash scripts/6_latent_analysis/run_latent_physics_trajectories_v13.sh
```

Luego, ejecución completa:

```bash
tmux new -s latent_physics_v13
bash scripts/6_latent_analysis/run_latent_physics_trajectories_v13.sh
```

Desacoplar la sesión: Ctrl+B, D.
Recuperar: tmux attach -t latent_physics_v13.

No se modifican los pesos ni los CSV de V13.
Se necesita que el checkpoint SWEA sea el de 256 frames
(sufijo _frames256). Para no ocultar huecos, se registran
los bloques ausentes de la caché; para exigir cobertura
completa se puede añadir --require-complete-cache a la
invocación directa del script Python.

## 6. Resultados

Directorio:
results/metrics/v13_latent_physics_trajectories/

- physical_features.csv:
  una fila por cruce catalogado, con z1-z3 MAG/SWEA,
  grupos, split y todos los parámetros medidos.
- latent_trajectories.csv:
  una fila por evento, instrumento y centro de ventana,
  con 3 coordenadas, desplazamiento en segundos y
  calidad de reconstrucción.
- trajectory_geometry.csv:
  longitud de trayectoria, velocidad de desplazamiento
  latente respecto del centro y pico de ese indicador.
- latent_physics_spearman.csv:
  Spearman de cada rasgo físico contra cada z en FIT
  y outer_holdout, con FDR en FIT.
- mag_spearman_fit.png / mag_spearman_outer_holdout.png.
- swea_spearman_fit.png / swea_spearman_outer_holdout.png.
- mag_physical_group_boxplots.png.
- swea_physical_group_boxplots.png.
- mag_median_latent_trajectories.png.
- swea_median_latent_trajectories.png.
- trajectory_examples/*.png:
  ejemplos individuales con MAG medido, espectro SWEA
  y ambas trayectorias latentes coloreadas por el
  desplazamiento de centro.
- missing_cached_events.csv: instrumentación no cacheada.
- latent_physics_report.json: resumen, discrepancias de
  reproducción del CSV original, tamaño de muestra y
  advertencias metodológicas.

Resumen:

```bash
python -m json.tool \
  results/metrics/v13_latent_physics_trajectories/latent_physics_report.json
```

## 7. Interpretación científica pendiente

Comparar especialmente:
1. ¿z3 MAG sigue los saltos de intensidad |B|, la rotación
   vectorial o la varianza de las fluctuaciones?
2. ¿Los grupos MAG difieren en morfología, o predomina
   la intensidad/época de adquisición?
3. ¿Las coordenadas SWEA se relacionan con el cambio
   de espectro entre ambos lados de la MPB?
4. ¿La trayectoria 3D cambia apreciablemente cerca de
   t_c=t_MPB, o su desplazamiento es gradual?
5. ¿Las asociaciones de FIT persisten en fold 4?

Estas respuestas permitirán plantear hipótesis físicas
verificables y decidir si el autoencoder requiere un
cuello de botella más interpretable.
