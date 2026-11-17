# V13b: SWEA 3D autoencoder + agrupamiento conjunto MAG/SWEA en 6D

## Objetivo

Comparar si los espectros de electrones de SWEA separan tipos de cruces
MPB diferentes a los obtenidos con MAG y si la combinación de ambas
representaciones genera grupos más separados y estables.

El estudio previo MAG V13 produjo 553/559 vectores latentes 3D y dos
grupos de 102 y 451 cruces. Su silhouette FIT era 0,493. Esos latentes
se **reutilizan sin volver a entrenar MAG**.

La detección/localización de una MPB con <=60 s en >=80 % de los
eventos sigue siendo el objetivo final, pero ESTE PASO es solamente
un análisis de agrupamiento exploratorio. No produce nuevas
predicciones autónomas de los tiempos MPB.

## SWEA autoencoder

- Entrada: 12 canales espectrales preprocesados de SWEA a la
  **cadencia real del instrumento**, en ventanas de 10 minutos.
  No se reinterpola SWEA a los tiempos MAG.
- Las ventanas con 12–256 frames medidos se rellenan de forma simétrica
  hasta un tensor 12×256. La máscara evita penalizar posiciones
  artificiales en la reconstrucción; los valores de esas posiciones
  son cero normalizado.
- Estandarización SWEA por canal calculada con las épocas únicas del
  conjunto FIT original. Ninguna muestra del fold externo se usa
  para ajustar normalizadores.
- Encoder: ocho Conv1d, reducción progresiva, capa lineal,
  espacio latente de **tres escalares** (s1, s2, s3).
- Decoder: ocho Conv1d con upsampling de activaciones, reconstruye
  12 canales SWEA; no crea mediciones físicas interpoladas.
- Dos cabezas auxiliares predicen presencia de MPB en ventana y
  desplazamiento de cruce en una ventana positiva.
- Misma pérdida V13 que para MAG:

    L = MSE_reconstruccion_mascarada
        + 0.30 BCE_ventana_MPB
        + 0.10 SmoothL1((offset_pred-offset_true)/60s)

Los positivos se desplazan entre -240 y +240 s del centro; los
negativos se seleccionan lejos de MPB y bow shock mediante la
misma lógica de ventanas físicamente seguras de V13 MAG.
El conjunto de ventanas de extracción de latentes se define por
las MPB **ya catalogadas**, que es aceptable para describir
morfologías pero NO para inferir cruces no identificados.

La versión inicial tenía un límite artificial de **32 frames**
y falló al encontrar una ventana medida de **38 frames**. La revisión
corrige ese límite sin descartar muestras ni interpolar datos: se
admiten hasta **256 frames por ventana** y se ignora el relleno en
la pérdida de reconstrucción. Se ajustaron las ocho capas del
encoder/decoder al nuevo tamaño, manteniendo tres variables latentes.
**Los checkpoints de 32 frames no son compatibles**; el directorio
de salida ahora lleva el sufijo `_frames256` para evitar mezclarlos.

## Clustering: MAG vs SWEA vs combinación

Para cada evento elegible:

    MAG  : m1, m2, m3     [del CSV V13 existente]
    SWEA : s1, s2, s3     [nuevo encoder SWEA]
    JOINT: m1,m2,m3,s1,s2,s3

Se hace KMeans por separado sobre:

1. Los 3 latentes de SWEA.
2. Los **6 latentes concatenados MAG+SWEA**, sin proyectarlos primero.

Cada coordenada se estandariza por media/desviación de TRAIN FIT.
Cada sensor aporta tres coordenadas estandarizadas, evitando que
las unidades de uno dominen el ajuste inicial solo por escala.
Se prueban K=2...6 y se elige K por silhouette FIT, descartando
particiones con clusters de menos de tres ejemplos.
El conjunto MONITOR y el fold 4 son asignados con KMeans
ya entrenado; no se utilizan para decidir K ni ajustar centroides.

**PCA NO forma los clusters.** Solo proyecta visualmente a 3D
los resultados del KMeans en 6D, ajustando sus componentes
exclusivamente en FIT. La suma de las varianzas explicadas
por las primeras tres PC queda registrada.

## Comparación de métodos

Se informan:

- Silhouette FIT en SWEA 3D.
- Silhouette FIT en MAG+SWEA 6D.
- Silhouette del agrupamiento MAG anterior, recalculada sobre los
  cruces FIT que poseen también SWEA.
- Silhouette del conjunto fold 4 utilizando grupos ya asignados.
- Cantidad de grupos, tamaños y distribución temporal por fold.
- Índice Rand ajustado (ARI) MAG–SWEA, MAG–JOINT y SWEA–JOINT.
  ARI~1 indica mismos agrupamientos, ARI~0 relación esperable
  al azar; no requiere alinear manualmente números de grupo.
- Grupos del catálogo originales y proporciones de errores V10
  dentro de +/-60 s en cada nuevo cluster (análisis POSTERIOR,
  no se usa V10 para ajustar clusters).
- Alineación exacta por timestamp UTC: nunca combinar filas
  MAG y SWEA por posición de fila.

**Cuidado:** un silhouette mayor no prueba mayor precisión
de localización ni que los clusters describan tipos físicos
reales. KMeans puede aislar ruido/outliers y representar
cambios de intensidad o instrumento, no un mecanismo MPB.

## División de eventos

El dataset previo MAG contiene 559 cruces: 379 FIT, 41 MONITOR y
139 OUTER-HOLDOUT (fold 4). La nueva ejecución reproduce exactamente
esas particiones y comprueba que coincidan con el CSV MAG anterior.
Para eventos sin coordenadas MAG/SWEA, la tabla conjunta deja
el grupo vacío y los excluye del ajuste conjunto.

## Ejecución en WSL2

Primero prueba funcional y revisión de errores (el piloto usa
una muestra pequeña, por lo que sus clusters no son
conclusiones científicas):

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

PILOT_EVENTS_PER_FOLD=16 \
OUTPUT_DIR=results/metrics/swea_mag_joint_clusters_v13_frames256_smoke \
bash scripts/5_training/run_swea_mag_joint_clusters_v13.sh
```

Con los tests aprobados y ejecución piloto sin error, correr
los 559 eventos:

```bash
tmux new -s swea_v13
bash scripts/5_training/run_swea_mag_joint_clusters_v13.sh
```

Desacoplar tmux: Ctrl+B, después D. Volver:
`tmux attach -t swea_v13`.

Se reutiliza la caché de mediciones MAG/SWEA de V13:
- MAG: data/cache/mag_events
- SWEA: data/cache/swea_events

El script primero verifica que exista
`results/metrics/mag_autoencoder_clusters_v13/crossing_latents_groups.csv`
y que los eventos coincidan. Solo se entrena el nuevo autoencoder
SWEA; no modifica checkpoints ni resultados anteriores de MAG.

## Salidas

```
results/metrics/swea_mag_joint_clusters_v13_frames256/
  discovery_last.pt
  discovery_best.pt
  discovery_final.pt
  discovery_summary.json
  swea_latents_groups.csv
  mag_swea_joint_groups.csv
  swea_latent_clusters_3d.png
  mag_swea_joint_pca3d.png
  training.log
```

- `swea_latents_groups.csv`: grupo y tres coordenadas SWEA.
- `mag_swea_joint_groups.csv`: tiempo MPB catalogado, fold,
  grupo MAG previo, grupo SWEA, grupo conjunto y las seis
  coordenadas.
- `discovery_summary.json`: métricas de reconstrucción/identificación
  SWEA, selección de K, silhouettes, ARI, matriz cruzada MAG/SWEA,
  distribución por folds, silhouette sobre el fold 4 y comparación posterior con V10.
- `discovery_final.pt`: pesos SWEA y normalizador, centroides
  y parámetros de estandarización de los clusters y PCA.

Ver el resumen:

```bash
python -m json.tool \
  results/metrics/swea_mag_joint_clusters_v13_frames256/discovery_summary.json
```

## Decisión científica posterior

Si el espacio combinado separa mejor los cruces (especialmente
en fold 4) Y la separación es físicamente interpretable,
investigaremos características MAG/SWEA de cada grupo.

Si la separación no mejora, no conviene forzar redes diferentes:
se deberá reconsiderar los objetivos del autoencoder, los
canales de SWEA o el criterio de agrupamiento.

En ambas opciones, la etapa siguiente para el objetivo
80 % <=60 s requiere un localizador temporal completo,
evaluado sobre bloques de 70 min y en eventos realmente
nuevos.
