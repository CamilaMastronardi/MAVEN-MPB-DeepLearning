# Experimento: V10 independiente para los dos grupos MAG — solo fold 4

## Objetivo

Probar si entrenar dos refinadores temporales V10 especializados según
los dos grupos latentes MAG de V13 mejora la localización de la MPB,
particularmente el porcentaje con error <=60 s, frente al V10 original.

**Se evalúa EXCLUSIVAMENTE el fold 4**, que fue la cohorte externa
en el entrenamiento del autoencoder/clustering MAG V13.

Datos del CSV de V13:
- Entrenamiento disponible: MAG1 72 cruces, MAG2 342 cruces,
  de los folds 1, 2 y 3.
- Evaluación fold 4: MAG1 30 cruces, MAG2 109 cruces,
  total 139 cruces.
- Seis eventos de los folds 1-3 sin latente MAG no entran al
  entrenamiento de ninguno de los dos expertos.

## Diseño de los dos modelos

Mantener exactamente la arquitectura y el entrenamiento de V10:
- V7b FOLD4 congelado como clasificador físico y buscador de
  candidatos de 10 minutos con MAG+SWEA;
- misma arquitectura de refinador V10 (RegionAnchoredTemporalRefiner),
  pérdida, ventanas positivas desplazadas y controles negativos
  alejados de las fronteras;
- normalizadores del checkpoint V7b fold 4;
- cadencias instrumentales originales, sin interpolar SWEA;
- evaluación de búsqueda gruesa en +/-30 min y corrección fina
  con parámetros alpha/método calibrados solo en TRAIN.

Se entrenan desde cero **dos refinadores distintos** con las
mediciones correspondientes a cada grupo MAG:

- **Experto MAG1:** FIT + MONITOR de los 72 eventos de grupo 1
  en folds 1-3; evaluación SOLO en sus 30 eventos del fold 4.
- **Experto MAG2:** FIT + MONITOR de los 342 eventos de grupo 2
  en folds 1-3; evaluación SOLO en sus 109 eventos del fold 4.

La elección de los meses MONITOR se realiza por separado para
cada grupo. Se exige al menos 5 eventos MONITOR y 12 FIT. Esta
separación es necesaria porque el conjunto MONITOR original
del autoencoder MAG tenía únicamente 2 eventos MAG1,
insuficientes para calibrar un experto de manera estable.

El criterio de early stopping sigue siendo la pérdida de
validación V10. La calibración temporal V10 elige alpha y método
media/máximo con datos del MONITOR de SU MISMO GRUPO. No hay
uso de etiquetas del fold 4 en el entrenamiento de las cabezas
de refinamiento ni en su calibración.

## Interpretación del experimento: importante

Los grupos MAG1/MAG2 se obtuvieron a partir de representaciones
de ventanas positivas **seleccionadas usando el instante MPB
catalogado**. Aunque el modelo V13 para grupos no se entrenó
con datos del fold 4, para asignar cada evento fold 4 a MAG1/MAG2
se usan ventanas centradas respecto de su anotación.

Esto es un experimento de **enrutamiento con etiquetas
retrospectivas/oracle**, no una solución autónoma: para aplicar
los expertos en datos no etiquetados tendremos que diseñar un
clasificador de grupo independiente de la anotación verdadera.
No se debe confundir una mejora de este experimento con
generalización end-to-end de un localizador.

Tampoco constituye una prueba ciega final: los mismos cuatro
folds ya se utilizaron en experimentos previos V7b-V13.

## Ejecución

Desde WSL2:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_v10_mag_groups_fold4.py' -v
```

Verificar particiones sin entrenar ni consultar BigQuery:

```bash
python scripts/5_training/5k_train_v10_mag_groups_fold4.py
```

Entrenar en tmux solo fold 4:

```bash
tmux new -s v10_mag_fold4
bash scripts/5_training/run_v10_mag_groups_fold4.sh
```

Separar terminal sin terminar entrenamiento: Ctrl+B, D.
Retomar: `tmux attach -t v10_mag_fold4`.

Se reutiliza la caché MAG/SWEA de los experimentos anteriores
si existe. Si faltan bloques, el cargador de V7b realiza
preflight de costos BigQuery antes de ejecutar consultas.
Los checkpoints se guardan por grupo y se recuperan con
`--resume`. Resultados antiguos de V10/MAG V13 no se modifican.

## Archivos

```
results/metrics/v10_mag_group_specialists_fold4/
  fold_04_maggroup_1_last.pt
  fold_04_maggroup_1_best.pt
  fold_04_maggroup_1_final.pt
  fold_04_maggroup_1_metrics.json
  fold_04_maggroup_1_event_metrics.csv

  fold_04_maggroup_2_last.pt
  fold_04_maggroup_2_best.pt
  fold_04_maggroup_2_final.pt
  fold_04_maggroup_2_metrics.json
  fold_04_maggroup_2_event_metrics.csv

  fold_04_summary.json
  training.log
```

El resumen global asegura que las dos partes cubran los
**139 eventos distintos del fold 4**, sin perder ni repetir
un cruce. Se comparan evento por evento:

- aciertos <=60, <=120, <=300 s por grupo;
- mediana, media y percentil 90 del error;
- número de eventos que mejoraron/empeoraron frente a V10
  original en ese mismo fold;
- precisión combinada de los 139 cruces, comparada contra
  el V10 original para exactamente esos cruces.

Para consultar:

```bash
python -m json.tool \
  results/metrics/v10_mag_group_specialists_fold4/fold_04_summary.json
```

**Objetivo experimental:** mejorar al V10 global; el 80 % dentro de
60 s es una meta ambiciosa, no una consecuencia garantizada de
entrenar modelos separados sobre menos ejemplos.
