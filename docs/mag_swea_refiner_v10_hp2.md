# V10-HP2: maximizar localizaciones MAG+SWEA con error <=60 s

## Contexto

- HP1 modificó los hiperparámetros de entrenamiento de V10 manteniendo V7b
  congelado. Primer fold reportado: mediana 42.7 s, 57.5% <=60 s,
  146/146 evaluables, alpha=0.75.
- Ese fold aislado **no prueba** una mejora global frente a HP1 original.
- La búsqueda original de alpha priorizaba el promedio del error absoluto
  truncado a 600 s y solo después el porcentaje <=60 s.
- Se crean dos experimentos **separados** para distinguir la mejora por
  calibración del efecto del nuevo entrenamiento.

## HP2A — recalibrar HP1, sin reentrenar

La arquitectura, los pesos HP1, el V7b congelado, los folds y las
predicciones del conjunto externo **no cambian**.

1. Para cada fold HP1 que ya tenga `_final.pt`, `_metrics.json` y
   `_event_metrics.csv`, se reconstruye EXACTAMENTE el mismo MONITOR
   interno (meses excluidos del FIT).
2. Se exige que el checkpoint de HP1 tenga la firma de la configuración
   por defecto de `run_mag_swea_refiner_v10_hp1.sh`. Esto protege
   contra la reutilización accidental de modelos con otros hiperparámetros,
   particiones o teacher.
3. Se calculan las propuestas V7b para los meses MONITOR utilizando
   exclusivamente bloques MAG/SWEA presentes en caché y se obtienen
   las dos estimaciones de offset con el refinador HP1 ya entrenado.
4. Se reproduce primero la calibración antigua; si no coincide con
   la registrada en el JSON de HP1, **se aborta**.
5. Se evalúan `alpha={0.0,0.1,...,1.0}` **más el alpha original de HP1**
   si no coincide con esa grilla (por ejemplo 0.75), para ofrecer
   la misma referencia en el MONITOR. Para cada alpha se prueban
   las estimaciones `mean` y `peak`.
6. Se selecciona el máximo `fraction_within_60s` **solo en MONITOR**;
   los desempates son menor error medio truncado a 600 s,
   menor mediana, menor alpha, preferencia por mean.
7. La calibración elegida se aplica a los offsets externos que HP1
   YA había guardado, sin volver a inferir ni escoger con OUTER.
   Se comparan porcentaje <=60, mediana y cambios pareados vs HP1.

Nunca se reentrena ni consulta BigQuery. Las cachés originales deben
estar presentes; cualquier falta provoca un error.

La calibración original de HP1 puede ser mejor sobre OUTER que la nueva:
el comportamiento en eventos no vistos no está garantizado por MONITOR.

### Ejecutar HP2A

Desde WSL, raíz del repositorio:

```bash
git pull origin main
source .venv/bin/activate
bash scripts/5_training/run_mag_swea_refiner_v10_hp2a.sh
```

De manera predeterminada recalibra **solo los folds HP1 completados**;
también es posible especificar `FOLDS=1` o `FOLDS=all` cuando estén
listos los cuatro. El cálculo puede consumir CPU porque se vuelven a
escanear los candidatos V7b de MONITOR; no entrena redes.

Salidas independientes:

```
results/metrics/mag_swea_refiner_v10_hp2a/
  fold_01_metrics.json
  fold_01_event_metrics.csv
  ...
  cv_summary.json
```

## HP2B — nueva red con pérdida más focalizada en tiempo

HP2B conserva V10, V7b y los mismos datos pero entrena nuevos
refinadores. Se implementa mediante parámetros de V10 ya existentes,
más opciones nuevas **compatibles con sus defaults anteriores**.

| Parámetro | HP1 | HP2B |
| --- | --- | --- |
| Max epochs | 150 | 150 |
| Min epochs | 20 | 20 |
| Patience | 20 | 20 |
| Learning rate | 0.0001 | 0.0001 |
| Batch size | 16 | 16 |
| Sigma gaussiana temporal | 32 s | **24 s** |
| Peso KL temporal | 2.0 | **3.0** |
| Peso offset SmoothL1 | 1.0 | **1.5** |
| Positivos por evento y época | 4 | **6** |
| Negativos por evento y época | 3 | 3 |
| Objetivo de calibración | Error medio truncado | **% <=60 s** |
| Grilla alpha | 0.25 | **0.1** |

Función de pérdida:

```
L = BCE_presencia + 3.0 * KL_temporal(sigma=24s) + 1.5 * SmoothL1_offset
```

La **selección de checkpoint** conserva el early stopping por pérdida
total sobre MONITOR. La selección de alpha sí prioriza <=60 s.

### Ejecutar HP2B

```bash
git pull origin main
source .venv/bin/activate
tmux new -s mpb_v10_hp2b
bash scripts/5_training/run_mag_swea_refiner_v10_hp2b.sh
```

Usar Ctrl+B y después D para separar la sesión; recuperar con
`tmux attach -t mpb_v10_hp2b`.

Para un piloto independiente (no es una medición global):

```bash
PILOT_EVENTS_PER_FOLD=8 \
bash scripts/5_training/run_mag_swea_refiner_v10_hp2b.sh
```

Si faltan datos en caché, el lanzador aborta **antes de BigQuery**.
Si hay datos completos, los reutiliza y entrena sin descargarlos.
Salidas separadas: `results/metrics/mag_swea_refiner_v10_hp2b/`.

## Interpretación y pruebas

- No usar OUTER para elegir la grilla alpha, los pesos ni sigma.
- Comparar HP1, HP2A y HP2B con exactamente los mismos cruces y separar
  aciertos que entran/salen del umbral de 60 s.
- Los cuatro folds existentes ya se observaron durante desarrollo.
  Una evaluación final requerirá meses adicionales intactos.
- El procedimiento continúa siendo **localización condicionada a una
  MPB catalogada cercana**, no búsqueda autónoma sobre toda la órbita.
- El refinador corrige un candidato de V7b hasta unos 240 s; si V7b
  elige otra transición muy alejada, el nuevo entrenamiento no garantiza
  resolver el evento.

Códigos: `scripts/5_training/5m_recalibrate_v10_hp2a.py`,
`scripts/5_training/run_mag_swea_refiner_v10_hp2a.sh`,
`scripts/5_training/run_mag_swea_refiner_v10_hp2b.sh`.
Pruebas: `tests/test_v10_hp2_calibration.py`.
