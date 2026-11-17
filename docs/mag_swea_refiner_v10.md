# V10 — MPB coarse-to-fine: conservar V7b y refinar su tiempo

## Motivación física y metodológica

V7b combinó MAG y SWEA con clasificación de magnetovaina/magnetosfera
inducida y ranking entre candidatos. Su validación sobre 559 MPB
catalogadas consiguió mediana 72 s y 250/559 (44,7 %) con error <=60 s.

V8 (ranking local) y V9 (cabeza densa sustitutiva) no mejoraron
claramente V7b en los primeros resultados. Entrenar más tiempo V7b
tampoco aportó una mejora sostenida.

La nueva hipótesis de V10 es que la cabeza de regiones **sí sabe proponer
transiciones físicas**, pero la transición elegida puede necesitar un
ajuste local. Por eso **no sustituimos V7b ni reentrenamos sus pesos**.

## Arquitectura: dos etapas

1. **Coarse (V7b original, congelado).** Cada ventana de MAG+SWEA de
   10 min se puntúa con la cabeza de transiciones de V7b. Se escanea
   una región de +/-30 min alrededor del cruce catalogado, con centros
   separados por 32 s, y se selecciona su máximo tal como lo hacía
   V7b. No se exige que el máximo esté en el instante etiquetado.
2. **Fine (nuevo refinador).** Sobre la ventana ganadora, se reutilizan
   encoders MAG (4 canales) y SWEA (12 energías entre 30 y 300 eV)
   de V7b, **sin permitirles aprender ni modificar sus pesos**.
   Se fusionan representaciones de ambos instrumentos y dos medidas
   temporales de cambio de MAG (~31 y ~121 s). Además se proporciona
   a la nueva cabeza el puntaje de transición y las dos salidas de
   clasificación física de V7b. El refinador genera una distribución
   sobre los tiempos MAG a +/-250 s alrededor del centro candidato,
   y una probabilidad auxiliar de que su ventana contenga una MPB.
3. **Predicción final:** tiempo de V7b más `alpha * corrección` del
   refinador, con correcciones limitadas a +/-240 s. El modo temporal
   (máximo o media de la distribución) y alpha en {0, .25, .5, .75,
   1} se eligen exclusivamente con los meses de MONITOR internos.

Si alpha=0, V10 reproduce exactamente la salida de V7b para ese fold.
Esto permite ver si el refinador realmente añade información temporal
sin imponer una corrección perjudicial.

## Entrenamiento

Se reutiliza la preparación científicamente conservadora de ventanas
de V9, *no* su arquitectura:

- Positivos de 10 min con MPB en posiciones distintas respecto del
  centro: -240, -180, -120, -60, 0, 60, 120, 180 y 240 s.
- Negativos físicos lejanos: -1500, -1200, -900, -600, 600, 900,
  1200 y 1500 s; se excluyen siempre que estén cerca de BS u otras
  fronteras catalogadas, siguiendo las reglas de V7b.
- Hasta tres positivos y tres negativos por evento y época.
- Los negativos **no reciben etiquetas temporales inventadas**.

La pérdida es:

    BCE_presencia_10min
      + 1.5 * KL(gaussiana del catálogo || distribución temporal)
      + 0.5 * SmoothL1(offset esperado / 60 s, offset real / 60 s)

La gaussiana de referencia tiene sigma 32 s. La KL y el término
de offset se calculan **solo** sobre ventanas positivas. Los
parámetros por defecto son una hipótesis experimental, no un óptimo
conocido.

El objetivo de clasificación de ventana es auxiliar; V10 sigue siendo
un **localizador condicionado a que existe una MPB en el intervalo
de 70 min**, no un detector autónomo de presencia/ausencia.

## Validación

- Mismos 4 folds temporales externos y mismos eventos de V7b.
- Normalización extraída del checkpoint V7b del mismo fold
  (ajustada solo con su TRAIN interno).
- Pesos V7b congelados y validación del puntaje original: V10
  vuelve a recorrer las mismas ventanas, exige que el instante
  propuesto coincida con el CSV del V7b original y falla si no.
- Early stopping sobre BCE+KL+offset usando solo meses del monitor
  interno. Calibración final de alpha y tipo de estadístico también
  sobre el monitor interno, nunca sobre el fold externo.
- Se informan media/mediana/p90, fracción <=60, <=120 y <=300 s, y
  cantidad de eventos mejorados, empeorados e iguales frente a V7b.
- Los resultados quedan en un directorio **nuevo**, sin alterar los
  cuatro checkpoints de V7b ni las variantes V8/V9.

## Ejecución

Detener primero cualquier V9 anterior, si sigue activo:

```bash
tmux attach -t mpb_v9
# luego Ctrl+C para interrumpir (y Ctrl+B, D si se requiere)
```

Lanzar V10 (CPU en WSL2):

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate
tmux new -s mpb_v10
bash scripts/5_training/run_mag_swea_refiner_v10.sh
```

El launcher ejecuta `py_compile` y toda la batería de unit tests antes
de iniciar el entrenamiento. Si fallan, **no** se comienza a entrenar.
Se conservan los topes BigQuery; las cachés MAG/SWEA de V7b se
reutilizan si están completas.

Salir de tmux dejando el proceso activo: Ctrl+B seguido de D.
Retomar consola: `tmux attach -t mpb_v10`.

Piloto opcional independiente:

```bash
PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_swea_refiner_v10_pilot \
bash scripts/5_training/run_mag_swea_refiner_v10.sh
```

El piloto usa igualmente los checkpoints V7b originales completos
correspondientes a cada fold. Para comparación científica completa
se debe usar la ejecución de todos los eventos.

Salidas:

    results/metrics/mag_swea_refiner_v10/
      fold_01_metrics.json
      fold_01_event_metrics.csv
      fold_01_best.pt
      fold_01_last.pt
      fold_01_final.pt
      ...
      cv_summary.json
      training.log

`--resume` permite continuar desde checkpoints de V10 con una firma
compatible. Nunca utilizar como salida la carpeta de V7b.

## Interpretación y límites

- La hipótesis V10 permite corregir unos minutos el pico de V7b.
  Si V7b elige una estructura equivocada a más de 240 s de la MPB,
  el refinador **no puede** recuperarla por completo.
- La calidad final puede no mejorar: la selección alpha=0 lo admite.
  Que el refinador mejore algunos eventos no demuestra mejorar la
  distribución global.
- El monitor para seleccionar alpha puede ser pequeño; reportar su
  tamaño y sensibilidad a los folds.
- Los folds externos ya se utilizaron en el proceso exploratorio de
  elección de arquitecturas. Para una afirmación final se requiere
  evaluación sobre meses adicionales, realmente intactos.
- SWEA contiene promedios angulares sobre canales medidos, no una
  medición omnidireccional calibrada.
