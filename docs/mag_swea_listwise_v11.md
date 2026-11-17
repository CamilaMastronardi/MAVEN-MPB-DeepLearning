# V11 — Ranking global de candidatos MPB + refinamiento temporal de V10

## Objetivo

Se mantiene la hipótesis de que **el intervalo de 70 min contiene una
MPB catalogada**. Queremos estimar el instante de esa MPB. No estamos
entrenando un clasificador de presencia/ausencia de MPB en órbitas completas.

**Meta cuantitativa:** >=80 % de los 559 eventos dentro de +/-60 s
(al menos 448 eventos). Referencia V10: aproximadamente 296/559 = 53 %
dentro de +/-60 s, mediana de error 55,5 s. El 80 % es una hipótesis
experimental ambiciosa, nunca un resultado prometido.

## Razón de cambiar la arquitectura

V10 conserva V7b y mejora la posición del candidato que V7b eligió como
máximo. Pero si ese máximo está a muchos minutos de la MPB, una corrección
limitada a +/-240 s no puede recuperar el cruce verdadero.

V11 puede elegir un **candidato totalmente distinto** en la región
explorada y después refinarlo con la cabeza temporal congelada de V10.

## Arquitectura y etapas

1. **Propuestas completas (V7b, congelado).** Para cada una de las
   ventanas de 10 min válidas, con centros cada 32 s dentro de
   +/-30 min alrededor de la MPB catalogada, se calculan el logit
   de transición de V7b, sus dos logits de clasificación de regiones
   (magnetovaina / magnetosfera inducida) y características físicas
   directamente de los datos MAG+SWEA medidos. Se conservan los
   encoders originales, la normalización TRAIN-only de su fold, sus
   dos series con cadencias propias y sus filtros de gaps.
2. **Ranking global nuevo.** Un TCN compacto con convoluciones
   temporales entre candidatos recibe 16 características por candidato.
   Usa características del candidato, vecinos y un resumen global
   promedio de los candidatos de ese evento. Predice un logit
   residual para **cada** candidato. No recibe la hora de la MPB,
   el índice del centro del evento, la dirección/BS anotada, ni
   el tiempo del candidato relativo a la MPB como features.
   Su ranking se combina con los logits V7b.
3. **Refinamiento fino (V10, congelado).** Una vez elegido el mejor
   candidato global, se aplica exactamente el refinador entrenado
   de V10 correspondiente al mismo fold, con su alpha y su método
   media/máximo seleccionados anteriormente en el monitor de V10.
4. **Calibración del ranking.** Se prueban pesos alpha de
   {0, 0.5, 1, 1.5, 2} para el residual y se selecciona el
   que obtiene mayor proporción <=60 s en los **meses internos
   de MONITOR del fold TRAIN**. Empates priorizan menor error
   medio truncado y menor alpha. Alpha=0 devuelve exactamente
   la selección original V7b seguida de V10: es una opción
   legítima si la nueva red no mejora.

La salida es un instante de cruce, no un umbral autónomo de detección.

## Entrenamiento listwise

Para cada evento de entrenamiento con datos reales, se puntúan
**todas** las ventanas válidas de su intervalo de 70 min. No
se excluyen de entrada las propuestas con bajo V7b score.
El objetivo de entrenamiento es una distribución gaussiana suave
de sigma=56 s alrededor del tiempo MPB catalogado:

    target_i = softmax_i(-0.5 * (offset_i / 56 s)^2)

El loss de ranking del evento es:

    L_event = -sum_i target_i * log_softmax_i(
        (v7b_logit_i + v11_residual_i) / 2
    )
    L = mean_events(L_event) + 0.0005 * mean(residual_i^2)

El offset respecto de MPB **solo entra para construir los targets**
durante el aprendizaje y para reportar errores. Nunca entra como
variable de entrada de la nueva red, al igual que los IDs de evento
y los tiempos absolutos. El modelo debe reconocer el cruce a través
de las mediciones y las predicciones físicas de V7b.

Se aprende con los meses FIT; se guardan checkpoints seleccionados
por pérdida listwise sobre meses de MONITOR de TRAIN. Ninguna
medición del fold OUTER se utiliza para calcular gradientes,
normalizadores, early stopping o elegir alpha.

## Evaluación y diagnósticos

- Cuatro folds cronológicos, los mismos 559 eventos de V7b y V10.
- Comparación **emparejada** V11 vs V10 vs V7b.
- Mediana, media, P90 y acierto <=60, 120 y 300 s.
- Número de eventos que mejoran/empeoran respecto de V10.
- Diagnóstico de cobertura entre los top-5 candidatos separados
  temporalmente de V7b: a qué distancia está el más cercano a la
  anotación. Este es un *análisis a posteriori* que utiliza la
  anotación, no una técnica de selección de candidatos.
- Archivo por evento con propuesta V7b, propuesta V11 y predicción
  refinada final para revisar errores grandes.

El benchmark NO aporta una validación final en eventos nunca
consultados: hemos usado los mismos folds para explorar V7b/V8/V9/V10.
Para conclusiones de tesis necesitaríamos después una cohorte temporal
inédita y, eventualmente, comparar contra incertidumbres manuales
de anotación MPB.

## Archivos del repositorio

- `src/detection/mpb_listwise_reranker_v11.py` — arquitectura y
  características físicas.
- `scripts/5_training/5g_train_mag_swea_listwise_v11.py` —
  caché de features, entrenamiento, evaluación y comparación.
- `tests/test_mag_swea_listwise_v11.py` — tests sintéticos, incluidos
  fallback exacto a V7b y aislamiento del target.
- `scripts/5_training/run_mag_swea_listwise_v11.sh` —
  launcher de entrenamiento con `--resume`.

## Ejecución en WSL2/CPU

Es conveniente probar primero la infraestructura:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_swea_listwise_v11_pilot \
bash scripts/5_training/run_mag_swea_listwise_v11.sh
```

Si se verifica que no hay errores y la salida es coherente,
ejecutar todos los eventos en un directorio diferente:

```bash
tmux new -s mpb_v11
bash scripts/5_training/run_mag_swea_listwise_v11.sh
```

Salir de tmux sin cortar: Ctrl+B, luego D. Volver:

```bash
tmux attach -t mpb_v11
```

El launcher hace `py_compile` y ejecuta pruebas específicas antes
de comenzar. **No se han ejecutado estas pruebas en la WSL del usuario
desde GitHub**. Conviene supervisar el resultado del piloto antes
del entrenamiento completo.

Se reutilizan `data/cache/mag_events` y
`data/cache/swea_events`, y se crea una caché adicional de
características en:

    results/metrics/mag_swea_listwise_v11/features/fold_XX/*.npz

Su finalidad es evitar recalcular las ~100 propuestas por evento tras
una interrupción, con verificación de hashes del modelo V7b y del
manifiesto. El primer cálculo de características por fold puede
llevar bastante tiempo en CPU. Si faltan bloques de medidas, la
carga heredada de V7b ejecuta un preflight BigQuery y respeta
límites configurados.

Resultados:

    results/metrics/mag_swea_listwise_v11/
      features/fold_01/*.npz
      fold_01_last.pt
      fold_01_best.pt
      fold_01_final.pt
      fold_01_metrics.json
      fold_01_event_metrics.csv
      ...
      cv_summary.json
      training.log

Al finalizar:

```bash
python -m json.tool results/metrics/mag_swea_listwise_v11/cv_summary.json
```

## Limitaciones

- El modelo puede cometer errores grandes si puntúa estructuras distintas
  a la MPB, aun después de comparar todos los candidatos.
- El entrenamiento compara un único instante catalogado por evento;
  múltiples cruces o imprecisiones de la anotación pueden complicar
  la función objetivo.
- El objetivo 80 % con tolerancia +/-60 s puede ser difícil si hay
  incertidumbre de etiquetado o transiciones físicas extendidas.
- La calibración por monitor favorece porcentaje <=60 s, pero sobre un
  subconjunto limitado; el desempeño externo no está garantizado.
- Al igual que V7b/V10, la evaluación supone ya conocida la presencia
  de MPB dentro de una ventana de ~70 min. No demuestra detección
  orbital autónoma.
