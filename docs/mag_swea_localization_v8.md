# V8 — mejorar la localización de la MPB cuando se sabe que existe

## Objetivo

Esta prueba **no** clasifica si una órbita/serie tiene una MPB o no.
Partimos de un intervalo de unos 70 minutos que contiene un cruce catalogado
y buscamos el centro temporal donde el puntaje de MPB es máximo, como en V7b.

La referencia es V7b completa: 559 eventos MAG+SWEA, mediana 72 s,
250/559 (44.7 %) a <=60 s, F1 de candidatos débiles 0.599. Esas métricas
provienen de la evaluación condicionada a eventos anotados.

## Cambio respecto de V7b

Los negativos explícitos de V7b estaban principalmente entre 600 y 1800 s
de la anotación. Esto puede separar MPB de estados estables, pero no obliga
a resolver un pico preciso cerca del cruce.

V8 comienza con **los pesos reales de cada fold de V7b**, conserva encoders
MAG/SWEA, clasificación de regiones, reconstrucción y ranking de controles
lejanos. Añade un objetivo de *ranking ordinal cerca del cruce*:

- Centro positivo: t = t_MPB del catálogo.
- Otros positivos binarios: t_MPB +/- 30 s, según la disponibilidad.
- Candidatos temporalmente próximos: +/- 96, 160, 224, 320 y 448 s.
- Los candidatos próximos **no son etiquetas negativas**. No se les aplica
  BCE binaria ni BCE de regiones; solo se pide de forma suave un puntaje
  inferior al del centro catalogado. Se excluyen ventanas cerca de BS u
  otras MPB/BS anotadas y las de cobertura MAG/SWEA insuficiente.
- Cada época usa hasta uno de cada lado entre los cercanos, más los
  controles originales de V7b (dos difíciles + dos aleatorios por evento).

La nueva pérdida de cercanía es:

    L_near = mean_events mean_near softplus(
        0.1 + logit_near - logit_center
    )

Se promedia primero por evento y se pondera por defecto 0.7. La pérdida
completa mantiene BCE_MP B + 0.5*BCE_regiones +
0.15*(Recon_MAG+Recon_SWEA) + 1.0*ranking_lejano + 0.7*L_near.
Los pesos son exploratorios; no están validados contra datos futuros.

## Protección de validación

- CV externa: mismos cuatro folds mensuales que V7b.
- Cada V8 se inicializa **solo con el checkpoint V7b correspondiente**,
  cuyo entrenamiento no vio los eventos de prueba de ese fold.
- Normalizadores: exactamente los guardados en ese checkpoint V7b,
  calculados usando exclusivamente su TRAIN interno.
- Selección de época: BCE detección + 0.5*BCE región +
  1.0*rank lejano + 0.7*rank cercano sobre meses de MONITOR interno.
- Evaluación externa: máximo dentro de +/-30 minutos en rejilla cada
  32 s, igual a V7b; mismo conjunto de eventos, sin obligar a alcanzar
  un umbral ni inventar intervalos negativos de 70 min.
- Salida: métricas y CSV por fold; en cada evento se guardan también
  el error previo V7b y la diferencia de errores. El resumen global
  calcula la mediana a partir de todos los errores individuales.
- La comparación no es una validación completamente nueva, porque ya
  utilizamos estos folds para decidir probar esta arquitectura. Hace
  falta una prueba temporal intacta para una estimación sin ese sesgo.

## Ejecutar entrenamiento completo (CPU)

    cd ~/Codigos/MAVEN-MPB-DeepLearning
    git pull origin main
    source .venv/bin/activate
    bash scripts/5_training/run_mag_swea_localization_v8.sh

Para desacoplarlo de la terminal (si el sistema dispone de tmux):

    tmux new -s mpb_v8
    bash scripts/5_training/run_mag_swea_localization_v8.sh

Salir de tmux sin detener el proceso: Ctrl+B, después D.
Reconectar: tmux attach -t mpb_v8.

El launcher hace py_compile y las pruebas unitarias ANTES de iniciar el
entrenamiento. Si falla una prueba no inicia ningún fold. Reutiliza las
cachés ya construidas; si encuentra fuentes ausentes ejecuta preflight
BigQuery con los topes existentes antes de consultar.

Para una prueba pequeña, usando igualmente checkpoints V7b completos:

    PILOT_EVENTS_PER_FOLD=8 OUTPUT_DIR=results/metrics/mag_swea_localization_v8_pilot bash scripts/5_training/run_mag_swea_localization_v8.sh

Por defecto, salida aislada:

    results/metrics/mag_swea_localization_v8/

Archivos relevantes:

- fold_XX_metrics.json
- fold_XX_event_metrics.csv
- fold_XX_best.pt, fold_XX_last.pt, fold_XX_final.pt
- cv_summary.json
- training.log

El flag --resume se usa por defecto en el launcher. Nunca escribir en la
carpeta original de V7b. Si cambia la configuración, usar otro OUTPUT_DIR.

## Qué consideramos mejora

Comparar sobre los mismos 559 eventos, especialmente:
1. Mediana global menor que 72 s.
2. Más de 250/559 (44.7 %) dentro de 60 s.
3. Reducción de grandes errores (>300 s).
4. Número de eventos que mejoran vs empeoran comparados de forma emparejada.

**El criterio de éxito es localización condicional, no detección autónoma.**
Para detectar presencia/ausencia de MPB en una serie arbitraria de 70 min
se necesitaría otro clasificador previo y un corpus independiente de
intervalos sin cruces (con verificación física de negativos).
