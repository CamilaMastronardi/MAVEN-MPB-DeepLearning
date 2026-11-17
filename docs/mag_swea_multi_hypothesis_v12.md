# V12 — Selector de múltiples hipótesis refinadas de MPB (MAG + SWEA)

## Objetivo y contexto

El modelo V10 localizó **53,0 %** de 559 MPB dentro de +/-60 segundos
(mediana de error: 55,5 s). El objetivo experimental es alcanzar al
menos **80 % dentro de +/-60 s**, que sobre esos mismos 559 eventos
equivale a **448 aciertos** (152 más que V10 aproximadamente).

El primer fold completo de V11 (ranking global) produjo 61,0 % dentro
de +/-60 s frente a 62,3 % de V10 y mejoró individualmente 55/146
eventos. V11 modificaba el candidato grueso ANTES de usar el refinador.
V12 modifica ese enfoque: **refina varias propuestas con V10 y
después aprende cuál de los instantes refinados es más plausible**.

80 % es una meta, no una cifra esperada ni garantizada.

## Arquitectura

**Etapa A — V7b congelado, búsqueda exhaustiva de propuestas.**

Se puntúan todas las ventanas válidas de 10 minutos, separadas por
32 s, en una región de +/-30 minutos alrededor de la MPB catalogada.
Se reutilizan las mediciones MAG y SWEA, normalización fold-TRAIN,
encoders y clasificación de regiones físicas de V7b.

De esa secuencia se conservan los **12 candidatos mejor puntuados**
por V7b con **separación mínima de 120 s** entre centros. La separación
busca evitar que los 12 pertenezcan a la misma fluctuación local.
El primer candidato es exactamente el máximo de V7b (mismas reglas
float32/sigmoid para empates).

**Etapa B — V10 congelado, refinamiento de cada propuesta.**

Cada uno de los 12 candidatos se procesa con el modelo V10 del
mismo fold, utilizando su calibración original para estimar una
corrección temporal de hasta +/-240 s. Para cada hipótesis se guarda:
- propuesta física (16 características MAG+SWEA y cabezas V7b);
- tiempo corregido según V10;
- probabilidad auxiliar de presencia en la ventana;
- media/máximo de la distribución temporal;
- entropía normalizada y dispersión temporal (incertidumbre);
- acuerdo entre las distintas hipótesis a 60, 120 y 240 s.

**Etapa C — Selector V12 entrenable.**

Una MLP pequeña compartida con contexto global (media y máximo
del embedding del conjunto) recibe **28 características por
hipótesis** y puntúa las 12. Es invariante al orden de entrada
(de hecho, equivarante: al permutar propuestas se permutan las
puntuaciones) y no recibe el tiempo absoluto, la distancia al
centro del intervalo ni la posición del candidato respecto de
la anotación de MPB.

Las características de consenso usan exclusivamente **diferencias
entre los tiempos estimados de las hipótesis**: permanecen
idénticas si todos los candidatos se trasladan temporalmente.

La predicción final es uno de los 12 tiempos ya refinados. Es
diferente de V10 porque puede elegir otra estructura, y diferente
de V11 porque la selección se basa en las hipótesis **después**
del refinamiento V10 y su incertidumbre.

## Entrenamiento de V12

Solo se entrenan los pesos del selector V12; V7b y V10 están
congelados. Se minimiza una pérdida listwise de elección de mejor
hipótesis. Los objetivos se construyen usando el error absoluto
entre el tiempo refinado de cada hipótesis y la MPB catalogada.

- Distribución objetivo suave con escala 90 s alrededor del
  mínimo error dentro de la lista.
- Pérdida de entropía cruzada entre esas probabilidades objetivo
  y el ranking de la red.
- Pérdida binaria auxiliar para reforzar propuestas de error
  <=60 s frente a propuestas claramente incorrectas (>=120 s).
  Las intermedias no participan en ese término.

**Importante:** esos errores temporales se calculan para construir
etiquetas y métricas; NO entran como variables de entrada de la red.

## Validación y protección contra fugas de información

- Cuatro folds temporales originales. Mismos 559 eventos para la
  comparación final.
- Los pesos de V7b/V10 y la normalización de mediciones corresponden
  al mismo fold y permanecen congelados.
- La normalización de las 28 características del selector se ajusta
  exclusivamente a eventos TRAIN.
- Early stopping: se prioriza porcentaje <=60 s en meses MONITOR
  internos; empates favorecen menor error medio truncado y pérdida.
- Mezcla de ranking con V7b: alpha entre 0 y 1, calibrada solo sobre
  MONITOR. Con **alpha=0**, se usa el candidato original y, para
  los resultados externos, la predicción literal del CSV V10,
  garantizando un fallback exacto.
- Se comprueba que la mejor propuesta recuperada coincide con
  el máximo del V7b original antes de evaluar cualquier fold.
- No se utiliza el fold externo para gradientes, normalizadores,
  early stopping o elección de alpha.

**Cuidado metodológico:** los mismos cuatro folds ya han servido
para elegir arquitecturas V7b a V12. Aunque la evaluación interna
esté separada por meses, una afirmación científica final necesita
nuevos datos temporales independientes.

## Diagnóstico imprescindible: ¿es posible llegar al 80 %?

Se reporta `oracle_shortlist_refined_within_60s`, proporción de
eventos donde **al menos una** de las 12 predicciones refinadas
termina dentro de +/-60 s del catálogo. Es un **techo teórico**
del selector para esta shortlist y estos modelos congelados.

- Si este techo es menor del 80 %, entonces V12 no puede alcanzar
  la meta cambiando solo el entrenamiento del selector.
- Si el techo supera 80 %, pero V12 queda lejos, el límite principal
  está en la selección y no en la cobertura de hipótesis.

Este techo se calcula utilizando la etiqueta SOLO como diagnóstico,
después de generar las predicciones. No interviene en la decisión
de cuál candidato elegir en inferencia.

## Archivos

- `src/detection/mpb_multi_hypothesis_v12.py`
- `scripts/5_training/5h_train_mag_swea_multihypothesis_v12.py`
- `tests/test_mag_swea_multihypothesis_v12.py`
- `scripts/5_training/run_mag_swea_multi_hypothesis_v12.sh`

Resultados separados:

    results/metrics/mag_swea_multi_hypothesis_v12/
      features/fold_01/AAAAmmddTHHMMSSffffff.npz
      fold_01_last.pt
      fold_01_best.pt
      fold_01_final.pt
      fold_01_metrics.json
      fold_01_event_metrics.csv
      ...
      cv_summary.json
      training.log

La caché por evento contiene puntajes V7b, el conjunto de 12 hipótesis
y las correcciones V10. Se reutiliza si se interrumpe el trabajo, con
verificación de hashes de fuentes/modelos.

## Ejecución

Primero una prueba funcional pequeña:

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate

FOLDS=1 PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_swea_multi_hypothesis_v12_smoke \
bash scripts/5_training/run_mag_swea_multi_hypothesis_v12.sh
```

Los 8 eventos NO proporcionan una estimación fiable del desempeño.
Sirven para comprobar que los modelos se cargan, se procesan
las 12 hipótesis por evento y se generan archivos sin errores.

Después la validación completa:

```bash
tmux new -s mpb_v12
bash scripts/5_training/run_mag_swea_multi_hypothesis_v12.sh
```

Desacoplar tmux: Ctrl+B, D. Retomar:

```bash
tmux attach -t mpb_v12
```

El script primero ejecuta los tests específicos. Si fallan, no se
inicia el entrenamiento. Las pruebas no se han ejecutado en el
entorno real WSL2 de la usuaria desde el conector GitHub.

## Resultados

Al terminar:

```bash
python -m json.tool \
    results/metrics/mag_swea_multi_hypothesis_v12/cv_summary.json
```

Analizar en cada fold:
1. acierto <=60 s V12 frente a V10;
2. mediana de error;
3. porcentaje de eventos que V12 mejora/empeora;
4. alpha del selector calibrado;
5. **techo teórico de la shortlist V10-refinada**.

El objetivo sigue siendo localización temporal CONDICIONADA a
que existe una MPB catalogada en el intervalo de 70 minutos.
Estos resultados no miden detección autónoma a lo largo de órbitas.
