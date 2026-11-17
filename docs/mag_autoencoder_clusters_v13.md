# V13: primera fase — Autoencoder CNN de 8 capas, espacio latente 3D y agrupamiento de MPB

## Idea de trabajo

La meta científica sigue siendo ubicar el **tiempo de cruce de la MPB**
dentro de +/-60 segundos en al menos **80 % de los eventos**. Los
resultados previos fueron: V10 53,0 %, V12 56,9 % sobre 559 eventos.
V12 además mostró que el 87,3 % de las ventanas evaluadas contenía
**al menos una** de sus 12 hipótesis refinadas dentro de +/-60 s,
aunque el selector rara vez identificaba sistemáticamente esa hipótesis.

Para V13 se eligió **cambiar de paradigma**. En vez de sumar otra
red de ranking, primero buscamos **estructuras morfológicas distintas
entre los cruces**. Luego, si los grupos son estables y físicamente
interpretables, se estudiará una red localizadora especializada
para cada uno.

**Esta primera fase no es el localizador final**. Se entrena un
autoencoder simple y se clasifican exploratoriamente los cruces,
sin utilizar aún la pertenencia a un grupo para inferir automáticamente
una MPB en un intervalo desconocido.

## Datos elegidos para la primera etapa

- Ventanas reales de **MAG** de 10 minutos, con cuatro canales:
  Bx, By, Bz y |B|.
- Cadencia MAG nativa de aproximadamente 1 s. No se interpolan
  las mediciones faltantes: la entrada se rellena con ceros
  hasta 640 muestras y se mantiene una **máscara** para ignorar
  el relleno en la pérdida de reconstrucción.
- Se reutiliza el conjunto de eventos con MAG y SWEA disponibles
  y los filtros conservadores de ventanas seguras de V9. En esta
  fase la CNN recibe **solo MAG**: SWEA conserva su cadencia nativa
  y se incorporará después como una rama separada o como
  herramienta para interpretar los grupos. No se inventan mediciones
  SWEA a 1 Hz.
- Normalización de MAG ajustada únicamente con las mediciones
  de los eventos de entrenamiento (épocas únicas, no muestras
  duplicadas entre ventanas).
- Positivos de 10 minutos con MPB en posiciones distintas respecto
  del centro: -240 a +240 s, en intervalos de 60 s. Negativos
  solo de regiones estables, lejos de MPB y BS catalogados.
  Estas ventanas evitan que toda etiqueta positiva esté
  exactamente centrada en t=0, lo que sería una fuga de
  información para aprender un localizador.

## Arquitectura

**Encoder:** 8 capas CNN-1D de convolución con stride=2,
con dimensiones de canales 4→16→24→32→40→48→56→64→64.
La dimensión temporal pasa de 640 a 3, y una capa lineal
comprime las activaciones a tres escalares:

    MAG[4 x 640] -> 8 Conv1d -> FC -> z=(z1,z2,z3)

**Decoder:** a partir de `z`, una capa lineal y
ocho bloques Conv1d con interpolación *de activaciones* (no de
las mediciones) reconstruyen la señal MAG:

    (z1,z2,z3) -> FC -> 8 bloques upsampling+Conv1d
               -> MAG reconstruida[4 x 640]

La reconstrucción es auxiliar para aprender una representación
compacta y no debe confundirse con la reconstrucción física
exacta de las mediciones: comprimir 2400 valores a 3 números
es extremadamente restrictivo.

Dos cabezas pequeñas reciben únicamente `z`:

- `presence_logit`: clasificación binaria de ventana que
  contiene una MPB frente a un control físicamente seguro.
- `offset_s`: regresión acotada a +/-240 s del instante
  MPB respecto del centro de la ventana. Para una ventana
  positiva, el tiempo propuesto es:

    t_MPB_estimado = centro_de_la_ventana + offset_s

La cabecera temporal se entrena únicamente con ventanas
positivas. Para controles no se asigna ningún tiempo MPB
ficticio. Estos diagnósticos **todavía no constituyen una
detección autónoma en ventanas de 70 minutos**.

## Función de pérdida

    L = L_reconstruccion
        + 0.30 * BCE_presencia_MPB
        + 0.10 * SmoothL1((offset_estimado-offset_real)/60s)

La reconstrucción utiliza MSE únicamente en posiciones con
mediciones MAG válidas. La BCE distingue positivos de
controles seguros. La regresión temporal se calcula solo
para ventanas positivas.

Los pesos son valores exploratorios. No garantizan que el
espacio latente de tres dimensiones separe los tipos
de MPB. Si mejora la reconstrucción pero empeora la
identificación, habrá que ajustar el balance de pérdidas o
revisar la capacidad del cuello de botella.

## Agrupamiento inicial de los cruces

Se entrena **un solo autoencoder de descubrimiento**, para
evitar que cuatro redes distintas definan cuatro espacios
latentes arbitrariamente rotados/escalados.

- Entrenamiento en tres folds y reserva de un cuarto fold
  temporal para diagnóstico independiente. Por defecto el
  fold 4 es la reserva.
- Una selección interna de meses del conjunto de entrenamiento
  determina la época del checkpoint.
- Para describir un cruce catalogado, se codifican varias
  ventanas positivas con desplazamientos distintos y se
  promedia su vector de 3 componentes. Esto busca que el
  grupo represente el cruce y no solo la posición de la
  MPB dentro de una ventana.
- Se estandarizan las coordenadas usando **solo los cruces FIT**.
- Se prueba KMeans con k=2,...,6 (si el tamaño muestral lo permite).
  Se elige k mediante *silhouette score* en FIT, descartando
  particiones con grupos de menos de tres cruces.
- Los meses de monitor y el fold temporal externo reciben
  asignaciones por `predict`, sin volver a ajustar los clusters.
- Se genera un CSV de todos los eventos con z1/z2/z3, grupo,
  cantidad de vistas válidas, MSE y probabilidad de MPB. También
  se guarda una figura del espacio latente tridimensional.

**Cuidado:** las vistas que se promedian para describir un
cruce se seleccionan con la anotación de la MPB. Esto es
apropiado para un **estudio descriptivo** de los cruces ya
catalogados, pero no demuestra que podamos asignar grupos
sin conocer previamente dónde está la MPB. Para usar estos
grupos en un localizador autónomo, después habrá que
probar un método de asignación de grupo a partir de datos
no centrados en el catálogo.

## Diagnósticos

En las ventanas de monitor y también en el fold externo
se informa:
- MSE MAG en ventanas reales (escala normalizada);
- BCE, ROC-AUC y PR-AUC para clasificación de presencia
  de MPB dentro de una ventana;
- error mediano de desplazamiento temporal en las
  **ventanas positivas conocidas** y fracción <=60 s;
- k seleccionado, silhouette TRAIN, tamaño de grupos;
- distribución de grupos por fold, por Group1/2/3 del
  catálogo y error de reconstrucción medio por grupo.

Estos porcentajes de ventana no deben compararse
directamente con el 56,9 % de V12: V12 localizaba
la MPB dentro de un bloque de 70 minutos con muchos
candidatos, mientras que V13 fase 1 trabaja en ventanas
de 10 minutos con etiqueta de positivo conocida
durante el análisis.

## Ejecutar en WSL

Piloto funcional (no sirve para elegir grupos científicamente):

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
git pull origin main
source .venv/bin/activate
PILOT_EVENTS_PER_FOLD=8 \
OUTPUT_DIR=results/metrics/mag_autoencoder_clusters_v13_smoke \
bash scripts/5_training/run_mag_autoencoder_clusters_v13.sh
```

Ejecución completa:

```bash
tmux new -s mpb_v13
bash scripts/5_training/run_mag_autoencoder_clusters_v13.sh
```

Salir de tmux sin detener el proceso: Ctrl+B, después D.
Reingresar: `tmux attach -t mpb_v13`.

El launcher ejecuta `py_compile` y tests sintéticos V13
antes de entrenar. Las pruebas deben ejecutarse y aprobarse
en WSL; no se han ejecutado allí a través del conector de GitHub.

## Salidas

    results/metrics/mag_autoencoder_clusters_v13/
      discovery_last.pt
      discovery_best.pt
      discovery_final.pt
      discovery_summary.json
      crossing_latents_groups.csv
      latent_clusters_3d.png
      training.log

Revisar:

```bash
python -m json.tool \
  results/metrics/mag_autoencoder_clusters_v13/discovery_summary.json
```

La tabla de asignación de grupos se abre con pandas o Excel.

## Siguiente fase (todavía NO implementada)

Una vez conocidos los grupos y sus propiedades físicas:

1. Examinar qué diferencia sus perfiles MAG, condiciones de
   magnetización, intensidad y cambios en el campo; comprobar
   luego si SWEA distingue los mismos grupos.
2. Revisar cuántos eventos hay por grupo y si se mantienen
   en distintos períodos. Evitar entrenar redes separadas
   si un grupo tiene pocas decenas de eventos.
3. Evaluar por grupo cuál es la dificultad de localización
   temporal de V10/V12, utilizando CSV de los mismos eventos.
4. Proponer entonces un localizador por grupo o una red
   compartida con *heads* específicos. La salida final
   deberá ser un instante UTC y se medirá el porcentaje
   dentro de +/-60 s mediante una búsqueda sobre los
   intervalos completos de 70 minutos, sin elegir la ventana
   centrada en la MPB verdadera.

El objetivo de >=80 % es experimental y debe evaluarse
en datos temporales no utilizados para escoger el modelo.
