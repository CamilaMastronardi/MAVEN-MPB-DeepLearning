# Localización temporal de la MPB de Marte con MAVEN

**Trabajo final — Machine Learning en Geofluidos · Universidad de Buenos Aires**

Este proyecto estudia la localización temporal de la **Magnetic Pileup Boundary (MPB)** de Marte utilizando las observaciones de **MAG** y **SWEA** de la misión MAVEN. Se desarrolla y evalúa **V7b**, una red neuronal convolucional multimodal que combina clasificación de transiciones, clasificación auxiliar de regiones plasmáticas, reconstrucción de las señales y *ranking* de ventanas candidatas.

### Entrega y enlaces principales

- **[Notebook del trabajo final — 01_MPB_V7b.ipynb](notebooks/01_MPB_V7b.ipynb)**: explicación física, metodología, arquitectura, entrenamiento, evaluación y análisis de las representaciones latentes. Es el **documento principal** de la entrega.
- **[Implementación del modelo](src/model/v7b.py)** y **[pérdida de ranking](src/model/ranking.py)**.
- **[Entrenador V7b](scripts/02_training/01_train_v7b.py)**: validación cruzada, selección de checkpoints y evaluación.
- **[Documentación técnica](docs/)**: preparación de datos, arquitectura, entrenamiento y experimentos.

> **Alcance de la evaluación.** El modelo busca el instante de máxima puntuación dentro de un intervalo de **±30 minutos de una MPB catalogada**. Esto evalúa la capacidad de **localizar** un cruce conocido; no demuestra todavía la detección autónoma de MPB en órbitas completas.

## 1. Datos y preparación

El catálogo de referencia contiene **569 cruces MPB**, de los cuales **559** tienen cobertura SWEA indicada en el manifiesto. La disponibilidad efectiva para cada análisis depende de la calidad y continuidad de las mediciones de ambos instrumentos.

| Instrumento | Variables de entrada | Cadencia aproximada | Preprocesamiento |
| --- | --- | --- | --- |
| **MAG** | \(B_x,B_y,B_z,\lvert\mathbf B\rvert\) (**4 canales**) | 1 s | Control de calidad y normalización por canal |
| **SWEA** | Intensidad electrónica en **12 canales energéticos** entre 30 y 300 eV | 32 s | Promedio angular, interpolación **en energía**, `log1p` y normalización |

Cada candidato es una ventana de **600 segundos** (±300 s alrededor de un centro). MAG y SWEA conservan **sus cadencias nativas**: no se interpola SWEA temporalmente para igualarla con MAG. Las ventanas incompletas, inválidas o con discontinuidades incompatibles se excluyen.

La normalización se ajusta exclusivamente con datos del subconjunto **FIT** correspondiente. Las etiquetas de transición y de región se construyen con los instantes catalogados de MPB y *bow shock* (BS); son **etiquetas débiles**, no un etiquetado independiente de todas las regiones a partir de las mediciones.

El conjunto de datos se divide en **cuatro folds externos agrupados por mes UTC**. Dentro de los tres folds de entrenamiento se separan, también por meses, FIT (actualización de pesos) y **MONITOR** (selección de checkpoints y umbrales), evitando superposiciones de los intervalos físicos usados.

## 2. Arquitectura V7b

Cada instrumento pasa por un **encoder convolucional 1D independiente**, con una primera convolución de 24 canales, bloques residuales dilatados y una salida de **16 series temporales latentes**. Las dilataciones de los bloques son 1, 2, 4 y 8 para MAG, y 1 y 2 para SWEA.

Para cada ventana candidata se consideran los tiempos relativos a su centro \(t=0\):

- **Lado anterior:** muestras válidas entre −300 y −60 s.
- **Región central excluida del *pooling*:** entre −60 y +60 s.
- **Lado posterior:** muestras válidas entre +60 y +300 s.

Se promedia **cada uno de los 16 canales latentes en el tiempo**, de manera independiente para MAG y SWEA y usando las máscaras y los tiempos propios de cada instrumento. Así se obtienen **16 valores por instrumento y por lado**, que se concatenan en un vector de **32 características anterior** y otro de **32 posterior**. No se dividen los canales en dos grupos: se resumen **los mismos 16 canales antes y después**.

La arquitectura tiene tres objetivos complementarios:

1. **Transición MPB:** una cabeza utiliza los vectores anterior y posterior, su diferencia y la magnitud de esa diferencia para producir un logit de compatibilidad con un cruce.
2. **Región plasmática:** una cabeza auxiliar clasifica cada lado como **magnetovaina** o **magnetosfera inducida** a partir de etiquetas débiles.
3. **Reconstrucción:** decodificadores específicos de MAG y SWEA reconstruyen las entradas normalizadas para regularizar las representaciones.

Los **16 canales latentes por instrumento** son una decisión de capacidad expresiva del modelo y no se interpretan como 16 variables físicas independientes.

## 3. Entrenamiento con ranking

Para cada evento se forman grupos de candidatos: hasta **dos positivos** centrados cerca del cruce catalogado, **dos negativos difíciles** identificados mediante las puntuaciones de épocas anteriores y **dos controles negativos aleatorios**. Todos pertenecen al mismo evento. La memoria de negativos difíciles se construye **solo con FIT**.

La pérdida del modelo de referencia es:

\[
\mathcal{L}_{\mathrm{V7b}}=
\mathcal{L}_{\mathrm{trans}}
+0.5\,\mathcal{L}_{\mathrm{región}}
+2\,\mathcal{L}_{\mathrm{rank}}
+0.15\,\mathcal{L}_{\mathrm{rec,MAG}}
+0.15\,\mathcal{L}_{\mathrm{rec,SWEA}}.
\]

La componente de ranking penaliza que un control negativo reciba un logit mayor o demasiado cercano al de un positivo del **mismo cruce**. Esto permite entrenar comparaciones relativas y no solamente una clasificación independiente por ventana.

El entrenamiento de referencia utiliza **AdamW** (tasa de aprendizaje \(3\times 10^{-4}\), *weight decay* \(10^{-3}\)), tamaño de lote **16**, semilla **42**, máximo **100 épocas** y *early stopping* con paciencia **8**. Los pesos, el margen de ranking y otras opciones pueden modificarse mediante argumentos del entrenador; los valores concretos deben comprobarse en las métricas de la corrida analizada.

## 4. Evaluación y análisis del notebook

En cada fold externo se recorren centros candidatos cada **32 s** dentro de los **±30 min** del instante catalogado. Se evalúan las ventanas que tienen datos válidos de ambos instrumentos y se selecciona el centro con **mayor puntuación de transición**:

\[
\hat t_{\mathrm{MPB}}=\operatorname*{arg\,max}_{t_i}
\sigma\bigl(s(t_i)\bigr),\qquad
\epsilon=\lvert \hat t_{\mathrm{MPB}}-t_{\mathrm{MPB}}\rvert.
\]

El notebook incluye:

| Secciones | Contenido |
| --- | --- |
| **1–2** | Contexto físico, catálogo, MAG/SWEA, cobertura y divisiones temporales |
| **3–4** | Candidatos positivos/negativos, ranking, encoders, decoders y cabezas de clasificación |
| **5** | Configuración y ejecución reproducible del entrenamiento V7b |
| **6.1–6.3** | Evolución de las pérdidas, métricas por fold y distribución del error temporal |
| **6.4–6.6** | Matrices de confusión de candidatos MPB/control y de regiones plasmáticas |
| **6.7** | Experimento **opcional e independiente** con pérdida temporal explícita |
| **7** | Curvas del score de transición, morfología de picos y relación con el error |
| **8** | Mapas latentes, PCA, diferencias entre lados, casos individuales y correlaciones de canales |
| **9** | Discusión, limitaciones y trabajos futuros |

Entre las métricas se reportan el **error absoluto mediano de localización**, la **fracción de cruces dentro de 60 y 120 s**, las métricas de clasificación de candidatos con etiquetas débiles y la clasificación de regiones. Las matrices de confusión por candidato **no equivalen** a las métricas de localización por evento.

En la sección 8.7 se muestran **dos matrices de correlación promedio de 16 × 16**, una para MAG y otra para SWEA, ordenadas mediante agrupamiento jerárquico. Son un análisis **exploratorio**: los canales de redes entrenadas independientemente **no están alineados semánticamente** entre folds, por lo que su promedio por índice no permite afirmar que representen las mismas características físicas.

## 5. Reproducibilidad

### Entorno

Desde la raíz del repositorio:

```bash
git clone https://github.com/CamilaMastronardi/MAVEN-MPB-DeepLearning.git
cd MAVEN-MPB-DeepLearning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter lab notebooks/01_MPB_V7b.ipynb
```

El notebook también contempla ejecución local o en Colab. Las celdas de inicialización permiten definir por separado el directorio de código y el de datos mediante `MAVEN_V7B_REPO_DIR` y `MAVEN_V7B_DATA_ROOT`.

**Para consultar resultados existentes:** abrir el notebook, ejecutar las celdas iniciales y disponer de los archivos de métricas, checkpoints y cachés usados en la corrida. La variable `RUN_TRAINING` está inicialmente en `False`: abrir el notebook **no ejecuta automáticamente** el entrenamiento. Las figuras que dependen de observaciones MAVEN o checkpoints no pueden reproducirse únicamente con el código fuente.

### Entrenamiento V7b de referencia

```bash
python scripts/02_training/01_train_v7b.py \
  --folds all \
  --rank-weight 2 --rank-margin 0.25 \
  --region-weight 0.5 --reconstruction-weight 0.15 \
  --output-dir results/metrics/v7b_manual \
  --max-epochs 100 --patience 8 \
  --resume --execute
```

El entrenador reutiliza datos MAG/SWEA en caché. Si faltan, puede acceder a **BigQuery** con autenticación y límites de consulta, por lo que esta operación no debe confundirse con la simple visualización del notebook. Los datos originales y las cachés voluminosas no se distribuyen junto con el código. El proyecto incluye los módulos de preparación y el manifiesto de validación cruzada.

### Experimento adicional: penalización temporal

La sección **6.7** usa un script **separado**, [`05_train_v7b_temporal_full.py`](scripts/03_experiments/05_train_v7b_temporal_full.py), que entrena **desde cero** con:

\[
\mathcal{L}_{\mathrm{nueva}}=
\mathcal{L}_{\mathrm{V7b}}+
5\mathcal{L}_{\mathrm{temporal}},
\]

donde \(\mathcal{L}_{\mathrm{temporal}}\) es la distancia absoluta al instante MPB catalogado ponderada por un **softmax diferenciable de los logits de los candidatos del mismo evento**, normalizada por 600 s.

```bash
python scripts/03_experiments/05_train_v7b_temporal_full.py \
  --folds all --temporal-weight 5 \
  --max-epochs 1000 --patience 20 --resume
```

Este experimento conserva los cuatro folds, la minería de negativos y la separación FIT/MONITOR. **Solo lee cachés locales** y se detiene si falta la cobertura de algún evento elegible; **no consulta BigQuery**. Guarda los resultados por separado en `results/metrics/v7b_temporal_full_weight5/`.

**Estado:** la sección y su script están implementados; **no se afirma aquí que el entrenamiento completo esté ejecutado ni que mejore los resultados**. El script [`04_quick_temporal_loss_pilot.py`](scripts/03_experiments/04_quick_temporal_loss_pilot.py) corresponde a un experimento anterior de **ajuste fino**, distinto del entrenamiento desde cero. Además, el cambio de máximo de épocas y paciencia impide atribuir eventuales diferencias únicamente a la nueva pérdida sin repetir un control con la misma configuración.

## 6. Organización del repositorio

```text
data/                         Catálogo y manifiesto de los folds
docs/                         Documentación técnica del proceso
notebooks/
  01_MPB_V7b.ipynb            Trabajo final (informe principal)
src/
  preprocessing/              BigQuery, cachés, etiquetas débiles, ventanas y folds
  model/
    v7b.py                    Arquitectura MAG+SWEA y reconstrucción
    ranking.py                Pérdida de ranking y negativos difíciles
    score_shape.py            Análisis de picos del score
scripts/
  01_data/                    Preparación de manifiestos
  02_training/                Entrenamiento V7b de referencia
  03_experiments/             Barridos y experimentos adicionales
    04_quick_temporal_loss_pilot.py
    05_train_v7b_temporal_full.py
tests/                        Pruebas unitarias sin datos BigQuery
requirements.txt              Dependencias de Python
```

Los directorios de caché y resultados generados en las ejecuciones se mantienen **localmente**, según las reglas de `.gitignore`. Para replicar todas las figuras y métricas se requieren también los datos y checkpoints de las corridas correspondientes.

## 7. Limitaciones

- Las etiquetas MPB/control y magnetovaina/magnetosfera inducida son **débiles** y dependen de anotaciones catalogadas.
- Los resultados de localización están **condicionados a la presencia de una MPB catalogada** en el intervalo estudiado; no miden precisión o *recall* de detección sobre órbitas completas.
- Las ventanas de un mismo evento se superponen temporalmente y **no constituyen observaciones independientes** para interpretar las matrices de confusión.
- El análisis del espacio latente es exploratorio; las correlaciones, por sí solas, **no prueban redundancia funcional ni identifican una magnitud física** en cada canal.
- La cobertura y reproducibilidad de las figuras dependen de los datos de MAG/SWEA y checkpoints disponibles localmente.
- Los barridos de hiperparámetros son **exploratorios**: comparar decisiones tomadas observando resultados de los folds externos no equivale a una estimación de desempeño completamente independiente.

### Documentación técnica

[Preparación de datos](docs/01_data_preparation.md) ·
[Modelo V7b](docs/02_model_v7b.md) ·
[Entrenamiento](docs/03_training.md) ·
[Experimentos de hiperparámetros](docs/04_hyperparameter_experiments.md)

**Misión:** [MAVEN — NASA](https://science.nasa.gov/mission/maven/) ·
**Licencia del código:** [LICENSE](LICENSE)
