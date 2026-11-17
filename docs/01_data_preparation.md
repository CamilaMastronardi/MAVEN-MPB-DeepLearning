# 01 · Datos y preprocesamiento

## Entradas

- **MAG:** `B_x, B_y, B_z, |B|` en nT, muestras cercanas a 1 Hz.
- **SWEA:** flujo electrónico promediado angularmente, 12 energías logarítmicas fijas entre **30 y 300 eV**, a su cadencia nativa. La consulta lee 25–330 eV para permitir interpolación **solo en energía**, sin interpolación temporal.
- Se transforman los flujos SWEA mediante `log1p`; MAG conserva sus componentes físicas.
- Cada instante candidato define una ventana de ±300 segundos. Se conservan los tiempos nativos por instrumento, máscaras de muestras válidas y relleno de batches.

Los datos provienen de `maven.mag_fields`, `maven.swea_svy3d` y `maven.swea_svy3d_axes` en BigQuery. Las cachés físicas quedan en `data/cache/mag_events/` y `data/cache/swea_events/`, fuera de Git.

## Catálogo y validación

- `data/labels/`: cruces MPB curados, etiquetas y grupos de procedencia.
- `data/splits/cv4_monthly_manifest.csv`: manifiesto reproducible de los cuatro folds agrupados por **mes UTC**.
- De 569 MPB catalogadas, 559 tienen marcada cobertura SWEA. Esto no garantiza que todas las ventanas sean físicamente utilizables.
- Las funciones FIT/MONITOR se extraen a `src/preprocessing/cv_splits.py` y conservan el algoritmo y las semillas previas de V7b.
- Se evita que el soporte físico ±35 minutos de un evento de entrenamiento se solape con un evento de validación.
- Los normalizadores por canal se ajustan usando solo **muestras temporales únicas de FIT**; no utilizan MONITOR ni el fold externo.
- `src/preprocessing/weak_regions.py` prepara etiquetas conservadoras de magnetovaina/magnetosfera inducida y negativos alejados de otras MPB/BS.

## Módulos

| Ruta | Responsabilidad |
| --- | --- |
| `src/preprocessing/bigquery_source.py` | Consultas MAG/SWEA, particiones, límites de procesamiento |
| `src/preprocessing/catalog.py` | Lectura/validación del manifiesto, matriz MAG |
| `src/preprocessing/cv_splits.py` | Separación temporal FIT/MONITOR |
| `src/preprocessing/mag_cache.py` | Caché MAG |
| `src/preprocessing/swea_cache.py` | Caché SWEA y variables espectrales |
| `src/preprocessing/native_windows.py` | Ventanas, normalizadores y batches |
| `src/preprocessing/weak_regions.py` | Grillas de candidatos y etiquetas débiles |
| `scripts/01_data/01_build_cv_manifest.py` | Reconstrucción explícita del manifiesto desde las anotaciones |

El preprocesamiento **no importa clases del modelo neuronal**.
