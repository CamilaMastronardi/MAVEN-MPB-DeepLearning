# Jupyter Notebooks — modelos supervisados MAG y SWEA

Estos notebooks están diseñados para el final de la materia. Deben servir como
**cuaderno de trabajo explicativo**, con decisiones, ecuaciones y figuras basadas
solamente en datos reales MAVEN.

## Archivos

- `01_mag_autoencoder.ipynb`: insumos MAG `Bx, By, Bz, |B|`, cuatro canales.
- `02_swea_autoencoder.ipynb`: espectros SWEA entre 30 y 100 eV, con eje energético físico.

Ambos notebooks **reemplazan cualquier prueba con señales artificiales**.
No contienen ni generan datos sintéticos. Están guardados **sin ejecutar**:
los contadores de ejecución están vacíos y no incluyen resultados.

## Prerrequisitos

1. Tener el entorno `.venv` activo y JupyterLab instalado.
2. Contar con el archivo **final** del normalizador del instrumento/fold,
   por ejemplo `results/metrics/fold_normalization/fold_01_mag_normalizer.json`.
   **No alcanza el checkpoint parcial.**
3. Tener disponibles permisos/credenciales de BigQuery.
4. Abrir el notebook desde la raíz del repositorio o desde `notebooks/`.

```bash
cd ~/Codigos/MAVEN-MPB-DeepLearning
source .venv/bin/activate
git pull origin main
jupyter lab
```

## Seguridad y reproducibilidad

La variable `EXECUTE_REAL_BIGQUERY = False` está desactivada por defecto.
No se envían consultas de mediciones hasta que la persona la cambie a
`True` deliberadamente y ejecute esa celda. El código consulta primero
el estimador de bytes y respeta `MAX_QUERY_GIB` (1 GiB inicialmente).

Al activarlo, el notebook selecciona **un evento de TRAIN del fold indicado**,
consulta un bloque de 70 minutos en BigQuery, genera ventanas y forma un lote
normalizado usando solamente las estadísticas preexistentes del TRAIN.
No guarda mediciones descargadas en archivos locales.

El notebook **no entrena** un autoencoder todavía. El desarrollo de la
arquitectura encoder → decoder + detector supervisado se realizará después
de inspeccionar las ventanas reales. La evaluación definitiva debe hacerse
por evento y con separación por fold, nunca interpretando ventanas altamente
superpuestas como muestras independientes.

No hay que subir los normalizadores JSON locales ni los arreglos científicos
a GitHub. Antes de publicar notebooks, revisar que no hayan quedado resultados
de celdas con datos científicos en el propio archivo `.ipynb`.
