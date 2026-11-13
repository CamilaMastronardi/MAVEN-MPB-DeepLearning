# Paso 4c — Ajuste completo de la normalización por fold

**Estado:** código implementado y pruebas offline agregadas; pendiente de ejecución en WSL con BigQuery.

## Objetivo

Los pilotos `4b` demostraron que podemos normalizar MAG y SWEA sin usar la validación para calcular estadísticos, pero sólo estimaron la media y desviación estándar con **un evento de entrenamiento**. El nuevo `scripts/4_preprocessing/4c_fit_fold_normalizers.py` acumula los **datos de entrenamiento de todos los cruces elegibles del fold** en pasos controlados y reanudables.

- Cada etiqueta de entrenamiento define un soporte de observaciones **±35 minutos** alrededor de la MPB (centros de ventana dentro de ±30 min y ventana de ±5 min).
- Los soportes que se superponen **se fusionan**. Las ventanas deslizantes **no** se materializan para calcular la normalización: usamos cada muestra temporal original como máximo una vez por fold.
- Los intervalos se cortan en medianoche UTC. Se construye una consulta BigQuery **por día UTC** con filtros temporales parametrizados y filtro de partición.
- Para MAG se calculan media y desviación estándar por canal `[Bx,By,Bz,|B|]`.
- Para SWEA se conservan sólo las energías físicas 30–100 eV, se calcula el promedio angular en BigQuery sin filtrar por `quality`, y se aplica `log10(1+flux)` antes de las estadísticas por energía.
- Si aparece otro conjunto de energías físicas para SWEA, la ejecución falla explícitamente; no mezcla canales no comparables ni interpola implícitamente.
- Las mediciones originales sólo se mantienen transitoriamente en RAM. En `results/metrics/fold_normalization/` se guardan **únicamente pequeños JSON** con recuentos, sumas estadísticas implícitas (media, M2), días procesados y, al terminar, media y escala para el fold. No se guardan señales ni ventanas.
- Los JSON parciales **no se deben usar para entrenar**. Sólo el archivo final `fold_XX_mag_normalizer.json` o `fold_XX_swea_normalizer.json` representa la totalidad de las observaciones de entrenamiento definidas por este criterio.

## Comandos

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
git pull origin main

# Pruebas sintéticas de ventanas, normalizadores y planificación de fechas
python -m unittest discover -s tests -v

# DRY RUN: próximos 3 días de MAG del fold 1 (sin datos reales transferidos)
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument mag

# Si el costo estimado es aceptable: consulta y guarda checkpoint pequeño
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument mag --execute

# Repetir el mismo comando para continuar con otros 3 días.
# Cambiar a SWEA cuando se compruebe el funcionamiento:
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument swea
python scripts/4_preprocessing/4c_fit_fold_normalizers.py --fold 1 --instrument swea --execute
```

La ejecución por defecto incluye como máximo **3 nuevas fechas UTC**; al repetir un comando, se recupera automáticamente el checkpoint y se continúan las fechas pendientes. `--max-days N` permite elegir otra cantidad de fechas para cada corrida y `--max-days 0` selecciona todas las fechas pendientes, lo que puede implicar un volumen importante de consultas. Usar primero una tanda chica.

Por seguridad, se hace `dry run` en BigQuery para **cada día de la tanda** antes de ejecutar cualquiera de sus consultas. Los límites por defecto son **1 GiB por consulta**, **2 GiB totales por tanda**, **120.000 filas por día**. Se rechaza la ejecución si el costo estimado supera alguno de los límites. El historial de precios/cuotas reales de BigQuery puede diferir de los bytes estimados.

## Al terminar

Para cada combinación `(fold, instrumento)` se obtiene una transformación reutilizable con:

- `mean`: media por canal de los datos de **entrenamiento** transformados.
- `scale`: desviación estándar por canal, con protección frente a canales constantes.
- `n_unique_train_epochs`: cantidad de observaciones temporales válidas utilizadas.
- `swea_selected_energies_ev`: energías físicas comparables (SWEA).

Las observaciones inválidas (NaN/Inf y flujo SWEA negativo) se excluyen de la estimación y se cuentan aparte, sin imputación. Estos estadísticos **no incluyen** eventos etiquetados del fold reservado para validación.

La normalización final se calcula sobre la **unión de los bloques de 70 minutos** asociados a las etiquetas de entrenamiento: no es una normalización de toda la misión MAVEN, ni constituye una selección validada de ejemplos negativos. Aún falta integrar estos JSON con el entrenamiento de los autoencoders y elegir la estrategia de muestreo de ventanas.
