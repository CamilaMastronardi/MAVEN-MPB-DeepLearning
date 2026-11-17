# 03 · Entrenamiento V7b

El entrenador y evaluador único es `scripts/02_training/01_train_v7b.py`.

Valores originales: AdamW, learning rate 0.0003, weight decay 0.001, batch size 16, seed 42, hasta 100 épocas, mínimo 10, early stopping con paciencia 8. Por defecto `rank-weight=0` reproduce V7. **Para V7b, pasar siempre `--rank-weight > 0`.**

El entrenamiento se realiza sobre cuatro folds agrupados por meses UTC. Dentro del TRAIN externo se reserva MONITOR por meses para elegir checkpoint y umbral. El tiempo de catálogo MPB/BS no entra como variable en la red.

```bash
python scripts/02_training/01_train_v7b.py \
  --folds all --rank-weight 2 --rank-margin 0.25 --region-weight 0.5 \
  --output-dir results/metrics/v7b_manual --resume --execute
```

El programa reutiliza observaciones MAG/SWEA en caché, hace dry-run de consultas BigQuery faltantes antes de descargar y registra firmas SHA256 de las configuraciones. `--resume` protege frente a la mezcla de checkpoints de distintas configuraciones.

Las métricas externas son error absoluto de localización, error mediano, fracciones dentro de 60 y 120 s, F1 de candidatos bajo etiquetas débiles y exactitud de regiones.

Pruebas sin acceso a BigQuery:

```bash
python -m unittest discover -s tests -v
```

**Alcance:** los hiperparámetros se exploraron previamente observando resultados de folds externos; esta comparación es exploratoria y no una estimación final independiente.
