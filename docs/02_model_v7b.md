# 02 · Modelo de localización V7b

V7b aprende las características que distinguen regiones plasmáticas alrededor de un candidato MPB usando dos ramas convolucionales independientes.

## Arquitectura

- Encoder MAG: cuatro canales físicos, 24 ocultos, bloques residuales dilatados **1, 2, 4, 8**, y **16 canales latentes temporales**.
- Encoder SWEA: doce energías, 24 canales ocultos, bloques residuales dilatados **1, 2**, y **16 canales latentes temporales**.
- Dos decoders reconstruyen MAG y SWEA como tarea auxiliar.
- A ambos lados de un tiempo candidato se promedian los mapas latentes para tiempos relativos menores a −60 s y mayores a +60 s.
- Por lado se fusionan 16 características MAG y 16 SWEA (32 en total).
- La cabeza de regiones predice magnetovaina (0) o magnetosfera inducida (1) para cada lado, según etiquetas débiles derivadas de MPB/BS.
- La cabeza de transición recibe izquierda, derecha, diferencia y diferencia absoluta (128 componentes) y entrega un logit por instante candidato.

**No** se trata de un autoencoder con un único vector latente de 16 dimensiones por evento: la representación intermedia conserva el eje temporal.

Código: `src/model/v7b.py`.

## Ranking

La revisión V7b añade comparaciones entre positivos y controles negativos **del mismo evento** y minería online de controles difíciles sobre TRAIN únicamente.

```text
L_total = BCE_transición
        + λ_region * BCE_regiones
        + λ_rec * (MSE_MAG + MSE_SWEA)
        + λ_rank * mean_event[softplus(margen + logit_neg - logit_pos)]
```

El margen actúa en el espacio de logits. Código: `src/model/ranking.py`.

La evaluación selecciona el candidato con puntuación máxima en una vecindad de ±30 minutos alrededor de una MPB **catalogada**. No es una búsqueda orbital autónoma.
