# Guion docente · Similitud y distancias

**Duración:** 60 minutos. **Fuente conceptual:**
`unidades/clustering/capitulo_clustering_intro.qmd` (similitud y distancias) y
la sección de UMAP de `unidades/clustering/capitulo_tsne_umap.qmd`. Cada
estado de la presentación tiene una nota (tecla `S`). Todos los números salen
de `sections/data.py`.

## Idea que recorre la clase

> Antes de agrupar hay que decidir **cómo medir**. Medir es asignar números que
> respeten la estructura; una **distancia** sólo merece ese nombre si cumple
> cuatro axiomas, y cada axioma nace de una regla que falla sin él. La
> similitud y las medidas de desempeño son otros objetos. Transformar el
> espacio (escala, correlación, dirección, proyección) cambia quién es vecino
> de quién; en alta dimensión las distancias globales pierden sentido y UMAP
> responde confiando sólo en los vecindarios locales.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Enunciar los cuatro axiomas de métrica y dar, para cada uno, una medida
   usada en la práctica que lo viola (sólo edad, tiempo de subida,
   euclidiana al cuadrado, coseno).
2. Distinguir medida de conjuntos, distancia, similitud y medida de desempeño,
   y convertir distancia en similitud con el núcleo gaussiano.
3. Explicar con una transformación del plano por qué la escala es un peso
   implícito y por qué Mahalanobis es la euclidiana después de blanquear.
4. Explicar con la esfera dentro del cubo por qué las distancias se concentran
   en alta dimensión.
5. Describir el grafo de UMAP (ρᵢ, σᵢ, unión difusa) y qué se puede y qué no
   se puede leer en un mapa de UMAP.

**Material:** proyector, pizarra, hojas para el reto en parejas y el ticket.

## Secuencia minuto a minuto

| Min. | Acto | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–5 | 1 · Ana | Cada quien escribe quién se parece más a Ana. Tres reglas → tres respuestas. Mapa de algoritmos que consumen distancias | Aceptan que "parecido" es una decisión |
| 5–14 | 2 · Medir | Varas: orden y suma. Ciudades A, B, C: el grupo propone condiciones. Tres reglas rotas. Los cuatro axiomas | Asocian cada axioma con su regla rota |
| 14–21 | 3 · ¿Para qué? | Poda con la triangular (6.2 − 1.6 = 4.6 > 1.2). Cuatro conceptos. Núcleo gaussiano moviendo σ | Clasifican silueta, coseno, F1, Jaccard, área |
| 21–31 | 4 · Geometría 2D | Taxi/pájaro/rey (7, 5, 4). Bola con p = 1, 2, ∞ y p = ½. Estirar el eje x cambia el vecino. Blanquear la nube: Mahalanobis | Predicen la forma de la bola; explican la elipse |
| 31–37 | 5 · Vistas 3D | Documentos: euclidiana vs. esfera. 1 − cos falla la triangular (0.59 < 1). Dos grupos que sólo se ven al girar | Explican qué se pierde al normalizar y al proyectar |
| 37–45 | 6 · Dimensión | Dardos: 80 % (π/4). Esfera en el cubo: 52.4 %. Barras hasta p = 20. Cáscara. Histogramas que se estrechan | Predicen la tendencia antes de verla |
| 45–54 | 7 · UMAP | Rollo: 6.3 por el aire vs. 68.6 sobre la hoja. Grafo k-NN. Regla de A: ρ = 1, σ = 2.27, pesos 1, 0.414, 0.171. Desenrollar. Qué no se lee | Explican por qué UMAP no es una métrica global |
| 54–60 | 8 · Cierre | Gower: Carla (0.482) → Daniel (0.384) con pesos. Reto de cuatro conjuntos. Mapa del capítulo. Ticket | Eligen y justifican una distancia por tipo de dato |

## Respuestas y números clave

- **Ana, sólo edad e ingreso (normalizados por rango, L2):** Daniel 0.118,
  Carla 0.890, Beto 0.962, Elena 1.157.
- **Gower, pesos iguales:** Carla 0.482, Beto 0.508, Daniel 0.538, Elena 0.909.
- **Gower, pesos (1, 1, ½, ½):** Daniel 0.384, Carla 0.477, Beto 0.511,
  Elena 0.878.
- **Ana ↔ Daniel:** d₁ = 0.152, d₂ = 0.118, d∞ = 0.111.
- **Poda:** d(q, c) = 6.2, radio 1.6, cota 4.6 > δ = 1.2.
- **p = ½:** ‖u‖ = ‖v‖ = 1 pero ‖u + v‖ = 4 > 2.
- **Cambio de unidades (x × 2.5):** el vecino pasa de (1, 0.3), a 1.04, a
  (0.3, 1.1), a 1.33.
- **Mahalanobis, ρ = 0.8:** a = (1, 1) → 1.05; b = (1, −1) → 3.16; ambos a
  √2 = 1.41 en euclidiana.
- **Documentos:** euclidiana d(d₁, d₂) = 3.61, d(d₁, d₃) = 4.36; coseno
  d(d₁, d₂) = 0, d(d₁, d₃) = 0.825 (θ = 80°).
- **Coseno no métrico:** con direcciones a 0°, 45° y 90°,
  0.293 + 0.293 = 0.59 < 1.
- **Bola/cubo:** p = 2: 78.5 %; 3: 52.4 %; 5: 16.4 %; 10: 0.25 %;
  20: 2.5 × 10⁻⁶ %.
- **Cáscara del 5 %:** p = 2: 9.8 %; 10: 40.1 %; 50: 92.3 %; 100: 99.4 %.
- **Contraste (d_max − d_min)/d_min**, 500 puntos uniformes, semilla 0 del
  capítulo: p = 2: 82.0; 10: 2.19; 100: 0.35; 1000: 0.12.
- **UMAP en la recta A–E, k = 3:** ρ_A = 1, σ_A = 2.27; pesos 1, 0.414,
  0.171.

## Errores frecuentes que conviene provocar

- "La euclidiana al cuadrado es una distancia más fuerte": no, viola la
  triangular; k-medias la usa porque minimiza varianza, no porque sea métrica.
- "Similitud = 1 − distancia siempre": la conversión no conserva los axiomas
  (coseno).
- "La silueta es una métrica": es una medida de desempeño de una partición;
  usa distancias, pero no mide la separación de un par.
- "Estandarizar es neutral": cambiar unidades es dar pesos; el vecino cambió.
- "En UMAP, los grupos lejanos son más distintos": UMAP sólo conserva
  vecindarios; tamaños y separaciones entre grupos no se interpretan.

## Reto en parejas (56–58 min)

| Conjunto | Respuesta esperada | Justificación |
|---|---|---|
| Canastas (sí/no) | Jaccard | las coausencias no informan |
| Reseñas de texto | Coseno | la longitud no es el tema |
| Sensores correlacionados | Mahalanobis | corrige escala y correlación |
| Clientes mixtos | Gower con pesos explícitos | combina tipos y maneja faltantes |

## Ticket de salida

1. Da un ejemplo de similitud que no sea métrica. ¿Qué axioma falla?
2. ¿Por qué no se leen las distancias entre grupos en un mapa de UMAP?

## Nota sobre el desenrollado

La animación del acto 7 muestra el **objetivo** de UMAP (una hoja plana que
conserva el grafo de vecinos), no la salida de una corrida concreta de
`umap-learn`. Con datos reales el resultado es aproximado y depende de la
semilla, de `n_neighbors` y de `min_dist`; conviene decirlo en voz alta.
