# Guion docente · Ensambles de árboles y XGBoost

**Duración:** 60 minutos. **Fuente conceptual:**
`unidades/clasificadores/capitulo_arboles_regresion_lineal.qmd` (árboles de
regresión, bagging, bosques aleatorios, AdaBoost) y `capitulo_xgboost.qmd`
(gradient boosting a mano, objetivo de XGBoost, pesos de hoja, ganancia).
Viene después de [Árboles de decisión](../06-06-arboles-decision-manim/guion-docente.md).

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Todos los números en pantalla se calculan en
`sections/data.py` (`python -m sections.data` imprime el resumen usado aquí).

## Idea que recorre la clase

> Un árbol solo es inestable. Hay dos maneras de combinar muchos: promediar
> árboles profundos y diversos (bagging, bosques) para bajar la **varianza**, o
> sumar árboles pequeños que corrigen los errores de los anteriores (boosting)
> para bajar el **sesgo**. XGBoost es boosting con una función objetivo
> regularizada y una aproximación de segundo orden.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Explicar por qué promediar reduce la varianza y qué papel juega la
   correlación entre modelos.
2. Construir una muestra bootstrap e identificar los ejemplos fuera de la bolsa.
3. Calcular una ronda de AdaBoost (ε, α y repesado).
4. Hacer gradient boosting a mano con tocones y tasa η.
5. Calcular pesos de hoja −G/(H + λ) y la ganancia de un corte en XGBoost.

**Material:** proyector, pizarra, calculadora; papel para el ejercicio.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–2 | Portada y ruta | Recordar la inestabilidad del árbol. | «Pedir varias opiniones». |
| 2–7 | **1 · Árbol de regresión** | Laptops: promedio, residuos, barra SSE; RAM ≤ 6 y ≤ 12. | Proponen dónde cortar antes de verlo. |
| 7–9 | **1 · Varianza** | Árbol profundo sobre 40 puntos. | Explican qué pasaría con otra muestra. |
| 9–13 | **2 · Condorcet** | Cinco jueces al 70 %; binomial; curvas para 70, 55 y 45 %. | Predicen si la mayoría acierta más que un juez. |
| 13–17 | **2 · Correlación** | Var = ρσ² + (1 − ρ)σ²/B; pisos. | Explican por qué copiar al mismo juez no ayuda. |
| 17–21 | **3 · Bootstrap y OOB** | Urna, muestra {A, A, C, D, F, F}; árbol y errores OOB; e⁻¹. | Calculan (5/6)⁶ = 0.335. |
| 21–24 | **3 · Promediar** | Árboles bootstrap y su promedio; errores 0.665 / 0.540 / 0.36. | Comparan las tres barras. |
| 24–27 | **3 · Bosques** | Variables al azar por nodo; hiperparámetros. | Explican cómo baja ρ. |
| 27–34 | **4 · AdaBoost** | Tres rondas: tocón, ε, α, repesado. | Eligen el tocón de la ronda 2 antes de verlo. |
| 34–37 | **4 · Suma y formalización** | Votos apilados; signo; pérdida exponencial. | Predicen el signo de F(10). |
| 37–43 | **5 · Gradient boosting** | Cinco casas, tres rondas, tabla de SSE; F = F + η·h. | Calculan los residuos de la ronda 1. |
| 43–46 | **5 · Gradiente** | Parábola ½(y − F)²; −g = residuo. | Explican el nombre «gradient». |
| 46–50 | **6 · Objetivo y Taylor; peso de hoja** | Recta tangente vs parábola; vértice −G/(H + λ) con λ móvil. | Calculan w con λ = 1 (−3.33). |
| 50–55 | **6 · Ganancia, γ, clasificación, extras** | Barras de ganancia y línea γ; h = p(1 − p). | Calculan 29.17; explican cuándo se poda. |
| 55–57 | **7 · Comparar y perillas** | Tabla bagging/boosting; hiperparámetros. | Asocian cada perilla con sesgo o varianza. |
| 57–59 | **7 · Ejercicio** | Cuatro puntos, una ronda. | Resultados abajo. |
| 59–60 | Cierre | Cuatro ideas y ticket. | Respuesta escrita. |

### Si el tiempo apremia

- Acto 2: omitir la curva de Condorcet (decir el 0.98 con 25 jueces).
- Acto 3: omitir la lámina de bosques y mencionar sólo «m variables al azar».
- Acto 4: hacer la ronda 3 sin repesado.
- Acto 6: omitir «Lo que agrega XGBoost».

## Cálculos de referencia

### Laptops (RAM, precio)

A (4, 6 500), B (4, 7 200), C (8, 10 500), D (8, 12 000), E (16, 21 000),
F (16, 24 000). Promedio 13 533.33; SSE₀ = 266.43·10⁶.

| Corte | Izquierda | Derecha | SSE | Reducción |
|---|---:|---:|---:|---:|
| RAM ≤ 6 | 6 850 | 16 875 | 132.43·10⁶ | 50.3 % |
| **RAM ≤ 12** | **9 050** | **22 500** | **25.23·10⁶** | **90.5 %** |

Bootstrap {A, A, C, D, F, F}: hojas 8 875 y 24 000; fuera de la bolsa B
(error 7 200 − 8 875 = −1 675) y E (21 000 − 24 000 = −3 000).
(1 − 1/n)ⁿ: n = 6 → 0.335; n = 40 → 0.363; n → ∞ → 0.368.

### Condorcet (p = 0.7)

P(k = 3, 4, 5) = 0.309, 0.360, 0.168 → mayoría 0.837. Con 25 jueces 0.983.

### Bagging (40 puntos, y = 2 sen x + 0.3x + ruido, σ² = 0.36)

Error esperado en datos nuevos (sesgo² + varianza respecto a la función real,
más σ²): un árbol sin límite (37 hojas) 0.665; promedio de 60 árboles bootstrap
0.540. (Semilla 7; el laboratorio del libro usa otra semilla y reporta el error
en 300 puntos de prueba, así que sus cifras difieren ligeramente.)

### AdaBoost (x = 1…10; y = + + + − − − + + + −)

| Ronda | Tocón | Falla | ε | α |
|---:|---|---|---:|---:|
| 1 | x ≤ 3.5 ⇒ + | 7, 8, 9 | 0.300 | 0.424 |
| 2 | x ≤ 9.5 ⇒ + | 4, 5, 6 | 0.214 | 0.650 |
| 3 | x ≤ 6.5 ⇒ − | 1, 2, 3, 10 | 0.182 | 0.752 |

Pesos tras la ronda 1: errores 0.1667, aciertos 0.0714 (los errores suman 0.5).
Tras la ronda 2: 4–6 pesan 0.1667; 7–9, 0.1061; 1–3 y 10, 0.0455.
F(x): 1–3 → +0.321, 4–6 → −0.526, 7–9 → +0.978, 10 → −0.321.

### Gradient boosting (x = 1…5, y = 2, 4, 9, 10, 15; η = 0.5)

| Ronda | F | Tocón (residuos) | SSE |
|---:|---|---|---:|
| 0 | 8, 8, 8, 8, 8 | — | 106.00 |
| 1 | 5.5, 5.5, 9.67, 9.67, 9.67 | x ≤ 2.5: −5 \| 3.33 | 43.50 |
| 2 | 4.83, 4.83, 9, 9, 12.33 | x ≤ 4.5: −1.33 \| 5.33 | 16.83 |
| 3 | 3.92, 3.92, 9.61, 9.61, 12.94 | x ≤ 2.5: −1.83 \| 1.22 | 8.43 |

### XGBoost, primera ronda (λ = 1, γ = 0)

g = F₀ − y = (6, 4, −1, −2, −7), h = 1. Corte x ≤ 2.5: w_L = −10/3 = −3.33,
w_R = 10/4 = 2.5 (contra −5 y 3.33 sin λ).

| Corte | G_L | G_R | Ganancia |
|---|---:|---:|---:|
| x ≤ 1.5 | 6 | −6 | 12.60 |
| **x ≤ 2.5** | **10** | **−10** | **29.17** |
| x ≤ 3.5 | 9 | −9 | 23.63 |
| x ≤ 4.5 | 7 | −7 | 17.15 |

La librería `xgboost` reporta la ganancia sin el ½ (58.33). Con γ > 29.17
ningún corte compensa. Clasificación: hoja con y = (1, 1, 0) y p = 0.5 →
G = −0.5, H = 0.75, w = 0.5/1.75 = 0.286.

### Ejercicio final (x = 1, 2, 3, 4; y = 1, 3, 6, 8)

1. F₀ = 4.5; residuos (−3.5, −1.5, 1.5, 3.5); SSE₀ = 29.
2. Tocón x ≤ 2.5 con −2.5 | 2.5; F₁ = (3.25, 3.25, 5.75, 5.75); SSE₁ = 10.25.
3. g = (3.5, 1.5, −1.5, −3.5): G_L = 5, G_R = −5, H = 2 y 2 → w = −1.67 y
   1.67; ganancia ½(25/3 + 25/3 − 0) = 8.33.

## Errores frecuentes que conviene anticipar

- **Creer que bagging reduce el sesgo.** Promediar no cambia el valor
  esperado; por eso se promedian árboles profundos.
- **Pensar que más árboles en un bosque sobreajustan.** No: el error converge.
  En boosting, en cambio, más rondas sí pueden sobreajustar.
- **Confundir los pesos de los ejemplos (wᵢ) con el peso del modelo (α) en
  AdaBoost.**
- **Ajustar el árbol a y en lugar de a los residuos** en gradient boosting.
- **Signo del gradiente.** g = F − y; el residuo es −g.
- **Leer w de una hoja de clasificación como probabilidad.** Es un cambio en
  log-odds, multiplicado por η.

## Ticket de salida

¿Por qué agregar árboles a un bosque no sobreajusta y agregar rondas de
boosting sí puede? ¿Qué hiperparámetro de XGBoost usarías para frenarlo?
(Respuesta esperada: el bosque promedia modelos independientes y su error
converge; el boosting sigue ajustando residuos, incluido el ruido. Frenarlo con
early stopping, η más pequeño, max_depth menor, o λ y γ más grandes.)
