# Guion docente · Clasificadores lineales

**Duración:** 60 minutos. **Fuente conceptual:**
`unidades/clasificadores/capitulo_regresion_clasificadores_lineales.qmd`
(geometría de wᵀx + b, mínimos cuadrados, Fisher, perceptrón),
`capitulo_regresion_logistica.qmd` (seis estudiantes, verosimilitud,
gradiente, IRLS) y `capitulo_svm.qmd` (kernel trick).

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Todos los números en pantalla se calculan en
`sections/data.py` (`python -m sections.data` imprime el resumen usado aquí).

## Idea que recorre la clase

> Un clasificador lineal decide con el signo de wᵀx + b: su frontera es un
> hiperplano. Mínimos cuadrados, Fisher y la logística son **criterios**
> distintos para elegir w y b; los dos primeros tienen fórmula cerrada porque
> su óptimo es un sistema lineal, la logística no porque β queda dentro de σ.
> Un kernel curva la frontera en x, pero sigue siendo un hiperplano en φ(x):
> el modelo es lineal en sus parámetros.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Interpretar w (dirección perpendicular) y b (desplazamiento) y calcular la
   distancia con signo g(x)/‖w‖.
2. Resolver a mano mínimos cuadrados en una variable y explicar por qué la
   solución es cerrada (ecuaciones normales).
3. Explicar por qué la regresión logística no tiene fórmula cerrada y por qué,
   aun así, el óptimo es único (convexidad).
4. Mostrar que la frontera de la logística (binaria o multinomial) es un
   hiperplano, en cualquier número de variables.
5. Explicar en qué sentido un modelo con kernel es lineal.

**Material:** proyector, pizarra, calculadora; papel para el ejercicio.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–2 | Portada y ruta | Plantear la tesis: «todo es un hiperplano». | — |
| 2–4 | **1 · Muchas rectas** | Dieciséis puntos; la recta gira, contador de errores. | Proponen una recta antes de verla. |
| 4–9 | **1 · Geometría** | w gira, b desliza, distancias +2.3 y −4.2; hiperplano en ℝᵈ. | Cuentan parámetros (d + 1). |
| 9–10 | **1 · Perceptrón** | Tres correcciones; otra recta también separa. | Predicen hacia dónde se mueve la recta. |
| 10–13 | **2 · MCO a mano** | Cuadrados rojos 6 → 3.2; Sxy = 7, Sxx = 17.5. | Verifican Sxy con la tabla. |
| 13–16 | **2 · Tazón** | Ecuaciones normales 2 × 2; un salto desde cualquier punto. | Explican «lineal en β». |
| 16–19 | **2 · Defecto** | Cinco puntos lejanos: 0 → 3 errores; (5.2 − 1)² = 17.8. | Votan si la frontera debe moverse. |
| 19–25 | **3 · Fisher** *(opcional)* | Sombras, J = 4.22 → 5.64; w ∝ S_W⁻¹Δm. | Eligen la dirección antes del giro. |
| 25–28 | **4 · Sigmoide** | MCO con 0/1 da −0.2 y 1.2; la sigmoide; p = ½ ⇔ z = 0. | Explican por qué sigue siendo lineal. |
| 28–30 | **4 · Verosimilitud** | Barras −log p por los pasos de Newton: 0.693 → 0.413. | Calculan −log 0.5. |
| 30–32 | **4 · Sin fórmula** | Lado a lado: Xᵀ(Xβ − y) contra Xᵀ(σ(Xβ) − y). | Señalan dónde queda atrapado β. |
| 32–35 | **4 · Valles** | Elipses contra curvas deformadas; Newton contra gradiente. | Explican por qué el mínimo es único. |
| 35–39 | **4 · Trampas** | Separabilidad perfecta; lejanos ya no pesan. | Proponen regularizar. |
| 39–44 | **5 · Dos variables** | Niveles de p paralelos; proyección sobre w y sigmoide 1D. | Explican por qué son paralelas. |
| 44–48 | **5 · Cubo y softmax** | Plano en un cubo que gira; tres clases y tres rayos. | Describen la frontera en 50 variables. |
| 48–52 | **6 · Elevar** | Umbral con 3 errores; (x, x²) y x² = 2; ±1.41. | Predicen la forma de la frontera de regreso. |
| 52–54 | **6 · Anillo** | Paraboloide, plano x₃ = 1.8, círculo r = 1.34. | — |
| 54–56 | **6 · Kernel trick y RBF** | 9 + 12 + 4 = 25 = (3 + 2)²; campanas escaladas. | Responden «¿es lineal?». |
| 56–57 | **7 · Comparar** | Tabla de cuatro criterios. | Explican qué columna no cambia. |
| 57–59 | **7 · Ejercicio** | w = (2, −1), b = −3; P, Q, R. | Resultados abajo. |
| 59–60 | Cierre | Cuatro ideas y ticket. | Respuesta escrita. |

### Si el tiempo apremia

- Omitir el acto 3 (Fisher): quitar `self.fisher` de `presentacion.py` y
  mencionarlo en la tabla final como «otra fórmula cerrada».
- Acto 4: pasar rápido por la separabilidad perfecta.
- Acto 6: omitir el anillo y quedarse con la recta elevada y el RBF.

## Cálculos de referencia

**Geometría.** Con w = (cos 45°, sin 45°) y la recta por (5.2, 4.3):
g(8.8, 4.0) = 0.707 · (12.8 − 9.5) = +2.33; g(1.5, 2.0) = 0.707 · (3.5 − 9.5) = −4.24.

**Perceptrón** (η = 0.2, w₀ = (0.26, 0.97), b₀ = −6): corrige el cuadrado
(7.8, 2.8), luego los círculos (2.5, 1.0) y (2.0, 3.5); termina en
w = (0.92, 0.63), b = −6.2 sin errores.

**MCO, seis estudiantes** (horas 1–6, y = −1, −1, +1, −1, +1, +1): x̄ = 3.5,
ȳ = 0, Sxy = 7, Sxx = 17.5, β₁ = 0.4, β₀ = −1.4, frontera x = 3.5, SSE = 3.2
(contra 6 con ŷ = 0). Ecuaciones normales:
[[6, 21], [21, 91]] β = (0, 7).

**Puntos lejanos.** Con la recta de MCO sin ellos, un punto lejano
(12, 7) tiene ŷ = 5.2 y residuo² 17.8. Con ellos, MCO comete 3 errores; la
logística (con λ = 0.01 porque la base es separable) comete 0, y en el punto
lejano p ≈ 1.

**Fisher.** m_A = (2, 2.67), m_B = (5, 5.33), S_W = [[4, 1], [1, 1.33]],
w* ∝ S_W⁻¹(m_A − m_B) = (−0.31, −1.77); J(medias) = 4.22, J(Fisher) = 5.64.

**Logística, seis estudiantes** (y = 0, 0, 1, 0, 1, 1): pérdida inicial 0.693,
gradiente en 0 = (0, −0.583); Newton: (0, 0) → (−2.80, 0.80) → (−3.88, 1.11)
→ (−4.22, 1.21) → (−4.25, 1.21); pérdida 0.413; frontera 3.5 h; p ajustadas
0.05, 0.14, 0.35, 0.65, 0.86, 0.95. El gradiente con η = 0.5 necesita del
orden de 3000 pasos para llegar al mismo punto.

**Separable** (y = 0, 0, 0, 1, 1, 1, sigmoide centrada en 3.5): pérdida 0.40,
0.25, 0.12, 0.043, 0.006 con β₁ = 0.5, 1, 2, 4, 8.

**Dos variables** (18 puntos): b = −7.94, w = (0.98, 0.72); niveles
logit(0.1, 0.3, 0.5, 0.7, 0.9) = −2.20, −0.85, 0, 0.85, 2.20.

**Kernels.** Recta: clase +1 en |x| < 1.41 ⇔ x² < 2. Anillo: radios internos
≤ 0.81, externos ≥ 1.75, plano x₃ = 1.8 ⇒ r = 1.34. Kernel cuadrático:
x = (1, 2), z = (3, 1), φ(x) = (1, 2.83, 4), φ(z) = (9, 4.24, 1),
φ(x)ᵀφ(z) = 25 = (xᵀz)². RBF con γ = 0.9, λ = 0.3: α = (K + λI)⁻¹y clasifica
bien los diez puntos.

## Ejercicio (respuestas)

w = (2, −1), b = −3:

| Punto | x | z | p = σ(z) | Clase |
|---|---|---:|---:|---|
| P | (3, 1) | +2 | 0.88 | +1 |
| Q | (1, 2) | −3 | 0.05 | −1 |
| R | (2, 1) | 0 | 0.50 | sobre la frontera |

Pregunta extra: duplicar w y b no mueve la frontera (z = 0 es la misma
recta), pero vuelve las probabilidades más extremas.

**Ticket de salida.** ¿Por qué un SVM con kernel RBF, cuya frontera es curva,
se considera un método lineal? *Esperado:* su frontera es un hiperplano en el
espacio de características φ(x); la función de decisión Σ αᵢK(xᵢ, x) + b es
lineal en los parámetros α.

## Errores frecuentes

- **«La logística es no lineal porque la sigmoide es curva.»** La sigmoide
  transforma el puntaje, pero la frontera p = ½ es z = 0, un hiperplano.
- **«Si no hay fórmula, puede haber muchos óptimos.»** La log-loss es convexa:
  el mínimo es único (si existe; con separación perfecta no existe).
- **Confundir w con la frontera.** w es perpendicular a ella.
- **Pensar que MCO falla por la fórmula.** La fórmula es exacta; el problema
  es el criterio, que castiga a los puntos «demasiado correctos».
- **«El kernel cambia el algoritmo.»** El algoritmo es el mismo; sólo se
  reemplazan productos punto por K(x, z).
