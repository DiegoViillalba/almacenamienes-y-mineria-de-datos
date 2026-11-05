# Guion docente · Introducción a los clasificadores

**Duración:** 60 minutos. **Fuente conceptual:**
`unidades/clasificadores/6-0-introduccion-clasificacion.qmd` (los diez
clientes, la regla de un corte, 1-NN y 3-NN, los clientes de prueba A–E) y
`6-0b-vecinos-cercanos.qmd` (escalamiento). Las métricas se profundizan en
`6-2-metricas_clasificacion.qmd` y la regresión logística en
`capitulo_regresion_logistica.qmd`.

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Todos los números en pantalla se calculan en
`sections/data.py` (`python -m sections.data` imprime el resumen usado aquí).

## Idea que recorre la clase

> Clasificar es partir el espacio de atributos en regiones. Hay muchas
> maneras de partirlo —un corte, los vecinos, una recta inclinada— y la
> matriz de confusión, medida en clientes que el modelo **no vio**, dice cuál
> se equivoca menos y **cómo** se equivoca.

Un solo ejemplo en toda la clase: diez clientes (ingreso en miles de pesos al
mes, deuda en % del ingreso, ¿pagó?). Positivo = **no paga**, la alarma que el
banco quiere encender.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Distinguir atributos `x`, etiqueta `y` y predicción `ŷ`, y explicar qué es
   una frontera de decisión.
2. Construir una matriz de confusión y calcular exactitud, precisión,
   sensibilidad y F1, explicando qué pregunta responde cada una.
3. Clasificar un caso a mano con k-NN y explicar el efecto de k y de las
   unidades.
4. Interpretar una regresión logística: puntaje, sigmoide, umbral y pesos.
5. Comparar dos clasificadores en datos de prueba y argumentar cuál conviene
   según el costo de cada error.

**Material:** proyector, pizarra; papel para los dos «tu turno» y el
ejercicio final.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–2 | Portada y ruta | Pedir ejemplos de decisiones «sí/no» automáticas. | Ejemplos de la clase. |
| 2–6 | **1 · Clasificar** | Tres decisiones cotidianas; la tabla con llaves `x` / `y`; filas que se vuelven puntos; cliente nuevo y `ŷ`. | Dicen qué columna se adivina; votan por el cliente nuevo. |
| 6–10 | **1 · Primeras reglas** | Referencia 6/10; la recta sube con contador de errores y se detiene en 44; `ŷ = f(x)`. | Gritan «¡alto!» cerca del mínimo; nombran la excepción (6). |
| 10–14 | **2 · Matriz** | Dos maneras de equivocarse; clientes que viajan a las celdas; nombres TP/FN/FP/TN. | Predicen la celda del cliente 6 (FP). |
| 14–17 | **2 · Métricas** | Diagonal = exactitud, columna = precisión, fila = sensibilidad; F1. | Señalan columna y fila antes de la fórmula. |
| 17–20 | **2 · Corte** | Mover el corte 37.5 → 44 → 51.5 → 59 con barras. | Predicen qué barra sube y cuál baja. |
| 20–22 | **2 · Paradoja y tu turno** | 97 % de exactitud con sensibilidad 0; matriz de fraude. | 0.94, 0.67, 0.80. |
| 22–28 | **3 · k-NN** | Cliente B; círculo creciente; lupa con Pitágoras; votación 3-NN. | Calculan √20; explican por qué gana «no paga». |
| 28–32 | **3 · Algoritmo y regiones** | Medir, ordenar, votar; regiones k = 1, 3, 9. | Predicen qué pasa con la isla del 6. |
| 32–34 | **3 · Unidades** | Deuda como proporción: cambian los vecinos. | Explican por qué hay que estandarizar. |
| 34–37 | **3 · Tu turno y prueba** | Cliente C con 3-NN; clientes A–E; tabla entrenamiento/prueba. | 10, 5, 6 → «paga» (falla); 1-NN 1.00 vs 0.60. |
| 37–41 | **4 · Recta y puntaje** | El corte gira; distancias con signo; recta numérica de `z`. | Señalan el «no paga» más seguro y el más dudoso. |
| 41–44 | **4 · Sigmoide y bandas** | Clientes suben a la curva; `p = 0.5 ⇔ z = 0`; bandas 0.1/0.5/0.9. | Leen p del cliente 6 (0.76). |
| 44–47 | **4 · Entrenar** | Multa −log p por cliente; la recta se acomoda; promedio baja a 0.324. | Explican por qué 6 y 10 conservan multa. |
| 47–52 | **4 · Fórmulas y límites** | Puntaje, sigmoide, decisión, log-loss; pesos y momios; patrón XOR; tabla k-NN vs logística. | Interpretan w₂ = +0.19. |
| 52–56 | **5 · Comparar** | Tabla de prueba; matrices lado a lado; umbral 0.2. | Argumentan qué error cuesta más. |
| 56–58 | **5 · Ejercicio** | Cliente (20, 50) en parejas. | Resultados abajo. |
| 58–60 | Cierre | Cuatro ideas y ticket de salida. | Respuesta escrita. |

### Si el tiempo apremia

- Acto 1: omitir el estado «tres decisiones» y empezar con la tabla.
- Acto 2: saltar F1 y recorrer sólo los cortes 44 y 59.
- Acto 3: omitir k = 9 y la trampa de unidades (mencionarla verbalmente).
- Acto 4: omitir el estado de límites (XOR).

## Cálculos de referencia

### Regla de un corte (deuda > t ⇒ no paga)

| t | errores | quiénes |
|---:|---:|---|
| 22.5 | 5 | 2, 3, 4, 5, 6 |
| 32.5 | 3 | 4, 5, 6 |
| 37.5 | 2 | 5, 6 |
| **44.0** | **1** | **6** |
| 51.5 | 2 | 6, 10 |
| 59.0 | 2 | 7, 10 |

### Matrices al mover el corte (entrenamiento, positivo = no paga)

| t | TP | FN | FP | TN | exactitud | precisión | sensibilidad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 37.5 | 4 | 0 | 2 | 4 | 0.80 | 0.67 | 1.00 |
| 44.0 | 4 | 0 | 1 | 5 | 0.90 | 0.80 | 1.00 |
| 51.5 | 3 | 1 | 1 | 5 | 0.80 | 0.75 | 0.75 |
| 59.0 | 2 | 2 | 0 | 6 | 0.80 | 1.00 | 0.50 |

F1 en t = 44: 2(0.80)(1.00)/1.80 ≈ 0.89.

**Tu turno (fraude):** TP 8, FN 2, FP 4, TN 86 → exactitud 0.94, precisión
8/12 ≈ 0.67, sensibilidad 0.80.

### k-NN

- Cliente B (31, 60): 6 a 2.24 (paga), 9 a 4.47, 8 a 9.00 (no pagan) →
  1-NN «paga», 3-NN «no paga».
- Deuda como proporción: vecinos de B pasan a 6, 3, 10 → 3-NN «paga».
- Cliente C (26, 45): 10 a 3.61 (no paga), 5 a 12.08, 6 a 13.60 (pagan) →
  3-NN «paga»; en realidad no pagó.
- Exactitud: k = 1 → 1.00 / 0.60 (entrenamiento / prueba); k = 3 → 0.90 /
  0.60; k = 9 → 0.60 / 0.60.

### Regresión logística (ajustada por máxima verosimilitud)

z = −7.57 − 0.073·ingreso + 0.188·deuda; frontera deuda ≈ 40.2 + 0.39·ingreso.

| cliente | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| z | −5.27 | −3.76 | −5.21 | −3.76 | −1.14 | 1.15 | 1.90 | 2.11 | 1.54 | −0.59 |
| p(no paga) | 0.01 | 0.02 | 0.01 | 0.02 | 0.24 | 0.76 | 0.87 | 0.89 | 0.82 | 0.36 |

Exactitud de entrenamiento 0.80 (fallan 6 y 10); log-loss 0.324. Momios:
e^0.188 ≈ 1.21 por punto de deuda; e^−0.073 ≈ 0.93 por mil de ingreso.

### Prueba (A–E)

| cliente | real | 3-NN | p logística | logística |
|---|---|---|---:|---|
| A (18, 35) | paga | paga ✓ | 0.09 | paga ✓ |
| B (31, 60) | no paga | no paga ✓ | 0.81 | no paga ✓ |
| C (26, 45) | no paga | paga ✗ | 0.27 | paga ✗ |
| D (36, 30) | paga | paga ✓ | 0.01 | paga ✓ |
| E (40, 50) | paga | no paga ✗ | 0.25 | paga ✓ |

- 3-NN: TP 1, FN 1, FP 1, TN 2 → 0.60 / 0.50 / 0.50.
- Logística (0.5): TP 1, FN 1, FP 0, TN 3 → 0.80 / 1.00 / 0.50.
- Logística (0.2): TP 2, FN 0, FP 1, TN 2 → 0.80 / 0.67 / 1.00.

### Ejercicio final: cliente (20, 50)

(a) Vecinos 10 (8.25), 7 (9.43), 8 (10.20): tres «no paga» → no paga.
(b) z = −8 − 0.10·20 + 0.25·50 = 2.5 → p ≈ 0.92 → no paga.
(c) Si pagó, ambos cometen un **falso positivo**.

## Errores frecuentes que conviene anticipar

- **«Positivo» = bueno.** Aquí positivo es «no paga»: la clase que se quiere
  detectar. Repetirlo cada vez que se nombre una celda.
- **Confundir precisión con sensibilidad.** Precisión: columna (lo que dijo el
  modelo); sensibilidad: fila (lo que pasó en realidad).
- **Creer que entrenamiento perfecto es buen modelo.** 1-NN acierta 10/10 en
  entrenamiento y 3/5 en prueba.
- **Medir distancias con unidades distintas.** La variable de mayor escala
  domina; estandarizar antes de k-NN.
- **Leer la salida de la logística como clase.** Es una probabilidad; la clase
  depende del umbral, que se elige según el costo de los errores.
- **Sacar conclusiones con cinco casos de prueba.** Un solo cliente cambia la
  exactitud en 0.20; en la práctica se usa validación cruzada.

## Ticket de salida

Da un ejemplo donde un falso negativo cueste más que un falso positivo. ¿Qué
métrica vigilarías y hacia dónde moverías el umbral? (Respuesta esperada:
p. ej., detección de una enfermedad o de fraude; vigilar la sensibilidad y
bajar el umbral.)
