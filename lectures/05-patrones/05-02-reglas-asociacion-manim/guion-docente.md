# Guion docente · Reglas de asociación: construir, ver, medir y decidir

**Duración:** 60 minutos. **Fuentes conceptuales:**
`unidades/reglas_asociacion/5-0-introduccion-patrones.qmd`, `5-1-introduccion.qmd`,
`5-1-apriori.qmd` y `5-2-metricas-extra.qmd`; notas previas en
`lectures/notes/guia-docente-asociacion.md` (clases 2 y 3).

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Este guion organiza el tiempo, los cálculos de
pizarra, las respuestas esperadas y los errores que conviene anticipar.
Todos los números en pantalla se calculan en `sections/data.py`
(`python -m sections.data` imprime el resumen que se usa aquí).

## Idea que recorre la clase

> Una regla A ⇒ B sólo dice algo nuevo si saber A cambia la probabilidad de B.
> En el cuadrado unitario, B es agua: bajo independencia se nivela a P(B); la
> asociación la hace subir dentro de la columna de A. La confianza es la altura
> del agua en esa columna y el lift es esa altura dividida entre el nivel.

El hilo es una sola tienda. Abre con el gerente que se entusiasma por un 60 %
que en realidad es la tasa base (intuición), pasa a cinco tickets para
construir reglas a mano, vuelve a la tienda para ver las métricas como áreas y
termina usando cuatro reglas de la misma tienda para decidir.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Construir reglas A ⇒ B desde una lista de tickets: matriz binaria, soporte,
   itemsets frecuentes (Apriori), confianza y lift.
2. Dibujar el cuadrado unitario de una regla y leer en él soporte (área),
   confianza (altura dentro de A) y lift (altura / nivel, o área observada /
   área esperada).
3. Calcular e interpretar leverage, conviction y Zhang, y explicar qué
   pregunta responde cada una.
4. Justificar una decisión (o la decisión de no actuar) con varias métricas y
   reconocer cuándo una regla no basta: pocos datos, ítems dominantes,
   causalidad y orden temporal.

**Material:** proyector, pizarra, hojas cuadriculadas para dibujar cuadrados.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–3 | Portada y objetivos | Preguntar qué decisiones tomarían sabiendo qué se compra junto. Anotar respuestas. | Lista de decisiones (combo, acomodo, recomendación). |
| 3–10 | **Intuición · 100 tickets** | Cuadrícula; pan dulce (30); el gerente: 18/30 = 60 %; votación; la leche en el resto (42/70); cuadrícula → cuadrado; pregunta guía. | El grupo explica que el 60 % no dice nada porque la leche ya está en 60 % de todos los tickets. |
| 10–20 | **1 · Construir** | Cinco tickets → matriz binaria → soportes → pares → pan ⇒ leche como condicional → seis reglas → lift. | Calculan supp, conf = 3/4 y lift = 0.9375; explican la antimonotonía con el queso. |
| 20–30 | **2 · Ver** | Cuadrado de café ⇒ pan dulce; nivel de independencia; el agua sube; conf = altura; lift = altura/nivel = área/área; animación hacia lift 1, < 1 y techo; tabla 2 × 2. | Predicen qué pasa con el agua; deducen lift > k ⇔ conf > k·P(B); explican el techo 2.5. |
| 30–40 | **3 · Medir** | Leverage (franja verde), conviction (errores rojos vs. amarillos), Zhang (dos alturas); cuatro cuadrados y tabla. | Señalan cada métrica en el dibujo; descubren que café ⇒ pan dulce mueve más tickets que vino ⇒ queso. |
| 40–50 | **4 · Decidir** | Red de asociación; mapa soporte × lift; cuatro fichas de decisión; filtro mental; cuándo no usar reglas. | Defienden una ficha en 30 s con métrica, acción y riesgo. |
| 50–57 | **Reto en parejas** | Papas y refresco en 200 tickets: dibujo, métricas, dirección y decisión. | Resultados del reto (abajo). |
| 57–60 | Cierre y ticket | Cuatro verbos: contar, comparar, medir, decidir. Ticket individual. | Respuesta escrita del ticket. |

### Si el tiempo apremia

- En el acto 1 · Construir, omitir el trío y la lista de pares (−2 min); basta
  con la fila resaltada de {leche, pan}.
- En el acto 4 · Decidir, mostrar la red sólo de paso y saltar a las fichas
  (−2 min).
- No recortar la predicción del agua (min 22) ni el reto: son la evidencia
  principal de comprensión.

## Hilo conductor y formalización

### Intuición: el dilema del gerente (min 3–10)

Cien tickets en una cuadrícula de 10 × 10; cada columna son 10 tickets. Las
tres primeras columnas tienen pan dulce. En **todas** las columnas los seis
tickets inferiores tienen leche:

| | con leche | total | proporción |
|---|---|---|---|
| con pan dulce | 18 | 30 | 0.60 |
| sin pan dulce | 42 | 70 | 0.60 |
| toda la tienda | 60 | 100 | 0.60 |

La regla pan dulce ⇒ leche tiene confianza 0.60 y lift 1: el pan dulce no
cambia nada. Al cerrar los huecos, la cuadrícula se vuelve un cuadrado de área
1 donde la leche es una **franja plana**: así se ve la independencia. Si
alguien propone «comparar con el 60 % de toda la tienda», acaba de inventar el
lift; dígalo.

### 1 · Construir (min 10–20)

Los cinco tickets del libro: T1 = {leche, pan, mantequilla}, T2 = {leche, pan},
T3 = {leche, mantequilla}, T4 = {pan, mantequilla}, T5 = {leche, pan, queso}.
Conteo mínimo 2 (soporte 0.40), confianza mínima 0.60.

- Soportes: leche 4, pan 4, mantequilla 3, queso 1 → queso no es frecuente.
- Pares: {leche, pan} = 3, {leche, mantequilla} = 2, {pan, mantequilla} = 2.
  Ningún par con queso se cuenta (antimonotonía). {leche, pan, mantequilla} = 1.
- `conf(pan ⇒ leche) = supp({pan, leche}) / supp({pan}) = (3/5)/(4/5) = 0.75`.
  Visualmente: se apagan los tickets sin pan; de los 4 que quedan, 3 tienen
  leche. La confianza es una probabilidad **condicional**: el universo se
  encoge.

| Regla | Conteo | Confianza | ≥ 0.60 | Lift |
|---|---|---|---|---|
| leche ⇒ pan | 3/4 | 0.75 | sí | 0.9375 |
| pan ⇒ leche | 3/4 | 0.75 | sí | 0.9375 |
| leche ⇒ mantequilla | 2/4 | 0.50 | no | 0.8333 |
| mantequilla ⇒ leche | 2/3 | 0.67 | sí | 0.8333 |
| pan ⇒ mantequilla | 2/4 | 0.50 | no | 0.8333 |
| mantequilla ⇒ pan | 2/3 | 0.67 | sí | 0.8333 |

Cuatro reglas pasan la confianza mínima y **todas** tienen lift < 1. Apriori
encuentra lo frecuente y lo confiable, no lo interesante. Pregunta de control:
*¿por qué leche ⇒ pan y pan ⇒ leche tienen la misma confianza?* Porque ambos
ítems tienen soporte 4: el denominador es el mismo.

### 2 · Ver: probabilidad como área (min 20–30)

Regla de la tienda: café ⇒ pan dulce, con 40 tickets con café, 30 con pan dulce
y 20 con ambos (n = 100).

1. **Soporte = área.** El cuadrado es todo (área 1). La columna azul (ancho
   P(A) = 0.40) son los tickets con café.
2. **Independencia = agua nivelada.** Si café y pan dulce fueran independientes,
   el pan dulce (30 % del área) llegaría a la misma altura 0.30 dentro y fuera
   de la columna. El rectángulo amarillo, 0.40 × 0.30 = 0.12, es lo esperado
   juntos por azar.
3. **Predicción.** ¿Qué pasa con el agua si quien compra café tiende a llevar
   pan dulce? Sube dentro de la columna (0.50 = 20/40) y baja fuera (10/60 ≈
   0.17). El área naranja total no cambia.
4. **Confianza = altura dentro de A** = P(B | A) = 0.50.
5. **Lift = altura / nivel** = 0.50 / 0.30 = 1.67. Equivalentemente,
   **área observada / área esperada** = 0.20 / 0.12. Como el cociente de áreas no
   distingue A de B, el lift es **simétrico**; la confianza no:
   conf(pan dulce ⇒ café) = 20/30 = 0.67.
6. **Mover los datos** (animación con lectura en vivo):
   - 12 tickets con ambos → agua nivelada → lift 1.
   - 2 tickets → agua dentro por debajo del nivel → lift 0.17.
   - 30 tickets (todo el pan dulce dentro de café) → altura 0.75 y fuera seco →
     lift 2.5 = **lift máximo** = 1 / max{P(A), P(B)}.
7. **Formalización.** Las cuatro regiones son la tabla de contingencia
   (20, 20 / 10, 50). De `lift = conf / P(B)` se sigue
   `lift > k ⇔ conf > k · P(B)`: con leche al 60 %, exigir lift > 1.5 obliga a
   conf > 0.90 (el «whole milk problem» de las notas de la clase 3).

### 3 · Medir: leverage, conviction y Zhang (min 30–40)

Todas sobre el mismo cuadrado de café ⇒ pan dulce:

| Métrica | En el dibujo | Cálculo | Lectura |
|---|---|---|---|
| Leverage | Franja verde entre el nivel (0.30) y el agua (0.50) en la columna A | 0.20 − 0.12 = 0.08 | 8 tickets de cada 100 aparecen juntos de más. Volumen absoluto. |
| Conviction | Errores esperados (amarillo, sobre el nivel) / errores observados (rojo, sobre el agua) | 0.28 / 0.20 = 1.40 | La regla se equivoca 1.4 veces menos que el azar. Dirigida: conv(pan dulce ⇒ café) = 1.80. |
| Zhang | Diferencia entre altura dentro (0.50) y fuera (0.17), entre la mayor | (0.50 − 0.17)/0.50 = 0.67 | Escala fija de −1 a 1. Z(pan dulce ⇒ café) = 0.57. |
| Lift relativo | Lift entre su techo | 1.67 / 2.5 = 0.67 | Qué fracción del máximo posible se alcanza. |

**Equivalencia de Zhang.** La forma del libro,
`Z = (P(A∪B) − P(A)P(B)) / max{P(A∪B)(1 − P(A)), P(A)(P(B) − P(A∪B))}`,
es idéntica a `(P(B|A) − P(B|¬A)) / max{P(B|A), P(B|¬A)}` (basta dividir
numerador y denominador entre P(A)(1 − P(A))). La segunda es la que se ve en el
cuadrado.

**Las cuatro reglas de la tienda (n = 100):**

| Regla | supp | conf | lift | lev × 100 | conv | Zhang |
|---|---|---|---|---|---|---|
| café ⇒ pan dulce | 0.20 | 0.50 | 1.67 | +8.0 | 1.40 | +0.67 |
| pan dulce ⇒ leche | 0.18 | 0.60 | 1.00 | 0.0 | 1.00 | 0.00 |
| vino ⇒ queso | 0.04 | 0.80 | 10.00 | +3.6 | 4.60 | +0.95 |
| té ⇒ café | 0.02 | 0.10 | 0.25 | −6.0 | 0.67 | −0.79 |

Pregunta clave: *¿cuál regla genera más tickets extra?* La intuición apunta a
vino ⇒ queso (lift 10), pero café ⇒ pan dulce suma 8 contra 3.6. Lift mide
fuerza relativa; leverage, volumen.

### 4 · Decidir (min 40–50)

- **Red de asociación:** nodos = productos, aristas = reglas; verde si lift > 1,
  gris si ≈ 1, rojo si < 1; grosor según |log₂ lift|. Sin flechas porque el lift
  es simétrico. Muestra fuerza, no volumen ni dirección.
- **Mapa soporte × lift** (lift en escala logarítmica, tamaño = |leverage|):
  acción amplia (arriba a la derecha), nicho por validar (arriba a la izquierda
  del umbral de 5 tickets), sin información (sobre lift = 1), sustitutos (abajo).
- **Fichas de decisión:**

| Regla | Acción | Justificación | Riesgo |
|---|---|---|---|
| café ⇒ pan dulce | Combo o promoción cruzada | lift 1.67, +8 tickets/100, soporte 20 % | Canibalizar ventas que ya ocurrían |
| pan dulce ⇒ leche | No mover la leche | lift 1, leverage 0 | Ninguno: la leche se vende sola |
| vino ⇒ queso | Recomendación dirigida | lift 10, conviction 4.6 | Sólo 4 tickets: validar con más días |
| té ⇒ café | No hacer combo | lift 0.25, leverage −6 | Son sustitutos: útil para surtido |

- **Filtro mental** (orden de las notas de la clase 2, ampliado):
  1. ¿Hay asociación? lift > 1, Zhang > 0.
  2. ¿Hay datos suficientes? Conteo de A ∪ B y estabilidad en el tiempo.
  3. ¿Cuánto mueve? leverage × n.
  4. ¿En qué dirección actúo? Confianza y conviction de A ⇒ B y de B ⇒ A.
  5. ¿Es causal? Sólo un experimento (A/B) lo responde.
- **Cuándo no confiar:** pocos tickets (lift inestable), ítem dominante
  (confianza inflada), preguntas causales y problemas donde importa el orden
  (patrones secuenciales, capítulo 5.3c).

## Reto en parejas (min 50–57)

200 tickets: 50 con papas (A), 80 con refresco (B), 30 con ambos.

| Paso | Resultado |
|---|---|
| Dibujo | Ancho P(A) = 0.25; nivel P(B) = 0.40; agua dentro 30/50 = 0.60; fuera 50/150 ≈ 0.33 |
| Confianza | conf(papas ⇒ refresco) = 0.60 |
| Lift | 0.60 / 0.40 = 1.50 |
| Leverage | 0.15 − 0.25 · 0.40 = 0.05 → +10 tickets de 200 |
| Conviction | conv(A ⇒ B) = 0.60 / 0.40 = 1.50; conv(B ⇒ A) = 0.75 / 0.625 = 1.20 |
| Zhang | 0.44 (A ⇒ B); 0.56 (B ⇒ A) |
| Decisión | Ofrecer refresco a quien lleva papas (la dirección con mayor confianza y conviction) y medirlo con un experimento |

**Pista si se atoran:** conf = altura dentro de A; lift = altura / nivel;
leverage × 200 = tickets. **Pregunta de control:** *¿por qué el lift es igual
en ambas direcciones pero la confianza no?* Porque el lift compara áreas y la
confianza divide entre el ancho de la columna del antecedente.

## Errores frecuentes a vigilar

- **«Confianza alta = regla buena».** Sin comparar con P(B) no hay forma de
  saberlo (pan dulce ⇒ leche, 60 %).
- **Confundir lift < 1 con lift = 1.** Lift < 1 es asociación negativa
  (sustitutos); lift = 1 es independencia.
- **Comparar lifts entre pares con soportes muy distintos.** El techo
  1 / max{P(A), P(B)} cambia: un lift de 10 sólo es posible con ítems raros.
- **Leer el soporte como interés.** El soporte mide frecuencia, no relación.
- **Tomar la regla como causa.** Una regla describe co-ocurrencia; la causa se
  prueba con un experimento.
- **Olvidar la dirección.** Lift y leverage son simétricos; confianza,
  conviction y Zhang no.

## Ticket de salida (individual)

> Una regla tiene conf(X ⇒ leche) = 0.90 y P(leche) = 0.90. ¿La usarías para
> una promoción? Justifica con el cuadrado. Bonus: si P(A) = P(B) = 0.5, ¿cuál
> es el lift más alto posible?

**Respuesta esperada:** no; lift = 0.90 / 0.90 = 1, el agua está nivelada y X
no aporta información. Bonus: 1 / 0.5 = 2.

## Notas sobre las fuentes

- La guía de `lectures/notes/guia-docente-asociacion.md` (clase 3, slide 5)
  escribe en un punto `P(A ∪ B) = max(P(A), P(B))` y `1/min`; lo correcto, como
  en `5-2-metricas-extra.qmd`, es que la co-ocurrencia está acotada por el ítem
  **menos** frecuente y `lift_max = 1 / max{P(A), P(B)}`. Esta clase usa la
  versión correcta (café ⇒ pan dulce: 1 / 0.40 = 2.5).
- `5-2-metricas-extra.qmd` describe la métrica de Zhang como simétrica. Con la
  fórmula del propio capítulo no lo es en general: Z(café ⇒ pan dulce) = 0.67
  y Z(pan dulce ⇒ café) = 0.57. La presentación lo muestra explícitamente;
  conviene revisar esa afirmación del capítulo.
