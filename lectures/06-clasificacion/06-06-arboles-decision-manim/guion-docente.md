# Guion docente · Árboles de decisión

**Duración:** 60 minutos. **Fuente conceptual:**
`unidades/clasificadores/6-1-clasificacion_arboles.qmd` (béisbol, entropía,
ID3, C4.5) y `6-1-clasificacion_cart.qmd` (laptops, Gini, CART, poda de
costo-complejidad). Retoma los diez clientes de
`6-0-introduccion-clasificacion.qmd`. Continúa con
[Ensambles y XGBoost](../06-07-ensambles-xgboost-manim/guion-docente.md).

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Todos los números en pantalla se calculan en
`sections/data.py` (`python -m sections.data` imprime el resumen usado aquí).

## Idea que recorre la clase

> Un árbol hace preguntas de sí o no; geométricamente, parte el espacio en
> rectángulos. La «mejor pregunta» es la que deja grupos más puros (entropía o
> Gini), pero un árbol que persigue la pureza total memoriza el ruido: hay que
> podarlo pagando un precio α por cada hoja.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Leer un árbol como reglas SI… ENTONCES y como una partición en rectángulos.
2. Calcular entropía, ganancia de información, razón de ganancia y Gini a mano.
3. Explicar el sesgo de la ganancia hacia atributos con muchos valores y cómo
   lo atenúa C4.5.
4. Encontrar el mejor corte de una variable numérica.
5. Explicar la poda de costo-complejidad (R(T) + α|T|, eslabón más débil) y
   por qué α se elige por validación.

**Material:** proyector, pizarra, calculadora con log₂; papel para el ejercicio.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–2 | Portada y ruta | Pedir ejemplos de diagramas de flujo cotidianos. | Ejemplos de la clase. |
| 2–7 | **1 · Partición** | La regla «deuda ≤ 44» como árbol; segunda y tercera pregunta sólo dentro de su rectángulo. | Explican por qué el corte vertical no cruza todo el plano. |
| 7–10 | **1 · Cliente nuevo y fórmula** | Bajar (33, 52) por el árbol; f(x) = Σ c·1[x∈R]. | Recorren el árbol en voz alta: «no paga». |
| 10–13 | **2 · Béisbol** | Tabla, votación de «una pregunta»; fichas en cubetas por Pronóstico y Temperatura. | Señalan el grupo puro (Nublado). |
| 13–16 | **2 · Entropía** | Medidor de mezcla; fórmula; H = 0.971. | Predicen la forma de la curva; verifican 0.971. |
| 16–19 | **2 · Ganancia** | Barras de área; ganancia 0.280; carrera de los cuatro atributos. | Ordenan atributos antes de ver las barras. |
| 19–23 | **2 · Recursión, reglas, Hunt** | Ramas Soleado y Lluvia; 5 reglas; algoritmo voraz. | Anticipan Humedad y Viento. |
| 23–28 | **3 · C4.5** | Atributo Día (ganancia 0.971); SplitInfo; tabla de razones. | Explican por qué Día no sirve para D16. |
| 28–31 | **3 · Numéricos y resumen** | Barrido de humedad; mejor corte 82.5 %; C4.5 en una lámina. | Gritan «¡alto!» cerca del máximo. |
| 31–34 | **4 · Impurezas** | Tres curvas; cuerda: error gana 0, Gini 0.08. | Explican por qué el error «no ve» la mejora. |
| 34–37 | **4 · Gini y CART** | Laptops: 0.459, ¿adulto? 0.357; tabla ID3 vs CART. | Calculan ¿joven? en parejas (0.394). |
| 37–43 | **4 · Plano y monotonía** | Barrido de cortes de los clientes; mismo árbol del acto 1; √ingreso. | Predicen dónde está el mínimo; contrastan con k-NN. |
| 43–46 | **5 · Sobreajuste** | Árbol de 18 hojas; 300 puntos de prueba; zonas verdaderas. | Predicen la exactitud de entrenamiento y la de prueba. |
| 46–50 | **5 · Costo-complejidad** | Rectas R(T) + α\|T\|; mover α. | Explican por qué aguanta el árbol de 4 hojas. |
| 50–54 | **5 · Eslabón y ruta** | Tabla g(t); curva entrenamiento/prueba. | Verifican R(T) = 0.224 de la raíz. |
| 54–56 | **6 · Límites** | Quitar un punto cambia la raíz; escalera ante una diagonal. | Proponen «promediar muchos árboles». |
| 56–59 | **6 · Ejercicio** | Ocho puntos, tres cortes. | Resultados abajo. |
| 59–60 | Cierre | Cuatro ideas y ticket. | Respuesta escrita. |

### Si el tiempo apremia

- Acto 2: omitir la lámina de reglas (mencionarlas oralmente).
- Acto 3: omitir la lámina «C4.5 en una lámina».
- Acto 4: omitir la tabla ID3 vs CART y la lámina de la transformación
  monótona.
- Acto 5: omitir la ruta de poda (última lámina).

## Cálculos de referencia

### Diez clientes (CART con Gini)

- Gini ponderado de los cortes en deuda: 22.5 → 0.444, 27.5 → 0.400,
  32.5 → 0.343, 37.5 → 0.267, **44 → 0.160**, 51.5 → 0.317, 56.5 → 0.419,
  59 → 0.300, 61 → 0.400. El mejor de ingreso es 13.5 → 0.400.
- Árbol completo: deuda ≤ 44 (5 pagan, puro); si no, ingreso ≤ 29 (3 no pagan,
  puro); si no, ingreso ≤ 32.5 (el cliente 6 | el cliente 9).
- Cliente nuevo (33, 52): no, no, no → hoja del cliente 9 → «no paga».

### Béisbol (9 sí, 6 no)

H(S) = −0.6·log₂0.6 − 0.4·log₂0.4 = 0.971.

| Atributo | H después | Ganancia | SplitInfo | Razón |
|---|---:|---:|---:|---:|
| Pronóstico | 0.691 | **0.280** | 1.566 | 0.179 |
| Humedad | 0.785 | 0.186 | 0.997 | **0.186** |
| Viento | 0.911 | 0.060 | 0.918 | 0.065 |
| Temperatura | 0.958 | 0.013 | 1.566 | 0.009 |
| Día | 0 | 0.971 | 3.907 | 0.249 |

- Pronóstico: (6/15)·0.918 + (4/15)·0 + (5/15)·0.971 = 0.367 + 0 + 0.324 = 0.691.
- Rama Soleado (2 sí, 4 no; H = 0.918): Humedad 0.918, Temperatura 0.252,
  Viento 0.109. Rama Lluvia (3 sí, 2 no; H = 0.971): Viento 0.971,
  Humedad 0.020, Temperatura 0.020.
- La razón de Temperatura es 0.0085; el libro la muestra truncada como 0.008
  y la presentación redondea a 0.009.
- Humedad numérica: mejor corte 82.5 % (≤: 6 sí, 1 no; >: 3 sí, 5 no),
  ganancia 0.186; le siguen 87.0 (0.145) y 85.5 (0.078).

### Impurezas y concavidad (padre p = 0.25, hijos 0.05 y 0.45 de igual tamaño)

| | padre | hijos | promedio | ganancia |
|---|---:|---|---:|---:|
| Error | 0.250 | 0.05, 0.45 | 0.250 | 0 |
| Gini | 0.375 | 0.095, 0.495 | 0.295 | 0.080 |
| Entropía | 0.811 | 0.286, 0.993 | 0.640 | 0.172 |

### Laptops (9 compran, 5 no)

G raíz = 0.459. ¿adulto? 4/0 y 5/5 → 0.357 (ΔG 0.102); ¿joven? 2/3 y 7/2 →
0.394 (0.066); ¿mayor? 3/2 y 6/3 → 0.457 (0.002). Mejor de cada atributo:
edad 0.102, estudiante 0.092, crédito 0.031, ingreso 0.016.

### Poda (60 puntos de entrenamiento, 300 de prueba)

| α | hojas | entrenamiento | prueba |
|---:|---:|---:|---:|
| 0 | 18 | 1.000 | 0.717 |
| 0.0141 | 14 | 0.967 | 0.743 |
| 0.0148 | 12 | 0.950 | 0.757 |
| 0.0152 | 10 | 0.933 | 0.773 |
| 0.0172 | 7 | 0.900 | 0.843 |
| 0.0188 | 4 | 0.867 | **0.853** |
| 0.0902 | 1 | 0.550 | 0.543 |

Árbol de 4 hojas: raíz x₂ ≤ 5.70 (g = 0.090), izquierdo x₁ ≤ 4.05
(g = 0.115), derecho x₁ ≤ 2.30 (g = 0.125). R(T) de la raíz:
13/60·0.260 + 21/60·0.308 + 0 + 20/60·0.180 = 0.224;
g = (0.495 − 0.224)/3 = 0.090.

Inestabilidad: sin el punto (9.7, 2.2) la raíz pasa a x₁ ≤ 8.75 y el árbol
completo tiene 20 hojas.

### Ejercicio final

Puntos (x₁, x₂, clase): (1,6,■) (2,7,■) (3,2,●) (4,8,■) (5,3,●) (6,6,●) (7,1,●)
(8,7,●).

1. G raíz = 1 − (3/8)² − (5/8)² = 0.469.
2. x₁ ≤ 2.5 → 0.208; **x₁ ≤ 4.5 → 0.188**; x₂ ≤ 4.5 → 0.300.
3. CART elige x₁ ≤ 4.5: a la derecha 4 círculos (puro).
4. A la izquierda (3 ■, 1 ●) basta x₂ ≤ 4: árbol de 3 hojas.

## Errores frecuentes que conviene anticipar

- **Promediar sin ponderar.** La entropía (o Gini) después de preguntar pesa
  cada grupo por su tamaño; el promedio simple sobreestima a los grupos chicos.
- **Elegir la pregunta con la menor entropía de UN hijo.** Se compara la
  entropía ponderada de todos los hijos, o la ganancia.
- **Creer que la mayor ganancia siempre es la mejor pregunta.** Identificadores
  (Día, folio, código postal fino) ganan por azar: se excluyen.
- **Pensar que el corte numérico se prueba en los valores.** Se prueba en los
  puntos medios entre valores consecutivos.
- **Confundir el número de hojas con la profundidad.**
- **Elegir α mirando la prueba.** Se elige con validación cruzada; la prueba se
  usa una vez al final.

## Ticket de salida

¿Por qué un árbol de 18 hojas puras puede predecir peor que uno de 4? Explica
qué papel juega α en la poda. (Respuesta esperada: las hojas pequeñas se
ajustan al ruido —sobreajuste—; α cobra un precio por hoja, de modo que sólo
sobreviven los subárboles cuya reducción de impureza por hoja supera α.)
