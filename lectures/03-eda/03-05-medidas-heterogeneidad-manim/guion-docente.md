# Guion docente · medidas de heterogeneidad

## Propósito de la sesión

Repaso aplicado para estudiantes que ya estudiaron el tema. La sesión no busca
recitar fórmulas: reconstruye cada medida desde una geometría y termina con una
transferencia a datos no usados en los ejemplos guiados.

Cadena que debe quedar instalada:

```text
pregunta → dibujo del reparto → proporciones → medida → convención → interpretación
```

Para una respuesta numérica agrupada:

```text
puntos → media global → medias grupales → SS_B + SS_W → η² → cautelas
```

## Ruta sugerida

| Minutos | Estados | Bloque | Evidencia observable |
|---:|---:|---|---|
| 0–4 | 1–5 | Dos preguntas y predicción | Distinguen composición categórica de separación numérica |
| 4–11 | 6–12 | Gini–Simpson geométrico | Interpretan el complemento del área diagonal |
| 11–18 | 13–19 | Shannon y categorías efectivas | Explican sorpresa ponderada, nats y normalización |
| 18–22 | 20–23 | Equilibrio y cambio de pregunta | Predicen el movimiento conjunto de los medidores |
| 22–31 | 24–32 | Descomposición entre/dentro | Reconstruyen `20 = 19 + 1` y acotan `η²` |
| 31–33 | 33 | Heterogeneidad cruzada | Leen media y tamaño en cada celda |
| 33–38 | 34–36 | Reto nuevo | Entregan cálculo, convención e interpretación |
| 38–39 | 37 | Cierre | Eligen la representación según la pregunta |

## Conducción por bloques

### 1. Reparto antes que fórmula

Los seis datos son:

```text
A, A, A, B, C, C
```

Transformación visible:

```text
f = (3, 1, 2) → p = (1/2, 1/6, 1/3) → reparto uniforme (2, 2, 2)
```

No presente “adivinar la categoría” como una definición formal. Úselo como una
pregunta intuitiva: la incertidumbre aumenta cuando las proporciones se
equilibran.

Preguntas útiles:

- ¿Qué cambió si `n=6` y `K=3` siguen fijos?
- ¿Contar categorías ocupadas basta para describir el reparto?
- ¿Qué objeto visual registra el desequilibrio sin ordenar las etiquetas?

### 2. Gini–Simpson como área

El cuadrado representa todos los pares de extracciones independientes. Los
bloques diagonales son los pares que coinciden en categoría:

```text
P(misma) = (1/2)² + (1/6)² + (1/3)²
          = 1/4 + 1/36 + 1/9
          = 7/18 = 0.3889
```

El complemento es:

```text
D = 1 - Σp² = 11/18 = 0.6111
```

Interpretación modelo:

> En dos extracciones independientes de esta distribución, la probabilidad de
> obtener categorías distintas es 61.11%.

No acepte “hay 61.11% de diversidad” sin declarar el evento. Si se eligen dos
registros distintos **sin reemplazo**, la fórmula es otra; esta presentación
adopta el esquema independiente/con reemplazo del índice `1-Σp²`.

Extremos con `K=3` fijo:

```text
(1,0,0)       → D = 0
(1/3,1/3,1/3) → Dmáx = 1 - 1/3 = 2/3
```

El número efectivo de Simpson invierte la probabilidad de coincidencia:

```text
1 / Σp² = 18/7 = 2.5714
```

No es `1/D`.

### 3. Shannon: sorpresa por frecuencia

Con `ln`, la sorpresa de cada categoría es `-ln(p_k)`:

| Categoría | `p_k` | `-ln(p_k)` | `-p_k ln(p_k)` |
|---|---:|---:|---:|
| A | 1/2 | 0.6931 | 0.3466 |
| B | 1/6 | 1.7918 | 0.2986 |
| C | 1/3 | 1.0986 | 0.3662 |

La categoría rara sorprende más cuando ocurre, pero pesa menos en el promedio.
La suma correcta es:

```text
H = 0.3466 + 0.2986 + 0.3662
  = 1.011404... nats
```

#### Intuición matemática: El pico en $1/e$ y la continuidad

La función de contribución $f(p) = -p \ln(p)$ es continua en $[0, 1]$.
Al derivar e igualar a cero:

$$f'(p) = -\ln(p) - 1 = 0 \implies \ln(p) = -1 \implies p = \frac{1}{e} \approx 0.3679$$

- **Cima de incertidumbre:** Una categoría con frecuencia $p \approx 0.368$ es la que más aporta a la entropía total ($f(1/e) = 1/e \approx 0.368$).
- **Por qué $C$ aporta más que $A$ o $B$:** Como $p_C = 1/3 \approx 0.333$, se encuentra a sólo $0.035$ del pico global. $B$ ($1/6 \approx 0.167$) tiene alta sorpresa pero poca frecuencia; $A$ ($1/2 = 0.5$) es muy frecuente pero aporta poca sorpresa.
- **Continuidad de $0 \ln 0 := 0$:** Por la regla de L'Hôpital, $\lim_{p \to 0^+} -p \ln p = \lim_{p \to 0^+} \frac{-\ln p}{1/p} = \lim_{p \to 0^+} \frac{-1/p}{-1/p^2} = \lim_{p \to 0^+} p = 0$. La curva aterriza suavemente en $(0,0)$. No es un dogma: es continuidad matemática pura.

#### Familia unificada de Números de Hill ($^q D$)

Tanto Shannon como Simpson pertenecen a la familia paramétrica de Hill (1973):

$$^q D = \left( \sum_{k=1}^K p_k^q \right)^{\frac{1}{1-q}}$$

- **$q = 0$ (Riqueza):** $^0 D = K = 3$ (cuenta categorías sin importar frecuencia).
- **$q \to 1$ (Shannon):** $^1 D = \exp(H) \approx 2.75$ (pondera proporcionalmente).
- **$q = 2$ (Simpson):** $^2 D = \frac{1}{\sum p_k^2} \approx 2.57$ (penaliza categorías raras).
- Propiedad fundamental: Para cualquier distribución no uniforme, $^0 D > ^1 D > ^2 D$. Se igualan únicamente bajo uniformidad perfecta.

Con `K=3`:

```text
Hmáx  = ln(3) = 1.098612...
Hnorm = H/ln(3) = 0.920620...
exp(H) = 2.749459... ≈ 2.75 categorías efectivas
```

Convenciones que se dicen en voz alta:

- usamos `ln`: la unidad es el nat;
- definimos `0 ln 0 := 0` por continuidad;
- `K` es el conjunto de niveles declarado para el análisis;
- cambiar la base cambia la unidad, no el orden;
- la normalización usa la misma base arriba y abajo.

### 4. Mover masa con `K=4`

Los tres escenarios proceden del QMD:

| Escenario | Proporciones | `D` | `Hnorm` | `exp(H)` |
|---|---|---:|---:|---:|
| Dominante | .82, .08, .06, .04 | .316 | .478 | 1.94 |
| Intermedio | .50, .25, .15, .10 | .655 | .871 | 3.35 |
| Uniforme | .25, .25, .25, .25 | .750 | 1.000 | 4.00 |

Haga que los estudiantes describan el movimiento: la masa sale de la categoría
dominante y llega a las minoritarias. Mantener `K` fijo es parte de la
comparación. Un valor alto sólo indica mayor equilibrio, no mayor calidad,
rendimiento o conveniencia comercial.

### 5. Respuesta numérica entre grupos

Datos:

```text
A = (4,5)    B = (8,9)    C = (5,5)
```

Media global y medias grupales:

```text
ȳ = 36/6 = 6
ȳ_A = 4.5    ȳ_B = 8.5    ȳ_C = 5
```

Variación total:

```text
SS_T = (4-6)² + (5-6)² + (8-6)² + (9-6)² + (5-6)² + (5-6)²
     = 4 + 1 + 4 + 9 + 1 + 1
     = 20
```

Componente entre grupos; el factor `n_g=2` no puede omitirse:

```text
SS_B = 2(4.5-6)² + 2(8.5-6)² + 2(5-6)²
     = 4.5 + 12.5 + 2
     = 19
```

Componente dentro de grupos:

```text
SS_W = (0.25+0.25) + (0.25+0.25) + (0+0) = 1
```

Identidad y cociente:

```text
SS_T = SS_B + SS_W      20 = 19 + 1
η² = SS_B/SS_T = 19/20 = 0.95
```

#### Geometría de ANOVA: La suma de cuadrados como área euclidiana

- La variación no es un número abstracto: es la **suma de las áreas de 6 cuadrados geométricos** levantados sobre los segmentos $|y_i - \bar{y}|$ en la recta numérica.
- **Partición física del área:** El área total acumulada ($20$) se redistribuye exactamente en:
  - $19$ unidades de área entre centros de grupos ponderados ($SS_B = 4.5 + 12.5 + 2$). Observe que el Grupo B por sí solo aporta más del 65% de la separación entre medias ($12.5 / 19$).
  - $1$ unidad de área dentro de grupos ($SS_W = 0.5 + 0.5 + 0$).

Interpretación modelo:

> El 95% de la suma de cuadrados total observada corresponde a la separación
> entre las medias grupales en estos datos.

Evite “los grupos causan 95% de la variación”. `η²` aquí es descriptiva: no
demuestra causalidad y no sustituye la revisión de medianas, dispersión, forma
ni tamaños de grupo. En muestras pequeñas, `η²` tiende a sobreestimar el efecto poblacional, por lo que la literatura avanzada reporta también `ω²` (omega cuadrada insesgada).

### 6. Heterogeneidad cruzada

Cuando dos variables categóricas interactúan, una media o mediana por celda
debe mostrarse junto con su conteo. El estado 33 usa un heatmap genérico para
hacer visible que una diferencia grande en una celda con `n=2` exige cautela.
No se calcula un índice nuevo: el objetivo es mirar simultáneamente patrón y
tamaño de celda, como indica el QMD.

## Reto de transferencia

Datos sintéticos nuevos, no usados en los ejemplos guiados:

```text
Canal      Web   App   Teléfono   Tienda
Frecuencia   9     6          3        2
```

En el estado 34 deje **tres minutos reales**. No muestre proporciones ni una
fórmula resuelta. Pida:

1. dibujar el reparto;
2. fijar `K=4`, `n=20` y `ln`;
3. calcular `D`, `Hnorm`, `exp(H)` y el inverso de Simpson `1/Σp²`;
4. escribir una interpretación con el evento probabilístico y los órdenes de Hill.

Solución independiente verificada:

```text
p = (.45, .30, .15, .10)
Σp² = .2025 + .09 + .0225 + .01 = .325
D = 1 - .325 = .675
```

Aportes de Shannon:

```text
.45[-ln(.45)] = .359328...
.30[-ln(.30)] = .361192...
.15[-ln(.15)] = .284568...
.10[-ln(.10)] = .230259...
H = 1.235347... ≈ 1.2353 nats
Hnorm = 1.2353/ln(4) = .8911
exp(H) = 3.44 categorías efectivas (Hill orden 1)
1 / Σp² = 1 / 0.325 = 40/13 ≈ 3.08 categorías efectivas (Hill orden 2)
```

Reporte modelo:

> El reparto es relativamente equilibrado, aunque no uniforme. Bajo dos
> extracciones independientes, los canales difieren con probabilidad 67.5%.
> La entropía alcanza 89.11% de su máximo para cuatro canales y equivale a unas
> 3.44 categorías igualmente frecuentes según Shannon, o 3.08 según Simpson.
> El orden 2 castiga más la presencia de categorías minoritarias (Tienda y Teléfono).

## Puentes curriculares con Minería de Datos y Machine Learning

Estas medidas descriptivas son las funciones de costo e impureza que gobiernan los algoritmos centrales del curso:

1. **Árboles de Decisión (CART):** La función de impureza en cada nodo de decisión $t$ es exactamente el índice de Gini–Simpson:
   $$I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2$$
   La reducción de impureza tras una partición binaria ($\Delta I_G$) determina el mejor split.
2. **Árboles C4.5 / ID3:** La entropía de Shannon $H(t)$ gobierna la *Ganancia de Información*. Para resolver el sesgo hacia variables con muchas categorías, C4.5 divide entre la entropía intrínseca del atributo (*Split Info*), obteniendo el *Gain Ratio*, que es formalmente un cociente de entropías idéntico conceptualmente a $H_{\mathrm{norm}}$.
3. **Clustering ($k$-means):** El objetivo analítico de $k$-means es minimizar la dispersión intra-cluster ($WSS = SS_W$), lo cual, dada la invariancia $TSS = BSS + WSS$, equivale exactamente a maximizar la separación entre clusters ($BSS = SS_B$). El índice de validación de **Calinski–Harabasz** es formalmente una versión de $\frac{\eta^2}{1-\eta^2}$ reescalada por grados de libertad.

## Dinámica con el Simulador Interactivo (`interactivo.html`)

Para consolidar la intuición en clase activa:

1. **Exploración de la colina de Shannon:** Proyecte `interactivo.html`. Pida a los estudiantes mover los controles para ubicar una sola categoría en $p = 1/e \approx 0.368$ y observar cómo esa categoría maximiza su contribución visual.
2. **Colapso de Gini–Simpson:** Arrastre una categoría hasta $p=0.95$ y observe cómo el cuadrado unitario queda copado por un solo bloque diagonal, comprimiendo el área de pares distintos ($D$) a una franja casi invisible.
3. **Balanza ANOVA en vivo:** En la pestaña 2, arrastre los puntos de los grupos A, B y C. Muestre cómo al fusionar los centros grupales la barra verde de $SS_B$ colapsa a 0% ($\eta^2 = 0$), mientras que al separar los grupos con mínima dispersión interna la barra verde llena el 99% ($\eta^2 \approx 1$).

## Correcciones incorporadas respecto del PDF

Estas decisiones son deliberadas y deben conservarse en futuras ediciones:

1. **Nombre del índice.** En las páginas 9–17 y 23–31 el PDF llama “Gini” a
   `1-Σp²`. La presentación usa **Gini–Simpson** y lo distingue del coeficiente
   de Gini de desigualdad.
2. **Base logarítmica.** Las páginas 19 y 22 usan `ln`, pero la tabla de la
   página 24 cambia a `log₂` sin explicar que los resultados están en bits. La
   presentación usa `ln` y nats de forma consistente.
3. **Índice de categorías.** La página 28 escribe `p₁,…,p_n`; las proporciones
   categóricas deben indexarse `p₁,…,p_K`. Aquí `n` se reserva para el número de
   observaciones.
4. **Universo ambiguo.** Los ejemplos concentrados de las páginas 6 y 15 omiten
   la categoría X, mientras los ejemplos uniforme e intermedio usan cinco
   categorías. Aquí `K` se fija antes de comparar y los ceros se conservan.
5. **Máximo yuxtapuesto al caso intermedio.** Las páginas 12–13 colocan
   `Dmáx=.80` junto a la distribución no uniforme `(20,15,10,3,2)`, cuyo valor
   real es `.7048`. La presentación separa el máximo teórico del valor observado.
6. **Interpretación de `.7048`.** No significa “70.48% de diversidad” en una
   escala universal. Sí es la probabilidad de categorías distintas bajo el
   esquema independiente; para `K=5`, el máximo bruto es `.8`.
7. **Afirmación sobre localización.** La página 4 dice en bloque que media,
   mediana y moda no se usan para cualitativos. Se corrige la sobregeneralización:
   la media no corresponde a nominales, la moda sí aplica a categóricas y la
   mediana puede tener sentido para ordinales.
8. **Analogía de adivinación.** La página 20 la presenta sin matiz. Aquí es una
   intuición de incertidumbre, no la tasa de error de un predictor modal.
9. **Condición “K fijo”.** “Gini alto = mayor equilibrio” requiere fijar el
   universo de categorías. La condición aparece explícitamente en pantalla y
   notas.

Los cálculos numéricos del QMD para el ejemplo `(A,A,A,B,C,C)` y la
descomposición `20=19+1` son correctos. Se conserva su contenido y se precisa
que el 95% se refiere a la suma de cuadrados observada, no a causalidad.

## Controles de cierre

Antes de terminar la clase, pida que completen oralmente:

- “D es la probabilidad de…”
- “H está en ___ porque usamos ___.”
- “El máximo de D depende de…”
- “Un número efectivo traduce el índice a…”
- “η² no demuestra…”
