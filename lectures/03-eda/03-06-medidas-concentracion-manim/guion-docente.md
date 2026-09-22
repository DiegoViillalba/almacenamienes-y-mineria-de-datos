# Guion docente · Medidas de concentración

**Duración:** 60 minutos. **Fuente conceptual:** `unidades/intro_mineria/2-7-concentracion.qmd`. La presentación tiene pausas con notas integradas (`S` en Reveal.js). Este guion organiza el tiempo, los cálculos de pizarra y la evaluación formativa. El mismo caso —ventas de cinco cafeterías— recorre toda la clase.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Definir unidad de análisis, masa no negativa, total positivo y participaciones; distinguir igualdad de concentración total.
2. Construir e interpretar `CR₂`, HHI, `HHI*`, número efectivo, puntos de Lorenz, Gini y `G*` para cinco unidades.
3. Predecir y comprobar cómo una transferencia de una unidad pequeña a una grande cambia `CR₂`, HHI, Lorenz y Gini.
4. Escoger una herramienta según la pregunta y reconocer pérdidas de información, diferencias de escala, ceros y cambios de agregación.

**Material:** proyector, pizarra y calculadora opcional. Abrir `index.html`; `→` avanza, `S` abre notas del presentador. Escribir al inicio `Norte 5, Sur 10, Centro 15, Oriente 20, Poniente 50` y conservarlo en la pizarra.

## Secuencia minuto a minuto

| Minutos | Estados / acción docente | Evidencia de aprendizaje |
|---|---|---|
| 0–7 | Portada, objetivos, tres repartos. Preguntar qué cambia si el total y la media son los mismos. Definir unidad, masa y `T=100`. | Distinguen `20,20,20,20,20`, `0,0,0,0,100` y el caso base. |
| 7–13 | Participaciones y `CR₂`. Hacer que localicen primero a las líderes. Contrastar A y B con `CR₁=.5`. | `CR₂=.70`; explican qué no ve `CR₁`. |
| 13–19 | Urna de 100 boletos: dos extracciones con reemplazo. Calcular los cinco cuadrados antes de revelar la suma. | `HHI=.325`; lo leen como coincidencia, `1−HHI=.675` como diferencia. |
| 19–23 | Comparar con igualdad y monopolio. Normalizar y obtener unidades efectivas. | `1/5≤HHI≤1`, `HHI*=.15625`, `1/HHI≈3.08`. |
| 23–26 | Transferir 5 ventas Norte → Poniente. Pedir predicción antes de animar. | `CR₂=.75`, `HHI=.375`, `ΔHHI=.05`; el total sigue en 100. |
| 26–30 | Acumular 0, 5, 15, 30, 50, 100. **Trabajo en parejas de 90 s:** escribir seis puntos de Lorenz. | Ordenan ascendentemente y obtienen `(0,0),(.2,.05),(.4,.15),(.6,.30),(.8,.50),(1,1)`. |
| 30–37 | Dibujar diagonal y revelar curva; trazar la curva tras transferir cinco ventas, y comparar con igualdad. | Explican que el 80% inferior reúne 50%; conocen el límite de curvas cruzadas. |
| 37–44 | Área entre Lorenz e igualdad; cinco trapecios. **Trabajo en parejas de 2 min:** promedios de alturas. Relacionar con diferencias por pares. | `B=.300`, `G=.400`; no confunden Gini con cuota del líder. |
| 44–46 | Comparar extremos de Gini para `K=5` y normalizar. | `Gmax=.8`, `G*=.5`. |
| 46–49 | Guía de decisión y A/B con mismo líder. Pedir una pregunta de negocio antes de elegir índice. | Justifican CR, HHI, Lorenz o Gini; A tiene `HHI=.325, G=.4`, B `.3125, .3`. |
| 49–54 | **Práctica en parejas:** `10,10,20,20,40`. No mostrar solución hasta completar el tiempo. | Calculan y escriben al menos un paso de cada método. |
| 54–57 | Puesta en común, respuestas escalonadas. | `CR₂=.60`, `HHI=.260`, `Nef≈3.85`, `L(.8)=.6`, `B=.36`, `G=.28`. |
| 57–60 | Ceros, agregación y puente a BigMart. Ticket individual de salida. | Proponen HHI o Lorenz/Gini para mirar el otro 50% y explican por qué `CR₁` no basta. |

## Hilo conductor y formalización

**1. ¿Qué contamos?** En el caso base, la sucursal es la unidad y las ventas son la masa. `T=Σxᵢ=100`, `sᵢ=xᵢ/T`; las participaciones son `.05,.10,.15,.20,.50`. `xᵢ≥0` y `T>0` son condiciones de las fórmulas usadas. La media de 20 no resume el reparto. Multiplicar todas las ventas por una constante positiva deja intactos los índices.

**2. ¿Cuánto reúnen las líderes?** Antes de mostrar símbolos, señalar las barras de 50 y 20: juntas producen 70 ventas. Formalizar `CRₖ=Σᵢ₌₁ᵏs₍ᵢ₎` con orden descendente. `CR₁=.50`, `CR₂=.70`. Contrastar `A=(50,20,15,10,5)` con `B=(50,12.5,12.5,12.5,12.5)`: `CR₁` empata, aunque la cola difiere. Pregunta de control: “¿Qué ocurriría si elijo un `k` distinto?” Debe declararse `k` y `K`.

**3. ¿Coinciden dos ventas?** Primero imaginar la urna. Poniente dos veces tiene probabilidad `.5²=.25`. Para cualquiera de las cinco sucursales, sumar los sucesos mutuamente excluyentes: `HHI=Σsᵢ²=.325`. Complemento de diferencia: `1−HHI=.675`. El mínimo con `K=5` es `.2`, el máximo es `1`; `HHI*=(.325−.2)/(.8)=.15625`. Número efectivo `1/.325≈3.08`. En escala 0–10 000, el mismo valor es 3250; no aplicar umbrales institucionales sin contexto. **Precisión:** el HHI es probabilidad de coincidencia, no una “probabilidad de dominancia” ni 32.5% del recorrido entre extremos.

**4. ¿Qué hace una transferencia?** Cambiar `(5,10,15,20,50)` a `(0,10,15,20,55)` mantiene `T=100`. `HHI` pasa de `.325` a `.375`; `CR₂` de `.70` a `.75`. Con receptora `a`, donante `b` y fracción transferida `δ`, `ΔHHI=2δ(sₐ−s_b)+2δ²=.05`. Pedir que el grupo prediga el signo antes de sustituir. La curva de Lorenz se arquea más y Gini también aumenta; el cambio de Gini es `.08` (de `.40` a `.48`).

**5. ¿Cómo se acumula desde abajo?** Orden ascendente; construir los seis puntos indicados en la tabla. Formalizar `uᵢ=i/K`, `Lᵢ=(Σⱼ₌₁ⁱx₍ⱼ₎)/T` y `L₀=0`. La diagonal es igualdad. En `(0.8,0.5)`, cuatro de cinco sucursales generan la mitad de ventas. Una curva por debajo de otra en todo punto indica mayor desigualdad en el sentido de Lorenz; si se cruzan, no existe un orden inequívoco por este criterio.

**6. ¿Cómo resumir la curva?** Construir el área primero. `G=1−2B`, donde `B` es el área bajo Lorenz. Cinco trapecios de ancho `.2` tienen alturas medias `.025,.100,.225,.400,.750`; `B=.2(1.5)=.3`, `G=.4`. Lectura alternativa: diferencia absoluta esperada entre dos sucursales aleatorias `16`; dividir entre `2×20` también da `.4`. Con cinco unidades, el máximo finito es `(K−1)/K=.8` y `G*=G/.8=.5`. Indicar si se reporta la versión normalizada.

**7. ¿Qué cuenta la historia completa?** Comparar cuatro lecturas del mismo vector: `CR₂=.70` responde por líderes; `HHI=.325` es coincidencia; Lorenz muestra acumulación; `G=.40` resume brechas. Ninguna por sí sola demuestra causa, monopolio o daño. Antes de llevarlo a BigMart, definir si las unidades son productos, categorías u outlets y si la masa son ventas monetarias o unidades vendidas.

## Respuestas de práctica y errores a vigilar

Para `10,10,20,20,40`: participaciones `.1,.1,.2,.2,.4`; `CR₂=.6`; `HHI=.01+.01+.04+.04+.16=.26`; `1/HHI≈3.846`; puntos acumulados `0,.1,.2,.4,.6,1`; promedios de alturas `.05,.15,.30,.50,.80`; `B=.2(1.8)=.36`; `G=1−.72=.28`. Si se pide normalización, `HHI*=(.26−.2)/.8=.075` y `G*=.28/.8=.35`.

- **Orden invertido:** `CRₖ` usa orden descendente; Lorenz, ascendente.
- **HHI como porcentaje de concentración máxima:** su mínimo para `K=5` es `.2`, no cero.
- **Gini como cuota del líder:** `G=.4` no significa que una sucursal venda 40%; el líder vende 50%.
- **Borrar ceros:** HHI bruto no cambia, pero sí `K`, Lorenz, Gini y las normalizaciones.
- **Agrupación:** dividir la sucursal de 50 en dos de 25 cambia los índices si se consideran unidades separadas. Exigir una definición sustantiva de unidad.

## Ticket de salida y seguimiento

Pedir por escrito: “Dos sistemas tienen `CR₁=.50`. ¿Qué añadirías para estudiar el otro 50% y por qué?” Respuesta esperada: HHI y/o curva de Lorenz con Gini, porque usan el resto de participaciones y permiten distinguir la estructura de la cola. Para extender la clase, usar el laboratorio interactivo del capítulo y después el caso BigMart `2-7-concentracion_bigmart.qmd`.
