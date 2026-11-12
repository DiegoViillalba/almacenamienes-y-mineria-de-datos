# Guion docente · Reducción de variables categóricas con χ²

**Duración:** 60 minutos. **Fuente conceptual:**
`lectures/04-preprocesamiento/04-05-chi-cuadrado.qmd`. Cada estado de la
presentación tiene una nota (tecla `S`). Todos los números salen de
`sections/data.py`.

## Idea que recorre la clase

> Una variable aporta si conocerla **cambia el perfil** del objetivo. χ² mide
> qué tan lejos está la tabla real del mundo donde eso no pasa; el azar fija
> la vara (valor p); V de Cramér dice qué tan fuerte es; con ambos se eliminan
> columnas que son ruido o que repiten a otra.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Leer una tabla de contingencia como barras de perfil y reconocer a ojo
   dependencia e independencia.
2. Construir la tabla esperada `E_ij = (fila i)(columna j)/n` y calcular χ² a
   mano para una tabla 2 × 2.
3. Explicar el valor p como "qué tan seguido el azar produce un χ² así" y
   calcular los grados de libertad `(r − 1)(c − 1)`.
4. Distinguir significancia (p) de fuerza (V de Cramér) y aplicar los filtros
   de relevancia y redundancia para reducir variables.

**Material:** proyector, pizarra, hojas para el reto en parejas y el ticket.

## Secuencia minuto a minuto

| Min. | Acto | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–6 | 1 · El problema | Cada quien escribe qué dos columnas eliminaría. "¿Promedio de ventana y pasillo?" | Reconocen que con categorías sólo se cuenta |
| 6–14 | 2 · Ver | Predicción antes de separar por retraso. Barras de perfil contra la línea del 50 %. Asiento como contraste | Dicen "aporta / no aporta" mirando barras |
| 14–21 | 3 · Mundo independiente | 50 % de 120 = 60. El grupo dicta las otras celdas de E. Residuos ±20 | Calculan E sin fórmula escrita |
| 21–27 | 4 · Construir χ² | Dos tiendas (40 vs 1,000 esperados). 400/60 y 400/40. Fórmula al final | Explican cada pieza de la fórmula |
| 27–38 | 5 · Azar y g. l. | Barajar tarjetas; seis barajadas en vivo, 400 en histograma; curva χ²₁; 3.84; p; celdas libres | Definen p con sus palabras; obtienen g. l. |
| 38–43 | 6 · Fuerza | Predicción: ¿mismo χ² con n ×10? V de Cramér | Separan "¿es real?" de "¿qué tanto?" |
| 43–53 | 7 · Reducir | Comparar con las elecciones del minuto 2. Matriz de V. Decisión y receta | Justifican cada columna eliminada |
| 53–59 | 8 · Reto | Parejas, 4 min; recorrer el salón | Hoja con E, χ², decisión, V |
| 59–60 | Cierre | Ticket de salida | Respuestas en papel |

## Números de referencia

**Ejemplo guiado** (retraso × satisfacción, n = 200):

| | Alta | Baja | Total |
|---|---|---|---|
| Sin retraso | 80 (E 60) | 40 (E 60) | 120 |
| Con retraso | 20 (E 40) | 60 (E 40) | 80 |
| Total | 100 | 100 | 200 |

Contribuciones 6.67, 6.67, 10, 10 → **χ² = 33.33**, 1 g. l.,
p ≈ 7.8 × 10⁻⁹, **V = 0.41**. Asiento (61/59, 39/41): χ² = 0.08, p = 0.77.

**Simulación de barajadas** (semilla 7, 400 repeticiones): máximo 12; el 4.5 %
supera 3.84 (lo esperado es 5 %); ninguna se acerca a 33.3.

**Significancia vs. fuerza:** tabla 30/20, 20/30 → χ² = 4, p = 0.046,
V = 0.20; la misma ×10 → χ² = 40, p ≈ 2.5 × 10⁻¹⁰, V = 0.20.

**Encuesta de 1,000 pasajeros** (idéntica al capítulo):

| Variable | χ² | g. l. | p | V |
|---|---|---|---|---|
| Retraso | 108.79 | 4 | < 0.001 | 0.233 |
| Clase | 74.03 | 4 | < 0.001 | 0.192 |
| Tipo de viaje | 9.79 | 2 | 0.007 | 0.099 |
| Frecuencia | 16.60 | 4 | 0.002 | 0.091 |
| Asiento | 2.05 | 4 | 0.727 | 0.032 |
| Momento de compra | 0.60 | 2 | 0.740 | 0.025 |

V entre predictoras útiles: clase–tipo de viaje 0.49, clase–frecuencia 0.37,
tipo–frecuencia 0.25, retraso con las demás < 0.10. Decisión: conservar
retraso y clase; tipo de viaje y frecuencia opcionales; eliminar asiento y
momento de compra.

**Reto (Wi-Fi a bordo, n = 120):** O = 28/32, 22/38; E = 25/35 en ambas
filas; residuos ±3; χ² = 9/25 + 9/35 + 9/25 + 9/35 = 1.23 < 3.84; p = 0.27;
V = 0.10 → sin evidencia de relación, candidata a eliminar.

## Errores frecuentes

- **Comparar conteos en vez de proporciones.** "Hay más satisfechos sin retraso"
  puede ser sólo porque ese grupo es más grande. Regresar a las barras al 100 %.
- **Sumar residuos.** Siempre dan cero; por eso el cuadrado.
- **Olvidar dividir entre E** en el reto (dan 36 en lugar de 1.23).
- **Leer p como probabilidad de que H₀ sea cierta.** p es la frecuencia con que
  el azar, suponiendo H₀, produce un χ² al menos así de grande.
- **Usar χ² como medida de fuerza.** Crece con n; para comparar usar V.
- **Eliminar las dos variables de un par redundante.** Se conserva la de mayor
  V con el objetivo.

## Ticket de salida

1. Con un dibujo de barras, explica por qué χ² = 0 cuando dos variables son
   independientes.
2. ¿Basta un p muy pequeño para decidir conservar una variable? ¿Qué más
   mirarías?
