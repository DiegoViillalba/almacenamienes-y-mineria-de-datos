# Guía Docente — Clase 2: Reglas de Asociación
## *Interpretación Crítica y Selección de Reglas Útiles*

> **Audiencia:** Profesor de Machine Learning  
> **Nivel:** Licenciatura (física, datos, ingeniería)  
> **Duración total:** ~60 minutos  
> **Prerequisito:** Clase 1 completada (soporte, confianza, lift, mlxtend básico)

---

## Tabla de Contenidos

1. [Slide 1 — ¿Dónde estamos?](#slide-1)
2. [Slide 2 — Configuración del entorno](#slide-2)
3. [Slide 3 — El Dataset](#slide-3)
4. [Slide 4 — Frecuencia de ítems](#slide-4)
5. [Slide 5 — Matriz de co-ocurrencia](#slide-5)
6. [Slide 6 — Generación de reglas](#slide-6)
7. [Slide 7 — El problema central](#slide-7)
8. [Slide 8 — Tabla de métricas](#slide-8)
9. [Slide 9 — Scatter interactivo](#slide-9)
10. [Slide 10 — Interpretando el scatter](#slide-10)
11. [Slide 11 — Caso 1: regla débil](#slide-11)
12. [Slide 12 — Caso 2: regla fuerte](#slide-12)
13. [Slide 13 — Selección de reglas](#slide-13)
14. [Slide 14 — Cuándo NO usar](#slide-14)
15. [Slide 15 — Resumen](#slide-15)

---

## Slide 1 — ¿Dónde estamos? {#slide-1}

### Objetivo del slide
Anclar la memoria de los estudiantes con la clase previa y crear tensión cognitiva que motive la sesión actual.

### Concepto clave
**Transición de "generar reglas" a "evaluar reglas"**. El estudiante pasó de la mecánica al juicio crítico.

### Lo que el profesor debe decir
> *"La semana pasada hicimos algo que es bastante fácil: le dijimos al algoritmo que nos diera reglas. Y las obtuvimos. Muchas. Hoy la pregunta es: ¿cuántas de esas reglas deberíamos creerle? ¿Cuántas son información real y cuántas son accidentales?"*

Mencionar que este problema —distinguir señal de ruido— es uno de los problemas centrales de toda la minería de datos, no solo de las reglas de asociación. Conectar con la idea de "sobreajuste" que los estudiantes probablemente conocen del contexto de modelos predictivos.

### Errores comunes de los estudiantes
- Creer que si el algoritmo lo devolvió, es verdad.
- Confundir la facilidad computacional de generar reglas con la validez estadística de las mismas.
- No cuestionar que `min_support` y `min_threshold` son parámetros arbitrarios del usuario, no umbrales con significado estadístico intrínseco.

### Preguntas sugeridas para discusión
- "¿Cuántas reglas obtuvimos la clase pasada? ¿Las revisamos todas?"
- "¿Ustedes confiarían en una regla que aparece en solo 2 de 50 transacciones?"

### Conexión con conceptos previos
Directa con Clase 1. El slide es deliberadamente corto para no repetir.

---

## Slide 2 — Configuración del entorno {#slide-2}

### Objetivo del slide
Ejecutar el entorno limpio antes de mostrar datos. Es funcional, no pedagógico.

### Lo que el profesor debe decir
> *"Ejecutamos este bloque primero para asegurarnos que todo funcione. Si hay errores de importación, los vemos aquí."*

### Nota técnica
Si `plotly` no está instalado, puede instalarse con `pip install plotly` en la terminal antes de clase. El `np.random.seed(42)` garantiza reproducibilidad si se usan muestras aleatorias.

---

## Slide 3 — El Dataset {#slide-3}

### Objetivo del slide
Mostrar el dataset concreto que se usará en toda la sesión. La transparencia del dataset es clave para el pensamiento crítico.

### Concepto clave
**Dataset pequeño = inestabilidad estadística**. 50 transacciones es suficiente para mostrar el algoritmo, pero insuficiente para confiar ciegamente en las reglas.

### Lo que el profesor debe decir
> *"Este dataset tiene 50 transacciones. Es una muestra representativa del dataset Groceries que vimos la semana pasada. ¿Por qué trabajamos con 50 y no con 9,000? Porque queremos que el código corra en tiempo real. Pero ojo: el tamaño tiene consecuencias en la interpretación."*

Señalar que `whole milk` aparecerá de forma desproporcionada, igual que en el dataset real. Eso no es un defecto del dataset —es un reflejo de la realidad de un supermercado— pero sí implica que cualquier regla que incluya `whole milk` como consecuente tendrá artificialmente alta confianza.

### Errores comunes
- Los estudiantes tienden a ignorar el tamaño del dataset al interpretar reglas.
- Es frecuente que confundan "transacción" con "cliente único". Aclarar que en Groceries una transacción puede ser de cualquier cliente en cualquier momento.

### Preguntas sugeridas
- "¿Qué pasaría con las reglas si solo tuviéramos 10 transacciones?"
- "¿Por qué creen que `whole milk` aparece tanto?"

---

## Slide 4 — Frecuencia de ítems {#slide-4}

### Objetivo del slide
Visualizar la distribución asimétrica de frecuencias y plantear el problema de los ítems dominantes.

### Concepto clave
**Sesgo por frecuencia base**: cuando un ítem aparece en el 60-70% de las transacciones, casi cualquier regla que lo incluya tendrá alta confianza —no por ser interesante, sino por ser ubicuo.

### Lo que el profesor debe decir
> *"Observen este gráfico. `whole milk` aparece en casi 30 de 50 transacciones. Eso es un 60%. Ahora imaginen la regla `{pan} → {whole milk}`. Si yo compro pan, ¿la regla dice algo útil? O simplemente... `whole milk` aparece siempre."*

Este es el momento de introducir el concepto de **frecuencia base** (base rate). En probabilidad:

$$P(\text{whole milk}) \approx 0.60$$

Una regla que diga `P(whole milk | X) = 0.65` no es impresionante si `P(whole milk) = 0.60`.

### Interpretación del gráfico
- La escala de color (azul oscuro = más frecuente) permite leer la jerarquía de ítems rápidamente.
- El gradiente revela la distribución de Pareto: pocos ítems concentran la mayoría de transacciones.

### Conexión con conceptos previos
- En Clase 1 los estudiantes calcularon soporte. Aquí ven visualmente por qué el soporte de algunos ítems es tan alto.
- Conecta directamente con la fórmula del lift de la siguiente sección.

### Preguntas sugeridas
- "¿Qué significa para el lift que `whole milk` tenga soporte ≈ 0.60?"
- "Si yo fuera gerente de supermercado, ¿este gráfico me sorprendería?"

---

## Slide 5 — Matriz de co-ocurrencia {#slide-5}

### Objetivo del slide
Mostrar co-ocurrencias brutas antes de calcular reglas formales. Esto permite que los estudiantes "vean" los patrones antes de que el algoritmo los formalice.

### Concepto clave
**Co-ocurrencia ≠ Asociación útil**. Dos ítems pueden co-ocurrir muchas veces simplemente porque ambos son frecuentes, no porque exista una relación real entre ellos.

### Lo que el profesor debe decir
> *"Esta matriz muestra cuántas veces aparecen juntos los 10 ítems más frecuentes. El color naranja más intenso = más co-ocurrencias. Pero antes de emocionarnos con las celdas naranjas: ¿qué otros ítems esperaríamos que co-ocurrieran con `whole milk`? Básicamente todos, ¿no? Porque `whole milk` está en casi todo."*

Señalar que la diagonal está en cero (un ítem no co-ocurre consigo mismo en este contexto). La matriz es simétrica, lo que visualmente se puede observar.

### Interpretación del gráfico
- Celdas muy brillantes relacionadas con `whole milk` y `bread` deben tomarse con cautela: son frecuentes por su alta frecuencia base.
- Celdas brillantes entre ítems que NO son los más frecuentes son más interesantes (ej. `granola-yogurt-fruit`).

### Preguntas sugeridas
- "¿Qué par de ítems menos frecuentes co-ocurre más de lo esperado?"
- "Si elimináramos `whole milk` del análisis, ¿cambiaría mucho el gráfico?"

### Extensión posible
Para clases más avanzadas: la co-ocurrencia normalizada (dividida por la frecuencia esperada) es esencialmente el lift. Esto conecta la visualización con la métrica.

---

## Slide 6 — Generación de reglas {#slide-6}

### Objetivo del slide
Ejecutar la pipeline conocida pero ahora como punto de partida para el análisis, no como objetivo final.

### Concepto clave
El código es idéntico a Clase 1 en estructura. El énfasis ahora está en **cuántas reglas se generan** y si eso es un número razonable.

### Lo que el profesor debe decir
> *"Aquí usamos exactamente el mismo código de la semana pasada. Noten el número de reglas que obtenemos. ¿Ese número les parece razonable? ¿Podemos revisarlas todas? ¿Deberíamos?"*

### Nota técnica
Con `min_support=0.10` y 50 transacciones se obtienen tipicamente entre 20-50 reglas. Es posible que el número varíe. Si hay muy pocas reglas, bajar a `min_support=0.08`. Si hay demasiadas, subirlo a `0.15`.

El `metric="lift"` con `min_threshold=0.5` es deliberadamente laxo para que entren reglas cuestionables que se analizarán en slides posteriores.

### Errores comunes
- Los estudiantes a veces interpretan el número de reglas como señal de riqueza del dataset. Más reglas ≠ más información.
- Es común que se fijen solo en la columna `confidence` al ver la tabla. Incentivar a leer `lift` primero.

---

## Slide 7 — El problema central {#slide-7}

### Objetivo del slide
Este es el slide conceptualmente más importante. Formalizar matemáticamente por qué la confianza sola es insuficiente.

### Concepto clave
$$\text{lift} = \frac{P(A \cup B)}{P(A) \cdot P(B)} = \frac{\text{confidence}(A \to B)}{P(B)}$$

La confianza ignora la frecuencia base del consecuente. El lift la normaliza.

### Lo que el profesor debe decir
> *"Vamos a ser muy concretos. Supongamos que `whole milk` aparece en el 60% de las transacciones. Ahora tengo la regla `{pan} → {whole milk}` con confidence = 0.65. ¿Es eso bueno? El lift sería 0.65 / 0.60 = 1.08. Prácticamente 1. La regla no dice nada nuevo: compres pan o no lo compres, la probabilidad de que también compres leche es aproximadamente la misma."*

Conectar con el concepto estadístico de **independencia**:
$$P(B | A) = P(B) \iff A \text{ y } B \text{ son independientes}$$

Cuando `lift = 1`, la ocurrencia de A no afecta la probabilidad de B. La regla es trivial [@tan2006introduction].

### Por qué aparecen reglas con lift ≤ 1
Porque el umbral en `association_rules()` se puso sobre lift (`min_threshold=0.5`), lo que permite reglas con lift < 1. Esto es intencional: sirve para mostrar ejemplos malos.

### Errores comunes
- Confundir lift < 1 (relación negativa: A inhibe B) con lift = 1 (independencia).
- Creer que confidence = 1.0 siempre es bueno.

### Preguntas sugeridas
- "¿Qué valor de lift esperarían para una regla realmente útil?"
- "¿Puede una regla tener confidence = 1.0 y lift < 1? ¿Cómo?"

### Respuesta a la pregunta difícil
Sí: si `P(B) > confidence`. Esto puede ocurrir cuando el umbral de confidence es bajo y el ítem B es muy frecuente. En la práctica, con confidence = 1.0 el lift siempre sería ≥ 1 (porque `confidence / P(B) = 1/P(B) ≥ 1`). El caso interesante es confidence alta pero no = 1, p.ej. confidence = 0.7 con P(B) = 0.8 → lift = 0.875 < 1.

---

## Slide 8 — Tabla de métricas {#slide-8}

### Objetivo del slide
Ofrecer una referencia visual rápida de las tres métricas. Es un slide de síntesis.

### Concepto clave
Cada métrica responde una pregunta diferente. Ninguna por sí sola es suficiente.

### Lo que el profesor debe decir
> *"Esta tabla es para que la tengan como referencia. No memorizarla, sino entender que cada métrica ve una faceta distinta del patrón. Support dice qué tan común es. Confidence dice qué tan predecible es B dado A. Lift dice si A y B tienen una relación real o casual."*

### Nota sobre la limitación del lift
Con datasets pequeños (como nuestras 50 transacciones), el lift puede ser muy inestable. Una co-ocurrencia que aparece 5 veces puede parecer muy significativa por azar. Esto conecta con el concepto de **significancia estadística** que se puede desarrollar en cursos avanzados (p-valor, prueba chi-cuadrado sobre tablas de contingencia).

### Extensión para grupos avanzados
Existen métricas adicionales: conviction, leverage, phi-coefficient. Mencionarlas como lectura opcional de Han et al. (2011), cap. 6.

---

## Slide 9 — Scatter interactivo {#slide-9}

### Objetivo del slide
Este es el gráfico central de la clase. Permite explorar el espacio completo de reglas de forma visual e interactiva.

### Concepto clave
**El espacio de reglas es tridimensional** (support, confidence, lift). El scatter comprime dos dimensiones en ejes y la tercera en tamaño de punto.

### Lo que el profesor debe decir
> *"Aquí está todo. Cada punto es una regla. El eje X es la confianza, el eje Y es el lift, y el tamaño del círculo representa el soporte. Pasen el mouse sobre cualquier punto para ver la regla completa. La línea roja es el umbral de independencia: lift = 1. Todo lo que está por debajo de esa línea es una regla que técnicamente representa una relación negativa o trivial."*

Dar tiempo a los estudiantes para explorar el gráfico (2-3 minutos). Pedir que identifiquen:
1. El punto más grande y más verde (mejor candidato).
2. Un punto con confidence alta pero lift bajo (trampa).
3. Un punto con lift muy alto pero tamaño pequeño (posible ruido).

### Interpretación del gráfico (detallada)

| Zona | Características | Interpretación |
|---|---|---|
| Arriba a la derecha, círculo grande | Alto lift, alta confidence, alto support | **Regla ideal**: frecuente, predecible y no trivial |
| Arriba a la izquierda, círculo pequeño | Alto lift, baja confidence, bajo support | **Curiosidad**: podría ser ruido estadístico |
| Abajo a la derecha | Lift ≤ 1, confidence alta | **Trampa**: parece buena pero es espuria |
| Abajo a la izquierda | Todo bajo | **Descartar** |

### Errores comunes en la interpretación
- Los estudiantes tienden a fijarse en los puntos más grandes sin ver el lift.
- Pueden creer que el punto más "a la derecha" (máxima confidence) es el mejor.

### Preguntas sugeridas durante la exploración
- "¿Pueden encontrar una regla con confidence > 0.8 y lift < 1.1?"
- "¿Cuál es la regla con el lift más alto? ¿Qué soporte tiene?"

---

## Slide 10 — Interpretando el scatter {#slide-10}

### Objetivo del slide
Codificar en texto lo que los estudiantes exploraron visualmente en el slide anterior.

### Concepto clave
Los cuadrantes del scatter son un marco de decisión, no una regla absoluta.

### Lo que el profesor debe decir
> *"Vamos a formalizar lo que exploraron. Hay tres zonas. La zona verde es donde queremos estar: alta confianza, alto lift, buen soporte. La zona roja es la trampa: confidence alta que engaña, pero lift cerca de 1. Y la zona amarilla es donde hay que tener cuidado: el lift es alto pero el soporte es tan bajo que puede ser ruido."*

### Respuesta a la pregunta del slide
*"¿Por qué una regla con confidence = 1.0 puede estar por debajo de la línea roja?"*

Técnicamente no puede, si calculamos bien. Confidence = 1.0 implica P(B|A) = 1.0, y lift = 1.0 / P(B). Como P(B) ≤ 1, el lift ≥ 1. La pregunta es provocadora para que los estudiantes razonen. La trampa real es confidence = 0.7-0.9 con P(B) alto.

Modificar en clase si lo considera apropiado: *"¿Por qué una regla con confidence = 0.85 puede estar por debajo de la línea roja?"* Esa sí puede ocurrir.

---

## Slide 11 — Caso 1: Regla débil {#slide-11}

### Objetivo del slide
Mostrar un ejemplo concreto de regla engañosa. La concreción es clave para el aprendizaje.

### Concepto clave
**Alto confidence + Bajo lift = Regla espuria**. El consecuente es tan frecuente que ocurre de todas formas.

### Lo que el profesor debe decir
> *"Aquí tenemos reglas con confidence > 0.5 ordenadas por lift de menor a mayor. El peor caso. Miren el lift: está cerca de 1, o incluso puede ser menor. Esto significa que A no está realmente asociado con B. B ocurre con o sin A."*

Tomar una regla específica del output (el resultado varía según el dataset) y calcular mentalmente con los estudiantes:
- Si `whole milk` tiene soporte 0.60 y la regla `X → whole milk` tiene confidence 0.65, el lift es 0.65/0.60 ≈ 1.08.
- Eso es prácticamente independencia.

### Conexión con el mundo real
Este fenómeno tiene nombre en epidemiología: **confusión por variable prevalente**. Si una enfermedad es muy común, cualquier factor de riesgo parecerá asociado a ella solo por la prevalencia base.

---

## Slide 12 — Caso 2: Regla fuerte {#slide-12}

### Objetivo del slide
Contrastar con el caso anterior. Mostrar cómo se ve una regla con evidencia real.

### Concepto clave
**Lift > 1.5 con soporte razonable = Relación no trivial**

### Lo que el profesor debe decir
> *"Ahora el otro extremo. Estas reglas tienen lift > 1.5. Eso significa que los ítems co-ocurren un 50% más de lo esperado por azar. Eso ya es una señal. Si además el soporte es decente —más del 10%— tenemos una regla en la que vale la pena confiar."*

Mostrar el cálculo explícito para la mejor regla del output:
- Lift = confidence / P(B)
- Si lift = 1.8 y confidence = 0.54, entonces P(B) ≈ 0.30
- La regla dice que B ocurre 80% más frecuente cuando A está presente que en promedio.

### Errores comunes
- Los estudiantes pueden creer que lift = 1.8 es "enorme". En realidad es modesto. En datasets reales, lifts de 5-10 son más convincentes. Con 50 transacciones, es lo mejor que podemos esperar.

### Preguntas sugeridas
- "¿Este lift de 1.8 justificaría un cambio en la distribución del supermercado?"
- "¿Cuántas transacciones necesitarían para confiar en esta regla?"

---

## Slide 13 — Selección de reglas útiles {#slide-13}

### Objetivo del slide
Dar al estudiante un criterio operacional de filtrado. Pasar de la filosofía a la práctica.

### Concepto clave
Los umbrales de filtrado son decisiones del analista, no verdades absolutas. Dependen del contexto de negocio.

### Lo que el profesor debe decir
> *"Aplicamos tres filtros simultáneos: lift > 1.2, confidence > 0.4, support ≥ 0.10. El resultado es un subconjunto mucho más pequeño de reglas. Esas son nuestras candidatas. ¿Son perfectas? No. ¿Son más confiables que la lista completa? Definitivamente sí."*

Reflexionar sobre los umbrales:
- `lift > 1.2`: conservador. En producción real se usaría ≥ 1.5 o más.
- `confidence > 0.4`: muy bajo. En retail se usan umbrales de 0.6-0.8.
- `support ≥ 0.10`: en un dataset de 50, esto es solo 5 transacciones. Con 9,000 el umbral natural sería 0.01-0.05.

### Nota pedagógica importante
El objetivo no es que los estudiantes memoricen estos umbrales, sino que entiendan que **los umbrales son decisiones** y que cambiarlos tiene consecuencias. Mostrar en vivo qué pasa al cambiar `lift > 1.5`: ¿cuántas reglas quedan?

### Preguntas sugeridas
- "¿Qué criterio cambiarían primero si fueran un gerente de supermercado?"
- "¿Qué significa el número de reglas que quedan después del filtrado?"

---

## Slide 14 — Cuándo NO usar reglas de asociación {#slide-14}

### Objetivo del slide
Pensamiento crítico sobre las limitaciones del método. Este slide distingue a un analista competente de uno que aplica herramientas mecánicamente.

### Concepto clave
**Toda herramienta tiene alcance y límites.** Conocer los límites es tan importante como conocer el método.

### Lo que el profesor debe decir
> *"Hasta ahora aprendimos a usar Apriori. Ahora vamos a aprender cuándo NO usarlo. Hay cuatro situaciones clave."*

**1. Dataset pequeño:**
Con 50-100 transacciones, el soporte de los itemsets es estadísticamente inestable. Un ítem que aparece 5 veces de 50 tiene soporte = 0.10, pero con una transacción más o menos ese valor cambia un 2%. En datasets grandes, la estabilidad mejora.

**2. Ítems muy frecuentes dominan:**
Ya lo vimos con `whole milk`. Una solución práctica: eliminar ítems cuyo soporte supere el 50% antes de correr Apriori, o analizar subgrupos del dataset.

**3. Buscas causalidad [@pearl2009causality]:**
Las reglas de asociación son descriptivas, no causales. `{pan} → {mantequilla}` no significa que comprar pan *cause* comprar mantequilla. Puede haber una variable confusora (ej. el perfil socioeconómico del comprador). Para causalidad se necesitan diseños experimentales o modelos causales.

**4. Necesitas secuencias temporales:**
Apriori ignora el orden. Si quieres saber que los clientes compran primero leche y luego yogurt (en visitas distintas), necesitas algoritmos de patrones secuenciales como PrefixSpan o GSP.

### Analogías útiles
- Correlación vs. causalidad: clásico ejemplo del helado y ahogados en verano.
- Reglas de asociación vs. modelos predictivos: las reglas describen lo que pasó, no predicen lo que pasará.

### Preguntas sugeridas
- "¿Podrían usar reglas de asociación para predecir el próximo ítem que comprará un cliente?"
- "¿Cómo distinguirían una regla causal de una correlacional en este contexto?"

---

## Slide 15 — Resumen: El Filtro Mental {#slide-15}

### Objetivo del slide
Cerrar la clase con un marco de decisión portable y memorable.

### Concepto clave
El árbol de decisión del slide codifica el orden correcto de evaluación de una regla.

### Lo que el profesor debe decir
> *"Esto es el resumen de todo lo que vimos hoy. Cuando vean una regla, la primera pregunta no es ¿tiene alta confianza? La primera pregunta es ¿tiene lift > 1? Si no lo tiene, descártenla sin mirar más. Si sí, pregunten: ¿tiene soporte razonable? Si no, traten con precaución. Solo si ambas respuestas son afirmativas, entonces evalúen la confianza en el contexto de su problema."*

La cita del final no es de ningún paper específico —es una síntesis conceptual. Si los estudiantes la cuestionan, es una buena señal de pensamiento crítico. Responder: *"Esta no es una cita directa, es una síntesis del enfoque de Tan et al. (2006) y Han et al. (2011) sobre la selección de patrones interesantes."*

### Reflexión final para la clase
> *"La diferencia entre un analista que usa minería de datos bien y uno que la usa mal no está en el código —el código es el mismo. Está en las preguntas que hace antes de creerle a los resultados."*

---

## Notas Generales de Gestión de Tiempo

| Sección | Slides | Tiempo estimado |
|---|---|---|
| Recordatorio + setup + dataset | 1-3 | 8 min |
| Visualizaciones | 4-5 | 10 min |
| Generación de reglas + problema central | 6-7 | 10 min |
| Tabla de métricas + scatter | 8-9 | 12 min |
| Exploración interactiva | 10 | 5 min (libre exploración) |
| Casos concretos | 11-12 | 8 min |
| Filtrado + cuándo NO usar | 13-14 | 10 min |
| Resumen + cierre | 15 | 5 min |
| **Total** | | **~68 min** |

Si el tiempo es ajustado, el Slide 5 (matriz de co-ocurrencia) puede omitirse o mostrarse rápidamente sin discusión profunda. El Slide 9 (scatter interactivo) y el Slide 7 (problema central) son **no negociables** para el aprendizaje.

---

## Referencias Bibliográficas

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*, 1215, 487–499.
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
- Tan, P.-N., Steinbach, M., & Kumar, V. (2006). *Introduction to Data Mining*. Addison-Wesley.
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.


# Guía Docente — Clase 3: Más Allá del Lift
## *Métricas Avanzadas, Sensibilidad a Parámetros y Grafos de Reglas*

> **Audiencia:** Profesor de Machine Learning  
> **Nivel:** Licenciatura (física, datos, ingeniería)  
> **Duración total:** ~60 minutos  
> **Prerequisito:** Clases 1 y 2 completadas

---

## Tabla de Contenidos

| # | Slide | Bloque | Tiempo |
|---|---|---|---|
| 1 | Agenda de hoy | Intro | 2 min |
| 2 | Setup y datos | Intro | 3 min |
| 3–6 | Bloque A: Límites del lift | A | 15 min |
| 7–11 | Bloque B: Métricas complementarias | B | 15 min |
| 12–16 | Bloque C: Sensibilidad a parámetros | C | 15 min |
| 17–24 | Bloque D: Grafo y caso de estudio | D | 15 min |
| 25 | Conclusión | Cierre | 5 min |
| 26 | Lecturas | Cierre | 2 min |

---

## Slide 1 — Agenda de Hoy

### Objetivo del slide
Ubicar esta sesión en el arco de tres clases y mostrar la estructura explícita del tiempo.

### Lo que el profesor debe decir
> *"Llevamos dos clases. En la primera aprendimos a generar reglas. En la segunda, aprendimos a cuestionar las que genera el algoritmo. Hoy vamos un paso más lejos: vamos a aprender a cuantificar con más rigor qué tan buena es una regla, y a entender qué tanto cambian los resultados según los parámetros que elegimos."*

La tabla de bloques no es decorativa — es un contrato pedagógico. Si el tiempo se ajusta, se puede señalar qué bloques son prioritarios (B y D) vs. opcionales (A y C pueden comprimirse).

---

## Slide 2 — Setup y Datos

### Objetivo del slide
Reutilizar exactamente el mismo dataset de las clases anteriores. Esto es intencional: la continuidad permite al estudiante comparar resultados y ver que los cambios en los resultados no vienen del dataset, sino de la metodología.

### Lo que el profesor debe decir
> *"El código de setup es idéntico al de la sesión pasada. El dataset es el mismo. Si obtienen resultados distintos a los de la clase anterior, la causa es metodológica, no de datos. Eso es importante para lo que viene."*

### Nota técnica
`sklearn` se importa en el bloque del radar de métricas (Slide 10). Si no está instalada: `pip install scikit-learn`. Verificar antes de clase.

---

## Slide 3 — El Lift es Asimétrico

### Objetivo del slide
Revelar una limitación no obvia del lift que los estudiantes casi nunca consideran: que el lift es simétrico pero la confianza no lo es.

### Concepto clave
$$\text{lift}(A \to B) = \text{lift}(B \to A)$$

Pero:
$$\text{conf}(A \to B) \neq \text{conf}(B \to A) \quad \text{en general}$$

### Lo que el profesor debe decir
> *"Aquí hay algo que parece trivial pero tiene implicaciones prácticas importantes. El lift es el mismo para la regla `yogurt → whole milk` y para `whole milk → yogurt`. Ambas relaciones tienen exactamente la misma 'fuerza' según el lift. Pero la confianza puede ser muy diferente. ¿Eso les parece correcto?"*

Dar tiempo para que los estudiantes razonen. La respuesta es que el lift mide si hay correlación entre los dos ítems, sin distinción de dirección. Pero si queremos tomar una decisión de negocio ("si el cliente compra X, ¿le ofrezco Y?"), la dirección sí importa.

### Errores comunes
- Los estudiantes tienden a ver lift como una métrica "simétrica de correlación" y no pensar en las implicaciones asimétricas para la toma de decisiones.
- Confundir "el lift es el mismo" con "la regla es igual en ambas direcciones".

### Preguntas sugeridas
- "¿Cuándo importaría la dirección en un sistema de recomendación real?"
- "¿Cuál dirección elegiríamos si queremos colocar productos cerca de `yogurt`?"

---

## Slide 4 — Asimetría de Confianza

### Objetivo del slide
Hacer explícita la diferencia de confianza entre las dos direcciones de una regla.

### Lo que el profesor debe decir
> *"Aquí vemos las dos reglas. La confianza de `yogurt → whole milk` es distinta a la de `whole milk → yogurt`. ¿Cuál es más alta? Y más importante: ¿por qué? La respuesta está en las frecuencias base. La regla con el consecuente más frecuente tendrá automáticamente más confianza, no porque la relación sea más fuerte, sino porque ese consecuente ocurre más."*

Este slide refuerza el mensaje de la Clase 2 desde un ángulo nuevo: la asimetría de la confianza como síntoma del problema de la frecuencia base.

---

## Slide 5 — El Problema del Lift Máximo

### Objetivo del slide
Introducir el concepto de lift normalizado. Este es contenido nuevo que va más allá de los libros de texto estándar.

### Concepto clave
El lift tiene un techo teórico: no puede ser arbitrariamente alto para cualquier par de ítems. El máximo depende de las frecuencias individuales. Un lift de 2.0 para dos ítems con soporte 0.10 es muy diferente a un lift de 2.0 para dos ítems con soporte 0.50.

### Lo que el profesor debe decir
> *"Aquí hay una idea que los libros de texto rara vez mencionan. El lift no tiene una escala universal. Si tengo dos ítems muy raros —soporte 0.01 cada uno— el lift puede ser 100 o más. Pero si tengo dos ítems frecuentes —soporte 0.5 cada uno— el lift máximo posible es solo 2. Entonces comparar lifts entre reglas con distintos soportes es como comparar porcentajes sin mencionar el denominador."*

La fórmula $\text{lift}_{\max} = 1/\max(P(A), P(B))$ se deriva directamente de la definición. En el mejor caso, los dos ítems co-ocurren siempre que ocurre el más frecuente, lo que da $P(A \cup B) = \max(P(A), P(B))$ y por tanto $\text{lift}_{\max} = \max(P(A), P(B)) / (P(A) \cdot P(B)) = 1/\min(P(A), P(B))$. La fórmula en el slide usa $\min$ en el denominador, equivalente a $1/\min(P(A),P(B))$. Verificar consistencia con los estudiantes si surge la pregunta.

### Preguntas sugeridas
- "¿Dos ítems con soporte 0.01 que tienen lift = 5 son más interesantes que dos ítems con soporte 0.5 y lift = 1.8?"
- "¿Cómo cambiaría su criterio de selección si normalizan el lift?"

---

## Slide 6 — Lift Relativo: Scatter

### Objetivo del slide
Visualizar la diferencia entre lift absoluto y lift relativo para las reglas del dataset.

### Interpretación del gráfico
- Eje X: lift observado (el que calculamos normalmente).
- Eje Y: qué fracción del máximo teórico representa ese lift.
- La línea punteada naranja marca el 50% del máximo teórico.
- Puntos sobre la línea naranja: reglas que alcanzan más de la mitad de su potencial máximo.
- Puntos debajo: reglas que, aunque tienen lift positivo, están lejos de su techo.

Un punto en la esquina inferior derecha (lift alto, lift_relativo bajo) es una regla entre ítems frecuentes: el lift parece alto pero el par está lejos de co-ocurrir siempre. Un punto en la esquina superior izquierda (lift bajo, lift_relativo alto) es una regla entre ítems raros que co-ocurren muy consistentemente.

---

## Slide 7 — Más Allá del Lift: Tres Métricas Clave

### Objetivo del slide
Formalizar las definiciones de leverage, conviction y Zhang's metric. Este es el slide más matemático del Bloque B.

### Concepto clave

**Leverage:**
- Diferencia absoluta entre $P(A \cup B)$ y $P(A) \cdot P(B)$.
- Tiene interpretación directa: "¿cuántas transacciones más tienen ambos ítems de lo que esperaríamos por azar?"
- Con 50 transacciones, leverage = 0.05 significa 2.5 transacciones extra sobre el baseline de independencia.
- Desventaja: sensible al soporte total (ítems raros siempre tendrán leverage pequeño aunque la relación sea fuerte).

**Conviction:**
- Mide qué tan "determinista" es la regla en la dirección A → B.
- Si confidence → 1, conviction → ∞.
- Conviction = 1 ↔ independencia (igual que lift = 1).
- Conviction < 1: A y B tienen relación negativa.
- Ventaja sobre lift: es asimétrico (como la confianza), lo que lo hace más adecuado para reglas dirigidas.

**Zhang's metric:**
- Rango [-1, 1], simétrico.
- Robusto a ítems muy frecuentes (normaliza mejor que leverage).
- Z = 0: independencia. Z = 1: co-ocurrencia perfecta. Z = -1: exclusión perfecta.
- Recomendado por Tan et al. (2006) como uno de los mejores indicadores de interés general.

### Lo que el profesor debe decir
> *"No se preocupen por memorizar estas fórmulas. Lo que importa es entender qué pregunta responde cada una. Leverage dice: ¿cuántas transacciones extra tenemos? Conviction dice: ¿qué tan determinista es la regla? Zhang dice: ¿qué fracción del máximo posible representa esta co-ocurrencia? Cada una mira desde un ángulo distinto."*

---

## Slide 8 — Calculando las Métricas Complementarias

### Objetivo del slide
Mostrar que `mlxtend` calcula leverage y conviction automáticamente, y que Zhang requiere cálculo manual.

### Nota técnica
Zhang's metric no está implementada en `mlxtend`. La implementación manual en el slide es correcta y puede usarse directamente. Si algún estudiante pregunta sobre el denominador: `max{P(AB)(1-P(A)), P(A)(P(B)-P(AB))}` evita divisiones por cero y normaliza correctamente en ambos extremos del rango.

### Preguntas sugeridas
- "¿Ordenar por zhang vs. ordenar por lift da el mismo ranking? ¿Por qué?"
- "¿Qué regla tiene conviction más alta? ¿Qué significa eso intuitivamente?"

---

## Slide 9 — ¿Coinciden las Métricas? Correlaciones

### Objetivo del slide
Mostrar empíricamente que las métricas no son redundantes: algunas están correlacionadas, otras no.

### Interpretación del heatmap
- Correlación alta (rojo intenso) entre lift y zhang: ambas miden "desviación de la independencia" aunque desde fórmulas distintas.
- Correlación moderada entre leverage y support: ítems más frecuentes tienden a tener mayor leverage absoluto.
- Conviction puede correlacionar poco con las demás porque es muy sensible a confidence cercana a 1.
- Donde hay correlaciones bajas o negativas: las métricas ven facetas diferentes. Ahí es donde el juicio analítico es más necesario.

### Lo que el profesor debe decir
> *"Si todas las métricas estuvieran perfectamente correlacionadas, bastaría con una sola. Pero no lo están. Eso significa que cada una detecta algo diferente. Y cuando discrepan, tenemos que decidir a cuál le creemos más según el contexto."*

---

## Slide 10 — Cuándo Usar Cada Métrica: Radar

### Objetivo del slide
Visualizar el "perfil" de las mejores reglas en un gráfico de radar normalizado.

### Interpretación del radar
- Un pentágono perfectamente regular (todas las métricas al máximo) sería la regla ideal — prácticamente imposible.
- Perfiles asiméticos revelan las fortalezas y debilidades de cada regla.
- Una regla con radio grande en `lift` pero pequeño en `support` es una candidata débil a pesar del lift.
- Una regla con perfil balanceado (hexágono razonablemente regular) es más robusta que una con un solo "pico".

### Nota sobre MinMaxScaler
La normalización con MinMaxScaler es solo para visualización. No implica que todas las métricas sean igual de importantes: el radar es una herramienta exploratoria, no un score final.

---

## Slide 11 — Tabla: Cuándo Usar Cada Métrica

### Objetivo del slide
Dar al estudiante una guía de referencia práctica.

### Lo que el profesor debe decir
> *"Esta tabla resume cuándo conviene mirar cada métrica. No hay una respuesta universal. En retail, conviction es útil porque queremos reglas casi deterministas para colocar productos. En medicina, leverage puede ser más relevante porque queremos saber cuántos pacientes extra estamos identificando. En detección de fraude, Zhang puede ser mejor porque las transacciones fraudulentas son raras y el lift puede inflarse artificialmente."*

---

## Slide 12 — El Problema del Umbral Arbitrario

### Objetivo del slide
Este slide planta la semilla del Bloque C: los parámetros de Apriori son decisiones del analista, no constantes del universo.

### Concepto clave
La crítica fundamental es epistemológica: cuando reportamos resultados de minería de datos sin reportar los parámetros usados, el resultado es irreproducible e ininterpretable.

### Lo que el profesor debe decir
> *"Aquí hay algo que pocos cursos mencionan explícitamente. Cuando corremos Apriori con `min_support=0.10`, estamos haciendo una elección. Si hubiéramos elegido 0.08, obtendríamos más reglas. Si hubiéramos elegido 0.15, obtendríamos menos. ¿Cuál es la correcta? No hay una respuesta correcta a priori. Pero sí hay una forma de entender las consecuencias de cada elección: análisis de sensibilidad."*

Conectar con la práctica científica: en un paper, siempre se reportan los hiperparámetros. En un proyecto de datos, lo mismo.

---

## Slide 13 — Experimento: Sensibilidad al Soporte

### Objetivo del slide
Ejecutar el análisis de sensibilidad y mostrar los resultados tabulados.

### Interpretación de los resultados esperados
- Al aumentar `min_support`, el número de itemsets frecuentes cae (menos ítems y combinaciones pasan el umbral).
- El número de reglas cae más rápido que el de itemsets porque las reglas requieren que el itemset sea frecuente Y que la métrica supere el umbral.
- El lift mediano tiende a subir: al filtrar itemsets raros, los que sobreviven tienden a ser más robustos.

### Preguntas sugeridas
- "¿Por qué el lift máximo puede subir al aumentar el soporte mínimo?"
- "Si usáramos `min_support=0.20`, ¿qué tipos de relaciones estaríamos perdiendo?"

---

## Slide 14 — Visualización de Sensibilidad

### Objetivo del slide
El gráfico de doble eje permite ver simultáneamente dos efectos opuestos: más soporte → menos reglas pero mejor calidad (lift mayor).

### Lo que el profesor debe decir
> *"Este gráfico muestra el trade-off fundamental de Apriori: puedo tener muchas reglas baratas o pocas reglas buenas. No puedo tener las dos cosas al mismo tiempo. En la práctica, siempre hay que decidir qué es más valioso en el contexto del problema."*

El cruce de las dos curvas (si ocurre) es un punto de interés: donde el número de reglas cae rápidamente pero el lift sigue siendo razonable, puede ser un buen umbral pragmático.

---

## Slide 15 — Sensibilidad al Umbral de Lift

### Objetivo del slide
Análogo al slide anterior pero variando el umbral de lift, con soporte fijo.

### Interpretación
La curva de "reglas que sobreviven" debería ser monótonamente decreciente. Si hay una caída abrupta entre lift=1.2 y lift=1.3, eso sugiere que la mayoría de las reglas tienen lift en ese rango — zona de baja robustez.

Si la curva cae de forma gradual, las reglas están distribuidas en todo el rango de lift — más diversidad.

---

## Slide 16 — El Mapa Soporte-Confianza

### Objetivo del slide
Este es el gráfico más poderoso del Bloque C. Muestra de una vez cómo el número de reglas depende simultáneamente de dos parámetros.

### Interpretación del heatmap
- Esquina inferior izquierda (bajo soporte, baja confianza): máximo de reglas, mínima calidad.
- Esquina superior derecha (alto soporte, alta confianza): mínimo de reglas (posiblemente cero), máxima calidad.
- La zona de trabajo razonable es la diagonal central: soporte moderado + confianza moderada.

### Lo que el profesor debe decir
> *"Este mapa les muestra todo el espacio de decisión de una vez. Cada celda responde: si pongo estos umbrales, ¿cuántas reglas obtengo? Ahora pregúntense: ¿qué celdas habrían elegido antes de ver este mapa? ¿Y ahora? La mayoría de los analistas elige umbrales por defecto sin saber qué hay en el resto del espacio."*

### Conexión con conceptos de ML
Análogo al análisis de hiperparámetros en modelos supervisados: la validación cruzada también produce mapas similares cuando se varían dos hiperparámetros.

---

## Slide 17 — Visualización como Grafo: Introducción

### Objetivo del slide
Motivar la visualización en grafo como herramienta complementaria al scatter.

### Lo que el profesor debe decir
> *"Hasta ahora hemos visto las reglas en tablas y en scatter plots. Pero hay una tercera forma de verlas: como una red. Los ítems son nodos, y las reglas son aristas. Eso nos permite ver la estructura global de las relaciones en el dataset."*

---

## Slide 18 — Grafo Interactivo

### Objetivo del slide
El grafo es la visualización más informativa para comunicar resultados de reglas de asociación a audiencias no técnicas.

### Interpretación del grafo
- **Tamaño del nodo:** importancia como suma de lifts de todas las reglas en las que participa el ítem.
- **Color del nodo:** mismo criterio de importancia, codificado por color (escala Plasma: oscuro = bajo, amarillo/blanco = alto).
- **Aristas:** cada arista es una regla que supera los umbrales. No tienen dirección en esta visualización (el grafo es no-dirigido).

**Lo que buscar en el grafo:**
1. Clústeres de ítems densamente conectados: indican "familias" de productos que co-ocurren regularmente.
2. Ítems con muchas aristas (hub): pueden ser ítems genéricos (cuidado: posible distorsión por frecuencia).
3. Ítems con pocas aristas pero el nodo es grande: ítems que participan en pocas relaciones pero todas son fuertes.
4. Ítems aislados (sin aristas): no tienen relaciones que superen los umbrales elegidos.

### Limitación del grafo del slide
El grafo usa layout circular, que no refleja la "distancia" entre ítems. Para clases avanzadas: layouts basados en fuerza (force-directed) como Fruchterman-Reingold dan información topológica más rica. La librería `networkx` + `pyvis` puede producirlos, aunque requiere más código.

### Preguntas sugeridas
- "¿Qué ítems formarían un clúster si usáramos un layout por fuerza?"
- "¿Por qué `whole milk` puede ser el nodo más grande aunque sus relaciones sean débiles?"

---

## Slide 19 — Leyendo el Grafo

### Objetivo del slide
Dar vocabulario para interpretar el grafo: hubs, puentes, periféricos.

### Lo que el profesor debe decir
> *"Hay una distinción importante entre ítems que son centrales porque son frecuentes y ítems que son centrales porque tienen relaciones genuinamente fuertes. `whole milk` puede ser el nodo más grande del grafo, pero eso puede deberse a su alta frecuencia, no a que tenga relaciones fuertes con muchos ítems. Contrastar con ítems como `granola` o `fruit` que pueden tener pocas conexiones pero todas muy específicas."*

### Analogía con redes sociales
En una red de colaboración académica, un investigador que escribe en muchas áreas puede tener muchas conexiones (hub) sin ser necesariamente el más influyente en ninguna de ellas. Un investigador especializado con pocas pero fuertes colaboraciones puede ser más influyente en su nicho.

---

## Slide 20 — Caso de Estudio: El "Whole Milk Problem"

### Objetivo del slide
Aplicar todo lo aprendido a un caso concreto que ilustra el problema más común en reglas de asociación.

### Lo que el profesor debe decir
> *"Vamos al caso más importante del dataset: `whole milk`. Aparece en el 60% de las transacciones. Eso significa que cualquier regla `X → whole milk` tiene una ventaja artificial enorme en confianza. La pregunta es: ¿podemos encontrar alguna regla `X → whole milk` que sea genuinamente interesante?"*

Mostrar la tabla de reglas con consecuente `whole milk`. Si ninguna supera lift > 1.2, eso es el resultado correcto y pedagógicamente valioso: *"a veces el análisis riguroso confirma que no hay nada interesante"*.

---

## Slide 21 — El Umbral Mínimo de Confianza Útil

### Objetivo del slide
Derivar matemáticamente el umbral de confianza que hace que una regla sea no-trivial, dado el soporte base del consecuente.

### Derivación para el pizarrón (si hay tiempo)
$$\text{lift} > 1 \iff \frac{\text{conf}(A \to B)}{P(B)} > 1 \iff \text{conf}(A \to B) > P(B)$$

Para lift > k:
$$\text{conf}(A \to B) > k \cdot P(B)$$

Con $P(\text{whole milk}) \approx 0.60$ y $k = 1.5$:
$$\text{conf} > 0.90$$

Esto significa que para que una regla con consecuente `whole milk` sea "sustancialmente no-trivial" (lift > 1.5), necesitamos que prácticamente 9 de cada 10 personas que compran el antecedente también compren leche. Eso es casi imposible de observar en un supermercado general.

### Conclusión pedagógica
> *"Este cálculo les muestra por qué es prácticamente imposible encontrar reglas interesantes con `whole milk` como consecuente. No es un defecto del dataset. Es una consecuencia directa de la frecuencia base. La solución no es cambiar el umbral — es reconocer que este ítem no es útil como consecuente."*

---

## Slide 22 — Generalización: Tabla de Umbrales por Ítem

### Objetivo del slide
Extender el razonamiento del slide anterior a todos los ítems del dataset.

### Lo que el profesor debe decir
> *"Ahora tienen una tabla que les dice: para cada ítem, ¿qué confianza necesita una regla para ser no-trivial? Guarden esta tabla. Antes de analizar cualquier regla, consulten la fila del consecuente. Si la confianza de la regla no supera ese umbral, la regla no tiene poder predictivo real."*

Este es el aporte práctico más directo del Bloque D.

---

## Slide 23 — Diagnóstico Completo de una Regla

### Objetivo del slide
Mostrar cómo se vería un análisis riguroso y completo de una sola regla.

### Lo que el profesor debe decir
> *"Esto es lo que debería verse en un análisis profesional: no solo tres números en una tabla, sino un diagnóstico completo que incluye prevalencia, cada métrica con su interpretación, y un veredicto fundamentado. ¿Tarda más? Sí. ¿Vale la pena? Absolutamente."*

### Conexión con práctica profesional
En proyectos reales de data science, este tipo de "fichas" de reglas son útiles para documentar los hallazgos y comunicarlos a stakeholders no técnicos.

---

## Slide 24 — El Pipeline Completo de Evaluación

### Objetivo del slide
Codificar el proceso de evaluación en una función reutilizable. Este es el entregable práctico de toda la clase.

### Lo que el profesor debe decir
> *"Esta función `evaluar_regla()` implementa todo lo que discutimos en tres sesiones. Pueden llevársela y usarla en sus propios proyectos. El sistema de puntos es arbitrario —pueden ajustarlo según su contexto— pero la lógica subyacente es la que hemos construido juntos."*

### Nota sobre los pesos del sistema de puntos
El sistema de puntos (lift: hasta 3 puntos, soporte: hasta 2, zhang: hasta 2, conviction: 1) refleja las prioridades relativas aprendidas en las tres clases. El lift sigue siendo la métrica más importante pero no la única. Modificar los pesos es un ejercicio útil para que los estudiantes reflexionen sobre sus prioridades.

---

## Slide 25 — Conclusión: Lo que Cambia con Este Marco

### Objetivo del slide
Cerrar el arco de las tres clases con una comparación explícita entre el enfoque inicial y el enfoque desarrollado.

### Lo que el profesor debe decir
> *"En tres sesiones pasamos de 'genero reglas y reporto las de mayor confianza' a 'genero reglas, las evalúo con múltiples métricas, verifico que no sean artefactos de los parámetros, y solo reporto las que son robustas según criterios claros'. La diferencia no es de herramientas —el código es casi el mismo— sino de mentalidad."*

La tabla comparativa es para que los estudiantes la copien/guarden. Resume el arco pedagógico completo.

---

## Slide 26 — Lecturas Recomendadas

### Objetivo del slide
Proporcionar un camino de lectura para quien quiera profundizar.

### Priorización para el estudiante promedio
1. **Han, Kamber & Pei, cap. 6** — primero. Es el más accesible y completo para nivel licenciatura.
2. **Tan et al. (2006), cap. 6** — segundo. Tiene la comparación de 21 métricas en una sola tabla — invaluable como referencia.
3. **Agrawal & Srikant (1994)** — tercero. Sorprendentemente legible para ser un paper de conferencia. Los autores mismos señalan limitaciones.
4. **Pearl (2009)** — opcional, para quienes quieran entender la frontera causalidad/correlación.

---

## Notas Generales de Gestión de Tiempo

| Bloque | Slides | Tiempo nominal | Colchón si hay preguntas |
|---|---|---|---|
| Intro | 1–2 | 5 min | No comprimir |
| A: Límites del lift | 3–6 | 15 min | Omitir Slide 6 (lift relativo scatter) |
| B: Métricas | 7–11 | 15 min | Comprimir tabla de métricas (Slide 11) |
| C: Sensibilidad | 12–16 | 15 min | Omitir Slide 16 (heatmap) si hay presión |
| D: Grafo y caso | 17–24 | 15 min | No comprimir (contiene los slides más ricos) |
| Cierre | 25–26 | 7 min | No comprimir |

**Si solo hay 45 minutos:** Omitir completamente el Bloque C (Slides 12–16) y mencionar brevemente el concepto de sensibilidad de forma oral.

---

## Errores Pedagógicos Frecuentes a Evitar

1. **Pasar demasiado tiempo en las fórmulas de Slide 7.** El objetivo es comprensión conceptual, no derivación. Las fórmulas están para referencia.

2. **No dejar tiempo libre para explorar el scatter y el grafo.** Al menos 3-4 minutos de exploración libre en cada visualización interactiva.

3. **Presentar el pipeline de evaluación (Slide 24) como "la respuesta correcta".** Es una propuesta, no un estándar universal. Invitar a los estudiantes a modificar los pesos.

4. **Saltar el "Whole Milk Problem" (Slides 20-22).** Es el momento pedagógico más poderoso de la sesión D: cuando el análisis riguroso confirma que no hay nada interesante, eso es un resultado valioso, no un fracaso.

---

## Referencias Bibliográficas

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*, 1215, 487–499.
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
- Tan, P.-N., Steinbach, M., & Kumar, V. (2006). *Introduction to Data Mining*. Addison-Wesley.
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.