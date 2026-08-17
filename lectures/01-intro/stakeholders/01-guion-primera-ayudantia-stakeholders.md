# Guion de clase: de una necesidad de negocio a un problema analítico

## Propósito general

Guiar al grupo para que comprenda que un proyecto de datos útil no comienza eligiendo datos, herramientas o algoritmos. Comienza aclarando una necesidad, identificando a las personas involucradas, entendiendo la decisión que se desea mejorar y formulando un objetivo analítico que pueda validarse.

La pregunta conductora de la sesión es:

> ¿Cómo pasamos de “necesito datos” a un proyecto que realmente ayuda a decidir?

## Objetivos de aprendizaje

Al finalizar la clase, el estudiantado será capaz de:

1. Reconocer la ambigüedad presente en una solicitud de datos aparentemente detallada.
2. Distinguir entre una petición, una necesidad de negocio, una decisión y un objetivo analítico.
3. Identificar stakeholders que solicitan, usan, habilitan, validan, bloquean o reciben las consecuencias de una decisión.
4. Clasificar stakeholders mediante dos lentes independientes:
   - su relación con la organización: internos o externos;
   - su papel respecto del proyecto: primarios, secundarios o clave.
5. Priorizar la estrategia de involucramiento de cada stakeholder mediante una matriz de poder e interés.
6. Formular preguntas de aclaración antes de solicitar datos o seleccionar métodos.
7. Redactar un objetivo analítico provisional que incluya población, resultado, periodo, decisión, restricciones y una medida inicial.
8. Explicar los supuestos e incertidumbres que todavía deben validarse con las personas involucradas.

## Conceptos involucrados

### Necesidad de negocio

Situación que motiva una intervención. Puede expresarse inicialmente de manera vaga, por ejemplo: “las ventas bajaron” o “necesitamos entender mejor a los clientes”. Todavía no indica con precisión qué decisión debe cambiar.

### Petición de datos

Producto o solución solicitada de antemano, como un reporte, un dashboard o una lista de clientes. La petición puede ser útil como punto de partida, pero no debe confundirse con el problema real.

### Decisión

Acción concreta que una persona o grupo debe tomar. En el caso de la clase, podría consistir en decidir qué segmento contactar, con qué oferta, durante qué periodo y bajo qué restricciones.

### Objetivo analítico

Formulación provisional de lo que se analizará para producir evidencia útil para una decisión. Debe ser comprensible, delimitado y validable antes de elegir técnicas o algoritmos.

Una plantilla de apoyo es:

> Analizar **[resultado]** en **[población]** durante **[periodo]** para que **[actor]** decida **[acción]**, sujeto a **[capacidad o restricción]**, y observar **[medida]**.

### Stakeholder

Persona, grupo u organización que puede influir, aportar, habilitar, bloquear o verse afectada por el proyecto o la decisión. No es únicamente quien solicita el análisis ni necesariamente alguien favorable al proyecto.

### Stakeholders internos y externos

- **Internos:** forman parte de la organización y participan directamente en sus decisiones, procesos o actividades. Ejemplos: propietarios, accionistas, directivos y empleados.
- **Externos:** no forman parte de la organización, pero pueden influir en ella o recibir las consecuencias de sus actividades. Ejemplos: clientes, comunidad, proveedores, gobierno, bancos y autoridades.

### Stakeholders primarios, secundarios y clave

- **Primarios:** experimentan directamente el resultado o impacto del proyecto.
- **Secundarios:** participan o reciben efectos de manera indirecta.
- **Clave:** poseen una capacidad significativa para habilitar, modificar, legitimar o bloquear el resultado.

Estas categorías no sustituyen la distinción interno/externo. Son lentes diferentes y pueden superponerse. Por ejemplo, el área de privacidad puede ser un stakeholder interno, secundario y clave al mismo tiempo.

### Poder e interés

- **Poder:** capacidad contextual de un stakeholder para habilitar, modificar, bloquear o legitimar una decisión.
- **Interés:** grado en el que el proceso o su resultado le importa o le afecta. Tener interés no implica estar de acuerdo con el proyecto.

La combinación de ambas dimensiones orienta el involucramiento:

| Poder | Interés | Estrategia sugerida |
|---|---|---|
| Bajo | Bajo | Monitorear |
| Bajo | Alto | Mantener informado |
| Alto | Bajo | Mantener satisfecho |
| Alto | Alto | Gestionar de cerca |

La posición no es permanente: depende del alcance, la fase, los datos utilizados y las consecuencias de la decisión.

### Alcance, supuestos y validación

- **Alcance:** define población, periodo, unidad de análisis, capacidades y restricciones.
- **Supuesto:** condición que se considera cierta provisionalmente y que influye en la formulación del problema.
- **Validación:** conversación mediante la cual los actores pertinentes confirman, corrigen o refinan el objetivo y sus términos.

### Medición

Forma inicial de observar si la acción produjo el resultado esperado. Debe distinguirse entre describir un cambio y atribuirle causalidad a una intervención.

## Guion sugerido de la sesión

### 1. Apertura y encuadre — 0:00 a 3:00

Presentarse brevemente, indicar dónde se encuentran los materiales y plantear la pregunta conductora.

**Mensaje docente:**

> Hoy no comenzaremos con una base de datos ni con un algoritmo. Comenzaremos con alguien que pide ayuda y con una pregunta: ¿cómo convertimos esa petición en algo que realmente mejore una decisión?

### 2. Afinidades y colaboración — 3:00 a 10:00

Organizar al grupo en tríos. Cada persona comparte su nombre, un problema que le gustaría comprender con datos, una habilidad que puede aportar y algo que desea aprender.

Solicitar que cada trío identifique una coincidencia y una diferencia valiosa.

**Idea que debe quedar clara:** un equipo útil no requiere personas idénticas; necesita intereses compartidos y capacidades complementarias.

### 3. La ambigüedad de una petición — 10:00 a 16:00

Presentar el caso:

> “Las ventas de ropa de mujer bajaron este mes. Necesito un reporte con todos los clientes y sus compras para lanzar un descuento.”

Preguntar:

- ¿Respecto de qué referencia “bajaron” las ventas?
- ¿Qué significa “todos los clientes”?
- ¿La caída se refiere a ingresos, unidades, margen o número de compras?
- ¿El descuento es una decisión acordada o una hipótesis?
- ¿Quién utilizará el reporte y qué acción podrá ejecutar?

Explicar la progresión:

> Petición → conversación → decisión → objetivo analítico.

Usar el ejemplo del Dashboard 360 para mostrar que integrar datos y construir muchas visualizaciones no garantiza que alguien pueda actuar con ellas.

**Mensaje docente:**

> El problema no es usar un dashboard. El problema es construirlo antes de acordar quién decidirá, sobre qué unidad, con qué capacidad y bajo qué criterio.

### 4. Identificación y clasificación de stakeholders — 16:00 a 25:00

Pedir que el grupo identifique quién puede solicitar, usar, aportar contexto, validar, bloquear o recibir las consecuencias de la decisión comercial.

Recuperar actores como:

- Mercadotecnia;
- categoría e inventario;
- clientes;
- finanzas;
- privacidad o área legal;
- dirección y proveedores, si el alcance lo justifica.

Presentar primero la distinción entre internos y externos. Después introducir el papel de primarios, secundarios y clave.

**Pregunta de comprobación:**

> ¿Puede un stakeholder ser interno, secundario y clave al mismo tiempo? ¿Qué supuesto justificaría esa clasificación?

Enfatizar que las categorías dependen del contexto y no son etiquetas universales.

### 5. Priorización mediante poder e interés — 25:00 a 29:00

Presentar la matriz de poder e interés y pedir al grupo que ubique a Mercadotecnia, clientes y privacidad.

Por cada ubicación solicitar una razón y preguntar:

> ¿Qué cambio en el alcance movería a este actor a otro cuadrante?

Relacionar cada cuadrante con una estrategia de comunicación o participación. Aclarar que la matriz es una hipótesis para organizar el trabajo, no una verdad permanente.

### 6. Actividad central: reformular la petición — 29:00 a 43:00

En equipos, solicitar un producto de una hoja con:

1. Entre cuatro y seis stakeholders.
2. La clasificación de los actores más relevantes.
3. Su ubicación en la matriz de poder e interés y una estrategia de involucramiento.
4. Al menos tres preguntas de aclaración.
5. Una decisión concreta que se desea mejorar.
6. Un objetivo analítico provisional.
7. Un supuesto o término pendiente de validar.

Si un equipo se bloquea, preguntar:

- ¿Quién toma la decisión?
- ¿Quién vive sus consecuencias?
- ¿Quién puede detener o modificar la acción?
- ¿Qué significa exactamente “bajaron”?
- ¿La organización puede actuar sobre todos los clientes?
- ¿Cómo observaría si la acción funcionó?

Evitar resolver el caso por el grupo. Pedir que haga explícitas sus razones.

### 7. Comparación y refinamiento — 43:00 a 49:00

Solicitar reportes breves de dos equipos. Comparar sus decisiones, actores prioritarios y supuestos.

Explicar que respuestas diferentes pueden ser válidas cuando declaran su alcance y justifican sus supuestos.

Presentar el ciclo de refinamiento:

> Contexto → decisión → medición → alcance → validación → reformulación.

Un ejemplo provisional podría ser:

> Comparar la variación reciente de compras de ropa de mujer entre segmentos relevantes para que Mercadotecnia priorice una campaña limitada por inventario y capacidad, evaluada inicialmente mediante el cambio en conversión y margen frente a un grupo de comparación.

Señalar que esta formulación todavía requiere validación y no determina por sí sola el método analítico.

### 8. Transferencia a intereses personales — 49:00 a 56:00

Pedir a cada estudiante que registre dos problemas que le interesa comprender, una habilidad que aporta y otra que desea desarrollar.

Si alguien propone una tecnología como tema, reformular con la pregunta:

> ¿Qué situación o decisión te gustaría comprender mediante esa tecnología?

**Ejemplo:** sustituir “quiero usar redes neuronales” por “quiero comprender qué factores se asocian con el abandono y qué intervención convendría evaluar”.

### 9. Cierre y evaluación rápida — 56:00 a 60:00

Reconstruir con participación del grupo la cadena completa:

> Necesidad → personas → preguntas → decisión → objetivo → posible proyecto de datos.

Solicitar un ticket de salida con respuestas breves:

1. ¿Qué preguntarías antes de pedir datos?
2. ¿Qué problema te interesaría explorar?
3. ¿Qué habilidad o interés aportarías?
4. ¿Qué duda todavía tienes?

Cerrar con la idea central:

> Un proyecto de datos útil no comienza con el algoritmo; comienza comprendiendo a las personas, la decisión y el problema. Los datos y los métodos vienen después.

## Evidencias de aprendizaje

Al terminar la sesión deberían existir las siguientes evidencias:

- una lista razonada de stakeholders;
- una clasificación que use ambos lentes sin confundirlos;
- una matriz de poder e interés acompañada de estrategias de involucramiento;
- tres o más preguntas de aclaración;
- una decisión expresada como acción concreta;
- un objetivo analítico provisional;
- al menos un supuesto o término marcado para validación;
- un registro individual de problemas e intereses.

## Errores frecuentes que conviene anticipar

- Comenzar discutiendo columnas, SQL, dashboards o algoritmos.
- Aceptar la solución solicitada como si fuera la necesidad real.
- Confundir stakeholder con cliente, usuario o patrocinador.
- Suponer que interno equivale a primario y externo a secundario.
- Tratar la matriz de poder e interés como una clasificación fija.
- Formular objetivos con verbos técnicos como “hacer clustering” sin indicar qué decisión mejorará.
- Omitir población, periodo, comparación, restricciones o medida de resultado.
- Confundir una asociación o un cambio antes/después con evidencia causal.

## Síntesis para el docente

La sesión debe conducir al grupo desde una petición ambigua hasta una formulación provisional y discutible. El aprendizaje central no es encontrar una única respuesta correcta para el caso, sino desarrollar el hábito de preguntar quién decide, quién resulta afectado, qué términos son ambiguos, qué restricciones existen y qué evidencia permitiría actuar. Solo después de validar esas respuestas tiene sentido investigar fuentes de datos y elegir métodos.
