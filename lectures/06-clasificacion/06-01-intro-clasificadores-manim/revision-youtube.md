# Revisión de la clase de clasificadores para YouTube

La clase «¿Pagará el préstamo?» tiene una base pedagógica sólida para video:
conserva un ejemplo, transforma tablas en geometría y muestra los errores en
lugar de esconderlos. Antes de grabar conviene corregir algunas frases,
reducir la carga de fórmulas y adaptar las actividades presenciales. Recomiendo
una edición introductoria de unos 22–28 minutos; es una meta editorial que
debe ajustarse después de ensayar la narración.

Esta revisión corresponde a `IntroClasificadores`, en `presentacion.py`.
Los números de estado siguen el manifiesto local de 49 estados, con resolución
declarada de 3840 × 2160. Se revisaron los cinco actos y sus cálculos y se
inspeccionaron fotogramas finales de los estados 1, 4, 8, 13, 14, 25, 30,
37, 40, 41, 42, 45, 47, 48 y 49. La revisión visual es de esa muestra;
conviene hacer un pase completo de reproducción antes de grabar.

## Lo que conservaría

- **El mismo conjunto de clientes.** Evita volver a explicar el contexto al
  cambiar de algoritmo y permite comparar predicciones concretas.
- **Tabla → puntos → frontera.** La formalización llega después de una imagen
  que ya tiene significado.
- **El cliente 6 como excepción.** Une la regla sencilla, el vecino más cercano
  y la equivocación de la logística.
- **La matriz construida con clientes.** Es más explicativa que memorizar cuatro
  siglas y tres cocientes.
- **La paradoja del 97 %.** Es una buena apertura para despertar una pregunta
  que el resto del video resolverá.
- **Color acompañado de forma.** Círculo, cuadrado y rombo ayudan a distinguir
  clases y consultas sin depender exclusivamente del color.

## Cambios antes de grabar

### Corregir la relación entre puntaje y distancia

**Estado 37, `sections/part04_linear.py`, método `linear`.** El título dice
«Distancia a la recta = puntaje». Para `z = b + wᵀx`, la distancia euclidiana
con signo es `d = z / ||w||`. El código dibuja esa distancia correctamente,
pero el título identifica dos cantidades diferentes.

Propuesta: **«El puntaje indica el lado de la frontera»**. En la narración:
«El puntaje es proporcional a la distancia con signo. Cero corresponde a la
frontera; el signo indica de qué lado estamos». La distancia depende también
de las unidades con que representamos los atributos.

### Aclarar que el ejercicio usa otro modelo

**Estados 47–48, `sections/part05_compare.py` y `sections/data.py`.** Los pesos
`−8, −0.10, +0.25` aparecen como «redondeados», aunque el ajuste de la clase
es aproximadamente `−7.574, −0.0731, +0.1882`. Esa sustitución modifica mucho
la probabilidad del cliente `(20, 50)`:

| Modelo | Puntaje aproximado | Probabilidad de impago |
|---|---:|---:|
| Ajustado durante la clase | 0.375 | 0.593 |
| Propuesto para el ejercicio | 2.5 | 0.924 |

Ambos predicen «no paga» con umbral 0.5, pero la probabilidad es distinta.
Propuesta: cambiar la etiqueta a **«Modelo simplificado para practicar»** y
decir explícitamente que es otro modelo. El borrador de guion usa esta opción.
Si se desea conservar exactamente el modelo ajustado, deben cambiarse también
el cálculo, la tabla de sigmoide y la solución.

### Separar validación y prueba

**Estados 34 y 44–46.** El uso de clientes nuevos está bien planteado. Sin
embargo, bajar el umbral después de mirar el resultado en esos cinco clientes
puede enseñar accidentalmente a ajustar sobre prueba.

Propuesta: añadir **«Entrenar → elegir k y umbral con validación → evaluar en
prueba»**. Presentar el umbral 0.2 como una demostración del efecto del corte,
sin decir que se ha encontrado un umbral óptimo. En una evaluación real se
fija antes de consultar la prueba final. Los cinco casos ilustran diferencias;
no permiten afirmar que la logística es mejor en general.
[Evaluación y validación](https://scikit-learn.org/stable/modules/cross_validation.html),
[elección del umbral](https://scikit-learn.org/stable/modules/classification_threshold.html).

### Explicar por qué la logística acierta menos en entrenamiento

**Estados 8, 40 y 44.** El corte acierta 9/10 y la logística 8/10 en
entrenamiento. Es una oportunidad didáctica que necesita una frase explícita:
**«La logística aprende minimizando log-loss; eso no equivale a minimizar el
número de errores con umbral 0.5»**. Así, la recta inclinada no parece una
mejora automática por utilizar dos atributos.

En el estado 40, sustituir «6 y 10: los casos que ninguna recta resuelve» por
**«6 y 10: errores de esta frontera con umbral 0.5»**. El corte horizontal de
44 ya clasifica correctamente al 10: el problema no es que ninguna recta pueda
acertar esos casos.

### Matizar la interpretación de los pesos y la escala

**Estados 42–43.** Al interpretar un peso, añadir **«manteniendo fijo el otro
atributo»**. Es una asociación del modelo, no una afirmación causal. Si se
conservan los momios, definirlos como `p / (1 − p)`; multiplicar los momios
por 1.21 no significa sumar 21 puntos porcentuales a la probabilidad.

En la tabla comparativa, cambiar «escala: poco sensible» de la logística por
**«escalar ayuda al optimizador y condiciona la regularización»**. En un ajuste
sin penalización que converge, cambiar unidades puede compensarse cambiando
coeficientes; eso no justifica una recomendación general de ignorar la escala.
El propio ejemplo estandariza los datos antes de ajustar.
[Documentación de regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

### Afinar las explicaciones de los vecinos y los cortes

**Estados 27 y 31.** Cambiar «no hay entrenamiento» por **«no ajustamos una
fórmula paramétrica; conservamos los datos de entrenamiento»**. Sí se decide
el preprocesamiento, la distancia y k. La media y la desviación para
estandarizar deben calcularse con entrenamiento y reutilizarse en datos nuevos.
[Ejemplo de escalamiento](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).

**Estados 15–18.** Evitar «al subir el corte siempre sube la precisión». En
este ejemplo la precisión pasa por 0.67, 0.80, 0.75 y 1.00: no es monótona.
La sensibilidad sí deja de aumentar al exigir más deuda con esta regla fija.
Narración sugerida: «Hay menos alarmas; observemos cuáles eran correctas y
cuáles falsas antes de concluir qué pasa con la precisión».

## Ajustes para la grabación

| Estados | Propuesta para video | Motivo |
|---|---|---|
| 1–3 y 19 | Abrir con el caso del 97 %; portada breve; resumir la ruta oralmente | Formular pronto la pregunta del video |
| 4–9 | Conservar tabla, transformación al plano, referencia y corte | Construir intuición antes de las métricas |
| 13–14 | Mostrar una métrica por vez; F1 como ampliación opcional | Evitar que compitan cuatro fórmulas |
| 20–21 | Suprimir o dejar como ejercicio adicional | El salto a fraude introduce otro contexto |
| 24–25 | Conservar una distancia detallada y resumir la segunda | Mantener la idea de Pitágoras sin repetir el procedimiento |
| 30–31 | Conservar la trampa de unidades | Es esencial para interpretar k-NN |
| 35–40 | Conservar frontera, puntaje, sigmoide y entrenamiento visual | Explicar qué aprende la logística |
| 41–43 | Reservar log-loss completa, momios y XOR para una ampliación | Reducir conceptos nuevos en la primera explicación |
| 44–46 | Conservar comparación y umbral con la aclaración metodológica | Conectar resultados con el costo de los errores |
| 47–49 | Cambiar «en parejas» por «pausa el video» y cerrar con una pregunta | Adaptar la participación a una grabación |

Mi elección inicial sería un video con capítulos. Si el ensayo supera los
30 minutos, los cortes naturales son: conceptos y errores; k-NN y escalamiento;
logística y comparación. Es una propuesta editorial, no una predicción del
rendimiento del video en YouTube.

En la muestra visual, los estados 13–14 y 41–42 concentran más notación. En la
lupa del estado 25 las etiquetas de catetos y clientes quedan muy próximas:
mostrar un triángulo cada vez ayudaría. El estado 30 conserva marcas de 20–60
en el eje mientras los puntos se comprimen: conviene cambiar el eje o rotular
**«Esquema de cambio de escala»**, porque la compresión visual usa un factor
distinto del 0.01 empleado en el cálculo.

Grabar una prueba de un minuto a 1920 × 1080 y verla en un teléfono permitirá
comprobar etiquetas, fórmulas y subtítulos. Dar prioridad a tamaños de 28–32
para texto explicativo y reducir contenido antes que encogerlo. Evitar colocar
la cámara sobre tablas o gráficas; la composición actual usa ambos lados.
Las pausas de ejercicios pueden durar 3–5 segundos, acompañadas de «pausa el
video», en lugar de conservar minutos de espera de una clase presencial.

## Propuesta de paleta oscura

El fondo azul actual ya funciona. Propongo una variante de pizarra gris azulada
con acentos algo menos saturados y la misma relación entre clases y formas.
La apariencia educativa depende también del espacio libre, la jerarquía
tipográfica y la estabilidad de las convenciones.

| Token | Color | Uso |
|---|---|---|
| `BG` | `#111827` | Fondo general |
| `PANEL` | `#1B2638` | Tarjetas y matrices |
| `INK` | `#F3F4F6` | Texto y fórmulas principales |
| `MUTED` | `#B8C4D6` | Etiquetas y texto secundario |
| `GRID` | `#64748B` | Ejes y contornos necesarios para leer |
| `BLUE` | `#67C7EB` | Paga, círculo |
| `ORANGE` | `#F2B56B` | No paga, cuadrado |
| `PURPLE` | `#B8A4E3` | Modelo logístico y puntaje |
| `YELLOW` | `#E5C76B` | Consulta, umbral y pregunta |
| `GREEN` | `#85C6A2` | Acierto acompañado de ✓ |
| `RED` | `#ED8A8A` | Error acompañado de ✗ |

Usar retícula secundaria con baja opacidad; mantener ejes y límites importantes
opacos. Reservar el texto de fórmulas para `INK` y destacar sólo el término que
se está explicando. Conservar círculo/cuadrado y añadir ✓/✗ en resultados:
verde y rojo por sí solos no deben comunicar acierto o error.

El contraste calculado del texto principal es 16.1:1 sobre el fondo y 13.8:1
sobre el panel; el texto secundario alcanza 10.1:1 y 8.6:1, respectivamente.
Esto comprueba el contraste de los colores opacos; los tamaños de letra y las
transparencias deben revisarse después en la presentación renderizada.

La propuesta se muestra en [la vista previa](propuesta-paleta-oscura.png) y en
[el SVG editable](propuesta-paleta-oscura.svg). La paleta se aplica ahora a la edición para video y a la escena de aula.

## Material preparado

[Borrador de guion para video](guion-youtube-draft.md), organizado por estados,
con texto hablado, indicaciones de animación y las aclaraciones anteriores.
El guion docente de 60 minutos se conserva para la clase presencial.

Esta revisión conserva el diagnóstico de la versión original de 49 estados.
Los cambios se implementaron después en la edición para video. El
[guion de grabación actualizado](guion-youtube.md) sigue la numeración nueva;
el [README](README.md) documenta la compilación. El PDF del libro no forma
parte de esta actualización.
