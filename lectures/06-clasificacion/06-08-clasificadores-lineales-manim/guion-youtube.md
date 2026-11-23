# Guion de grabación · Clasificadores lineales

Edición española para YouTube · Diego Villalba · Facultad de Ciencias, UNAM.

**65 estados**, con narración completa y notas generadas desde `narracion.json`.
El orden se verifica contra `slides/ClasificadoresLineales.json`. Las notas del
HTML contienen este mismo texto y las indicaciones del presentador.

**Duración propuesta: 36–42 minutos.** El texto hablado contiene 4220 palabras.
A 115–130 palabras por minuto, su lectura ocupa 32–37 minutos;
se añaden demostraciones, énfasis y pausas breves. Ajustar después del ensayo.
Las animaciones pueden reproducirse mientras se habla: no sumar ambas duraciones
mecánicamente. Los tiempos de MP4 siguientes son referencias del archivo sin voz,
no marcas definitivas del futuro video narrado.

## Preparar la grabación

- Abrir el HTML, usar `S` para notas y grabar únicamente la presentación.
- Avanzar al estado indicado, dejar que termine la construcción y explicar
  su fotograma estable. Durante los giros, acompañar el objeto que cambia.
- Las invitaciones a pausar dejan 2–3 segundos; no insertar minutos de silencio.
- Si se usa el MP4 en un editor, prolongar sus fotogramas estables hasta cubrir
  la narración. El archivo sólo contiene animaciones, sin pista de voz ni música.
- Mantener libres los gráficos y la columna derecha al colocar cámara o subtítulos.

## Capítulos sugeridos

| Estados | Bloque | Tiempo editorial orientativo |
|---|---|---|
| 01–13 | Geometría y perceptrón | 0–7 min |
| 14–23 | Mínimos cuadrados y contraejemplo | 7–13 min |
| 24–28 | Fisher | 13–16 min |
| 29–41 | Regresión logística y optimización | 16–25 min |
| 42–47 | Más variables y softmax | 25–29 min |
| 48–59 | Características y kernels | 29–37 min |
| 60–65 | Comparación, ejercicio y cierre | 37–42 min |

Estos tiempos son una pauta de ensayo, no capítulos de YouTube listos para publicar.

## Narración por estado

### E01 · La pregunta del video

**MP4 sin voz:** 00:00–00:01. **Pausa del HTML:** 1.

Si dos rectas clasifican correctamente los mismos datos, ¿cuál elegirías? El número de aciertos no permite distinguirlas. Necesitamos decir qué queremos optimizar. Soy Diego Villalba, de la Facultad de Ciencias de la UNAM, y en este video vamos a conectar una imagen muy sencilla, una recta que separa puntos, con varias maneras de aprender un clasificador.

**Presentador:** Abre con la pregunta; deja dos segundos antes de dar el propósito.

### E02 · El recorrido

**MP4 sin voz:** 00:01–00:04. **Pausa del HTML:** 2.

Primero vamos a leer la geometría: qué hacen los pesos y el intercepto. Después elegiremos una frontera mediante cuadrados, proyecciones y probabilidades. Finalmente veremos más variables, más clases y fronteras curvas obtenidas al ampliar las coordenadas. La pregunta que nos acompañará es siempre la misma: ¿qué criterio utilizamos para aprender? No necesitamos que todos los métodos respondan igual.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E03 · Dos nubes

**MP4 sin voz:** 00:04–00:09. **Pausa del HTML:** 3.

Cada círculo azul pertenece a la clase menos uno y cada cuadrado naranja a la clase más uno. La forma acompaña al color durante todo el video. Imagina una recta que deje cada clase de un lado. Antes de continuar, intenta describir su inclinación. Aquí estamos mirando los datos de entrenamiento; todavía no sabemos cómo responderá una frontera ante observaciones nuevas.

**Presentador:** Invita a imaginar una recta; conserva círculos y cuadrados como leyenda.

### E04 · Muchas soluciones

**MP4 sin voz:** 00:09–00:18. **Pausa del HTML:** 4.

La recta gira y el contador nos muestra cuándo aparecen errores. En este conjunto hay varias orientaciones que separan las dos clases. Como la muestra es finita y queda un margen estricto entre los puntos y una recta separadora, podemos perturbarla un poco y seguir separando. Tenemos infinitas soluciones. Separar estos datos no basta para decidir cuál es preferible.

**Presentador:** Señala el contador durante el giro. La pausa termina con margen estricto.

### E05 · El vector de pesos

**MP4 sin voz:** 00:18–00:25. **Pausa del HTML:** 5.

La flecha morada representa el vector de pesos, w. Es perpendicular a la frontera y apunta hacia el lado que predecimos como más uno. El puntaje g se obtiene con un producto punto y un intercepto. Al cambiar la dirección de w, cambia la inclinación de la frontera. Observa que la flecha y la recta giran juntas conservando el ángulo recto.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E06 · Mover el intercepto

**MP4 sin voz:** 00:25–00:32. **Pausa del HTML:** 6.

Ahora mantenemos w fijo y cambiamos sólo b. La frontera se desplaza paralela a sí misma. Si b aumenta, más posiciones tienen puntaje positivo y la región predicha como más uno gana terreno. Después regresamos a la posición inicial. La interpretación de estos movimientos depende de las unidades de los ejes, por eso aquí las escalas geométricas son iguales.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E07 · Puntaje y distancia

**MP4 sin voz:** 00:32–00:35. **Pausa del HTML:** 7.

El signo del puntaje indica el lado, pero su magnitud no es automáticamente una distancia. Para obtener la distancia euclidiana con signo dividimos entre la norma de w, que suponemos distinta de cero. En este dibujo w tiene norma uno: el cuadrado está a unas más 2.3 unidades y el círculo a unas menos 4.2. Las líneas punteadas muestran la perpendicular más corta.

**Presentador:** Señala primero el segmento perpendicular y después el denominador.

### E08 · La regla de decisión

**MP4 sin voz:** 00:35–00:39. **Pausa del HTML:** 8.

Ya podemos escribir la regla: decidir por el signo de w transpuesta por x más b. No importa cuántos atributos contenga x: primero calculamos un solo número y después preguntamos si es positivo o negativo. Reservamos el valor cero para la frontera. En una implementación debemos fijar cómo resolver exactamente ese empate; la función signo, por sí sola, no elige una de las dos clases.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E09 · Qué es un hiperplano

**MP4 sin voz:** 00:39–00:44. **Pausa del HTML:** 9.

Con w distinto de cero, la ecuación g igual a cero define un hiperplano. En una variable es un punto, en dos una recta y en tres un plano. En d variables tiene dimensión d menos uno. Escribimos d pesos y un intercepto, aunque hay redundancia: multiplicarlos todos por la misma constante positiva conserva la frontera y las decisiones. Más adelante veremos que puede cambiar las probabilidades.

**Presentador:** Aclara la redundancia de escala y el empate antes de continuar.

### E10 · Primera corrección del perceptrón

**MP4 sin voz:** 00:44–00:50. **Pausa del HTML:** 10.

El perceptrón aprende corrigiendo un punto cada vez. Si la etiqueta multiplicada por el puntaje es menor o igual a cero, actualiza w sumando eta por la etiqueta por el punto; b recibe eta por la etiqueta. Aquí eta vale 0.2. El cuadrado marcado impulsa la primera corrección. El algoritmo no busca directamente una probabilidad ni un margen máximo: busca dejar de equivocarse.

**Presentador:** Señala el cuadrado marcado durante la primera corrección.

### E11 · Segunda corrección

**MP4 sin voz:** 00:50–00:53. **Pausa del HTML:** 11.

La actualización anterior dejó un círculo del lado incorrecto. Ahora su etiqueta negativa cambia la dirección de la corrección. Mira cómo se mueve la misma frontera y cómo cambian sus regiones. Cada actualización puede arreglar un caso y desacomodar otros: el perceptrón sólo atiende un error a la vez. El orden en que recorremos la muestra influye en la trayectoria.

**Presentador:** Sigue el círculo marcado sin adelantar la tercera actualización.

### E12 · Tercera corrección

**MP4 sin voz:** 00:53–00:57. **Pausa del HTML:** 12.

El tercer punto marcado provoca otra actualización. Después de este paso la frontera separa los dieciséis puntos. Tres correcciones bastaron en este ejemplo y con esta inicialización; no es una garantía de tres pasos para cualquier conjunto. El teorema de convergencia requiere una muestra linealmente separable y una tasa positiva apropiada para la regla clásica que estamos utilizando.

**Presentador:** Espera a que aparezca cero errores antes de comentar el resultado.

### E13 · ¿Por qué esta recta?

**MP4 sin voz:** 00:57–01:00. **Pausa del HTML:** 13.

La recta punteada también separa y, sin embargo, es distinta. El inicio y el orden pueden cambiar la respuesta del perceptrón. Para preferir una frontera necesitamos otro criterio: mínimos cuadrados busca reducir residuos, Fisher separa proyecciones y logística optimiza verosimilitud. Una SVM considera el margen. Hoy desarrollaremos los tres primeros y después ampliaremos el espacio con kernels. Un buen ajuste de entrenamiento aún necesita evaluación fuera de muestra.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E14 · Seis estudiantes y cuadrados

**MP4 sin voz:** 01:00–01:05. **Pausa del HTML:** 14.

Pasemos a un ejemplo que podemos calcular a mano. Tenemos seis estudiantes, con una a seis horas de estudio. Codificamos reprobó como menos uno y aprobó como más uno. Los cuadrados rojos representan residuos al cuadrado respecto a una recta de puntajes. Si empezamos prediciendo cero para todos, cada residuo tiene magnitud uno: el área total, o suma de cuadrados, vale seis.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E15 · El ajuste de mínimos cuadrados

**MP4 sin voz:** 01:05–01:09. **Pausa del HTML:** 15.

Al mover la recta hacia el ajuste, la suma de cuadrados baja de seis a 3.2. El puntaje ajustado es menos 1.4 más 0.4 por las horas. Su cruce con cero está en 3.5 horas. Con el signo como regla, nos equivocamos con quien aprobó a las tres horas y quien reprobó a las cuatro. Minimizar residuos cuadrados no equivale a minimizar el número de errores de clasificación.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E16 · Centrar los datos

**MP4 sin voz:** 01:09–01:12. **Pausa del HTML:** 16.

Vamos a reconstruir ese ajuste sin una caja negra. La tabla muestra las horas, sus desviaciones respecto a la media y las etiquetas. La media de las horas es 3.5; la de las etiquetas es cero porque tenemos tres de cada clase. Centramos para separar la variación de x del nivel promedio. Observa la última fila: cada desviación de horas multiplica su etiqueta.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E17 · Sumar dos cantidades

**MP4 sin voz:** 01:12–01:14. **Pausa del HTML:** 17.

Los productos de la última fila suman siete: ése es S sub x y. Los cuadrados de las desviaciones de las horas suman 17.5: ése es S sub x x. No estamos usando aquí las versiones divididas entre el tamaño de muestra; para el cociente ese factor común se cancelaría. Puedes pausar y verificar las dos sumas antes de seguir.

**Presentador:** Invita a pausar para sumar Sxy y Sxx.

### E18 · Pendiente e intercepto

**MP4 sin voz:** 01:14–01:16. **Pausa del HTML:** 18.

La pendiente es siete entre 17.5: obtenemos 0.4. Para recuperar el intercepto calculamos la media de y menos la pendiente por la media de x: cero menos 0.4 por 3.5, que da menos 1.4. Igualar el puntaje a cero devuelve el corte de 3.5 horas. Estos valores describen un ajuste de etiquetas numéricas; no interpretamos sus salidas como probabilidades.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E19 · El plano de parámetros

**MP4 sin voz:** 01:16–01:24. **Pausa del HTML:** 19.

Ahora cada punto del dibujo representa una pareja de parámetros, beta cero y beta uno, y no un estudiante. Las curvas unen ajustes con la misma pérdida: cerca del centro la pérdida es menor. Son elipses porque la suma de residuos cuadrados es cuadrática en beta. Derivar produce una ecuación lineal en esos parámetros. Ésa es la razón de que podamos resolver las ecuaciones normales.

**Presentador:** Aclara que los ejes ahora representan parámetros, no atributos.

### E20 · Resolver un sistema

**MP4 sin voz:** 01:24–01:30. **Pausa del HTML:** 20.

X contiene una columna de unos para el intercepto y otra con las horas. Las ecuaciones normales forman el sistema de dos por dos que aparece en pantalla. Aquí las columnas son independientes: X tiene rango completo y la solución es única. La fórmula con inversa resume el resultado algebraico; en software conviene resolver el problema con QR o SVD para evitar formar esa inversa explícitamente.

**Presentador:** No recomiendes formar inversas en una implementación numérica.

### E21 · Qué significa un salto

**MP4 sin voz:** 01:30–01:34. **Pausa del HTML:** 21.

Las flechas resumen una resolución directa del sistema, sin una trayectoria de búsqueda dependiente de un punto inicial. Decir un salto es una metáfora del contraste con la optimización iterativa: resolver un sistema sí requiere cómputo. Si X tuviera columnas dependientes, esa inversa no existiría y tendríamos que usar otra descripción, por ejemplo una pseudoinversa o regularización. La geometría del tazón también cambiaría.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E22 · Una predicción antes de cambiar datos

**MP4 sin voz:** 01:34–01:38. **Pausa del HTML:** 22.

Volvamos a puntos con dos atributos. En esta muestra de trece observaciones, la frontera obtenida con mínimos cuadrados no comete errores. Ahora imagina cinco cuadrados nuevos, todos bastante lejos y del lado correcto. ¿Esperarías que la frontera cambiara mucho? Pausa y formula una predicción. Vamos a conservar los mismos puntos y mover únicamente el ajuste para ver qué exige su criterio.

**Presentador:** Pide una predicción antes de introducir los cinco lejanos.

### E23 · El costo de estar muy lejos

**MP4 sin voz:** 01:38–01:45. **Pausa del HTML:** 23.

Los cinco puntos llegan y la frontera cambia hasta cometer tres errores. Con el ajuste anterior, el primer lejano tenía un puntaje de aproximadamente 5.2, aunque su etiqueta era más uno. El residuo al cuadrado era aproximadamente 17.8 usando el valor sin redondear. Para mínimos cuadrados, más uno es una meta numérica exacta. Una predicción positiva demasiado grande también cuesta. El problema nace del objetivo, no de un error al resolverlo.

**Presentador:** La cifra 17.8 usa el puntaje sin redondear.

### E24 · Fisher: mirar las sombras

**MP4 sin voz:** 01:45–01:51. **Pausa del HTML:** 24.

Fisher propone otra pregunta: ¿desde qué dirección se ven mejor separadas las clases? Proyectamos los puntos sobre una recta y observamos sus sombras. Queremos que los centros proyectados queden lejos y que cada clase quede concentrada. La dirección que une las medias es una primera intuición, pero puede ignorar cómo se dispersan las nubes. Los puntos originales conservan sus formas mientras cambia la dirección de observación.

**Presentador:** Explica que las sombras están centradas en el punto medio.

### E25 · Girar la dirección

**MP4 sin voz:** 01:51–01:58. **Pausa del HTML:** 25.

Al girar, las sombras se abren o se comprimen. El cociente J compara separación de medias al cuadrado con dispersión interna. En la dirección de las medias vale aproximadamente 4.22; cerca de ochenta grados alcanza 5.64. No estamos maximizando sólo la distancia entre centros: también importa la variación de los puntos. La misma dirección sirve al cambiar su signo, porque J no distingue entre mirar hacia adelante o hacia atrás.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E26 · Escribir el cociente

**MP4 sin voz:** 01:58–02:01. **Pausa del HTML:** 26.

La notación resume lo que acabamos de ver. Arriba medimos la separación entre medias proyectadas; abajo la dispersión interna proyectada. w aparece en ambas partes, así que multiplicarlo por una constante no cambia el cociente. Podemos buscar una dirección óptima sin decidir todavía cuánto mide el vector. Separar dirección y umbral evitará confundir la proyección de Fisher con una regla completa de clasificación.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E27 · Las matrices de Fisher

**MP4 sin voz:** 02:01–02:04. **Pausa del HTML:** 27.

Definimos delta m como la diferencia de las medias y S sub B como delta m por su transpuesta. S sub W suma las matrices de dispersión dentro de las clases. Con estos seis datos se obtiene la matriz numérica de la pantalla. Para definir y optimizar este criterio no necesitamos suponer que los datos siguen distribuciones gaussianas. Esa interpretación probabilística requerirá supuestos adicionales.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E28 · Dirección y umbral

**MP4 sin voz:** 02:04–02:09. **Pausa del HTML:** 28.

Si S sub W es definida positiva, la dirección óptima es proporcional a su inversa por la diferencia de medias. El vector escrito apunta hacia A; dibujamos su negativo hacia B, que da el mismo J. La dirección no fija por sí sola el umbral. Usamos el punto medio bajo previas y costos iguales. Con clases gaussianas de covarianza común, esta regla coincide con LDA; cambiar las previas puede mover el umbral.

**Presentador:** Menciona S_W definida positiva, signo arbitrario y previas/costos iguales.

### E29 · ¿Una probabilidad de 1.2?

**MP4 sin voz:** 02:09–02:16. **Pausa del HTML:** 29.

Regresemos a los seis estudiantes, ahora con etiquetas cero y uno. Al transformar el ajuste de mínimos cuadrados, la recta alcanza menos 0.2 y 1.2 en los extremos del eje mostrado. Esos valores no pueden ser probabilidades. Necesitamos una transformación que conserve el orden del puntaje y mantenga la salida entre cero y uno. La curva verde nos dará esa posibilidad, pero sus parámetros deben aprenderse con otro objetivo.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E30 · Puntaje y sigmoide

**MP4 sin voz:** 02:16–02:20. **Pausa del HTML:** 30.

La sigmoide transforma un puntaje z en uno dividido entre uno más e elevado a menos z. z sigue siendo una expresión afín de las horas: beta cero más beta uno por x. La animación de doblar la recta conecta las dos formas visualmente; el ajuste logístico no se obtiene simplemente doblando los coeficientes de mínimos cuadrados. La curva final usa los coeficientes aprendidos por máxima verosimilitud.

**Presentador:** La transformación visual no es un procedimiento para ajustar logística.

### E31 · Probabilidad y decisión

**MP4 sin voz:** 02:20–02:23. **Pausa del HTML:** 31.

Si elegimos el umbral de probabilidad 0.5, la decisión cambia exactamente cuando z vale cero. Aquí eso vuelve a ocurrir a las 3.5 horas. Es una coincidencia de la simetría de este ejemplo, no una igualdad general entre mínimos cuadrados y logística. El umbral tampoco es obligatorio: otros costos pueden justificar otro valor, elegido con validación. Interpretar p como probabilidad estimada no garantiza que esté bien calibrada.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E32 · La multa de cada observación

**MP4 sin voz:** 02:23–02:26. **Pausa del HTML:** 32.

¿Cómo aprendemos los parámetros? Si alguien aprobó, su contribución es menos logaritmo de p; si reprobó, menos logaritmo de uno menos p. Siempre usamos la probabilidad asignada a lo que realmente observamos. Con todos los parámetros en cero, p vale un medio y cada multa vale aproximadamente 0.693. Mira las seis barras: empezamos con la misma pérdida para todos y optimizamos el promedio.

**Presentador:** Define la pérdida para ambas etiquetas; no afirmes que todas las barras bajan.

### E33 · Entrenar con Newton

**MP4 sin voz:** 02:26–02:34. **Pausa del HTML:** 33.

Ahora los parámetros avanzan mediante Newton y la curva cambia con ellos. La pérdida promedio baja hasta aproximadamente 0.413. Algunas barras pueden crecer: aceptar más pérdida en ciertos casos puede reducir el total. No estamos buscando acertar cada etiqueta a toda costa. Los estudiantes de tres y cuatro horas son contradictorios con una tendencia creciente y quedan difíciles de explicar. Todo lo mostrado hasta aquí se evalúa sobre entrenamiento.

**Presentador:** Señala las barras centrales que crecen al bajar el promedio.

### E34 · Dos condiciones de óptimo

**MP4 sin voz:** 02:34–02:39. **Pausa del HTML:** 34.

Comparemos las dos ecuaciones. En mínimos cuadrados, beta aparece multiplicando a matrices conocidas: podemos reorganizarlo como un sistema lineal. En logística aparece dentro de la sigmoide. Aunque el puntaje sea lineal en los atributos, la ecuación para encontrar sus parámetros es no lineal. La forma geométrica de la frontera y el método para aprenderla son cuestiones distintas. Una frontera recta puede requerir optimización iterativa.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E35 · Por qué recurrimos a iteraciones

**MP4 sin voz:** 02:39–02:43. **Pausa del HTML:** 35.

Al escribir la suma vemos que distintos valores de x introducen distintas exponenciales de los parámetros. En general no tenemos una fórmula cerrada para el ajuste logístico y usamos métodos numéricos. Ésta es una explicación de la estructura del problema, no una prueba de que ningún caso especial pueda resolverse algebraicamente. En estas expresiones x incluye la coordenada constante que representa el intercepto.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E36 · Comparar los valles

**MP4 sin voz:** 02:43–02:49. **Pausa del HTML:** 36.

A la izquierda, mínimos cuadrados conserva sus niveles elípticos y su solución directa. A la derecha, la pérdida logística tiene niveles no cuadráticos. Es convexa: no aparecen mínimos locales engañosos. Pero esa propiedad, por sí sola, no garantiza que exista una solución finita ni que sea única. Primero vamos a observar dos rutas en este ejemplo y después separaremos cuidadosamente esas condiciones.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E37 · Gradiente y Newton

**MP4 sin voz:** 02:49–02:56. **Pausa del HTML:** 37.

La trayectoria gris resume unos tres mil pasos de descenso de gradiente, con tasa 0.5 y estos atributos sin reescalar. La amarilla muestra los primeros pasos de Newton, que utiliza también información de curvatura y queda muy cerca en cuatro o cinco pasos. Son cifras de este ejemplo. Cambiar escala, tasa, tolerancia o algoritmo cambia los conteos; no estamos comparando tiempos de ejecución ni garantizando una ventaja universal.

**Presentador:** Los 4–5 pasos son una aproximación visual, no la tolerancia final del solver.

### E38 · Existencia, unicidad y convergencia

**MP4 sin voz:** 02:56–02:57. **Pausa del HTML:** 38.

En esta muestra la matriz de diseño tiene rango completo y no hay separación completa ni cuasi completa. Por eso existe un óptimo finito y es único. Convexidad no sustituye esas condiciones. Además, un método necesita condiciones de convergencia: una tasa mal elegida puede arruinar el descenso y Newton puede requerir amortiguación. Si hay columnas redundantes, diferentes coeficientes pueden representar el mismo puntaje aunque la pérdida sea convexa.

**Presentador:** Distingue existencia, unicidad y convergencia.

### E39 · La trampa de los datos separables

**MP4 sin voz:** 02:57–03:07. **Pausa del HTML:** 39.

Ahora cambiamos las etiquetas centrales para que la separación sea perfecta. Podemos hacer la sigmoide cada vez más empinada, reducir la pérdida y seguir mejorando sin alcanzar un mínimo con coeficientes finitos. La norma de beta crece sin límite. Una penalización de tamaño, con lambda positivo, detiene ese crecimiento. Aquí no penalizamos el intercepto y hay ambas clases; la regularización expresa una preferencia adicional sobre los pesos.

**Presentador:** El valor 0.000 está redondeado; con parámetros finitos la pérdida sigue siendo positiva.

### E40 · Volver a los lejanos

**MP4 sin voz:** 03:07–03:15. **Pausa del HTML:** 40.

En el ejemplo de los puntos lejanos, la logística regularizada cambia poco su frontera y sigue sin errores de entrenamiento. Los lejanos correctos reciben una probabilidad casi uno para su clase: su pérdida y su contribución al gradiente son pequeñas. La recta gris recuerda el ajuste de mínimos cuadrados con esos puntos. Estamos comparando dos objetivos sobre los mismos datos, no demostrando que uno siempre clasifique mejor.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E41 · El alcance de la comparación

**MP4 sin voz:** 03:15–03:16. **Pausa del HTML:** 41.

Los dos ajustes logísticos de esta comparación usan pérdida promedio más lambda sobre dos por la norma de w al cuadrado, con lambda igual a 0.01. Lo declaramos porque la muestra base es separable. Esto no hace a logística resistente a cualquier atípico: uno mal clasificado con atributos grandes puede influir mucho. Para escoger modelo y regularización necesitamos validación y una evaluación final separada de estos ejemplos de entrenamiento.

**Presentador:** Indica el valor de lambda y limita la conclusión a este ejemplo.

### E42 · Dos variables, niveles paralelos

**MP4 sin voz:** 03:16–03:23. **Pausa del HTML:** 42.

Con dos atributos seguimos calculando un puntaje y después una sigmoide. Ahora las franjas representan niveles de probabilidad, desde 0.1 hasta 0.9. Dos observaciones cruzadas evitan la separación de la muestra. La línea amarilla indica 0.5 y las verdes otros valores. Antes de ver la ecuación, pregúntate por qué las líneas son paralelas: ¿qué parte del modelo comparten todas las franjas?

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E43 · Deshacer la sigmoide

**MP4 sin voz:** 03:23–03:26. **Pausa del HTML:** 43.

Fijar una probabilidad entre cero y uno equivale a fijar z en el logaritmo de p entre uno menos p. Todas las líneas comparten w y sólo cambia la constante. Por eso son paralelas y perpendiculares a w. Con el modelo y las unidades fijos, la probabilidad depende de la distancia con signo, no sólo de la distancia absoluta. Dos puntos a la misma distancia en lados opuestos no reciben la misma probabilidad.

**Presentador:** Di distancia con signo, con pesos y unidades fijos.

### E44 · Toda la nube se resume en z

**MP4 sin voz:** 03:26–03:35. **Pausa del HTML:** 44.

Proyectamos los puntos sobre la dirección de w. Cada uno se mueve conservando su puntaje, a lo largo de su línea de nivel. Después llevamos esos puntajes al eje horizontal de la gráfica de la derecha; la altura de cada marcador indica su etiqueta observada. La curva muestra la probabilidad del modelo. No estamos convirtiendo las etiquetas en probabilidades: son dos objetos distintos sobre la misma escala de puntajes.

**Presentador:** Los marcadores copiados conservan etiquetas observadas, no probabilidades.

### E45 · Tres variables y más

**MP4 sin voz:** 03:35–03:45. **Pausa del HTML:** 45.

En tres atributos, z igual a cero es un plano. El cubo gira para mostrar que no estamos dibujando una superficie curva. Los puntos mantienen las formas de las clases. Con cincuenta atributos no podemos dibujar la figura completa, pero calculamos igual: producto punto, intercepto y sigmoide. Si w es distinto de cero, la frontera tiene dimensión cuarenta y nueve. El dibujo es ilustrativo; la fórmula conserva el significado.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E46 · Más de dos clases

**MP4 sin voz:** 03:45–03:51. **Pausa del HTML:** 46.

Softmax asigna un puntaje afín a cada clase y transforma esos puntajes en probabilidades que suman uno. Elegir la probabilidad mayor equivale a elegir el puntaje mayor. La tercera clase usa triángulos rosas para distinguirla también por forma. Cada región reúne posiciones donde una clase gana frente a todas las demás. No necesitamos inventar una sola recta que divida simultáneamente las tres clases en dos lados.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E47 · Los empates que sí son frontera

**MP4 sin voz:** 03:51–03:53. **Pausa del HTML:** 47.

Dos puntajes empatan sobre un hiperplano dado por la diferencia de sus pesos e interceptos. Sin embargo, sólo cuenta como frontera visible la parte donde ambos superan a todas las demás clases. Por eso aquí aparecen tres rayos y no tres rectas completas. Las regiones ganadoras son intersecciones de semiplanos y son convexas. En general la frontera multiclase es una colección de porciones de hiperplanos, no un único hiperplano.

**Presentador:** Señala los tres rayos; evita llamar frontera a toda la recta de empate.

### E48 · Cuando un umbral no alcanza

**MP4 sin voz:** 03:53–04:01. **Pausa del HTML:** 48.

Ahora los cuadrados están en el centro de una recta y los círculos en ambos extremos. Un solo umbral, en cualquiera de sus orientaciones, no puede separar perfectamente esas etiquetas. El mejor corte deja tres errores en estos diez puntos. El problema no es encontrar mejor el umbral: la forma disponible no alcanza. Necesitamos ampliar las características si queremos conservar un modelo lineal en otro espacio.

**Presentador:** Invita a probar mentalmente ambas orientaciones del umbral.

### E49 · Agregar una coordenada

**MP4 sin voz:** 04:01–04:10. **Pausa del HTML:** 49.

Elevamos cada punto a la posición x, x al cuadrado. Las clases conservan sus formas mientras se mueven. En este nuevo plano, la recta horizontal de altura dos separa. El puntaje es dos menos x al cuadrado: los pesos sobre las nuevas coordenadas son cero y menos uno. Es una expresión lineal en las características elegidas, aunque una de ellas sea una función no lineal de x.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E50 · Regresar al eje original

**MP4 sin voz:** 04:10–04:14. **Pausa del HTML:** 50.

La recta de altura dos corta la parábola en más y menos raíz de dos, aproximadamente más y menos 1.41. Al regresar al eje original obtenemos dos umbrales y un intervalo positivo entre ellos. No es una frontera lineal en x. Sí viene de una frontera lineal en el espacio ampliado. Siempre que usemos la palabra lineal conviene preguntar: ¿lineal respecto a qué coordenadas?

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E51 · Elevar un anillo

**MP4 sin voz:** 04:14–04:20. **Pausa del HTML:** 51.

La misma idea funciona con un centro naranja y una corona azul en dos atributos. Agregamos x uno al cuadrado más x dos al cuadrado como tercera coordenada. Los puntos centrales suben menos y los exteriores suben más. La malla representa el paraboloide que contiene las nuevas posiciones. La altura se comprime visualmente para que la escena quepa; usamos los valores originales para decidir la separación.

**Presentador:** La altura se comprime visualmente; no es una escala isotrópica en 3D.

### E52 · Del plano al círculo

**MP4 sin voz:** 04:20–04:29. **Pausa del HTML:** 52.

Un plano horizontal en la coordenada nueva, a altura 1.8, separa las clases. Al proyectar de regreso, la intersección con el paraboloide deja el círculo x uno al cuadrado más x dos al cuadrado igual a 1.8. Su radio es raíz de 1.8, aproximadamente 1.34. La vista vuelve casi desde arriba para reconocer la frontera circular. El plano no se volvió curvo: cambió el espacio en que lo observamos.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E53 · Preparar el kernel cuadrático

**MP4 sin voz:** 04:29–04:32. **Pausa del HTML:** 53.

Los ejemplos anteriores construyeron coordenadas nuevas explícitamente. Ahora queremos calcular un producto punto en un espacio ampliado sin construirlo entero. Tomemos x igual a uno, dos y z igual a tres, uno. Usaremos un mapa cuadrático distinto del paraboloide anterior: sus coordenadas son x uno al cuadrado, raíz de dos por x uno por x dos, y x dos al cuadrado.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E54 · Construir las coordenadas

**MP4 sin voz:** 04:32–04:34. **Pausa del HTML:** 54.

Para x obtenemos uno, dos raíz de dos y cuatro. Para z obtenemos nueve, tres raíz de dos y uno. La pantalla redondea las coordenadas intermedias, pero calculamos con sus valores exactos. El factor raíz de dos hace que el producto del término cruzado produzca el dos necesario al expandir el cuadrado de un producto punto. Sin ese factor, el atajo que buscamos no coincidiría.

**Presentador:** Los decimales de φ se redondean sólo al mostrarlos.

### E55 · Dos caminos al mismo número

**MP4 sin voz:** 04:34–04:37. **Pausa del HTML:** 55.

Multiplicando las coordenadas ampliadas obtenemos nueve más doce más cuatro: veinticinco. En el espacio original, el producto punto es uno por tres más dos por uno: cinco. Elevar cinco al cuadrado también da veinticinco. El kernel cuadrático calcula exactamente ese producto ampliado a partir de las coordenadas originales. No aproximamos una nube de puntos: aprovechamos una identidad algebraica.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E56 · El atajo en una función de decisión

**MP4 sin voz:** 04:37–04:40. **Pausa del HTML:** 56.

Un kernel válido representa un producto interno en algún espacio de características. Para algoritmos que admiten esa formulación podemos escribir la decisión como una suma de kernels ponderados más un intercepto. No cualquier algoritmo acepta automáticamente sustituir productos punto. Con centros y parámetros del kernel fijos, aprendemos los coeficientes alfa. La frontera puede ser curva en x y seguir siendo un hiperplano en el espacio de características.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E57 · Campanas del kernel RBF

**MP4 sin voz:** 04:40–04:49. **Pausa del HTML:** 57.

Volvemos a los diez puntos del eje. Cada centro genera una campana RBF, con gamma fijo igual a 0.9. Aprendemos cuánto escalar cada campana, incluso con pesos negativos. Las curvas individuales conservan su forma y anchura: sólo cambia su amplitud. Aquí los colores identifican la clase del centro, no el signo de alfa. La suma final será nuestro puntaje; esas campanas no son probabilidades de clase.

**Presentador:** El color de cada campana identifica su centro; alfa puede tener otro signo.

### E58 · Qué optimiza kernel ridge

**MP4 sin voz:** 04:49–04:50. **Pausa del HTML:** 58.

La fórmula alfa igual a K más lambda I, inversa, por y corresponde aquí a kernel ridge sin intercepto. K es la matriz de productos kernel entre observaciones y lambda vale 0.3. El objetivo suma residuos cuadrados y penaliza alfa transpuesta K alfa, no la norma euclidiana de alfa. Con un kernel semidefinido positivo y lambda positivo, K más lambda I es invertible. Especificar el objetivo explica la fórmula.

**Presentador:** Declara suma, penalización RKHS, lambda y ausencia de intercepto.

### E59 · Sumar y decidir

**MP4 sin voz:** 04:50–04:55. **Pausa del HTML:** 59.

La curva gruesa suma las campanas escaladas. Donde el puntaje es positivo predecimos la clase naranja y donde es negativo la azul. En estos datos recupera el grupo central y clasifica los diez puntos, pero ese resultado de entrenamiento no garantiza generalización. Elegir gamma y lambda requiere validación. Si cambiamos la pérdida cuadrática por una logística, la misma representación con kernels normalmente vuelve a requerir optimización iterativa.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E60 · Comparar criterios

**MP4 sin voz:** 04:55–05:01. **Pausa del HTML:** 60.

La tabla separa tres preguntas: qué objetivo usamos, cómo se ajustan los parámetros y en qué espacio describimos la frontera. El perceptrón y la logística iteran; mínimos cuadrados y Fisher admiten las fórmulas indicadas bajo los supuestos que discutimos. Kernel no es por sí solo una pérdida: combina una representación con un criterio. Para evaluar un clasificador faltan validación, métricas y costos adecuados al problema real.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E61 · Ejercicio para pausar

**MP4 sin voz:** 05:01–05:06. **Pausa del HTML:** 61.

Te propongo un modelo sencillo para practicar: w igual a dos, menos uno y b igual a menos tres. Calcula z igual a dos por x uno menos x dos menos tres para los tres puntos de la tabla. Después aplica la sigmoide y decide con umbral 0.5. Uno cae exactamente en la frontera. Pausa el video; al volver revelaremos una fila a la vez.

**Presentador:** Di pausa el video y espera tres segundos; no insertes minutos de silencio.

### E62 · Resolver P

**MP4 sin voz:** 05:06–05:08. **Pausa del HTML:** 62.

Para P, que es tres, uno, calculamos dos por tres menos uno menos tres. El puntaje es dos y la sigmoide vale aproximadamente 0.88. Como supera 0.5, predecimos más uno. Fíjate en las tres operaciones: producto punto, suma del intercepto y transformación a probabilidad estimada. No necesitamos dibujar una curva nueva para cada observación: reutilizamos el mismo modelo.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E63 · Resolver Q

**MP4 sin voz:** 05:08–05:12. **Pausa del HTML:** 63.

Para Q, que es uno, dos, tenemos dos por uno menos dos menos tres. El puntaje es menos tres y la probabilidad estimada es aproximadamente 0.05. Con el mismo umbral predecimos menos uno. El signo decide y la sigmoide expresa el grado de confianza del modelo. Todavía no sabemos si esas probabilidades están calibradas en una población real; éste es un ejercicio de lectura del modelo.

**Presentador:** Deja terminar la animación y señala el objeto o término nuevo; conserva la pausa para explicar antes de avanzar.

### E64 · Resolver R y el empate

**MP4 sin voz:** 05:12–05:15. **Pausa del HTML:** 64.

Para R, que es dos, uno, obtenemos cuatro menos uno menos tres: cero. Su probabilidad es exactamente un medio y está en la frontera. Debemos fijar una convención si queremos devolver una de las clases. Como reto, duplica todos los pesos y el intercepto: la frontera no cambia, pero los puntajes no nulos se duplican y sus probabilidades se hacen más extremas. R sigue en un medio.

**Presentador:** En el empate no devuelvas una tercera clase: fija una convención binaria.

### E65 · Cerrar la pregunta inicial

**MP4 sin voz:** 05:15–05:19. **Pausa del HTML:** 65.

Dos rectas pueden acertar y representar criterios distintos. La geometría explica qué hace el modelo; el objetivo explica cómo se aprende. Logística conserva una frontera lineal con atributos originales, mientras un kernel puede hacerla curva allí y lineal en otro espacio. Para cerrar, explica con tus palabras en qué espacio es lineal una SVM con kernel RBF. Si distingues espacio, criterio y evaluación, ya tienes una base para comparar clasificadores con sentido.

**Presentador:** Cierra recuperando la pregunta inicial y deja el reto visible.

## Supuestos y fuentes para preparar respuestas

La [revisión editorial](revision-youtube.md) recoge los supuestos de rango,
separación, Fisher/LDA, regularización y kernels, con referencias a los capítulos
del libro y fuentes técnicas. `validar_calculos.py` contrasta los ejemplos con
SVD y un optimizador independiente de Newton. No se deduce capacidad de
generalización de los aciertos de entrenamiento mostrados.
