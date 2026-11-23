# Guion de grabación de clasificadores para YouTube

**Título propuesto:** Cómo funciona un clasificador con k-NN y regresión logística

**Autor:** Diego Villalba · Almacenes y Minería de Datos · Facultad de Ciencias, UNAM.

**Audiencia:** estudiantes que conocen tablas y coordenadas; no se presupone
experiencia entrenando clasificadores. **Meta de edición:** 22–28 minutos,
sujeta al ensayo de voz y animaciones. Los intervalos siguientes son una pauta
de montaje, no capítulos definitivos para publicar.

Las indicaciones entre corchetes son de producción. El resto es texto hablado.
La selección corresponde a la edición `IntroClasificadores`. Los estados son
pausas de presentación, no diapositivas estáticas: avanzar cuando termine la
explicación del estado actual. La narración requiere grabarse por separado.
La edición de aula se conserva como `IntroClasificadoresClase`.

## Apertura con una pregunta

**Referencia de montaje:** 0:00–1:00. **Pantalla:** estados 1–3.

[Mostrar los cien clientes. Revelar primero la exactitud y después la sensibilidad.]

Un modelo acierta el 97 por ciento de sus decisiones. ¿Le confiarías tu dinero?
Antes de responder, mira estos cien clientes. Noventa y siete pagan y tres no.
Nuestro modelo dice exactamente lo mismo para todos: «sí paga».

Acierta noventa y siete veces. Pero no detecta a ninguno de los tres clientes
que no pagan. Si lo que necesitábamos era una alarma, esta alarma nunca suena.

Entonces, ¿cómo sabemos si un clasificador está haciendo algo útil?

Soy Diego Villalba. Vamos a construir esa respuesta con diez clientes, una
gráfica y dos estrategias: preguntar a los vecinos y aprender una frontera
con regresión logística. Al terminar podrás seguir una predicción y explicar
de qué manera se equivoca.

[Portada durante una frase. Pasar al ejemplo; la ruta completa puede omitirse.]

## De los clientes a una frontera

**Referencia de montaje:** 1:00–4:00. **Pantalla:** estados 4–9.

Una financiera tiene el registro de diez clientes. De cada uno conocemos dos
atributos: su ingreso mensual, expresado en miles de pesos, y su deuda como
porcentaje del ingreso. También sabemos si pagó.

Hay una diferencia de tiempo importante. El ingreso y la deuda se conocen
antes de tomar la decisión. Si pagó o no lo sabemos después. Esa última
columna es la etiqueta que queremos anticipar para un cliente nuevo.

Usaremos la letra x para los atributos, y para lo que ocurrió, e y con un
sombrero para la predicción. Una predicción todavía puede equivocarse.

[Transformar las filas en puntos; señalar un cliente durante la transformación.]

Cada fila se vuelve un punto. Hacia la derecha crece el ingreso; hacia arriba,
la deuda. Los círculos azules son clientes que pagaron. Los cuadrados naranjas
son clientes que no pagaron. El rombo amarillo será una consulta nueva.

Ya podemos inventar una primera regla: decir «paga» para todos. Seis de estos
diez clientes pagaron, así que acertamos seis veces sin mirar los atributos.
Ese sesenta por ciento es nuestra referencia. Más adelante preguntaremos si
otras reglas aportan algo útil, según los errores que nos interesen.

Ahora sí usemos los datos. Podemos poner una línea horizontal: si la deuda
supera cierto valor, predecimos «no paga»; en caso contrario, «paga».

[Animar el barrido. Dejar que se lea el contador; detenerse en 44.]

Al mover la línea cambian los errores. Un corte en cuarenta y cuatro deja un
solo error entre estos diez clientes: el número seis, que tenía una deuda alta
y aun así pagó. Cuarenta y cuatro es una elección dentro del intervalo entre
cuarenta y cuarenta y ocho que produce esta misma separación en entrenamiento.

La línea es una frontera de decisión. Divide el plano en regiones donde la
respuesta del clasificador es diferente. Acabamos de construir una función
que recibe atributos y devuelve una categoría.

Nueve aciertos de diez suenan bien. Pero estos son los mismos clientes con los
que elegimos el corte. Todavía falta saber cómo funciona con clientes nuevos.
Y falta una pregunta: ¿todos los errores cuestan lo mismo?

## Los errores tienen dos direcciones

**Referencia de montaje:** 4:00–7:30. **Pantalla:** estados 10–19.

Para contar los errores fijemos una convención: la clase positiva será «no
paga». Positivo significa que se enciende la alarma que queremos estudiar.

Si decimos «no paga» y el cliente sí pagó, tenemos una falsa alarma: un falso
positivo. Si decimos «paga» y después no paga, la alarma no sonó: un falso
negativo. Estas dos equivocaciones pueden tener costos distintos.

[Construir la matriz con los clientes. Mantener visibles «real» y «predicción».]

La matriz de confusión organiza dos preguntas. Las filas muestran lo que pasó
en realidad. Las columnas muestran lo que predijo el modelo. Esta orientación
es la de nuestra gráfica; siempre hay que leer los encabezados de una matriz.

Con el corte en cuarenta y cuatro detectamos a los cuatro que no pagaron.
También acusamos al cliente seis, que sí pagó. Los otros cinco clientes que
pagaron quedaron correctamente del lado de «paga».

[Destacar una región de la matriz cada vez.]

La exactitud pregunta: del total, ¿cuántos aciertos hubo? Cuatro más cinco,
entre diez: noventa por ciento.

La precisión hace otra pregunta: cuando se enciende la alarma, ¿cuántas veces
tiene razón? Hubo cinco alarmas y cuatro eran correctas: ochenta por ciento.

La sensibilidad mira a quienes realmente no pagaron. De esos cuatro,
¿cuántos detectamos? Los cuatro: cien por ciento.

Ahora se entiende el modelo del inicio. Tenía mucha exactitud y sensibilidad
cero. Cada métrica responde una pregunta diferente.

[Mostrar los cortes 44, 51.5 y 59.]

Si exigimos más deuda para encender la alarma, sonarán menos alarmas. Pero
algunas personas que no pagan empezarán a escapar. En el último corte ya no
tenemos falsas alarmas, aunque sólo detectamos a la mitad de quienes no pagan.

La precisión no necesariamente sube en cada paso: depende de qué clientes
dejamos fuera. Lo que buscamos es entender el intercambio y relacionarlo con
el costo de equivocarnos.

## Preguntar a los vecinos

**Referencia de montaje:** 7:30–12:00. **Pantalla:** estados 20–30.

Podemos clasificar de otra manera. En lugar de trazar una línea, busquemos
clientes parecidos a la persona nueva.

Llega B, con ingreso treinta y uno y deuda sesenta. Si sólo pudiéramos consultar
a un cliente conocido, ¿cuál elegirías?

[Expandir el círculo hasta el cliente 6. Dejar dos segundos para observar.]

El más cercano es el seis. Como pagó, un clasificador de un vecino predice
«paga». Pero justamente el seis era nuestra excepción. Quizá convenga escuchar
a más personas antes de decidir.

Primero aclaremos qué significa cerca. Entre B y el seis hay una diferencia
de uno en ingreso y dos en deuda. En el plano forman dos catetos. La distancia
es la raíz de uno al cuadrado más dos al cuadrado: raíz de cinco,
aproximadamente dos punto veinticuatro.

[Construir un solo triángulo a la vez. Después, mostrar la segunda distancia.]

Para el cliente nueve, las diferencias son cuatro y dos. La distancia es raíz
de veinte, aproximadamente cuatro punto cuarenta y siete. Es Pitágoras
aplicado a los atributos, con la escala numérica que estamos usando.

Si elegimos los tres vecinos más cercanos aparecen seis, nueve y ocho. El seis
pagó; nueve y ocho no. La votación es dos contra uno: ahora predecimos «no paga».

Eso es k vecinos más cercanos: medir distancias, seleccionar los k menores y
votar. No ajustamos una fórmula paramétrica como la que veremos después;
conservamos los ejemplos de entrenamiento para hacer las consultas.

[Colorear las regiones para k igual a uno, tres y nueve.]

Podemos hacer esa consulta en todo el plano y colorear las respuestas. Con un
vecino se forma una isla alrededor del cliente seis. Con tres, esa excepción
pierde influencia. Con nueve, en este conjunto siempre gana «paga»: sólo hay
cuatro clientes de la otra clase entre los diez.

Un k muy pequeño puede seguir demasiado las particularidades de los ejemplos.
Uno demasiado grande puede borrar diferencias útiles. El valor adecuado se
elige mediante validación.

Hay otra decisión escondida: las unidades. Si escribimos sesenta por ciento
como cero punto sesenta y dejamos igual el ingreso, los clientes no cambiaron,
pero sí cambiaron los números con los que calculamos la distancia.

[Mostrar el cambio de representación y el nuevo orden de vecinos.]

Ahora B tiene como vecinos a seis, tres y diez. La votación cambia a «paga».
Por eso el preprocesamiento forma parte del modelo. Una opción común es
estandarizar cada variable: restar su media y dividir por su desviación.
Esos valores se calculan con entrenamiento y se reutilizan al recibir casos
nuevos. Las unidades elegidas no deberían decidir accidentalmente qué atributo
manda en la comparación.

## Probar con clientes nuevos

**Referencia de montaje:** 12:00–14:30. **Pantalla:** estados 31–34.

Vamos con C: ingreso veintiséis y deuda cuarenta y cinco. Sus tres vecinos son
diez, cinco y seis. Uno no pagó y dos pagaron. ¿Qué responde el modelo?

[Esperar tres segundos antes de revelar.]

Responde «paga». Pero C no pagó. La votación se ejecutó correctamente y aun
así la predicción fue incorrecta. Un algoritmo puede funcionar como fue
diseñado y cometer errores.

Para estimar esos errores necesitamos datos nuevos. Con un vecino acertábamos
todos los clientes de entrenamiento, porque cada cliente era su propio vecino
a distancia cero. Eso mide cuánto recordamos esos ejemplos; no demuestra que
podamos anticipar casos distintos.

En los cinco clientes nuevos acertamos tres: sesenta por ciento.

[Mostrar la tabla y avanzar al estado 34 con entrenamiento, validación y prueba.]

En un trabajo real separaríamos tres tareas. Entrenamiento construye el modelo.
Validación ayuda a elegir k, preprocesamiento y umbrales. Prueba queda para
evaluar la configuración final. Aquí usamos un conjunto diminuto para poder
seguir cada cálculo a mano.

## Aprender una frontera y una probabilidad

**Referencia de montaje:** 14:30–19:00. **Pantalla:** estados 35–42.

Volvamos a la línea horizontal. Ya era una frontera lineal, pero sólo utilizaba
la deuda. Una recta inclinada permite que también intervenga el ingreso.

[Mover la frontera. Mostrar después la ecuación del puntaje.]

La regresión logística empieza calculando un puntaje: una constante más un
peso por el ingreso y otro peso por la deuda. Lo llamamos z.

La frontera está donde z vale cero. Su signo indica de qué lado cae el cliente.
El puntaje es proporcional a la distancia con signo; para obtener la distancia
geométrica hay que dividir por la longitud del vector de pesos.

Podemos conservar toda la información del puntaje en lugar de quedarnos sólo
con «de un lado» o «del otro». La sigmoide lo transforma en un número entre
cero y uno que el modelo interpreta como probabilidad de impago.

[Hacer viajar los clientes desde el eje de puntajes hasta la curva.]

Si z vale cero, obtenemos cero punto cinco. Si es muy positivo, la salida se
acerca a uno; si es muy negativo, a cero. Para el cliente seis obtenemos
aproximadamente cero punto setenta y seis de impago. Sin embargo, pagó.
Es la estimación del modelo, que también puede fallar; no es una certeza
sobre esa persona.

Para pasar de probabilidad a clase usamos un umbral. Con cero punto cinco,
todo lo que esté en ese valor o por encima se clasifica como «no paga».
Aprender probabilidades y decidir cuándo encender una alarma son dos pasos
distintos.

[Mostrar las bandas brevemente y pasar a las barras de pérdida.]

¿Cómo encuentra los pesos? Durante el entrenamiento penalizamos que el modelo
asigne poca probabilidad a lo que realmente ocurrió. Equivocarse con mucha
confianza recibe una penalización grande. La animación muestra cómo cambian
los pesos mientras baja esa penalización promedio, llamada log-loss.

Aquí hay un detalle importante: la logística acierta ocho de diez en
entrenamiento, mientras que el corte sencillo acertaba nueve. ¿Por qué?
Porque aprender minimizando log-loss no equivale a minimizar el número de
errores después de aplicar un umbral de cero punto cinco. Estamos optimizando
objetivos distintos.

En este ajuste, más deuda aumenta el puntaje de impago si mantenemos fijo el
ingreso. Más ingreso lo reduce si mantenemos fija la deuda. Es una descripción
de las asociaciones aprendidas con estos datos. Las fórmulas avanzadas y la interpretación con momios quedan en la edición de aula.

## Comparar decisiones y mover el umbral

**Referencia de montaje:** 19:00–22:00. **Pantalla:** estados 43–45.

Pasemos los cinco clientes nuevos por ambos modelos. Con tres vecinos
acertamos tres. Con la logística y umbral cero punto cinco, cuatro.

[Revelar las filas y después las matrices.]

Mira la diferencia concreta. Ambos dejan escapar a C: predicen «paga» y no
pagó. Tres vecinos además produce una falsa alarma con E. La logística evita
esa falsa alarma con este umbral.

En estos cinco casos acertó más la logística. Eso no demuestra que sea mejor
para cualquier problema. Una sola persona cambia la exactitud en veinte
puntos porcentuales. Para una conclusión práctica necesitaríamos más datos y
una evaluación adecuada.

Ahora imaginemos que queremos detectar más impagos. Podemos bajar el umbral.
Con cero punto dos, C, cuya probabilidad es aproximadamente cero punto
veintisiete, activa la alarma. Pero E, con alrededor de cero punto veinticinco,
también la activa, aunque sí pagó.

[Mostrar el cambio de matriz y señalar C y E.]

Detectamos los dos impagos, pero aparece una falsa alarma. La exactitud sigue
en cuatro de cinco; lo que cambia es el tipo de error. El mismo porcentaje
puede esconder decisiones muy diferentes.

Estamos usando este cambio para entender el efecto del umbral. En una
evaluación real lo elegiríamos con validación y los costos relevantes antes
de consultar la prueba final. No lo ajustaríamos mirando qué valor favorece
a estos cinco clientes.

## Ejercicio y cierre

**Referencia de montaje:** 22:00–25:00. **Pantalla:** estados 46–48.

[Señalar el rótulo «otro modelo, para practicar» antes de sustituir valores.]

Te dejo un ejercicio. El nuevo cliente tiene ingreso veinte y deuda cincuenta.
Para tres vecinos, usa las distancias de la tabla. Para practicar la logística
usaremos otro modelo, con números sencillos: z igual a menos ocho, menos cero
punto diez por el ingreso, más cero punto veinticinco por la deuda. Estos pesos
son distintos de los que ajustamos hace un momento.

¿Qué predice cada método? Y, si después sabemos que el cliente sí pagó,
¿qué tipo de error cometieron? Pausa el video y haz los cálculos.

[Mantener el ejercicio limpio durante cuatro segundos; después revelar la solución.]

Los vecinos diez, siete y ocho no pagaron. Los tres votos coinciden:
predecimos «no paga».

En el modelo simplificado, z es menos ocho, menos dos, más doce punto cinco:
dos punto cinco. La sigmoide da aproximadamente cero punto noventa y dos.
Con umbral cero punto cinco también predecimos «no paga».

Si en realidad pagó, ambos produjeron un falso positivo. Encendieron una alarma
que, en este caso, era falsa.

[Cierre con una idea por aparición.]

Clasificar es usar atributos para asignar una categoría. Los vecinos lo hacen
comparando ejemplos; la logística aprende un puntaje y lo convierte en una
probabilidad. La matriz de confusión permite entender cómo se equivocan.
Y la comparación necesita datos que no hayamos utilizado para elegir el modelo.

La pregunta inicial ya tiene una respuesta: un noventa y siete por ciento de
aciertos no basta para saber si un clasificador nos sirve. También necesitamos
saber qué detecta, qué deja escapar y cuánto cuestan esas decisiones.

Piensa ahora en un caso donde un falso negativo cueste más que un falso
positivo. ¿Qué métrica vigilarías? ¿Qué cambiaría si bajaras el umbral?
Puedes dejar tu razonamiento en los comentarios. En la siguiente clase veremos
otra manera de dividir el espacio: los árboles de decisión.
