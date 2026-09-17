# Guion docente · Misión BigMart

Clase-taller de 50 minutos sobre medidas de localización y variabilidad con
pandas, NumPy y Plotly. Este documento acompaña la presentación
`03-03-localizacion-variabilidad-bigmart.qmd` y el notebook
`notebook-alumnos.ipynb`.

## Propósito

La clase no busca que el grupo memorice métodos de pandas. Busca que pueda
pasar por esta cadena de razonamiento:

```text
pregunta → predicción → cálculo → visualización → interpretación → precaución
```

Al finalizar, cada pareja debe poder defender un dictamen de cuatro líneas:

1. qué medida usaría;
2. cuál es su valor y unidad;
3. qué evidencia visual la respalda;
4. qué limitación impide una conclusión más fuerte.

## Materiales incluidos

- `03-03-localizacion-variabilidad-bigmart.qmd`: presentación Reveal.js.
- `clase-interactiva.css`: estilos de actividades y temporizador.
- `timer.html`: temporizador local integrado en la presentación.
- `notebook-alumnos.ipynb`: práctica que los alumnos completan.
- `README.md`: instrucciones de descarga y ejecución.

El dataset se conserva en `data/bigmart_sales.csv`. El notebook lo busca en
el repositorio y, si no lo encuentra, intenta leer la copia pública de GitHub.

## Preparación antes de la clase

Desde la raíz del repositorio:

```bash
# Comprobar Quarto y Jupyter
quarto check jupyter

# Renderizar la presentación
quarto render lectures/03-eda/03-03-localizacion-variabilidad-bigmart/03-03-localizacion-variabilidad-bigmart.qmd

# Abrir el notebook para probarlo
.env/bin/jupyter lab lectures/03-eda/03-03-localizacion-variabilidad-bigmart/notebook-alumnos.ipynb
```

La salida esperada de la presentación está en:

```text
lectures/_output/03-eda/03-03-localizacion-variabilidad-bigmart/03-03-localizacion-variabilidad-bigmart.html
```

Lista de comprobación:

- [ ] La presentación abre sin internet.
- [ ] El logo FC aparece grande y centrado en la portada, y pequeño después.
- [ ] La flecha derecha recorre por pasos los bloques de código extensos.
- [ ] El temporizador aparece en la primera actividad y comienza solo.
- [ ] La primera celda del notebook muestra `(8523, 12)`.
- [ ] Plotly muestra una figura interactiva con *hover* y zoom.
- [ ] El enlace de Colab abre el notebook después de publicarlo en GitHub.
- [ ] Existe una pareja o máquina de respaldo.
- [ ] El HTML renderizado está abierto antes de que llegue el grupo.

## Uso del temporizador

Las diapositivas de actividad incluyen un tiempo específico. Al entrar en
ellas, el contador aparece en la esquina superior derecha y comienza de forma
automática.

| Acción | Control |
|---|---|
| Pausar o continuar | Botón `Ⅱ/▶` o `Alt+T` |
| Reiniciar | Botón `↺` o `Alt+R` |
| Ocultar o mostrar | `Alt+H` |
| Abrir notas del presentador | `S` |

El temporizador se vuelve ámbar en los últimos 30 segundos y rojo al terminar.
No avanza automáticamente: quien presenta decide cuándo cerrar la actividad.

En las diapositivas con código, la flecha derecha resalta primero una operación
y luego la siguiente. Cada paso agrupa líneas que cumplen una misma función;
aproveche el cambio para preguntar qué resultado intermedio esperan. El código
ya fue ejecutado al renderizar: estos avances solo controlan la explicación.

## Contrato de atención

Explíquelo al comienzo en menos de 30 segundos:

1. Nadie usa “Ejecutar todo” al iniciar.
2. Antes de cada cálculo se registra una predicción.
3. Cuando el tiempo termina, manos fuera del teclado y mirada al frente.
4. En cada pareja una persona opera y otra interpreta.
5. Los roles cambian al comenzar el reto 2.

La presentación nunca expone contenido nuevo durante más de cinco minutos
seguidos. Cada explicación desemboca en una predicción, votación o decisión.

## Ruta exacta de 50 minutos

| Minutos | Bloque | Evidencia observable |
|---:|---|---|
| 0–3 | Instalar la misión | El grupo conoce el dictamen final |
| 3–8 | Repaso pandas y Plotly | Respuesta B en la primera votación |
| 8–10 | Semáforo técnico | Todas las parejas ven `(8523, 12)` |
| 10–14 | Dos rutas con igual promedio | Distinguen centro de variabilidad |
| 14–22 | Reto 1: venta típica | Eligen media o mediana según propósito |
| 22–31 | Reto 2: extremo y Tukey | Identifican medidas sensibles y robustas |
| 31–40 | Reto 3: comparación por grupo | Interpretan centro e IQR por tipo |
| 40–46 | Giro de faltantes | Detectan pérdida estructural de grupos |
| 46–48 | Defensa en pareja | Dictamen verbal con evidencia |
| 48–50 | Exit ticket | Respuesta D con justificación |

## Guion por bloque

### 0–3 · La misión

Frase de apertura sugerida:

> “BigMart publicará mañana una cifra de venta típica. El código puede ejecutar
> perfectamente y aun así producir una conclusión equivocada. Hoy ustedes
> deciden qué firmarían y qué se negarían a afirmar.”

Presente el dictamen final. Forme parejas y asigne roles:

- **Piloto:** ejecuta y completa código.
- **Analista:** anticipa el resultado y formula la interpretación.

No explique todavía media, mediana o IQR.

### 3–8 · Repaso mínimo de herramientas

En la diapositiva de pandas, limite la explicación a dos objetos y cuatro
verbos. La frase clave es:

> “Una `Series` es la columna que analizamos; `groupby` separa, aplica y combina.”

Primera pregunta de opción múltiple:

> ¿Qué devuelve `df.groupby("Outlet_Type")["Item_Outlet_Sales"].median()`?

**Respuesta:** B, una mediana por tipo de outlet.

En Plotly, dibuje la ruta mental:

```text
DataFrame → elegir geometría → agregar referencia → etiquetar → explorar
```

No enseñe personalización estética. Demuestre únicamente *hover*, zoom y
restablecimiento de ejes. El objetivo es leer y producir evidencia.

### 8–10 · Semáforo técnico

Pida ejecutar solo la sección inicial del notebook.

- **Verde:** aparece `(8523, 12)`.
- **Amarillo:** carga, pero la forma es distinta.
- **Rojo:** error de archivo o importación.

Una persona en rojo se integra de inmediato a una pareja verde. No consuma la
clase instalando paquetes. Use el HTML renderizado si la falla es general.

### 10–14 · Calentamiento de las rutas

Ambas rutas tienen media de 40 minutos. Dé 60 segundos para votar y justificar.

Resultados:

| Ruta | Media | Desviación estándar muestral | Rango |
|---|---:|---:|---:|
| A | 40.00 | 1.41 | 4 |
| B | 40.00 | 11.47 | 30 |

Pregunta de recuperación si responden únicamente “A”:

> “¿Qué propiedad cuantitativa hace más defendible esa elección?”

Respuesta esperada: menor variabilidad o mayor consistencia. Aclare que pandas
usa `ddof=1` en `Series.std()`; NumPy usa `ddof=0` si no se especifica.

### 14–22 · Reto 1: venta típica

Dé cinco minutos de trabajo y tres de puesta en común.

Preguntas para circular:

- ¿Qué representa una fila?
- ¿Hacia dónde mueve la cola derecha a la media?
- ¿“Típico” significa repartir el total o describir una observación central?

Resultados:

- media: `2,181.29 u.m.`;
- mediana: `1,794.33 u.m.`;
- forma: cola a la derecha.

Conclusión esperada:

> Para describir una venta típica por producto–outlet usaríamos la mediana de
> 1,794.33 unidades monetarias, porque la distribución tiene una cola derecha
> que eleva la media.

Acepte la media únicamente si el equipo cambia explícitamente la pregunta a
reconstruir o proyectar el total. La lección es propósito, no una jerarquía
universal entre medidas.

### 22–31 · Reto 2: llega un extremo

Cambie los roles. El nuevo valor se agrega a una copia, no al DataFrame
original.

Resultados aproximados:

| Medida | Original | Con extremo | Cambio porcentual |
|---|---:|---:|---:|
| Media | 2,181.29 | 2,192.76 | 0.53% |
| Mediana | 1,794.33 | 1,794.33 | 0.00% |
| Rango | 13,053.67 | 99,966.71 | 665.81% |
| Desviación estándar | 1,706.50 | 2,008.57 | 17.70% |
| IQR | 2,267.05 | 2,266.72 | -0.01% |

Punto importante: la media se mueve poco porque la muestra tiene 8,523
observaciones. Eso no convierte a la media en robusta.

La regla de Tukey aplicada a ventas produce:

- límite inferior: `-2,566.33`;
- límite superior: `6,501.87`;
- candidatos: `186` (`2.18%`).

Pregunta de opción múltiple oral:

> Encontramos 186 puntos fuera de los límites. ¿Qué hacemos?

- A. Los borramos.
- B. Los sustituimos por la media.
- C. Investigamos procedencia y reglas de negocio.
- D. Concluimos fraude.

**Respuesta:** C.

### 31–40 · Reto 3: la media global oculta grupos

Pida escribir primero la pregunta: “¿Cambian el centro y la dispersión de las
ventas por tipo de outlet?”. Después se ejecuta `groupby`.

Resultados:

| Outlet_Type | n | Media | Mediana | Desv. est. | IQR |
|---|---:|---:|---:|---:|---:|
| Grocery Store | 1,083 | 339.83 | 257.00 | 260.85 | 304.94 |
| Supermarket Type1 | 5,577 | 2,316.18 | 1,990.74 | 1,515.97 | 1,984.75 |
| Supermarket Type2 | 928 | 1,995.50 | 1,655.18 | 1,375.93 | 1,721.09 |
| Supermarket Type3 | 935 | 3,694.04 | 3,364.95 | 2,127.76 | 2,931.18 |

Lectura esperada:

- Type3 tiene la mayor venta típica;
- también tiene la mayor dispersión central absoluta;
- una diferencia descriptiva no implica causalidad.

Precaución de granularidad: una fila es un producto–outlet. Type3 contiene una
sola tienda física (`OUT027`), por lo que no se está estimando la variabilidad
entre varias tiendas Type3 ni estabilidad a través del tiempo.

### 40–46 · Giro de auditoría

Plantee la afirmación que deben auditar:

> “Como las medias globales cambian poco, podemos eliminar las filas con
> `Item_Weight` faltante.”

Resultados:

| Tipo de outlet | % faltante en Item_Weight |
|---|---:|
| Supermarket Type3 | 100.00% |
| Grocery Store | 48.75% |
| Supermarket Type1 | 0.00% |
| Supermarket Type2 | 0.00% |

El filtrado elimina:

- 1,463 filas (`17.17%`);
- `OUT019` y `OUT027` completas;
- todos los registros de 1985;
- todo `Supermarket Type3`.

Conclusión esperada:

> No es defendible declarar bajo riesgo de sesgo basándose solo en que cambian
> poco las medias globales; la muestra filtrada pierde segmentos completos.

La pregunta de cierre del bloque es: “¿Qué reporte ya no podríamos producir?”.
Respuesta: cualquier comparación representativa que incluya Type3 usando casos
completos de `Item_Weight`.

### 46–48 · Defensa en pareja

Cada integrante lee el dictamen de la otra persona. La escucha debe detectar:

- una medida sin propósito;
- una cifra sin unidades;
- una gráfica descrita sin patrón;
- una afirmación que excede la evidencia.

Si escucha dos equipos, seleccione uno que haya elegido mediana y otro que haya
defendido un uso legítimo de la media.

### 48–50 · Exit ticket

Respuesta correcta: **D**.

> Medida, gráfica, granularidad y patrón de ausencia deben interpretarse juntas.

Exija una justificación de una oración. “Porque es la más completa” no basta;
deben mencionar al menos una tensión concreta, por ejemplo cola derecha,
extremos o pérdida de grupos.

## Respuestas de las preguntas de opción múltiple

| Momento | Respuesta | Idea evaluada |
|---|---|---|
| Repaso `groupby` | B | Una medida por grupo |
| Tukey | C | Candidato atípico no equivale a error |
| Exit ticket | D | Interpretación conjunta y contextual |

## Errores frecuentes y preguntas de recuperación

| Error | Pregunta del docente |
|---|---|
| “La mediana siempre es mejor” | ¿Qué medida permite recuperar el total con `n`? |
| Comparar solo medias | ¿Qué dicen IQR o desviación estándar? |
| Borrar los 186 candidatos | ¿Qué evidencia demuestra que son errores? |
| Llamar “tiendas” a las 8,523 filas | ¿Cuál es la unidad de observación? |
| Decir que Type3 es estable en el tiempo | ¿Dónde está la variable temporal repetida? |
| Concluir que `dropna()` es seguro | ¿Qué grupos desaparecen después del filtro? |
| Obtener otra desviación con NumPy | ¿Qué `ddof` usa cada biblioteca? |
| Promediar medias de grupos | ¿Los grupos tienen el mismo tamaño? |

## Plan de contingencia

### Sin internet

El dataset existe dentro del repositorio y la presentación se renderiza con
recursos incrustados. Use la copia HTML ya preparada.

### Sin Jupyter en algunas máquinas

Forme parejas con una máquina funcional. Mantenga la predicción y la
interpretación como participación obligatoria para ambas personas.

### Sin Jupyter en todo el laboratorio

Proyecte el HTML con resultados y convierta cada celda en una predicción. El
grupo puede completar el dictamen en papel. No intente instalar durante la
sesión.

### Solo quedan 40 minutos

- reduzca el repaso a una diapositiva;
- muestre el código completo del reto 2 y conserve la predicción;
- no elimine el giro de faltantes ni el exit ticket.

### Sobran cinco minutos

Use la actividad opcional del notebook:

- encontrar la moda de `Item_Type`;
- reconstruir la media global como promedio ponderado de medias de grupo;
- explicar por qué el promedio simple de las cuatro medias es incorrecto.

## Criterio rápido de logro

Asigne un punto por cada elemento del dictamen:

| Criterio | Evidencia |
|---|---|
| Medida apropiada | La elección responde a una pregunta explícita |
| Número interpretable | Incluye valor y unidad |
| Evidencia visual | Describe cola, centro o dispersión observable |
| Precaución | Reconoce granularidad, atípicos o faltantes |

Con 3 de 4 puntos, el objetivo esencial de la sesión está cubierto. Con 4 de 4,
el equipo conectó correctamente cálculo, visualización y contexto.
