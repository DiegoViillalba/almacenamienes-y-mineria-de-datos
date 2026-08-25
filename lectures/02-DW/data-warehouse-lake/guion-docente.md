# Guion docente · Construyamos un Data Warehouse

Clase-taller de 60 minutos. La presentación contiene notas por diapositiva; este documento funciona como hoja de ruta y plan de contingencia.

## Resultado observable

Al finalizar, el grupo debe poder explicar y mostrar una cadena completa:

```text
pregunta de negocio → grano → esquema estrella → ETL → pruebas → consulta OLAP
```

Además, debe distinguir entre “apareció un JSON” y “existen requisitos que justifican evaluar una lakehouse”.

## Progresión didáctica

| Tiempo | Propósito | Acción del profesor | Acción del grupo |
|---:|---|---|---|
| 0–3 | Instalar la misión | Presentar la pregunta rectora | Reconocer el producto final |
| 3–10 | Activar conocimientos previos | Facilitar tres preguntas | Votar y justificar |
| 10–15 | Traducir negocio a modelo | Definir venta neta y preguntar por el grano | Comparar granos posibles |
| 15–22 | Contrastar OLTP y OLAP | Dibujar fuente normalizada y estrella | Ubicar hechos, dimensiones y medidas |
| 22–24 | Elegir modalidad | Abrir notebook local o dar enlace Colab | Preparar entorno o predecir salidas |
| 24–40 | Construir | Ejecutar por etapas, pausando antes de cada salida | Predecir, comprobar, explicar |
| 40–45 | Validar | Provocar la pregunta “¿cómo sabemos que está bien?” | Conciliar filas, importes y claves |
| 45–51 | Consultar | Ejecutar *roll-up* y proponer *drill-down* | Interpretar el resultado |
| 51–58 | Crear tensión con datos diversos | Mostrar esquema cambiante en JSON | Proponer cómo preservar y refinar |
| 58–60 | Recuperar evidencia | Solicitar *exit ticket* | Responder tres frases |

## Preparación antes de entrar al salón

Desde la raíz del repositorio:

```bash
# Render local de la presentación
quarto render lectures/02-DW/data-warehouse-lake/02-data-warehouse-en-vivo.qmd

# Notebook local; el directorio .env ya es el entorno virtual del proyecto
.env/bin/jupyter lab lectures/02-DW/data-warehouse-lake/laboratorio-data-warehouse.ipynb
```

Abra también la presentación renderizada en:

```text
lectures/_output/02-DW/data-warehouse-lake/02-data-warehouse-en-vivo.html
```

La presentación usa `embed-resources: true`; después de renderizar, ese HTML puede abrirse sin internet. El notebook sólo usa la biblioteca estándar de Python (`sqlite3`, `json`, `csv`, `pathlib`, `tempfile`).

## Dos modalidades de facilitación

### A. Demostración y pizarrón

Úsela si no hay red, si el grupo no trae computadora o si los teléfonos fragmentan la atención.

1. Dibuje primero la fuente OLTP.
2. No muestre el esquema estrella terminado: constrúyalo con respuestas del grupo.
3. Antes de cada celda pregunte qué tabla, cantidad o error se espera.
4. Invite a una persona a conciliar manualmente un renglón.
5. Mantenga visibles tres contratos: `7 filas`, `$702.50`, `0 huérfanas`.

Presione `S` en Reveal para abrir las notas del presentador. El pizarrón físico se usa para reconstruir el modelo antes de revelar cada solución.

### B. Seguimiento en Colab desde teléfono o computadora

Enlace:

<https://colab.research.google.com/github/DiegoViillalba/almacenamienes-y-mineria-de-datos/blob/main/lectures/02-DW/data-warehouse-lake/laboratorio-data-warehouse.ipynb>

Pida ejecutar una sección a la vez, no “Ejecutar todas” al inicio. En teléfono, la meta es observar y responder las predicciones; escribir SQL largo no es requisito. El enlace funcionará cuando el notebook exista en la rama pública `main` de GitHub.

## Respuestas que conviene hacer explícitas

- **Ventaja OLAP vs OLTP:** OLAP está diseñado para lecturas históricas, agregaciones y navegación multidimensional; no es “más rápido” para cualquier operación.
- **Escalamiento vertical:** más capacidad en una máquina; operación sencilla y techo físico/económico.
- **Escalamiento horizontal:** distribución entre nodos; elasticidad a cambio de partición, red y coordinación.
- **Grano:** un renglón de pedido completado. Permite reagrupar después sin perder categoría o producto.
- **Claves sustitutas:** desacoplan el almacén de los identificadores operacionales y habilitan integración e historia.
- **Pedido cancelado:** queda fuera por definición de venta neta; la regla debe documentarse y probarse.
- **Lakehouse:** no es una consecuencia automática de JSON. Se evalúa ante datos diversos a escala, conservación del crudo, evolución de esquema, múltiples cargas y necesidad de tablas gobernadas sobre archivos.

## Errores frecuentes y cómo convertirlos en preguntas

| Error | Pregunta de recuperación |
|---|---|
| Elegir “una fila por pedido” | ¿Qué ocurre si un pedido mezcla categorías? |
| Usar precio de lista actual | ¿Podemos reconstruir lo realmente cobrado hace seis meses? |
| Declarar éxito porque el código terminó | ¿Coinciden filas, importes y claves con la fuente? |
| Cargar pedidos cancelados | ¿Qué significa exactamente “venta” para Finanzas? |
| Afirmar que JSON exige lakehouse | ¿Qué volumen, consumidores, gobierno y conservación requiere el caso? |
| Presentar horizontal como gratuito | ¿Quién paga partición, red, coordinación y observabilidad? |

## Plan de contingencia

- **Sin internet:** use el HTML ya renderizado y el notebook local. Ningún dato se descarga.
- **Sin Jupyter:** proyecte las salidas esperadas incluidas en las diapositivas y ejecute el notebook más tarde; el diseño y las predicciones siguen siendo la actividad central.
- **Falla en una validación:** no salte el error. Compare cantidad en fuente, filtro de estado, cálculo de neto y carga de claves.
- **Sólo 45 minutos:** haga una sola pregunta de repaso, construya dimensiones y hecho en el pizarrón, ejecute el notebook completo y conserve validación + cierre lakehouse.
- **Sobran 5 minutos:** pida un *drill-down* de venta neta por ciudad o diseñar el tratamiento de un producto desconocido.

## Criterio rápido de logro

El grupo logró el objetivo si puede reconstruir, sin memorizar código:

1. qué decisión responde el DW;
2. cuál es el evento atómico;
3. qué distingue hechos de dimensiones;
4. cómo se concilia el destino con la fuente;
5. por qué los requisitos de datos diversos abren la conversación sobre lakehouse.
