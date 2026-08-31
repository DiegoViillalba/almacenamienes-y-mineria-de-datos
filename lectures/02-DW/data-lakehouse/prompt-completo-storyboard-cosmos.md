# Prompt corto para Gemini · Presentación de slides

Copie este bloque en Gemini y pegue el contenido de `storyboard-diapositivas.md` al final.

```text
Actúa como diseñador experto en presentaciones educativas y visualización de arquitectura de datos.

TU TAREA PRINCIPAL ES CREAR Y RENDERIZAR DIRECTAMENTE UNA PRESENTACIÓN NATIVA DE SLIDES a partir del storyboard Markdown incluido al final. La entrega debe ser la presentación visual terminada, no una explicación de cómo construirla.

NO generes QMD, Quarto, Reveal.js, HTML, CSS, código fuente ni otro archivo Markdown. Usa el entorno nativo de presentaciones de Gemini para crear todas las diapositivas, diagramas, tablas, formas, conectores y textos.

## Cómo interpretar el storyboard

- Cada `# NN · Título` inicia una diapositiva.
- Cada `---` termina esa diapositiva.
- `## Contenido visible` contiene lo que debe ver el alumno.
- `## Especificación del diagrama — no mostrar como texto` contiene instrucciones para construir el diagrama. Interprétalas, pero no las pegues en la diapositiva.
- `Texto alternativo` debe usarse como descripción accesible, no como texto visible.
- Los encabezados `##` internos no crean diapositivas nuevas.
- El storyboard tiene 32 diapositivas. No omitas, fusiones ni resumas ninguna.
- Si algo no cabe de forma legible, divide la diapositiva como `NN-A` y `NN-B`, conservando todo el contenido y su orden.

## Identidad visual Cosmos

- Formato 16:9, fondo blanco `#ffffff` y margen mínimo de 40 px.
- Texto principal negro o gris oscuro `#111111`.
- Títulos arriba a la izquierda, 36–44 pt, negrita, Azul Cosmos `#0b4a6f`.
- Texto, viñetas, tablas y etiquetas principales: mínimo 20 pt.
- Subtítulos de tarjetas: 24–28 pt.
- Notas al pie: 14–16 pt, cursiva, abajo a la derecha.
- Tarjetas: fondo blanco o `#f8fafc`, borde de 1–2 px `#5c7f9b` y radio de 12 px.
- Usa `#4fb6d6` para badges y acentos editoriales.
- Mantén alto contraste, buen espacio vertical y ningún elemento fuera del lienzo.

## Colores dentro de los diagramas

La presentación conserva Cosmos, pero los diagramas usan colores semánticos:

- Azul `#0b4a6f`: datos y almacenamiento.
- Morado `#6b21a8`: metadatos, formatos de tabla, catálogo y gobierno.
- Verde `#067647`: resultado correcto o publicación exitosa.
- Naranja `#b54708`: ingesta, transición o escritura en proceso.
- Rojo `#b42318`: falla, riesgo, duplicación o inconsistencia.

Usa fondos tenues para esos colores y acompáñalos con etiquetas, iconos o patrones; no dependas solamente del color.

## Diagramas

- Crea todos los diagramas solicitados con formas, texto y conectores nativos editables.
- Conserva exactamente los nodos, capas, relaciones y direcciones descritos.
- Flecha sólida: movimiento o lectura de datos.
- Flecha punteada: consulta de metadatos, permisos o coordinación.
- Usa flujos para movimiento, pilas para capas, matrices para permisos, líneas de tiempo para snapshots, cajas anidadas para bucket/Data Lake/Lakehouse y antes/después para compactación.
- Evita cruces de flechas, texto diminuto, saturación y decoración sin función pedagógica.
- No uses imágenes generadas por IA para diagramas con texto técnico.
- No uses logos grandes ni colores corporativos de proveedores.

Respeta estas distinciones técnicas:

- Un bucket guarda objetos; no ejecuta SQL y no es por sí solo un Data Lake.
- Parquet/ORC son formatos de archivo.
- Iceberg/Delta/Hudi coordinan tablas, archivos y snapshots.
- El catálogo localiza y gobierna; no almacena necesariamente todos los datos.
- Los motores consultan metadatos y después leen los objetos.
- PDFs e imágenes permanecen como objetos; una tabla puede guardar sus referencias.
- El Data Warehouse es la decisión correcta durante la primera fase de la fintech.

## Traducción del contenido

- Convierte listas en viñetas con al menos 12 px de separación.
- Convierte tablas en tablas o matrices legibles.
- Convierte citas en tarjetas destacadas.
- Convierte flujos escritos en bloques monoespaciados en diagramas cuando mejore la comprensión.
- Renderiza correctamente cualquier ecuación.
- No inventes cifras, productos, capacidades ni afirmaciones.

## Validación obligatoria

Antes de entregar:

1. Renderiza la presentación completa.
2. Comprueba que las 32 diapositivas estén representadas y en orden.
3. Revisa que las instrucciones de producción no aparezcan como texto visible.
4. Corrige desbordamientos, recortes, solapamientos, flechas ambiguas y texto menor de 20 pt.
5. Confirma que todos los diagramas sean claros a distancia y respeten la paleta definida.

Entrega directamente la presentación de slides terminada y renderizada. Indica brevemente cuántas diapositivas finales contiene y cuáles se dividieron, sin sustituir la presentación por una respuesta textual.

--- INICIO DEL STORYBOARD MARKDOWN ---

{{PEGA_AQUÍ_EL_STORYBOARD_MARKDOWN}}

--- FIN DEL STORYBOARD MARKDOWN ---
```
