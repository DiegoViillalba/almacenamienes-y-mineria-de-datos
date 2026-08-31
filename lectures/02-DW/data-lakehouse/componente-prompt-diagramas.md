# Componente de prompt para interpretar el storyboard

Este bloque puede copiarse dentro de una solicitud futura para convertir `storyboard-diapositivas.md` a QMD, Reveal.js, HTML, PowerPoint u otro formato. Sustituya los valores entre `{{...}}` cuando sea necesario.

## Bloque reutilizable

```text
<protocolo_de_interpretacion_visual>

Objetivo
Convierte el storyboard proporcionado en una presentación visual precisa, legible y coherente. El storyboard ya define el contenido de cada diapositiva y la intención de sus diagramas. Tu trabajo no es reinventar la arquitectura ni añadir teoría: debes materializar fielmente las relaciones descritas.

Fuente
- Storyboard: {{RUTA_O_CONTENIDO_DEL_STORYBOARD}}
- Formato de salida: {{FORMATO_DE_SALIDA}}
- Relación de aspecto: 16:9
- Idioma visible: español
- Audiencia: estudiantes que pueden no conocer almacenamiento de objetos, Data Lakes ni Lakehouses.

Cómo leer el storyboard
1. Cada encabezado `# NN · Título` inicia una diapositiva.
2. Cada separador `---` termina una diapositiva.
3. `## Contenido visible` contiene el único texto que debe aparecer como contenido principal de la diapositiva.
4. `## Especificación del diagrama — no mostrar como texto` son instrucciones de producción. Interprétalas y conviértelas en un diagrama; nunca pegues esos párrafos dentro de la diapositiva.
5. `Texto alternativo` debe implementarse como descripción accesible del diagrama, no como un párrafo visible.
6. `Revelado sugerido` define el orden de fragmentos o animaciones. Si el formato no admite animación, produce un estado final que mantenga el mismo orden de lectura.

Orden de interpretación de cada diagrama
1. Identifica primero el mensaje conceptual que debe comprenderse en menos de cinco segundos.
2. Extrae todos los nodos, grupos, capas, estados y etiquetas que la especificación menciona explícitamente.
3. Extrae cada relación y conserva su dirección: movimiento de datos, consulta de metadata, contención, secuencia temporal, comparación o dependencia.
4. Selecciona la gramática visual adecuada:
   - flujo para movimiento de datos;
   - pila para capas y responsabilidades;
   - matriz para permisos o comparaciones exactas;
   - línea de tiempo para snapshots y evolución;
   - cajas anidadas para bucket, Data Lake y Lakehouse;
   - antes/después para compactación y migración;
   - balanza para decisiones y trade-offs.
5. Aplica jerarquía visual y estilo sólo después de que las relaciones sean correctas.
6. Verifica el diagrama contra el texto alternativo: ambos deben contar la misma historia.

Reglas de fidelidad
- Conserva los títulos, cifras, etiquetas técnicas y nombres de producto escritos en el storyboard.
- No agregues capacidades, equivalencias entre proveedores, porcentajes, costos o benchmarks que no estén especificados.
- No transformes una comparación condicionada en una afirmación de que Lakehouse siempre es mejor.
- No conviertas marcas en capas arquitectónicas. Muestra siempre la responsabilidad que cada producto cumple en ese ejemplo.
- Si una marca puede cumplir más de un rol, representa sólo el rol indicado en esa diapositiva.
- Si una descripción y una preferencia estética entran en conflicto, prioriza en este orden: significado, relaciones, etiquetas, accesibilidad y finalmente decoración.
- Si el diagrama resulta demasiado denso, divide visualmente el contenido mediante grupos o paneles. No reduzcas el texto hasta hacerlo ilegible y no elimines una relación necesaria.

Guardas conceptuales obligatorias
- Bucket/Container: guarda objetos; no ejecuta SQL y no es por sí solo un Data Lake.
- Data Lake: arquitectura alrededor del almacenamiento; no es sinónimo de un bucket.
- Parquet/ORC: formatos de archivo; no proporcionan por sí solos un commit de tabla multiarchivo.
- Iceberg/Delta/Hudi: formatos o protocolos de tabla; coordinan conjuntos de archivos y sus versiones.
- Catálogo: resuelve nombre, ubicación, esquema, metadata y gobierno; no debe dibujarse como si almacenara todos los bytes.
- Motor: consulta el catálogo/metadata y lee los objetos; almacenamiento y cómputo son responsabilidades distintas.
- PDFs e imágenes: permanecen como objetos. Una tabla puede guardar sus URI, hashes, clasificación y resultados; no los dibujes como si fueran archivos Parquet administrados directamente por Iceberg o Delta.
- Snapshot: una escritura fallida no mueve el puntero de la versión visible. Los archivos no referenciados pueden existir y requieren limpieza posterior.
- Las garantías se expresan normalmente por tabla; no sugieras transacciones globales entre todas las tablas.
- Data Warehouse: sigue siendo una opción válida; en la fase inicial de la fintech es la elección correcta.

Sistema visual común
- Azul: datos, archivos y almacenamiento.
- Morado: metadata, formatos de tabla, catálogo y gobierno.
- Verde: consumo confiable, publicación o resultado correcto.
- Naranja: ingesta, transición o escritura todavía no publicada.
- Rojo: falla, duplicación, riesgo o costo evitable.
- Gris: motores de cómputo y herramientas intercambiables.
- No dependas únicamente del color. Añade texto, forma, patrón o icono que permita distinguir cada estado.
- Flecha sólida: movimiento o lectura de datos.
- Flecha punteada: consulta de metadata, permisos o coordinación.
- Mantén la misma semántica de color y flechas en toda la presentación.

Composición y legibilidad
- Diseña para 16:9 y conserva una zona segura mínima del 5 % en los cuatro bordes.
- Mantén una sola idea principal por diapositiva.
- Usa un orden de lectura evidente: izquierda a derecha, arriba abajo o centro hacia ramas.
- Evita cruces de flechas. Si son inevitables, usa puentes, carriles o agrupaciones claras.
- Los títulos deben dominar; el texto de nodos debe ser legible desde el fondo de un salón.
- No comprimas tablas o diagramas mediante tipografía diminuta. Simplifica decoración o divide paneles antes de reducir tamaño.
- No uses párrafos dentro de nodos. Convierte explicaciones en etiquetas breves conservando la terminología esencial.
- Usa alineaciones, tamaños y espacios consistentes para que elementos equivalentes parezcan equivalentes.
- En comparaciones entre proveedores, alinea las mismas responsabilidades en la misma altura.
- Evita logos grandes, capturas de consola o imágenes decorativas que compitan con la relación conceptual.

Elección de tecnología visual
- Prefiere diagramas nativos, editables y accesibles: HTML/CSS, SVG o componentes del formato de destino.
- Para QMD/Reveal.js, usa HTML/CSS/SVG cuando se requiera control exacto de capas, flechas, estados o fragmentos.
- Usa Mermaid sólo para flujos simples donde produzca una composición legible y permita las etiquetas requeridas.
- No uses una imagen raster generada por IA para diagramas con texto técnico exacto.
- No dependas de recursos de red para iconos esenciales; el resultado debe poder renderizarse de forma reproducible.
- Las tablas verdaderas deben permanecer como tablas accesibles cuando comuniquen mejor la relación que un dibujo.

Animación y fragmentos
- Cada revelado debe introducir una sola relación o cambio conceptual.
- No animes por decoración.
- Mantén visibles los elementos anteriores cuando sean necesarios para entender el siguiente paso.
- En la escritura fallida: estado inicial → archivos nuevos → falla → lectura ambigua.
- En snapshots: snapshot vigente → archivos preparados → rama de falla → commit exitoso.
- En el Lakehouse completo: fuentes → ingesta → almacenamiento → formato/catálogo → consumidores.
- En decisiones: presenta las alternativas antes de mostrar dónde cae cada fase de la fintech.

Accesibilidad
- Implementa el `Texto alternativo` en el mecanismo accesible del formato de destino.
- No dupliques el texto alternativo de forma visible si repite el diagrama.
- Conserva contraste suficiente entre texto y fondo.
- Acompaña color con etiquetas, formas o patrones.
- Mantén el orden del DOM equivalente al orden visual y narrativo.
- Las flechas y conectores deben tener etiquetas cuando su significado no sea obvio.

Control de calidad obligatorio
1. Genera o actualiza la presentación completa.
2. Renderiza el resultado en su formato final.
3. Inspecciona visualmente todas las diapositivas a 16:9; presta atención especial a las diapositivas 12, 16, 18, 22, 24, 25, 26, 27, 28 y 29.
4. Comprueba que no existan desbordamientos, recortes, solapamientos, flechas ambiguas ni texto demasiado pequeño.
5. Comprueba que todos los separadores produzcan exactamente una diapositiva y que la numeración sea continua.
6. Comprueba que las descripciones de producción no aparezcan como texto visible.
7. Comprueba la consistencia semántica de colores y flechas.
8. Verifica que las versiones sin animación y la vista de impresión sigan siendo comprensibles.
9. Corrige y vuelve a renderizar hasta que estas validaciones pasen.

Contrato de salida
- Entrega la presentación completa en {{FORMATO_DE_SALIDA}}.
- Conserva el archivo de storyboard como fuente conceptual; no lo sobrescribas salvo solicitud explícita.
- Indica qué archivos creaste o modificaste.
- Resume cualquier división visual necesaria, pero no omitas contenido sin señalarlo.
- Informa que el render y la revisión visual fueron completados, o especifica claramente qué impidió verificarlos.

</protocolo_de_interpretacion_visual>
```

## Ejemplo de uso mínimo

```text
Convierte el storyboard en una presentación QMD con Reveal.js.

<protocolo_de_interpretacion_visual>
[PEGAR AQUÍ EL BLOQUE ANTERIOR]

Fuente:
- Storyboard: lectures/02-DW/data-lakehouse/storyboard-diapositivas.md
- Formato de salida: QMD con Reveal.js
</protocolo_de_interpretacion_visual>

Usa los estilos existentes del repositorio cuando sean compatibles. Renderiza, inspecciona visualmente y corrige la presentación antes de entregarla.
```

## Qué consigue este componente

- Separa contenido visible de instrucciones de producción.
- Evita que un agente trate la descripción del diagrama como texto para pegar.
- Mantiene las relaciones técnicas correctas entre almacenamiento, archivo, tabla, catálogo y motor.
- Define una gramática visual estable para las 32 diapositivas.
- Exige renderizado e inspección, no sólo generación de código.
- Mantiene accesibilidad, legibilidad y reproducibilidad como condiciones de entrega.
