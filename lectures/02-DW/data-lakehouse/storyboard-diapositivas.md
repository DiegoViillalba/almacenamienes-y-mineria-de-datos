<!--
Storyboard en Markdown. No es un QMD ni una presentación renderizable todavía.
Cada separador horizontal `---` marca el inicio de una nueva diapositiva.
El apartado "Contenido visible" sí pertenece a la diapositiva.
El apartado "Especificación del diagrama" contiene instrucciones de producción y no debe mostrarse como texto en la diapositiva final.
-->

# 01 · De silos a Lakehouse

## Contenido visible

**De silos a Lakehouse**

Cuándo vale la pena cambiar la arquitectura de datos

**Caso de estudio:** fintech de pagos móviles y préstamos digitales

## Especificación del diagrama — no mostrar como texto

- **Tipo:** portada conceptual dividida.
- **Composición:** a la izquierda, dos torres aisladas: `Data Warehouse` y `Data Lake`, unidas por varias flechas rojas de copia. A la derecha, una base azul común con una capa morada de tablas y catálogo; arriba aparecen `BI`, `ML` y `Fraude` como tres consumidores.
- **Jerarquía:** el título ocupa el tercio superior; el contraste arquitectónico ocupa los dos tercios inferiores.
- **Color:** Warehouse azul oscuro, Lake azul claro, copias rojas, metadata morada y consumidores verdes.
- **Revelado sugerido:** primero los silos; después la arquitectura compartida.
- **Texto alternativo:** «Comparación entre dos sistemas aislados que intercambian copias y una arquitectura Lakehouse con almacenamiento común, tablas gobernadas y varios consumidores».

---

# 02 · La pregunta que guía la clase

## Contenido visible

> ¿Cuándo compartir una base abierta y gobernada para BI, streaming y ML justifica operar un Lakehouse?

```text
beneficio de compartir datos
            vs.
responsabilidad de operar la plataforma
```

## Especificación del diagrama — no mostrar como texto

- **Tipo:** balanza de decisión.
- **Composición:** platillo izquierdo con `menos copias`, `datos abiertos`, `BI + ML + streaming`; platillo derecho con `compactación`, `gobierno`, `contratos`, `observabilidad`.
- **Estado:** la balanza debe quedar equilibrada, no inclinada hacia Lakehouse. La elección todavía está abierta.
- **Mensaje visual:** no existe un ganador universal; la respuesta depende del caso.
- **Texto alternativo:** «Una balanza contrapone los beneficios de compartir datos y las responsabilidades adicionales de operar un Lakehouse».

---

# 03 · Una transacción, dos respuestas

## Contenido visible

**Transacción `tx_8472` · $3,200 · 10:02**

- Finanzas: **incluida en ingresos**
- Fraude: **bloqueada**

¿Cómo pueden ser correctos ambos procesos y producir verdades distintas?

## Especificación del diagrama — no mostrar como texto

- **Tipo:** flujo bifurcado.
- **Origen:** tarjeta central izquierda `tx_8472 · aprobada · 10:02`.
- **Ruta superior:** `réplica → ELT → Warehouse → Finanzas`, con sello de actualización `08:00`.
- **Ruta inferior:** `Kafka → archivos → modelo → Fraude`, con sello `10:02:03`.
- **Franja entre rutas:** `copias · desfase · reglas distintas` en rojo.
- **Flechas:** sólidas para el movimiento de datos.
- **Revelado sugerido:** evento, resultado de Finanzas, resultado de Fraude y finalmente las dos rutas.
- **Texto alternativo:** «La misma transacción viaja por un pipeline batch hacia Finanzas y por otro de tiempo real hacia Fraude, produciendo respuestas distintas por desfase, copias o reglas».

---

# 04 · Dos sistemas especializados

## Contenido visible

| Data Warehouse | Data Lake |
|---|---|
| SQL y reportes | Datos crudos y diversos |
| Esquema controlado | Almacenamiento económico |
| Alto desempeño analítico | Flexibilidad para ingeniería y ML |
| Menor flexibilidad | Sin transacciones de tabla por defecto |

**El problema no es que uno sea malo; es operarlos como mundos separados.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** dos tarjetas simétricas.
- **Tarjeta izquierda:** edificio ordenado `Data Warehouse`, con iconos de tabla, SQL y tablero.
- **Tarjeta derecha:** lago o conjunto de objetos `Data Lake`, con iconos de JSON, Parquet, PDF e imagen.
- **Centro:** espacio vacío o muro punteado con el texto `frontera organizacional y técnica`.
- **Evitar:** representar al Lake como desorden inevitable o al Warehouse como tecnología obsoleta.
- **Texto alternativo:** «Warehouse y Lake aparecen como sistemas con fortalezas diferentes, separados por una frontera que dificulta compartir definiciones y procesos».

---

# 05 · El impuesto de los silos

## Contenido visible

Cada copia agrega:

- otro esquema;
- otro control de calidad;
- otra política de acceso;
- sincronización;
- riesgo de usar versiones diferentes.

> La deuda no está sólo en los bytes duplicados, sino en las decisiones duplicadas.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** dos carriles con una factura lateral.
- **Carril superior:** `fuentes → ETL → Warehouse → BI`.
- **Carril inferior:** `eventos/documentos → ingesta → Lake → ML`.
- **Cruces:** flecha roja `Lake → Warehouse: preparar para BI` y flecha roja `Warehouse → ML: exportar`.
- **Factura lateral:** cinco fichas apiladas con los costos visibles de la lista.
- **Revelado sugerido:** carriles, cruces y factura final.
- **Texto alternativo:** «Dos pipelines especializados intercambian datos mediante copias y acumulan esquemas, controles, permisos y sincronización duplicados».

---

# 06 · Fintech, años 0–3

## Contenido visible

- **10,000 transacciones diarias**
- **≈500 GB por año**
- Datos relacionales
- Reportes de ingresos, morosidad y crecimiento

```text
PostgreSQL → Fivetran → BigQuery/Snowflake → dbt → Metabase/Tableau
```

## Especificación del diagrama — no mostrar como texto

- **Tipo:** pipeline lineal sencillo.
- **Nodos:** base operacional, ingesta, Warehouse, transformación SQL y tablero.
- **Etiquetas de rol debajo de cada nodo:** `opera`, `replica`, `almacena/consulta`, `modela`, `muestra`.
- **Diseño:** amplio espacio vacío para comunicar simplicidad; una sola ruta sin bifurcaciones.
- **Revelado sugerido:** construir el pipeline de izquierda a derecha.
- **Texto alternativo:** «Los datos relacionales de la fintech pasan de PostgreSQL por Fivetran a BigQuery o Snowflake, dbt los modela y Metabase o Tableau los presenta».

---

# 07 · ¿Qué hace cada producto?

## Contenido visible

| Producto | Responsabilidad en este ejemplo |
|---|---|
| PostgreSQL | Registrar pagos, usuarios y cuentas |
| Fivetran | Replicar datos |
| BigQuery o Snowflake | Guardar tablas analíticas y ejecutar SQL |
| dbt | Transformar y probar con SQL |
| Metabase o Tableau | Mostrar métricas |

**Una marca no es la arquitectura completa.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** cadena de tarjetas con “credencial de rol”.
- **Composición:** cinco tarjetas alineadas; cada una muestra nombre comercial arriba y responsabilidad en una banda inferior.
- **Conectores:** flechas simples de izquierda a derecha.
- **Énfasis:** resaltar BigQuery como `Warehouse en esta configuración`.
- **Nota visual:** agregar una pequeña etiqueta `el rol puede cambiar según la configuración` sin introducir todavía BigLake.
- **Texto alternativo:** «Cinco productos aparecen asociados a responsabilidades diferentes: operación, ingesta, Warehouse, transformación y visualización».

---

# 08 · Por qué el Warehouse era correcto

## Contenido visible

- Configuración aproximada: **una semana**
- Equipo: **un analista de datos**
- Costo: **cientos de dólares al mes**
- Problema dominante: **reporting SQL**

**Un Lakehouse todavía no ofrecía retorno suficiente para pagar su operación.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** tarjeta de decisión con sellos.
- **Centro:** el pipeline de la diapositiva 06 en miniatura.
- **Sellos verdes:** `rápido`, `simple`, `equipo pequeño`, `costo alineado`.
- **A un lado:** caja gris `Lakehouse` con una etiqueta `capacidad futura, operación inmediata` y un signo de interrogación.
- **Mensaje:** la arquitectura inicial es una decisión racional, no una etapa inferior.
- **Texto alternativo:** «El Warehouse recibe sellos de rapidez, simplicidad y costo apropiado, mientras el Lakehouse aparece como complejidad aún sin retorno claro».

---

# 09 · Año 4: cambió el problema

## Contenido visible

- **5 millones de usuarios**
- **4 TB al mes** de eventos y telemetría
- Kafka y fraude en tiempo real
- PDFs, OCR, biometría e imágenes
- ML necesita terabytes
- Copias y costos crecientes

## Especificación del diagrama — no mostrar como texto

- **Tipo:** tubo central sometido a seis presiones.
- **Centro:** el pipeline angosto de la fase inicial.
- **Cuñas alrededor:** `volumen`, `velocidad`, `variedad`, `nuevos consumidores`, `movimiento de datos`, `economía`.
- **Etiquetas numéricas:** `4 TB/mes ≈ 48 TB/año antes de crecimiento y réplicas` y `5 millones de usuarios`.
- **Color:** las cuñas pasan de naranja a rojo conforme se acumulan.
- **Texto alternativo:** «El pipeline inicial queda presionado simultáneamente por mayor volumen, streaming, datos diversos, ML, copias y costo».

---

# 10 · No elijamos una marca todavía

## Contenido visible

¿Qué requisito intentamos resolver?

1. Escalar el Warehouse
2. Añadir un Data Lake separado
3. Crear una base compartida tipo Lakehouse

**La elección depende del cuello de botella, no de la moda.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** bifurcación de tres caminos.
- **Origen:** caja `Año 4: reporting + streaming + ML + documentos`.
- **Camino 1:** `más capacidad SQL`, con fortaleza `operación sencilla`.
- **Camino 2:** `Lake separado`, con fortaleza `almacenamiento económico` y riesgo `dos autoridades`.
- **Camino 3:** `Lakehouse`, con fortaleza `datos compartidos` y costo `mayor disciplina operativa`.
- **Estado:** ningún camino debe aparecer marcado aún como ganador.
- **Texto alternativo:** «Tres alternativas arquitectónicas parten del mismo problema y muestran una ventaja y un costo distintos».

---

# 11 · Objeto, bucket y key

## Contenido visible

- **Objeto:** bytes + nombre/*key* + metadatos
- **Bucket:** contenedor con nombre para guardar objetos
- **Key:** localizador único dentro del bucket

```text
s3://fintech-datos-prod/documents/loan-8472.pdf
│            │                 └─ key del objeto
│            └─ bucket
└─ servicio
```

## Especificación del diagrama — no mostrar como texto

- **Tipo:** URI anotada.
- **Composición:** la URI ocupa el centro con tres llaves o llamadas que identifican servicio, bucket y key.
- **Abajo:** un objeto como tarjeta con tres compartimentos: `datos`, `key`, `metadata`.
- **Ejemplos de metadata:** `application/pdf`, `680 KB`, `PII`, `cifrado`.
- **Evitar:** dibujar el bucket como una tabla o base relacional.
- **Texto alternativo:** «Una dirección S3 se descompone en servicio, nombre del bucket y key; el objeto contiene datos y metadatos».

---

# 12 · Anatomía de un bucket

## Contenido visible

**Bucket: `fintech-datos-prod`**

```text
bronze/kafka/2026/08/25/part-0001.json
documents/loan-8472.pdf
silver/transactions/part-0042.parquet
models/fraud/v12/model.bin
```

Los prefijos ayudan a organizar; el bucket conserva objetos, no filas ni `JOIN`.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** contenedor con objetos heterogéneos.
- **Contenedor:** rectángulo azul `fintech-datos-prod`, región y política de acceso en el encabezado.
- **Objetos internos:** cuatro tarjetas con iconos JSON, PDF, Parquet y modelo.
- **Prefijos:** colorear `bronze/`, `documents/`, `silver/`, `models/` y unirlos a la etiqueta `organización mediante nombres/prefijos`.
- **Banda inferior roja:** `no conoce por sí solo esquema de tabla, joins ni commit multiarchivo`.
- **Equivalencias pequeñas:** `AWS: S3 bucket`, `Google: Cloud Storage bucket`, `Azure: container/file system`.
- **Texto alternativo:** «Un bucket contiene objetos JSON, PDF, Parquet y modelos identificados por keys; sus prefijos organizan, pero no crean tablas ni ejecutan SQL».

---

# 13 · Bucket, Data Lake y Lakehouse

## Contenido visible

```text
Bucket/Container
  + ingesta, organización, seguridad y catálogo
  = Data Lake

Data Lake
  + formato de tabla, gobierno y motores coordinados
  = Lakehouse
```

## Especificación del diagrama — no mostrar como texto

- **Tipo:** cajas anidadas o crecimiento por capas.
- **Centro:** caja pequeña `Bucket/Container: guarda objetos`.
- **Segunda envolvente:** `Data Lake`, con ingesta, zonas, seguridad y catálogo básico.
- **Tercera envolvente:** `Lakehouse`, con formato de tabla transaccional, gobierno y varios motores.
- **Flechas:** una flecha ascendente muestra adición de capacidades, no sustitución física.
- **Mensaje:** un bucket es componente de infraestructura; Lake y Lakehouse son arquitecturas.
- **Texto alternativo:** «El bucket está contenido dentro de un Data Lake, y el Data Lake recibe capacidades adicionales de tablas y gobierno para formar un Lakehouse».

---

# 14 · Una base física compartida

## Contenido visible

En almacenamiento de objetos pueden convivir:

- JSON, logs y eventos;
- PDFs e imágenes;
- archivos Parquet/ORC;
- resultados procesados.

**Ejemplos:** Amazon S3, Google Cloud Storage, Azure Data Lake Storage Gen2 y OneLake.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** convergencia de fuentes a una base común.
- **Arriba:** `PostgreSQL/CDC`, `Kafka`, `logs`, `PDFs`, `imágenes`.
- **Centro:** gran base azul de almacenamiento dividida en `crudo`, `columnar`, `refinado`.
- **Abajo:** consumidores potenciales `SQL`, `Spark/Ray`, `streaming`, todavía separados por signos de interrogación.
- **Ventajas a la izquierda:** `escala`, `costo por capacidad`, `formatos abiertos`.
- **Preguntas a la derecha:** `¿qué versión?`, `¿qué esquema?`, `¿commit completo?`, `¿quién puede leer?`.
- **Texto alternativo:** «Fuentes diversas convergen en almacenamiento de objetos; la ubicación queda unificada, pero todavía faltan semántica de tabla, versión y permisos».

---

# 15 · El bucket no sabe qué es una tabla

## Contenido visible

```text
transactions/
  part-0001.parquet
  part-0002.parquet
  part-0003.parquet
  part-0003-retry.parquet
  _temporary-part-0004.parquet
```

¿Qué archivos pertenecen a la tabla?

**Una carpeta y varios Parquet no constituyen un protocolo de commit.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** explorador de objetos con ambigüedad.
- **Composición:** panel estilo listado; archivos normales azules, reintento naranja y temporal rojo.
- **Lateral:** lector SQL con tres globos de duda: `¿duplicado?`, `¿temporal?`, `¿escritura completa?`.
- **Mensaje:** el nombre de la ruta no determina de forma confiable la versión de la tabla.
- **Texto alternativo:** «Una ruta contiene archivos normales, un reintento y un temporal; sin metadata de tabla el lector no puede identificar de manera confiable la versión válida».

---

# 16 · Cuatro responsabilidades diferentes

## Contenido visible

1. **S3 / GCS / ADLS:** guarda objetos
2. **Parquet / ORC:** organiza columnas en un archivo
3. **Iceberg / Delta / Hudi:** coordina archivos como tabla
4. **Catálogo:** encuentra, describe y gobierna la tabla

> El motor pregunta arriba y lee los datos abajo.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** pila vertical de cuatro capas.
- **Base azul:** almacenamiento de objetos.
- **Segunda capa azul claro:** formato de archivo.
- **Tercera capa morada:** formato de tabla con `snapshots · esquema · particiones · commits`.
- **Capa superior morado oscuro:** catálogo con `nombre · ubicación · propietario · permisos`.
- **Motores a la derecha:** `Trino`, `Spark`, `motor SQL`; flecha punteada al catálogo y flecha sólida al almacenamiento.
- **Texto alternativo:** «Una pila diferencia almacenamiento, archivo, tabla y catálogo; los motores consultan metadata y después leen los objetos».

---

# 17 · Sin protocolo: una escritura parcial se vuelve visible

## Contenido visible

Estado inicial: `A B C`

La escritura intenta agregar: `D E F`

Falla después de `E`.

El lector encuentra: `A B C D E`

**¿La actualización está completa? No puede saberlo.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** historieta de cuatro pasos.
- **Paso 1:** archivos `A B C` azules.
- **Paso 2:** job escribe `D` y `E` en naranja; `F` aparece punteado.
- **Paso 3:** rayo rojo `falla` antes de F.
- **Paso 4:** lector enumera la carpeta y recibe cinco archivos mezclados.
- **Flechas:** secuencia temporal horizontal.
- **Texto alternativo:** «Una escritura multiarchivo falla después de publicar dos de tres archivos y el lector observa un estado parcial sin saber si es válido».

---

# 18 · Con snapshots: publicar todo o nada

## Contenido visible

```text
actual → snapshot 17 → A B C
```

Los nuevos archivos se escriben sin publicarse.

- Si falla: los lectores conservan `snapshot 17`
- Si tiene éxito: `actual → snapshot 18 → A B C D E F`

## Especificación del diagrama — no mostrar como texto

- **Tipo:** doble línea de tiempo, falla y éxito.
- **Estado inicial:** puntero morado `actual` conectado a `snapshot 17`, que enumera `A B C`.
- **Rama de falla:** `D E` aparecen grises como no publicados; el puntero no cambia.
- **Rama de éxito:** después de escribir `D E F`, un único movimiento cambia el puntero a `snapshot 18`.
- **Nota pequeña:** `archivos no referenciados → limpieza posterior`.
- **Énfasis:** el commit visible es el cambio de versión, no la escritura individual de cada objeto.
- **Texto alternativo:** «Los nuevos archivos permanecen fuera de la versión visible hasta que un commit mueve el puntero del snapshot 17 al 18; una falla deja visible la versión anterior».

---

# 19 · Evolución de esquema

## Contenido visible

```text
Snapshot 41 · esquema v1
amount [field 2]

Snapshot 42 · esquema v2
gross_amount [field 2]
```

La identidad del campo permanece; cambia su nombre lógico.

**El soporte depende del formato, la configuración y los motores.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** línea temporal de dos snapshots.
- **Arriba:** v1 con `customer_id [1]` y `amount [2]`.
- **Flecha central:** `rename lógico`.
- **Derecha:** v2 con `customer_id [1]` y `gross_amount [2]`.
- **Abajo:** archivos históricos siguen mostrando `amount`; un mapa morado conecta `field 2` con el nombre vigente.
- **Advertencia:** no sugerir que cualquier cambio de tipo o semántica es gratuito.
- **Texto alternativo:** «La metadata conserva la identidad del campo 2 mientras su nombre lógico cambia, permitiendo leer archivos históricos sin reescribirlos cuando existe soporte compatible».

---

# 20 · El catálogo es el mapa central

## Contenido visible

```text
fintech.silver.transactions
ubicación: s3://fintech/silver/transactions/
metadata vigente: v018
propietario: Risk Data Team
clasificación: PII · financiera
```

El motor consulta el catálogo y después lee los objetos autorizados.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** resolución de nombre.
- **Izquierda:** motores `SQL`, `Spark`, `streaming`.
- **Centro:** ficha grande de catálogo con los campos visibles.
- **Derecha:** almacenamiento de objetos y metadata de tabla.
- **Flechas:** punteadas de motores a catálogo; sólidas de motores a objetos después de resolver ubicación y permisos.
- **Analogía visual:** mapa o índice de biblioteca, sin convertir el catálogo en el lugar donde viven todos los datos.
- **Texto alternativo:** «Varios motores resuelven el nombre de una tabla mediante un catálogo que contiene ubicación, versión, propietario y clasificación antes de leer sus objetos».

---

# 21 · Gobierno: quién puede ver qué

## Contenido visible

| Identidad | Acceso ilustrativo |
|---|---|
| Finanzas | Gold de ingresos; PII enmascarada |
| Fraude | Transacciones y señales de dispositivo |
| Data Science | Silver seudonimizada |
| Auditoría | Historial y logs de acceso |

**Ejemplos:** Unity Catalog; AWS Glue Data Catalog + Lake Formation.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** matriz de identidades contra activos.
- **Columnas:** `Gold financiero`, `Silver transacciones`, `señales de dispositivo`, `documentos PII`, `historial`.
- **Filas:** las cuatro identidades.
- **Celdas:** permiso completo, enmascarado, seudonimizado o denegado; usar icono y texto, no sólo color.
- **Banda superior:** `identidad · permisos · linaje · auditoría`.
- **Mensaje:** conocer la ruta física no equivale a tener autorización.
- **Texto alternativo:** «Una matriz muestra que Finanzas, Fraude, Data Science y Auditoría reciben permisos diferentes sobre tablas, señales, documentos e historial».

---

# 22 · Fintech, año 5: arquitectura Lakehouse

## Contenido visible

```text
fuentes diversas
→ ingesta batch y streaming
→ almacenamiento de objetos
→ tablas Bronze, Silver y Gold
→ BI, ML y fraude
```

**Una base gobernada; motores especializados.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** arquitectura integral horizontal.
- **Fuentes:** `transacciones/CDC`, `Kafka`, `logs`, `PDFs/OCR`, `imágenes/biometría`.
- **Ingesta:** dos entradas `batch/CDC` y `streaming`.
- **Base compartida:** contenedor S3/GCS/ADLS con franjas `Bronze`, `Silver`, `Gold`.
- **Llave morada:** abraza tablas tabulares con `Iceberg/Delta/Hudi: snapshots, esquema, transacciones`.
- **Binarios:** PDFs e imágenes permanecen como objetos; una tabla `document_index` apunta a ellos mediante URI y clasificación.
- **Banda superior:** `catálogo + gobierno + linaje + calidad`.
- **Consumidores:** tres motores separados para `SQL/BI`, `Spark/Ray/ML` y `streaming/fraude`.
- **Texto alternativo:** «Datos transaccionales, eventos y documentos llegan a almacenamiento de objetos; tablas Bronze, Silver y Gold están gobernadas y son usadas por motores separados de BI, ML y fraude».

---

# 23 · Mismos contratos, distintos consumos

## Contenido visible

- **Finanzas:** productos Gold certificados
- **ML:** entidades Silver y objetos autorizados
- **Fraude:** señales gobernadas y flujo en tiempo real

No es “una tabla para todo”.

Es una base común con contratos, versiones y responsables compartidos.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** tres carriles de consumo desde una base común.
- **Centro izquierdo:** tabla Gold, tablas Silver y objetos/document index.
- **Carril superior:** motor SQL → tablero financiero.
- **Carril medio:** Spark/Ray → entrenamiento ML.
- **Carril inferior:** motor de streaming → decisión de fraude.
- **Interruptores separados:** cada motor muestra escala y costo independientes.
- **Tachaduras:** `exportar terabytes al portátil` y `copiar para cada consumidor`.
- **Texto alternativo:** «Finanzas, ML y Fraude usan productos diferentes de una base gobernada mediante motores independientes, sin exportaciones masivas».

---

# 24 · Tecnologías por responsabilidad

## Contenido visible

| Responsabilidad | Ejemplos |
|---|---|
| Ingesta | Fivetran, Airbyte, Kafka, Kinesis, Pub/Sub |
| Warehouse | BigQuery, Snowflake, Redshift, Fabric Warehouse |
| Objetos | S3, Cloud Storage, ADLS Gen2, OneLake |
| Tabla | Iceberg, Delta Lake, Hudi |
| Catálogo/gobierno | Glue + Lake Formation, Unity Catalog, BigLake, Fabric/Purview |
| Cómputo | Athena, Trino, BigQuery, Databricks SQL, Spark |
| Consumo | Tableau, Power BI, Looker, Metabase |

## Especificación del diagrama — no mostrar como texto

- **Tipo:** mapa de capas con fichas de productos.
- **Columna izquierda fija:** responsabilidades en el orden `ingerir → guardar → coordinar tabla → catalogar/gobernar → procesar → consumir`.
- **Derecha:** chips con nombres de productos; no usar logotipos grandes.
- **Leyenda:** borde punteado para proyectos/formatos abiertos; borde sólido para servicios administrados.
- **Advertencias visibles al pie:** `S3 no es el Lakehouse completo`; `Iceberg/Delta/Hudi no son nubes`; `un producto puede cubrir más de una responsabilidad`.
- **Texto alternativo:** «Un mapa relaciona las responsabilidades estables de una plataforma con ejemplos abiertos y comerciales, sin equiparar una marca con toda la arquitectura».

---

# 25 · Tres patrones completos

## Contenido visible

### Warehouse

`PostgreSQL → Fivetran → BigQuery → dbt → Looker`

### Data Lake consultable

`Kafka → S3/Parquet → Glue → Athena/Spark`

### Lakehouse

`Kafka → S3 → Iceberg → Glue/Lake Formation → Athena/EMR`

## Especificación del diagrama — no mostrar como texto

- **Tipo:** tres filas comparables.
- **Alineación:** cada fila usa las mismas columnas: `fuente`, `ingesta`, `almacenamiento`, `administración`, `cómputo`, `consumo`.
- **Warehouse:** Warehouse administrado ocupa almacenamiento, tablas y SQL.
- **Data Lake:** S3/Parquet y catálogo, pero mostrar una pieza vacía `commit/snapshot de tabla`.
- **Lakehouse:** llenar esa pieza con Iceberg y añadir gobierno.
- **Énfasis:** la diferencia visual principal entre las últimas dos filas es la administración transaccional de tablas, no solamente S3.
- **Texto alternativo:** «Tres pipelines completos comparan un Warehouse, un Data Lake consultable sin formato transaccional y un Lakehouse con Iceberg, catálogo, gobierno y varios motores».

---

# 26 · Cuatro implementaciones de Lakehouse

## Contenido visible

| Ecosistema | Piezas principales |
|---|---|
| AWS | S3 + Iceberg + Glue/Lake Formation + Athena/EMR/Redshift |
| Databricks | S3/GCS/ADLS + Delta + Unity Catalog + SQL/Spark |
| Microsoft Fabric | OneLake + Delta + Fabric Lakehouse + SQL/Spark/Power BI |
| Google Cloud | Cloud Storage + BigLake/Iceberg + BigQuery/Spark |

**Las marcas cambian; las responsabilidades permanecen.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** cuatro mini-pilas verticales.
- **Cada pila:** base de almacenamiento, formato de tabla, catálogo/gobierno y motores.
- **Alineación:** las cuatro capas deben quedar a la misma altura para facilitar la comparación.
- **No afirmar:** equivalencia exacta entre servicios o soporte idéntico de todas las operaciones.
- **Nota:** BigQuery puede aparecer como Warehouse con tablas nativas o como motor sobre datos externos; destacar el rol de esta configuración.
- **Texto alternativo:** «AWS, Databricks, Microsoft Fabric y Google Cloud implementan las mismas responsabilidades de almacenamiento, formato de tabla, gobierno y cómputo con productos diferentes».

---

# 27 · El impuesto de los archivos pequeños

## Contenido visible

Streaming continuo → miles de archivos pequeños → consultas más lentas

**Compactar:** combinar archivos y publicar un nuevo snapshot.

También hay que:

- expirar snapshots;
- limpiar huérfanos;
- actualizar estadísticas;
- revisar partición y clustering.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** antes, mantenimiento y después.
- **Antes:** sesenta fichas azules pequeñas `1–5 MB`; un cronómetro rojo y muchas operaciones de apertura.
- **Centro:** engrane morado `compactar`, que lee pequeños, escribe grandes y publica un snapshot.
- **Después:** cuatro bloques grandes con cronómetro verde.
- **Banda inferior:** las otras cuatro tareas operativas como iconos independientes; no sugerir que compactar las reemplaza.
- **Texto alternativo:** «Numerosos archivos pequeños elevan la planificación y las aperturas; la compactación publica menos archivos grandes, pero siguen siendo necesarias retención, limpieza y estadísticas».

---

# 28 · Compartir datos también comparte el riesgo

## Contenido visible

Una tabla Silver alimenta:

- cierre financiero;
- características de fraude;
- entrenamiento de ML.

Un cambio semántico incompatible puede afectar a los tres.

**Defensas:** contratos, pruebas, compatibilidad, despliegue gradual, linaje y recuperación.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** nodo central con radio de impacto.
- **Centro:** `transactions v18`.
- **Salidas verdes:** Finanzas, Fraude y ML.
- **Cambio rojo:** `amount cambia de semántica sin versión compatible`; onda roja hacia los tres consumidores.
- **Anillo defensivo:** seis segmentos con las defensas indicadas.
- **Mensaje:** eliminar copias reduce inconsistencias, pero concentra dependencia.
- **Texto alternativo:** «Una tabla compartida alimenta tres consumidores; un cambio incompatible se propaga, mientras un anillo de contratos, pruebas y recuperación limita el impacto».

---

# 29 · La arquitectura se elige por el cuello de botella

## Contenido visible

| Criterio | Warehouse | Lake + Warehouse | Lakehouse |
|---|---|---|---|
| Datos | Relacionales | Usos separados | Diversos y compartidos |
| Prioridad | SQL rápido | Aislar economías | Reducir copias |
| Operación | Menor carga | Sincronización doble | Gobierno y mantenimiento |
| Riesgo | Rigidez/costo | Dos verdades | Radio de impacto común |

## Especificación del diagrama — no mostrar como texto

- **Tipo:** matriz de decisión.
- **Marcadores:** ficha `Años 0–3` en Warehouse y ficha `Año 5` en Lakehouse.
- **Tercera ficha punteada:** `otra empresa puede decidir distinto`.
- **Color:** no usar verde para toda una columna; sólo resaltar el ajuste entre una fase y sus requisitos.
- **Nota visual:** añadir debajo `beneficio esperado > carga operativa` como condición de la transición.
- **Texto alternativo:** «Una matriz compara tres arquitecturas; la fase inicial de la fintech se ajusta al Warehouse y la fase de escala al Lakehouse, sin declarar un ganador universal».

---

# 30 · Resultado de la transición

## Contenido visible

- Almacenamiento masivo alineado con economía de objetos
- Menos exportaciones y copias
- BI, fraude y ML sobre datos gobernados
- Cómputo independiente por carga
- Tres ingenieros sostienen capacidades con retorno directo

**No garantiza que toda consulta sea más barata.**

## Especificación del diagrama — no mostrar como texto

- **Tipo:** antes y después con métricas cualitativas.
- **Antes:** copias rojas entre Warehouse, Lake y portátiles; una factura ascendente.
- **Después:** base común con tres motores y una factura de almacenamiento descendente, mientras aparece una nueva ficha de costo `ingeniería y operación`.
- **Balance:** mostrar beneficios verdes y responsabilidades moradas en la misma escala.
- **Evitar:** porcentajes inventados o una promesa de ahorro universal.
- **Texto alternativo:** «La transición reduce movimiento y alinea el almacenamiento, pero reemplaza parte del costo por nuevas responsabilidades de ingeniería y operación».

---

# 31 · La idea completa

## Contenido visible

```text
silos y copias
→ base común en objetos
→ archivos coordinados por snapshots
→ catálogo y permisos
→ motores distintos sobre contratos compartidos
→ mantenimiento y control del impacto
→ elección condicionada por el caso
```

> El Lakehouse no elimina la complejidad: la mueve.

## Especificación del diagrama — no mostrar como texto

- **Tipo:** cadena causal de siete estaciones.
- **Composición:** camino de izquierda a derecha o serpiente en dos filas; cada estación usa un icono simple y el texto exacto.
- **Transición cromática:** rojo en silos, azul en almacenamiento, morado en metadata/gobierno, gris en motores y naranja en operación/decisión.
- **Final:** no colocar trofeo; usar una balanza para recordar que la elección sigue condicionada.
- **Texto alternativo:** «Una secuencia resume el paso de silos a almacenamiento común, snapshots, catálogo, motores compartidos y responsabilidades operativas antes de decidir».

---

# 32 · Comprobación final

## Contenido visible

Completa las frases:

1. Un bucket con Parquet no es todavía una tabla porque…
2. El catálogo sirve para…, mientras el formato de tabla sirve para…
3. La migración de la fintech se justifica cuando…, pero obliga a…

## Especificación del diagrama — no mostrar como texto

- **Tipo:** tres tarjetas de respuesta vacía.
- **Composición:** una tarjeta por frase, con espacio visual para respuesta; no mostrar soluciones en la vista inicial.
- **Iconos:** bucket/archivo, mapa/snapshot y balanza/engranes.
- **Revelado opcional posterior:**
  1. `falta una versión transaccional del conjunto de archivos`;
  2. `el catálogo encuentra y gobierna; el formato coordina esquema, snapshots y commits`;
  3. `compartir datos diversos a escala supera el costo; exige mantenimiento, contratos y gobierno`.
- **Texto alternativo:** «Tres frases incompletas comprueban la distinción entre bucket y tabla, catálogo y formato, y beneficio y responsabilidad del Lakehouse».
