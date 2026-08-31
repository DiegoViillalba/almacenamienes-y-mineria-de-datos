# Guion docente · De silos a Lakehouse: cuándo vale la pena cambiar

Clase conceptual de 60 minutos basada en `unidades/almacenes/1-1-datalakehouse.qmd`. Este documento define la narración, las preguntas al grupo y la especificación visual de los diagramas. No es todavía una presentación Reveal.js.

## Propósito de la clase

La clase no busca vender el Lakehouse como reemplazo universal del Data Warehouse. Su pregunta rectora es:

> **¿En qué momento compartir una base abierta y gobernada para BI, streaming y ML justifica asumir la operación adicional de un Lakehouse?**

Al finalizar, el alumnado debe poder:

1. explicar por qué un Data Warehouse fue la elección correcta durante la primera etapa de la fintech;
2. localizar la duplicación, la latencia y los controles dobles de una arquitectura con silos;
3. explicar qué son un objeto y un *bucket*, y distinguir almacenamiento de objetos, formato de archivo, formato de tabla, catálogo, gobierno y motor de cómputo;
4. narrar cómo un *snapshot* evita que una escritura incompleta se vuelva visible;
5. justificar la transición de la fintech usando volumen, variedad, velocidad, cargas de trabajo, costo y capacidad operativa;
6. reconocer que compactación, evolución de esquemas y radio de impacto siguen siendo responsabilidades del equipo.

## Idea que debe sobrevivir a la clase

```text
Lakehouse = almacenamiento de objetos
          + formato de tabla transaccional
          + catálogo y gobierno
          + motores especializados sobre datos compartidos
          + operación continua
```

No es “un lago junto a un almacén” ni “guardar Parquet en S3”. Es hacer que conjuntos de archivos se comporten como tablas confiables sin perder la apertura y la separación entre almacenamiento y cómputo.

## Vocabulario mínimo para un grupo que empieza

No asumir que el alumnado conoce la infraestructura de nube. Estas definiciones deben aparecer cuando se use cada término por primera vez, no como examen de memorización al inicio.

| Término | Explicación cotidiana | Ejemplo en la fintech | Qué no significa |
|---|---|---|---|
| **Objeto** | Una unidad de datos guardada en la nube: contiene bytes, un nombre o *key* y metadatos. | Un PDF de contrato, una imagen o un archivo Parquet. | No es necesariamente una fila ni una tabla. |
| **Bucket** | Un contenedor con nombre donde un proveedor de nube guarda objetos. | `s3://fintech-datos-prod/` contiene objetos de transacciones, logs y documentos. | No es un motor SQL, una base relacional ni todo el Data Lake. |
| **Key o nombre del objeto** | El localizador único del objeto dentro del bucket. Puede parecer una ruta. | `bronze/kafka/2026/08/25/part-0001.json`. | Los separadores `/` no garantizan que exista una carpeta tradicional; con frecuencia son prefijos de organización. |
| **Data Lake** | Una arquitectura para conservar y procesar datos diversos sobre almacenamiento económico, con organización, ingesta, seguridad y gobierno. | Buckets/containers con eventos, documentos, Parquet, políticas y procesos. | Un bucket aislado no se convierte automáticamente en Data Lake. |
| **Data Warehouse** | Un sistema analítico orientado a tablas, SQL y datos preparados para decisiones. | BigQuery o Snowflake con tablas de ingresos y morosidad. | No es la base operacional que autoriza cada pago. |
| **Motor de cómputo** | El software que realiza el trabajo: consulta, transforma, agrupa o entrena. | BigQuery, Trino, Databricks SQL o Spark. | El almacenamiento de objetos no procesa una consulta por sí solo. |
| **Formato de archivo** | La manera de organizar los bytes dentro de un archivo. | Parquet guarda columnas y compresión. | No coordina por sí mismo toda una tabla con muchos archivos. |
| **Formato de tabla** | El protocolo que decide qué archivos pertenecen a una tabla y conserva cambios, esquema y snapshots. | Iceberg, Delta Lake o Hudi. | No sustituye por sí solo al almacenamiento, al catálogo o al motor. |
| **Catálogo** | El mapa que relaciona un nombre de tabla con ubicación, esquema, propietario y metadata vigente. | `fintech.silver.transactions` apunta a una tabla en S3. | No es solamente una lista de archivos ni garantiza calidad sin procesos y responsables. |

### Analogía breve que puede repetirse

> “El bucket es el contenedor; los objetos son los paquetes; la *key* es la etiqueta que permite localizarlos. Parquet organiza el contenido de ciertos paquetes. Iceberg, Delta o Hudi mantienen el inventario versionado que dice qué paquetes forman una tabla. El catálogo permite pedir esa tabla por nombre y verificar si podemos usarla. El motor es quien abre los paquetes y hace el cálculo.”

La analogía sirve para separar responsabilidades, pero no debe extenderse demasiado: en la nube los objetos no son cajas físicas y los prefijos que parecen carpetas pueden tener semánticas distintas según el servicio.

## Criterios de precisión que debe conservar la futura presentación

- Los costos y volúmenes del caso son supuestos didácticos; no son cotizaciones de proveedores.
- `4 TB/mes` equivale aproximadamente a `48 TB/año`. La ruta hacia petabytes supone crecimiento y retención de varios años; no se debe afirmar que 4 TB mensuales ya son un petabyte.
- El problema de una escritura sobre objetos no es que S3, GCS o Azure Blob necesariamente “pierdan consistencia”. El problema es que una modificación compuesta por varios archivos necesita un **commit de tabla** atómico.
- Parquet u ORC son formatos de **archivo**. Iceberg, Delta Lake y Hudi son formatos/protocolos de **tabla** que coordinan muchos archivos y sus versiones.
- PDFs, imágenes y otros binarios permanecen como objetos. Las tablas pueden gobernar sus metadatos, resultados extraídos y referencias; no conviene sugerir que una fotografía “se convierte en Iceberg”.
- Las garantías ACID suelen expresarse por tabla. No se debe prometer automáticamente una transacción atómica entre muchas tablas.
- La evolución de esquemas y la interoperabilidad exacta dependen del formato, su configuración y el soporte de cada motor.
- Un Data Warehouse moderno puede ser elástico, admitir datos semi-estructurados y consultar almacenamiento externo. La decisión se toma por requisitos y economía, no por una caricatura tecnológica.

## Mapa de los 60 minutos

| Minutos | Bloque | Pregunta que mueve la narración | Evidencia esperada |
|---:|---|---|---|
| 0–4 | Apertura | ¿Cómo puede una transacción tener dos verdades? | El grupo identifica copia, desfase o definición distinta. |
| 4–11 | El problema de los silos | ¿Qué trabajo se duplica al mantener Warehouse y Lake separados? | Menciona pipelines, esquemas, calidad, seguridad y sincronización. |
| 11–17 | Fintech, años 0–3 | ¿Por qué no comenzar directamente con un Lakehouse? | Defiende simplicidad, SQL, tiempo a valor y bajo costo operativo. |
| 17–24 | Punto de quiebre | ¿Qué cambió realmente en el año 4? | Clasifica volumen, velocidad, variedad, ML, copias y costo. |
| 24–30 | Almacenamiento único | ¿Qué resuelve el bucket y qué no resuelve? | Separa costo/apertura de semántica/transacciones. |
| 30–39 | Formatos de tabla | ¿Cómo pasan archivos sueltos a ser una tabla confiable? | Explica metadatos, snapshot, commit y evolución de esquema. |
| 39–45 | Catálogo y gobierno | ¿Cómo sabe cada motor qué leer y qué puede ver? | Distingue mapa de tablas, versión vigente y políticas de acceso. |
| 45–51 | Fintech Lakehouse | ¿Cómo usan BI, ML y fraude los mismos datos sin un único motor? | Identifica cómputo desacoplado y contratos compartidos. |
| 51–56 | Trade-offs | ¿Qué trabajo nuevo aparece al compartir la plataforma? | Compactación, mantenimiento, contratos, observabilidad y costos. |
| 56–60 | Decisión y cierre | ¿Qué arquitectura elegirían en cada fase y por qué? | Argumento condicionado, no una respuesta por moda. |

## Convención visual para todos los diagramas

Usar la misma semántica de color durante toda la sesión:

- **Azul:** datos y almacenamiento.
- **Morado:** metadatos, formatos de tabla y catálogo.
- **Verde:** consumo confiable o resultado publicado.
- **Naranja:** ingesta y movimiento.
- **Rojo:** duplicación, falla, riesgo o costo evitable.
- **Gris:** cómputo y herramientas intercambiables.

Las flechas sólidas indican movimiento o lectura de datos. Las flechas punteadas indican consulta de metadatos, permisos o coordinación. Ningún diagrama debe depender únicamente del color: cada caja y flecha necesita texto o icono. Para las animaciones, revelar máximo un cambio conceptual por paso.

---

# Guion minuto a minuto

## 0–4 min · Apertura: una transacción, dos respuestas

### Diagrama 1 · «La transacción de las 10:02»

**Propósito.** Crear un problema concreto antes de introducir vocabulario arquitectónico.

**Composición visual.** Lienzo horizontal dividido en tres zonas:

1. A la izquierda, una tarjeta de evento con el texto `tx_8472 · $3,200 · 10:02` y una etiqueta pequeña `aprobada`.
2. Arriba a la derecha, una pantalla de Finanzas: `Ingresos del día: incluye tx_8472`, con sello de actualización `08:00`.
3. Abajo a la derecha, una pantalla del equipo de Fraude: `tx_8472: bloqueada`, con sello `10:02:03`.

Desde la transacción salen dos rutas. La ruta superior pasa por tres cajas naranjas: `réplica → ELT → Warehouse`. La inferior pasa por `Kafka → archivos → modelo`. Entre ambas rutas hay una franja roja con las palabras `copias`, `latencia`, `reglas distintas`. No presentar todavía el Lakehouse.

**Secuencia de revelado.**

1. Mostrar solamente la transacción y preguntar si existe una única realidad operacional.
2. Revelar Finanzas y Fraude con resultados distintos.
3. Revelar las dos rutas y sus horarios de actualización.
4. Revelar la franja roja después de escuchar hipótesis.

**Texto alternativo.** «Una misma transacción viaja por dos pipelines separados. Finanzas la ve en un Warehouse con una actualización anterior; Fraude la procesa en tiempo real desde archivos. Las copias y reglas separadas pueden producir respuestas distintas».

### Narración sugerida

> “No empecemos por una definición. Esta fintech tiene una transacción de 3,200 pesos a las 10:02. Finanzas la cuenta; el sistema de fraude la bloquea. Los dos equipos pueden haber ejecutado correctamente su propio proceso. Entonces, ¿cómo terminamos con dos verdades?”

Recoger tres respuestas sin corregir de inmediato. Escribir en el pizarrón: `desfase`, `copia`, `regla`. Cerrar el gancho:

> “El Lakehouse intenta reducir estas fronteras, pero sólo vale la pena si el problema es suficientemente grande para pagar su complejidad. Hoy decidiremos cuándo ocurre eso.”

---

## 4–11 min · El escenario tradicional: el problema de los silos

### Diagrama 2 · «Dos caminos que deben fingir que son uno»

**Propósito.** Mostrar las fortalezas legítimas de cada sistema y localizar el costo en la frontera, no declarar que uno de los dos es “malo”.

**Composición visual.** Dos carriles paralelos de izquierda a derecha:

- Carril superior, azul oscuro: `Sistemas transaccionales → Fivetran/ETL → Data Warehouse → Finanzas y BI`. Debajo del Warehouse: `SQL rápido · esquema controlado · reportes`.
- Carril inferior, azul claro: `Apps/Kafka/PDFs/imágenes → ingesta → Data Lake → ciencia de datos y ML`. Debajo del Lake: `objetos económicos · datos crudos · flexibilidad`.

En el centro, entre carriles, colocar dos flechas rojas diagonales y bidireccionales:

- `Lake → Warehouse: preparar y copiar para BI`.
- `Warehouse → entorno ML: exportar y volver a copiar`.

A la derecha, una columna roja titulada `Impuesto de coordinación` con cinco fichas apiladas: `dos esquemas`, `dos controles de calidad`, `dos políticas`, `sincronización`, `versiones distintas`.

En la parte inferior debe aparecer una nota: `Silo = sistemas con contratos, ciclos y autoridades separadas; no significa que jamás intercambien datos`.

**Animación.** Revelar primero cada carril y pedir una fortaleza. Después dibujar las flechas cruzadas una por una. Finalmente acumular las cinco fichas como si fueran una factura.

**Texto alternativo.** «Un pipeline lleva datos relacionales al Warehouse para BI; otro lleva datos diversos al Data Lake para ML. Para cruzar usos, la organización copia datos en ambas direcciones y duplica esquemas, controles, permisos y sincronización».

### Narración sugerida

> “El Warehouse resuelve muy bien una clase de problema: tablas controladas, SQL y reportes. El Lake resuelve otra: conservar grandes volúmenes y tipos diversos a bajo costo. El conflicto histórico no es poseer dos cajas; es sostener dos caminos que deben representar el mismo negocio.”

Usar un cambio concreto:

> “Producto agrega `device_risk_score`. ¿En cuántos lugares debe acordarse su significado para que Finanzas y Fraude puedan usarlo?”

### Pregunta al grupo

Pedir que enumeren el recorrido del cambio. Respuesta que se debe consolidar:

1. capturar el campo;
2. cambiar el esquema de aterrizaje;
3. cambiar la transformación del lago;
4. cambiar la tabla del Warehouse;
5. duplicar pruebas de calidad;
6. sincronizar el momento de publicación;
7. actualizar permisos y consumidores.

Remate:

> “La deuda no está únicamente en los bytes duplicados. Está en las decisiones duplicadas.”

---

## 11–17 min · Caso Fintech, fase 1: el Data Warehouse era suficiente

### Diagrama 3 · «Años 0–3: una arquitectura pequeña para un problema pequeño»

**Propósito.** Evitar que la historia se interprete como una escalera de madurez donde Lakehouse siempre es el destino correcto.

**Composición visual.** Un flujo lineal, con bastante espacio vacío alrededor:

`PostgreSQL / servicios de pagos` → `Fivetran` → `Snowflake o BigQuery` → `dbt` → `Metabase / Tableau`.

Sobre el flujo, cuatro tarjetas numéricas:

- `10,000 transacciones/día`.
- `≈500 GB/año`.
- `datos relacionales`.
- `reportes por lotes`.

Debajo, tres sellos verdes: `1 semana de configuración`, `1 analista`, `cientos USD/mes`. En una esquina, una caja gris deliberadamente vacía que diga `¿Lakehouse? Operación sin retorno claro todavía`.

**Animación.** Construir el pipeline en una sola dirección. Revelar los sellos verdes al preguntar por qué funcionaba. Mostrar la caja de Lakehouse al final.

**Texto alternativo.** «Durante los primeros tres años, datos relacionales y de volumen moderado fluyen mediante Fivetran a un Warehouse, dbt los transforma y BI los consume. La operación requiere poco personal y tiene costo controlado».

### Narración sugerida

> “Con 10,000 transacciones diarias y unos 500 GB al año, el problema dominante no era almacenar cualquier cosa a escala masiva. Era conocer ingresos, morosidad y crecimiento rápidamente. Un servicio administrado, dbt y un analista entregaron valor en una semana.”

Preguntar:

> “Si fuéramos el equipo fundador, ¿qué obtendríamos al contratar tres ingenieros para operar un Lakehouse desde el primer año?”

Respuesta a conducir: quizá flexibilidad futura, pero todavía sin un caso que compense personal, observabilidad, mantenimiento e integración. Concluir:

> “Elegir el Warehouse aquí no fue falta de visión. Fue ajustar la arquitectura al problema y al equipo.”

### Pausa de traducción: ¿qué hace cada producto en esta fase?

Dedicar alrededor de 60 segundos. No presentar las marcas como una sola “caja de datos”. Señalar cada una sobre el diagrama 3:

- **PostgreSQL, MySQL o una base operacional administrada** registra usuarios, cuentas y pagos de la aplicación. Está optimizada para que el negocio opere, no para escanear años de historia.
- **Fivetran** es un servicio comercial de ingesta: extrae o replica datos desde las fuentes hacia el destino. **Airbyte** es otra alternativa habitual con oferta abierta y administrada.
- **BigQuery**, **Snowflake**, **Amazon Redshift** y **Microsoft Fabric Warehouse** son ejemplos de servicios que pueden cumplir el rol de Data Warehouse analítico. No es necesario contratar los cuatro: son alternativas de plataforma.
- **dbt** ejecuta transformaciones y pruebas principalmente mediante SQL sobre el Warehouse; no es el lugar principal donde viven los datos.
- **Metabase**, **Tableau**, **Power BI** o **Looker** presentan tableros y métricas a los usuarios de negocio.

Ejemplo que un alumno debe poder narrar:

```text
PostgreSQL → Fivetran → BigQuery → dbt → Metabase
operación      copia      almacena/   modela    muestra
                           consulta
```

> “En esta configuración BigQuery cumple el rol de Warehouse: guarda tablas analíticas administradas y ejecuta SQL. Más adelante veremos que el mismo producto también puede funcionar como motor de consulta sobre datos externos. La marca no determina por sí sola la arquitectura; importa cómo se usa.”

---

## 17–24 min · Año 4: el punto de quiebre

### Diagrama 4 · «El mismo tubo, seis presiones nuevas»

**Propósito.** Hacer visible que la transición no se justifica por una sola palabra como JSON, Kafka o ML, sino por la combinación de fuerzas.

**Composición visual.** En el centro, reutilizar el pipeline angosto de la fase 1. Alrededor, seis cuñas que lo comprimen:

1. `Volumen` — `4 TB/mes ≈ 48 TB/año antes de crecimiento y réplicas`.
2. `Velocidad` — `Kafka y fraude en tiempo real`.
3. `Variedad` — `eventos, logs, PDFs, OCR, biometría`.
4. `Nuevos consumidores` — `BI + científicos de datos + fraude`.
5. `Movimiento` — `exportaciones de terabytes y copias locales`.
6. `Economía` — `almacenamiento/transformación propietarios: decenas de miles USD/mes en el supuesto`.

Al pie, una línea temporal: `10 mil transacciones/día → 5 millones de usuarios`. Marcar un punto rojo `Año 4: el problema dejó de ser sólo reporting`.

**Animación.** Mantener el pipeline intacto y agregar una cuña a la vez. Después de cada dos cuñas, preguntar si bastaría con aumentar cómputo. El tubo se torna rojo únicamente al final.

**Texto alternativo.** «El pipeline inicial queda presionado simultáneamente por más volumen, streaming, datos no estructurados, ML, copias y costo. El punto de quiebre ocurre por la combinación, no por un tipo de archivo aislado».

### Actividad de clasificación, 3 minutos

Pedir al grupo que asigne cada síntoma a una de seis columnas en el pizarrón: `volumen`, `velocidad`, `variedad`, `consumidores`, `movimiento`, `operación/costo`.

Presentar estos síntomas:

- millones de clics móviles;
- PDFs de contratos;
- modelo de fraude que necesita minutos o segundos;
- científicos que descargan terabytes;
- dos equipos calculan una característica de riesgo;
- la factura de transformaciones se dispara.

### Pregunta de decisión antes de mostrar la solución

> “¿Escalamos el Warehouse, añadimos un Lake separado o rediseñamos una base compartida? No elijan producto todavía: digan qué requisito intentan satisfacer.”

Registrar las tres opciones como hipótesis. No cerrar aún; se recuperarán en el minuto 56.

---

## 24–30 min · La base: almacenamiento único sobre objetos

### Subdiagrama 5A · «Anatomía de un bucket»

**Propósito.** Dar una imagen concreta a quienes nunca han usado almacenamiento de objetos. Dedicar como máximo 90 segundos; esta explicación se integra en el bloque de seis minutos.

**Composición visual.** Dibujar un gran contenedor azul con la etiqueta:

```text
bucket: fintech-datos-prod
proveedor: Amazon S3
región: us-east-1
```

Dentro, colocar cuatro tarjetas de objeto. Cada tarjeta muestra **key + datos + metadata**:

```text
key: bronze/kafka/2026/08/25/part-0001.json
datos: { bytes JSON }
metadata: application/json · 2 MB · cifrado
```

```text
key: documents/loan-8472.pdf
datos: { bytes PDF }
metadata: application/pdf · 680 KB · PII
```

```text
key: silver/transactions/part-0042.parquet
datos: { bytes Parquet }
metadata: application/octet-stream · 256 MB
```

```text
key: models/fraud/v12/model.bin
datos: { bytes del modelo }
metadata: version=12 · owner=risk-team
```

Un corchete señala los fragmentos `bronze/`, `documents/`, `silver/` y dice `prefijos para organizar; se ven como carpetas`. Añadir un aviso: `el bucket conserva objetos, pero no conoce por sí solo filas, joins o transacciones de tabla`.

A la derecha, mostrar las equivalencias comerciales:

- `Amazon Web Services → Amazon S3 → bucket`.
- `Google Cloud → Cloud Storage → bucket`.
- `Microsoft Azure → ADLS Gen2 / Blob Storage → container o file system`.
- `Microsoft Fabric → OneLake → almacenamiento unificado administrado por Fabric`.

**Animación.** Mostrar contenedor, agregar dos objetos conocidos —PDF e imagen—, añadir Parquet y finalmente revelar la advertencia sobre SQL/tablas. Las equivalencias de proveedores aparecen al final.

**Texto alternativo.** «Un bucket llamado fintech-datos-prod contiene objetos identificados por keys que parecen rutas. Cada objeto tiene bytes y metadatos. S3 y Cloud Storage llaman bucket al contenedor; Azure usa también los términos container o file system. El contenedor almacena, pero no ejecuta SQL ni decide qué archivos forman una tabla».

### Narración para principiantes

> “Un bucket es un contenedor de nube con nombre, ubicación y políticas. Dentro puede haber millones de objetos de cualquier tipo. Cada objeto se recupera por una key. Pensemos en `s3://fintech-datos-prod/documents/loan-8472.pdf`: `s3` identifica el servicio, `fintech-datos-prod` es el bucket y el resto es la key del objeto.”

Escribir las tres variantes sin pedir que memoricen su sintaxis:

```text
s3://fintech-datos-prod/documents/loan-8472.pdf
gs://fintech-datos-prod/documents/loan-8472.pdf
Azure ADLS Gen2: container "documents" dentro de una cuenta de almacenamiento
```

Preguntar: `¿Cuál de estas partes podría ejecutar SELECT SUM(amount)?` Respuesta: ninguna por sí sola; hace falta un motor que interprete los datos y, para tablas confiables, metadata de tabla y catálogo.

### Tres distinciones que deben quedar explícitas

1. **Bucket ≠ archivo.** El bucket contiene objetos; el PDF o Parquet es el objeto.
2. **Bucket ≠ Data Lake.** Un Data Lake utiliza uno o más buckets/containers junto con ingesta, organización, seguridad, catálogo y operación.
3. **Bucket ≠ Lakehouse.** El Lakehouse añade tablas transaccionales, catálogo, gobierno y motores coordinados sobre esa base.

### Diagrama 5 · «Una base física compartida no es todavía una base de datos»

**Propósito.** Explicar qué aporta el almacenamiento de objetos y establecer el problema que resolverá el formato de tabla.

**Composición visual.** Tres niveles:

1. **Fuentes**, arriba: `PostgreSQL/CDC`, `Kafka`, `logs`, `PDFs`, `imágenes`.
2. **Almacenamiento de objetos**, al centro, como un gran rectángulo azul rotulado `S3 / Google Cloud Storage / Azure Data Lake Storage Gen2 / OneLake`. Dentro, tres zonas conceptuales:
   - `objetos crudos: JSON, PDFs, imágenes`;
   - `archivos columnares: Parquet/ORC`;
   - `resultados refinados`.
3. **Consumidores potenciales**, abajo: `SQL`, `Spark/Ray`, `streaming`.

Las fuentes convergen hacia un solo rectángulo. Los consumidores no deben conectarse todavía: colocar signos de interrogación entre ellos y el almacenamiento.

A la izquierda del bucket, sellos verdes: `costo por capacidad`, `escala`, `formatos abiertos`. A la derecha, signos rojos: `¿qué archivos están vigentes?`, `¿cuál es el esquema?`, `¿se completó la escritura?`, `¿quién puede leer?`.

**Animación.** Hacer converger las fuentes, revelar las ventajas y detenerse antes de conectar consumidores. Revelar las cuatro preguntas como límite.

**Texto alternativo.** «Fuentes diversas convergen en almacenamiento de objetos económico con datos crudos y archivos Parquet. Esta base unifica ubicación, pero por sí sola no define tabla, versión, transacción ni permisos».

### Narración sugerida

> “La primera decisión del Lakehouse es desacoplar los bytes del motor que los procesa. Guardamos una vez en objetos y podemos asignar cómputo distinto a SQL, ML o streaming. Pero un bucket sabe almacenar claves y objetos; no sabe por sí solo qué conjunto representa la tabla `transactions`.”

Mostrar el contenido hipotético:

```text
transactions/
  part-0001.parquet
  part-0002.parquet
  part-0003.parquet
  part-0003-retry.parquet
  _temporary-part-0004.parquet
```

Preguntar:

> “Si consulto esta carpeta, ¿cuáles archivos son parte de la tabla y cuál es un reintento?”

La respuesta importante es que el nombre de la carpeta no constituye un protocolo confiable de commit.

---

## 30–39 min · La capa de inteligencia: formatos de tabla abiertos

### Diagrama 6 · «Archivo, tabla y catálogo no son sinónimos»

**Propósito.** Dar un modelo mental por capas antes de explicar snapshots.

**Composición visual.** Una pila vertical de cuatro niveles, de abajo hacia arriba:

1. Azul: `Almacenamiento de objetos` — «conserva bytes y rutas».
2. Azul claro: `Parquet / ORC` — «organiza columnas dentro de cada archivo».
3. Morado: `Iceberg / Delta Lake / Hudi` — «coordina archivos, snapshots, esquema y cambios de una tabla».
4. Morado oscuro: `Catálogo` — «resuelve nombre, ubicación, versión/metadata vigente y gobierno».

A la derecha, tres motores grises (`Trino`, `Spark`, `motor SQL`) apuntan con flechas punteadas al catálogo y con flechas sólidas a los objetos. Una llave visual explica: `pregunta arriba; lee abajo`.

**Animación.** Construir de abajo hacia arriba. Al revelar cada capa, formular «¿qué sabe y qué no sabe?». Añadir motores al final.

**Texto alternativo.** «El almacenamiento guarda objetos, Parquet organiza un archivo, un formato de tabla coordina conjuntos de archivos y un catálogo permite encontrar y gobernar la tabla. Los motores consultan metadatos y luego leen los objetos».

### Narración sugerida

> “Parquet no ofrece una transacción sobre veinte archivos. El formato de tabla mantiene la lista autorizada de archivos, esquemas, particiones y snapshots. El catálogo permite que el nombre de negocio apunte a esa metadata. Son responsabilidades diferentes aunque una plataforma pueda ocultarlas detrás de una interfaz.”

No convertir este momento en una comparación exhaustiva de marcas. Decir:

> “Iceberg, Delta y Hudi comparten esta intención general, pero no son idénticos. La elección depende de motores compatibles, patrones de actualización, ecosistema y operación.”

### Diagrama 7 · «Una escritura fallida que los lectores nunca ven»

**Propósito.** Hacer tangible el commit atómico mediante snapshots.

**Composición visual.** Dos historietas en paralelo.

**Panel A: carpeta sin protocolo de tabla.**

- Estado inicial: archivos `A`, `B`, `C` en azul.
- Un job intenta añadir `D`, `E`, `F`.
- Se escriben `D` y `E`; antes de `F` aparece un rayo rojo `falla`.
- Un lector enumera la carpeta y observa `A B C D E` con la pregunta `¿actualización completa?`.

**Panel B: tabla con snapshots.**

- Un puntero morado `actual → snapshot 17` enumera `A B C`.
- El job escribe `D` y `E` como archivos grises «no publicados» y falla antes del commit.
- El puntero continúa en `snapshot 17`; los lectores observan `A B C`.
- En una segunda viñeta de éxito, el job escribe `D E F` y realiza un solo cambio visible: `actual → snapshot 18`, que enumera `A B C D E F`.
- Nota pequeña: `los archivos no referenciados requieren limpieza posterior`.

**Animación.** Sincronizar ambos paneles paso a paso: estado inicial, archivos nuevos, falla, lectura. Después revelar la viñeta de commit exitoso.

**Texto alternativo.** «Sin protocolo, un lector puede encontrar una escritura parcial. Con una tabla basada en snapshots, los nuevos archivos se preparan sin publicarse; sólo un commit exitoso mueve el puntero a la nueva lista. Una falla conserva visible el snapshot anterior».

### Narración sugerida

> “La atomicidad no exige esconder cada objeto durante la escritura. Exige que los lectores tengan una lista autorizada y que el cambio de una lista completa a otra sea el commit visible. Los archivos huérfanos pueden existir físicamente y aun así no pertenecer a la tabla; después se limpian con mantenimiento.”

Preguntar: `¿Qué vería un lector que inició antes del commit?` Respuesta: una versión consistente, normalmente el snapshot que resolvió al comenzar, según el aislamiento y el motor.

### Diagrama 8 · «Evolucionar el esquema sin reescribir la historia»

**Propósito.** Mostrar el rol de metadatos y evitar la falsa idea de que cambiar un nombre siempre modifica terabytes de archivos.

**Composición visual.** Línea temporal con dos snapshots:

- `Snapshot 41 · esquema v1`: `customer_id [field 1]`, `amount [field 2]`.
- Flecha morada: `rename lógico`.
- `Snapshot 42 · esquema v2`: `customer_id [field 1]`, `gross_amount [field 2]`.

Debajo, tres archivos históricos continúan mostrando físicamente `amount`. Un mapa morado conecta `field 2` con el nombre lógico vigente `gross_amount`. Agregar una llamada lateral: `el soporte seguro de rename depende de IDs/mapeo de columnas, configuración y motores`.

**Animación.** Mostrar v1 y archivos, hacer el rename en metadata, revelar que los archivos no cambian y finalmente mostrar la advertencia.

**Texto alternativo.** «Un campo conserva una identidad estable aunque cambie su nombre lógico. La metadata nueva puede mapear archivos históricos sin reescribirlos, siempre que el formato y los motores soporten correctamente esa evolución».

### Narración sugerida

> “La metadata desacopla parte de la identidad de una columna de su representación física. Esto habilita evolución de esquemas y viaje en el tiempo. No significa que cualquier cambio sea gratuito: cambiar tipos incompatibles, semántica o partición puede exigir migración y pruebas.”

---

## 39–45 min · La capa de gestión: catálogo y gobierno

### Diagrama 9 · «El mapa, la dirección vigente y la caseta de acceso»

**Propósito.** Distinguir tres preguntas: dónde está la tabla, qué versión se lee y quién puede usarla.

**Composición visual.** En el centro, una tarjeta morada grande `Catálogo unificado`. Dentro, una ficha expandida:

```text
fintech.silver.transactions
ubicación: s3://fintech/silver/transactions/
metadata vigente: .../v018.metadata.json
esquema: transaction_id, user_id, amount, device_risk...
propietario: Risk Data Team
clasificación: PII · financiera
```

A la izquierda, tres motores grises: `Trino/SQL`, `Spark/Ray`, `stream processor`. Sus flechas punteadas preguntan al catálogo y sus flechas sólidas continúan hacia la tabla azul en el almacenamiento.

A la derecha, una matriz de gobierno:

| Identidad | Acceso ilustrativo |
|---|---|
| Finanzas | Gold de ingresos; PII enmascarada |
| Fraude | transacciones + señales de dispositivo |
| Data Science | Silver seudonimizada; entorno controlado |
| Auditoría | historial y logs de acceso |

Una banda superior, morado oscuro, cruza catálogo y almacenamiento con las palabras `identidad · permisos · linaje · auditoría`. Rotular como ejemplos: `Unity Catalog` o `AWS Glue + Lake Formation`; no presentar Lake Formation como idéntico al catálogo de Glue.

**Animación.** Primero resolver el nombre, luego la metadata vigente, después aplicar identidad/permisos y finalmente leer objetos. La matriz aparece una fila a la vez.

**Texto alternativo.** «Los motores consultan un catálogo que resuelve el nombre de la tabla, su ubicación y metadata vigente. El gobierno aplica permisos según identidad y clasificación antes de que el motor lea los objetos».

### Narración sugerida

> “El catálogo es el mapa central: convierte `fintech.silver.transactions` en ubicación, esquema y metadata vigente. El gobierno responde otra pregunta: aunque yo sepa dónde están los objetos, ¿tengo derecho a verlos y con qué nivel de detalle?”

Enfatizar:

> “Un catálogo no mueve mágicamente los datos ni garantiza calidad por existir. Necesita propietarios, políticas, clasificación y auditoría. Y los usuarios no deberían evadirlo leyendo rutas directas con credenciales amplias.”

### Comprobación rápida

Leer cuatro frases y pedir que respondan `almacenamiento`, `formato de archivo`, `formato de tabla` o `catálogo/gobierno`:

1. «Organiza columnas y compresión dentro de `part-0007`» → formato de archivo.
2. «Decide que los archivos A, B y C forman la versión 17» → formato de tabla.
3. «Resuelve dónde vive `silver.transactions` y verifica permisos» → catálogo/gobierno.
4. «Conserva el PDF original y los bytes de Parquet» → almacenamiento de objetos.

---

## 45–51 min · Fase 2: la arquitectura Lakehouse de la fintech

### Diagrama 10 · «Una base gobernada, tres caminos de cómputo»

**Propósito.** Integrar el caso completo y mostrar que compartir datos no implica ejecutar todas las cargas en el mismo clúster.

**Composición visual.** Arquitectura horizontal de izquierda a derecha con una banda transversal superior.

**Columna 1, fuentes.** Cinco tarjetas: `transacciones/CDC`, `Kafka: clics y telemetría`, `logs de auditoría`, `PDFs/OCR`, `imágenes/biometría`.

**Columna 2, ingesta.** Dos entradas naranjas: `batch/CDC` y `streaming`.

**Columna 3, base compartida sobre S3/GCS.** Tres franjas dentro del mismo contenedor:

- `Bronze · evidencia recibida`: eventos crudos, referencias a PDFs/imágenes y réplicas.
- `Silver · entidades validadas`: transacciones deduplicadas, usuarios seudonimizados, señales de dispositivo, resultados de OCR.
- `Gold · productos`: ingresos, morosidad, cartera y métricas certificadas.

Una llave morada abraza las tablas tabulares y dice `Iceberg/Delta/Hudi: snapshots, esquema y transacciones`. Los PDFs e imágenes se dibujan como objetos azules fuera de esa llave, pero una tabla Silver `document_index` apunta a ellos mediante `object_uri`, clasificación y estado de procesamiento.

**Banda superior.** `Catálogo + gobierno + linaje + calidad`, extendida sobre Bronze, Silver y Gold.

**Columna 4, cómputo desacoplado.** Tres carriles grises independientes:

1. `Motor SQL (Trino / BigQuery / Databricks SQL)` → `BI financiero`.
2. `Spark / Ray` → `entrenamiento de ML`.
3. `motor de streaming` → `fraude en tiempo real`.

Las tres rutas parten de tablas gobernadas, pero no de la misma capa obligatoriamente: BI usa Gold; ML combina Silver autorizado y objetos; fraude consume streaming y actualiza/consulta señales Silver. Agregar interruptores de costo independientes junto a cada motor para representar que el cómputo escala por carga.

En rojo tenue, mostrar dos flechas antiguas tachadas: `exportar terabytes al portátil` y `copiar Lake → Warehouse por cada consumo`.

**Animación.**

1. Ingresar fuentes a Bronze.
2. Refinar Bronze → Silver → Gold.
3. Superponer formato de tabla y catálogo/gobierno.
4. Conectar BI, ML y fraude uno por uno.
5. Tachar las exportaciones y separar los interruptores de cómputo.

**Texto alternativo.** «La fintech conserva datos diversos en almacenamiento de objetos. Tablas Bronze, Silver y Gold están coordinadas por un formato de tabla y un catálogo gobernado; los binarios permanecen como objetos referenciados. Motores SQL, ML y streaming escalan por separado y usan los mismos contratos sin exportaciones masivas».

### Narración sugerida

> “En el año 5 la fintech no cambia sólo de contenedor. Cambia el contrato: los bytes viven en una base económica y abierta; las tablas tienen versiones confiables; el catálogo controla su interpretación y acceso; cada carga usa el cómputo apropiado.”

Recorrer cada consumidor:

- **Finanzas:** consulta productos Gold certificados. No debería calcular ingresos directamente desde JSON Bronze.
- **ML:** lee entidades Silver autorizadas y resultados derivados de documentos, sin extraer terabytes a almacenamiento local.
- **Fraude:** procesa el flujo en tiempo real y consulta/escribe señales gobernadas según las garantías del pipeline.

Pregunta:

> “¿Dónde está ahora la fuente única de verdad?”

Respuesta que se debe matizar: no es una sola tabla ni una sola definición para todos los usos; es una base de objetos y contratos de datos versionados, catalogados y con responsables comunes.

### Resultado económico que se puede afirmar

> “En el supuesto, el almacenamiento masivo se alinea con precios de objetos y se reducen copias. Los tres ingenieros ahora sostienen capacidades que generan retorno: fraude de baja latencia, datos gobernados para ML y escala creciente. Esto no garantiza que toda consulta sea más barata; el cómputo y la operación aún deben controlarse.”

### Diagrama 10B · «Los conceptos tienen implementaciones comerciales»

**Propósito.** Conectar las capas abstractas con herramientas que el alumnado encontrará en ofertas de trabajo, documentación y proyectos. Dedicar 60–90 segundos en la sesión principal. No leer todas las marcas; construir primero una columna completa y después mostrar alternativas.

Antes de mostrar la tabla, usar una leyenda sencilla:

- **Estándares/proyectos abiertos:** Parquet, Iceberg, Delta Lake, Hudi, Spark, Kafka y Trino. Pueden tener distribuciones o servicios comerciales alrededor.
- **Servicios/plataformas administradas:** S3, BigQuery, Snowflake, Redshift, Databricks, Microsoft Fabric, Fivetran y las demás ofertas operadas por un proveedor.
- **Combinación frecuente:** una plataforma comercial implementa o administra tecnologías abiertas; por ejemplo, Databricks administra tablas Delta y AWS ofrece motores que trabajan con Iceberg.

**Composición visual.** Una tabla por responsabilidades. Las filas son estables y los productos son ejemplos, no equivalencias perfectas:

| Responsabilidad | Tecnologías y servicios frecuentes | Cómo aparece en el caso |
|---|---|---|
| Fuente operacional | PostgreSQL, MySQL, SQL Server, Cloud SQL, Amazon RDS, Azure SQL | Usuarios, cuentas, cobros y transferencias. |
| Ingesta batch/CDC | Fivetran, Airbyte, AWS Database Migration Service, Google Datastream, Azure Data Factory | Replica cambios de la operación. |
| Eventos/streaming | Apache Kafka, Confluent Cloud, Amazon Kinesis, Google Pub/Sub, Azure Event Hubs | Clics, telemetría y señales de fraude. |
| Data Warehouse administrado | BigQuery, Snowflake, Amazon Redshift, Microsoft Fabric Warehouse, Azure Synapse Analytics | Tablas SQL para ingresos, cartera y morosidad. |
| Almacenamiento de objetos | Amazon S3, Google Cloud Storage, Azure Data Lake Storage Gen2; OneLake lo unifica dentro de Fabric | Conserva JSON, Parquet, PDFs, imágenes y otros objetos. |
| Formato de archivo | Apache Parquet, ORC, Avro, JSON | Organiza bytes dentro de cada archivo. |
| Formato de tabla | Apache Iceberg, Delta Lake, Apache Hudi | Publica snapshots, esquema e inventario de archivos. |
| Catálogo y gobierno | AWS Glue Data Catalog + Lake Formation, Databricks Unity Catalog, capacidades de catálogo/gobierno de BigQuery y BigLake, Microsoft Purview/Fabric | Descubrimiento, permisos, linaje y auditoría. |
| SQL sobre el lago/lakehouse | Amazon Athena, Trino/Starburst, Databricks SQL, BigQuery/BigLake, Redshift Spectrum, endpoints SQL de Fabric | BI consulta tablas Gold. |
| Ingeniería y ML | Apache Spark, Databricks, Amazon EMR, Google Dataproc, Microsoft Fabric Spark, Ray | Transforma Silver y entrena modelos. |
| Consumo | Tableau, Power BI, Looker, Metabase, Apache Superset | Presenta métricas o resultados. |

Debajo de la tabla, incluir tres advertencias visibles:

1. `S3/GCS/ADLS son almacenamiento, no el Lakehouse completo`.
2. `Iceberg/Delta/Hudi son formatos abiertos, no proveedores de nube`.
3. `BigQuery, Snowflake, Redshift, Databricks y Fabric pueden cubrir más de una responsabilidad según la configuración`.

**Animación.** Resaltar una fila a la vez sobre el diagrama 10: ingesta, almacenamiento, tabla, catálogo, cómputo y consumo. En la segunda pasada cambiar los productos, pero mantener inmóviles los nombres de las responsabilidades. Así se comunica que las marcas cambian y la arquitectura permanece.

**Texto alternativo.** «Un mapa relaciona responsabilidades de una plataforma de datos con ejemplos comerciales y abiertos. S3, Cloud Storage y ADLS almacenan; Parquet organiza archivos; Iceberg, Delta y Hudi coordinan tablas; Glue, Lake Formation o Unity Catalog catalogan y gobiernan; Athena, BigQuery, Databricks SQL, Trino o Fabric SQL procesan; Tableau, Power BI, Looker o Metabase consumen».

### Arquitecturas concretas para comparar en clase

En la sesión principal mostrar completos sólo los ejemplos A y C o D: uno representa la fase inicial y otro la fase Lakehouse. Los demás quedan como tarjetas de consulta o apéndice para no convertir seis minutos en un catálogo de proveedores.

#### Ejemplo A · Warehouse sencillo en Google Cloud

```text
PostgreSQL → Fivetran → tablas nativas de BigQuery → dbt → Looker/Metabase
```

- BigQuery guarda y consulta las tablas analíticas administradas.
- La ruta funciona bien para la fase inicial relacional y orientada a BI.
- No necesita llamarse Lakehouse para ser una buena arquitectura.

#### Ejemplo B · Data Lake consultable en AWS, todavía sin tablas transaccionales

```text
Kafka/Kinesis → Amazon S3 con JSON y Parquet
                         ↓
                AWS Glue Data Catalog
                         ↓
                 Athena o Spark/EMR
```

- S3 conserva objetos; Glue registra esquemas; Athena o Spark ejecutan consultas.
- Si las tablas son sólo carpetas de Parquet sin Iceberg, Delta o Hudi, el diagrama muestra un Data Lake consultable, no todavía las garantías completas de una tabla Lakehouse.
- Puede ser una solución válida cuando las escrituras son simples y la organización acepta esas garantías.

#### Ejemplo C · Lakehouse modular en AWS

```text
Kafka/Kinesis → S3 → Apache Iceberg → Glue Data Catalog + Lake Formation
                                           ↓
                               Athena / EMR-Spark / Redshift
```

- S3 almacena los archivos.
- Iceberg coordina snapshots y tablas.
- Glue Data Catalog registra metadata y Lake Formation aplica gobierno fino.
- Athena, Spark en EMR o Redshift son motores diferentes sobre datos compartidos, con capacidades que deben verificarse para cada operación.

#### Ejemplo D · Lakehouse administrado con Databricks

```text
Fivetran/Kafka → S3, GCS o ADLS → Delta Lake → Unity Catalog
                                                    ↓
                                  Databricks SQL / Spark / ML
```

- El proveedor de nube conserva los objetos.
- Delta Lake aporta el protocolo de tabla.
- Unity Catalog centraliza descubrimiento, permisos, linaje y auditoría.
- Databricks SQL atiende BI y Spark atiende ingeniería, streaming y ML sobre la misma plataforma gobernada.

#### Ejemplo E · Lakehouse integrado con Microsoft Fabric

```text
Data Factory/Eventstreams → OneLake → tablas Delta de Fabric Lakehouse
                                             ↓
                                  Spark / SQL / Power BI
```

- OneLake es la base unificada de almacenamiento de Fabric, construida sobre capacidades de Azure Data Lake Storage.
- Fabric Lakehouse organiza archivos y tablas Delta.
- Spark, el endpoint SQL y Power BI reutilizan los datos dentro del entorno Fabric.

#### Ejemplo F · Ruta abierta en Google Cloud

```text
Datastream/Pub/Sub → Google Cloud Storage → BigLake/Iceberg
                                                   ↓
                                          BigQuery / Spark
```

- Cloud Storage guarda los objetos.
- BigLake y las tablas Iceberg permiten gobernar/consultar datos abiertos según el tipo de tabla y la configuración elegida.
- BigQuery puede ser Warehouse de tablas nativas o motor que consulta datos donde viven; por eso se debe preguntar siempre **qué almacena, qué cataloga y qué procesa**.

### Pregunta relámpago con productos

Mostrar `S3 + Parquet + Tableau` y preguntar: `¿qué piezas faltan para llamarlo Lakehouse confiable?`

Respuesta esperada: como mínimo un formato/protocolo de tabla con commits y snapshots, catálogo/gobierno, un motor compatible y procesos operativos. Tableau es consumo; no convierte archivos en tablas.

---

## 51–56 min · Trade-offs y operación

### Diagrama 11 · «El impuesto de los archivos pequeños»

**Propósito.** Mostrar por qué el streaming crea trabajo de mantenimiento aunque cada escritura sea correcta.

**Composición visual.** Antes y después:

- **Antes:** un minuto de streaming produce 60 archivos diminutos. Dibujar muchas fichas azules de `1–5 MB`. Un motor de consulta debe abrir y planificar cada una; arriba, un cronómetro rojo `mucho overhead`.
- **Mantenimiento:** engrane morado `compactar` que lee archivos pequeños, escribe 3–4 archivos grandes y publica el reemplazo mediante un nuevo snapshot.
- **Después:** cuatro bloques de tamaño objetivo, con un cronómetro verde `menos aperturas, mejor lectura`.

Debajo, una línea separada con otras tareas que no deben confundirse con compactar: `expirar snapshots según retención`, `limpiar huérfanos`, `actualizar estadísticas`, `revisar partición y clustering`.

**Animación.** Los archivos pequeños llegan rápidamente, la consulta se ralentiza, aparece el engrane y finalmente se publica un nuevo snapshot. Revelar las tareas inferiores al final.

**Texto alternativo.** «El streaming crea numerosos archivos pequeños y eleva el costo de planear y abrir una consulta. Una tarea de compactación escribe menos archivos grandes y publica el reemplazo de forma transaccional; además se requieren retención, limpieza y estadísticas».

### Narración sugerida

> “El Lakehouse desacopla almacenamiento y cómputo, pero no desacopla al equipo de la física de los archivos. Sesenta archivos válidos pueden ser mucho más lentos que cuatro archivos bien dimensionados. Compactar es una operación recurrente, no una limpieza ocasional.”

### Diagrama 12 · «Una fuente compartida también comparte el radio de impacto»

**Propósito.** Contraponer la reducción de silos con el riesgo de una ruptura central.

**Composición visual.** En el centro, una tabla Silver `transactions v18`. De ella salen tres flechas verdes a `cierre financiero`, `features de fraude` y `entrenamiento ML`. Luego aparece una modificación roja: `amount cambia de moneda/semántica sin versión compatible`. La onda de impacto alcanza a los tres consumidores.

Alrededor de la tabla, cinco barreras defensivas formando un anillo:

- `contrato y propietario`;
- `pruebas de esquema y calidad`;
- `compatibilidad o vista de transición`;
- `despliegue por etapas`;
- `linaje + observabilidad + rollback/time travel`.

**Animación.** Mostrar primero el beneficio de tres consumidores; introducir el cambio rompiente; propagar la onda roja; finalmente construir el anillo de controles.

**Texto alternativo.** «Una tabla compartida alimenta Finanzas, Fraude y ML. Un cambio semántico incompatible puede afectar a todos; contratos, pruebas, transición, linaje y recuperación limitan el impacto».

### Narración sugerida

> “Eliminar copias reduce inconsistencias, pero concentra dependencia. Un cambio roto ya no daña un pipeline aislado: puede afectar simultáneamente el cierre financiero y el modelo de fraude. La fuente compartida exige más disciplina de producto de datos.”

### Lista breve de responsabilidades operativas

- compactación y dimensionamiento de archivos;
- retención de snapshots y limpieza de archivos huérfanos;
- calidad, contratos y evolución compatible;
- permisos, auditoría y prevención de accesos directos no gobernados;
- observabilidad de latencia, frescura, costo y fallas;
- pruebas reales de interoperabilidad entre cada formato y motor;
- presupuestos y apagado/escalamiento del cómputo desacoplado.

---

## 56–60 min · Decidir, no evangelizar

### Diagrama 13 · «La arquitectura se elige por el cuello de botella»

**Propósito.** Cerrar comparando las dos fases con criterios explícitos.

**Composición visual.** Matriz con filas de criterio y tres columnas de alternativa. No usar ganadores absolutos; usar frases condicionales.

| Criterio | Warehouse administrado | Lake + Warehouse separados | Lakehouse |
|---|---|---|---|
| Datos dominantes | Relacionales y preparados | Crudos diversos con usos separados | Diversos y compartidos entre BI/ML/streaming |
| Prioridad | Rapidez de adopción y SQL | Aislar economías o equipos | Reducir copias con contratos comunes |
| Cómputo | Principalmente SQL/BI | Motores y autoridades separadas | Motores distintos sobre tablas compartidas |
| Operación | Menor carga para equipo pequeño | Coordinación y sincronización dobles | Metadatos, compactación y gobierno rigurosos |
| Señal económica | El servicio administrado sigue alineado | La separación tiene límites claros | Movimiento/copias y almacenamiento propietario dominan el costo |
| Riesgo principal | Rigidez o costo al crecer | Dos verdades y latencia | Complejidad y radio de impacto compartido |

Debajo de la matriz, dos fichas que se mueven con la animación:

- `Fintech años 0–3` cae en **Warehouse administrado**.
- `Fintech año 5` cae en **Lakehouse**, por la combinación de volumen, variedad, streaming, ML, duplicación y equipo capaz de operarlo.

Una tercera ficha punteada dice: `Otra empresa podría permanecer en Warehouse o usar una arquitectura híbrida`. Esto impide presentar la elección como ley universal.

**Texto alternativo.** «Una matriz compara alternativas según tipo de datos, prioridad, cómputo, operación, economía y riesgo. La fase inicial de la fintech favorece el Warehouse; la etapa de escala favorece el Lakehouse, pero otras organizaciones pueden decidir distinto».

### Debate final, 2 minutos

Dividir el grupo en dos mitades:

- Mitad A defiende la elección de Warehouse en los años 0–3 usando dos datos del caso.
- Mitad B defiende Lakehouse desde el año 5 usando dos requisitos y una responsabilidad operativa.

Exigir la forma de argumento:

> “Elegimos ___ porque ___ y ___; aceptamos el costo/riesgo de ___.”

Respuestas modelo:

> “Elegimos un Warehouse administrado al inicio porque los datos son relacionales y el objetivo es reporting rápido; aceptamos menor flexibilidad futura a cambio de operar con una sola persona.”

> “Elegimos Lakehouse al escalar porque BI, ML y fraude necesitan reutilizar datos diversos y el movimiento de terabytes ya domina el costo; aceptamos operar compactación, contratos y gobierno con un equipo especializado.”

### Exit ticket, último minuto

Pedir completar, en una sola frase cada una:

1. `Un bucket con Parquet no es todavía una tabla porque...`
2. `El catálogo sirve para..., mientras el formato de tabla sirve para...`
3. `La migración de la fintech se justifica cuando..., pero obliga a...`

Respuestas mínimas esperadas:

1. falta una lista/versionado transaccional que determine qué archivos forman la tabla;
2. el catálogo permite encontrar y gobernar; el formato coordina snapshots, esquema y commits;
3. varias cargas comparten datos diversos a escala y las copias/costos se vuelven dominantes; a cambio se operan mantenimiento, gobierno y contratos.

Cerrar con:

> “El Lakehouse no elimina la complejidad: la mueve. Cambia la complejidad de copiar y reconciliar silos por la complejidad de operar correctamente una base compartida. La arquitectura es buena cuando ese intercambio favorece al negocio.”

---

# Banco de preguntas y respuestas docentes

## «¿Un bucket es una carpeta gigante?»

Es una aproximación útil durante los primeros segundos: ambos ayudan a contener y organizar. Técnicamente, un bucket es el contenedor de objetos del servicio. En muchos servicios, una key como `bronze/2026/file.parquet` se muestra como carpetas por sus prefijos aunque el modelo base no sea el mismo que el sistema de archivos de una computadora. Algunas ofertas incorporan además un espacio de nombres jerárquico real. Para esta clase basta recordar `bucket → objetos identificados por key`.

## «¿Puedo hacer SQL directamente sobre un bucket?»

El bucket no ejecuta SQL. Un motor como Athena, BigQuery/BigLake, Trino, Spark o Databricks SQL puede leer archivos del bucket. Para tratarlos como tablas confiables necesita conocer el esquema y el conjunto vigente de archivos mediante catálogo y metadata de tabla.

## «¿S3 es un Data Lake?»

S3 es un servicio de almacenamiento de objetos y suele ser la base física de un Data Lake. El Data Lake es la arquitectura más amplia: ingesta, organización, archivos, catálogo, seguridad, gobierno, cómputo y operación. La misma distinción aplica a Google Cloud Storage y Azure Data Lake Storage.

## «¿Entonces el Data Warehouse ya no sirve?»

No. En la fase inicial fue la opción más eficiente y puede continuar sirviendo productos Gold o cargas de BI. La decisión Lakehouse aparece cuando almacenamiento abierto y múltiples motores sobre datos compartidos producen más valor que su carga operativa.

## «¿Guardar Parquet en S3 ya crea un Lakehouse?»

No. Eso resuelve almacenamiento y formato de archivo. Aún hacen falta administración de tablas, commits consistentes, catálogo, permisos, calidad, mantenimiento y motores compatibles.

## «¿Iceberg, Delta Lake y Hudi son bases de datos?»

Son formatos/protocolos de tabla que describen cómo coordinar datos y metadatos sobre almacenamiento. Un catálogo y uno o más motores completan la experiencia de lectura y escritura; una plataforma puede empaquetar todo como servicio.

## «¿El catálogo guarda todos los datos?»

Normalmente guarda o resuelve metadatos: nombres, ubicaciones, esquemas, propietarios, políticas y punteros. Los archivos de datos permanecen en almacenamiento de objetos.

## «¿Fuente única de verdad significa una tabla para todo?»

No. Significa autoridades y contratos coherentes. Bronze, Silver y Gold pueden coexistir, y distintos productos pueden publicar vistas adecuadas a sus decisiones sin crear definiciones incompatibles y no gobernadas.

## «¿Por qué no poner también todos los PDFs dentro de una tabla?»

Los binarios pueden permanecer como objetos, donde su almacenamiento es natural. Una tabla de índice puede registrar URI, hash, propietario, clasificación, estado de OCR y relaciones, haciendo gobernable su uso sin confundir objeto binario con tabla columnar.

## «¿Separar cómputo siempre abarata?»

No automáticamente. Permite escalar y apagar motores por carga, pero consultas ineficientes, falta de límites o tareas permanentes pueden elevar el gasto. Se requieren presupuestos, cuotas y observabilidad.

## «¿Time travel sustituye los respaldos?»

No necesariamente. Depende de políticas de retención, eliminación de objetos, aislamiento de cuentas y requisitos de recuperación. Es una capacidad de versionado útil, no una estrategia completa de continuidad por sí sola.

# Errores frecuentes y preguntas de recuperación

| Error del grupo | Pregunta para recuperar el concepto |
|---|---|
| «El Lakehouse es Warehouse + Lake» | ¿Qué capa hace que los archivos se comporten como una tabla y evite mantener dos verdades? |
| «Parquet garantiza ACID» | ¿Puede un formato dentro de un archivo publicar atómicamente una lista de veinte archivos? |
| «S3 es inconsistente» | ¿El fallo está en almacenar cada objeto o en decidir cuándo el conjunto completo se vuelve una versión de tabla? |
| «Todos deben consultar Bronze» | ¿Qué garantías necesita Finanzas que un evento recién recibido todavía no ofrece? |
| «Un catálogo garantiza calidad» | ¿Quién define, prueba y responde por la semántica de `amount`? |
| «Formato abierto significa compatibilidad perfecta» | ¿El motor soporta las mismas operaciones, versiones y extensiones del formato? |
| «Lakehouse siempre cuesta menos» | ¿Estamos contando cómputo, ingeniería, mantenimiento, observabilidad y movimiento, además del almacenamiento? |
| «La fintech debía migrar desde el día uno» | ¿Qué retorno habrían producido tres ingenieros cuando sólo había 500 GB/año y reporting SQL? |
| «La fuente compartida elimina todos los riesgos» | ¿Qué consumidores se rompen si cambia una tabla Silver central sin compatibilidad? |

# Plan de contingencia temporal

## Si sólo hay 45 minutos

- Reducir el debate inicial a una hipótesis.
- Mantener completos los diagramas 2, 3, 4, 7, 9, 10, 11 y 13.
- Presentar los diagramas 5 y 6 como una sola pila.
- Omitir el diagrama 8, pero mencionar evolución de esquema como beneficio condicionado.
- Conservar el exit ticket; es la evidencia de aprendizaje.

## Si sobran 5 minutos

Entregar este cambio propuesto:

```text
El campo amount, antes en MXN, llegará en moneda original junto con currency.
```

Pedir diseñar una migración segura: nuevo contrato, columna explícita, tabla/vista compatible, backfill si procede, pruebas, consumidores afectados, despliegue por etapas y fecha de retiro.

## Si el grupo confunde marcas con capas

Ocultar todos los nombres de productos y pedir reconstruir la arquitectura usando sólo responsabilidades:

```text
guardar → coordinar tablas → localizar/gobernar → procesar → consumir
```

Después colocar S3/GCS, Iceberg/Delta/Hudi, Unity Catalog/Glue + Lake Formation, Trino/Spark/SQL y BI/ML en su responsabilidad correspondiente.

# Lista de verificación para convertir después este guion a Reveal.js

- Una idea principal y una pregunta docente por diapositiva.
- Mantener la convención de colores y los tipos de flecha.
- Incluir texto alternativo equivalente a cada descripción visual.
- Animar snapshots por estados; no mostrar todo el diagrama 7 de una vez.
- No introducir logos de proveedores como si fueran capas arquitectónicas.
- Conservar las cifras del caso como supuestos visibles y no como benchmarks.
- Colocar las respuestas del docente en notas del presentador, no en la vista inicial.
- Dejar tiempo visible para las preguntas de predicción antes de revelar soluciones.
- Evitar diagramas con texto menor al tamaño legible desde el fondo del salón; dividir el diagrama 10 en dos diapositivas si fuera necesario.
- Presentar el subdiagrama 5A antes de usar por primera vez `bucket`, `objeto` y `key` como si fueran vocabulario conocido.
- En el mapa comercial, mantener fijas las responsabilidades y cambiar sólo las marcas; esto evita que el alumnado confunda producto con arquitectura.
- Usar en la sesión principal únicamente dos pilas comerciales completas y dejar las demás como apéndice.
- Terminar con la matriz de decisión, no con una lista de productos.

# Referencias oficiales para preparar las diapositivas

Estas referencias son para el docente y para verificar etiquetas de producto al construir el QMD; no es necesario proyectarlas todas durante la clase.

- [Amazon S3: buckets, objetos y keys](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html).
- [Google Cloud Storage: objetos dentro de buckets](https://docs.cloud.google.com/storage/docs/introduction).
- [Azure Data Lake Storage sobre Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction).
- [BigQuery como plataforma analítica y Data Warehouse administrado](https://docs.cloud.google.com/bigquery/docs/introduction).
- [BigLake para consultar y gobernar datos en almacenamiento externo](https://docs.cloud.google.com/bigquery/docs/biglake-intro).
- [AWS Lake Formation, Glue Data Catalog y gobierno de datos en S3](https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html).
- [Databricks Lakehouse: Delta Lake, Unity Catalog y Spark](https://docs.databricks.com/aws/en/lakehouse/).
- [Microsoft Fabric OneLake](https://learn.microsoft.com/en-us/fabric/onelake/onelake-overview).
- [Opciones de almacenamiento Lakehouse y Warehouse en Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/fundamentals/store-data).

# Criterio rápido de logro

El grupo alcanzó el objetivo si puede reconstruir verbalmente esta cadena:

```text
silos y copias
→ base común en objetos
→ archivos coordinados por snapshots
→ catálogo y permisos
→ motores distintos sobre contratos compartidos
→ mantenimiento y control del radio de impacto
→ elección condicionada por el caso
```

La respuesta sobresaliente añade que el Data Warehouse fue correcto en la fase 1 y que la transición se volvió razonable sólo cuando el beneficio de compartir datos diversos a escala superó el costo de operar el Lakehouse.
