# Presentaciones de clase

Los fuentes están ordenados con la misma numeración y secuencia temática de
`../_quarto.yml`. La configuración común de Reveal.js vive en `_quarto.yml`;
el tema y los logos están en `assets/`.

## Estructura

```text
lectures/
├── 03-eda/               # análisis exploratorio
├── 04-preprocesamiento/  # faltantes, PCA y chi cuadrado
├── 05-patrones/          # reglas de asociación
├── 06-clasificacion/     # métricas y clasificadores
├── assets/               # tema, logos e imágenes compartidas
├── references/           # bibliografías
├── notebooks/            # talleres y material ejecutable
├── notes/                # guías docentes
├── tools/                # utilidades de mantenimiento
├── _drafts/              # material que no se compila
├── _templates/           # punto de partida para nuevas clases
└── _output/              # HTML generado (no se versiona)
```

## Crear una clase

Desde la raíz del repositorio:

```bash
cp lectures/_templates/clase.qmd lectures/06-clasificacion/06-XX-mi-tema.qmd
quarto preview lectures/06-clasificacion/06-XX-mi-tema.qmd
```

La presentación hereda automáticamente el tema UNAM, el logo, el pie de
página, la numeración y las opciones comunes. El bloque YAML de cada clase
solo necesita título, subtítulo, autor y fecha. Si una clase requiere una
opción especial, puede declararla localmente.

## Compilar

```bash
# Todas las presentaciones
quarto render lectures

# Una sola presentación
quarto render lectures/04-preprocesamiento/04-02-datos-faltantes.qmd
```

Los resultados se escriben en `lectures/_output/`. El sitio principal publica
únicamente esa carpeta y la galería `../presentaciones.qmd` enlaza las rutas
organizadas. No guardes HTML ni carpetas `*_files` junto a los fuentes.

Para actualizar el sitio después de editar una clase:

```bash
quarto render lectures
quarto render presentaciones.qmd --no-execute
```
