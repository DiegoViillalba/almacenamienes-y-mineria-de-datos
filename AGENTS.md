# Guía operativa del repositorio

Este repositorio publica un libro bilingüe de Quarto y sus presentaciones de
clase. Antes de modificar o validar contenido, usa este documento como mapa del
flujo actual. Las instrucciones más específicas de un subdirectorio tienen
prioridad; en particular, cualquier trabajo dentro de `lectures/` también debe
seguir `lectures/AGENTS.md`.

## Estructura y fuentes de verdad

- `_quarto.yml` configura el libro principal en español. Sus fuentes están en
  la raíz y en `unidades/`; su salida HTML va a `docs/`.
- `en/_quarto.yml` configura el libro en inglés. Sus fuentes están en `en/` y
  su salida HTML va a `docs/en/`.
- `lectures/_quarto.yml` configura las presentaciones Reveal.js. Los fuentes
  viven en `lectures/` y el render local va a `lectures/_output/`.
- `presentaciones.qmd` integra las presentaciones publicadas en el libro.
- `_quarto-es.yml` y `_quarto-en.yml` son configuraciones de perfil separadas;
  no son las configuraciones que usan `build.sh` ni el workflow actual.
- `requirements.txt` es el entorno Python normal. Las dependencias pesadas de
  PyTorch/JAX están separadas en `requirements-torch-jax.txt` y solo deben
  instalarse en un entorno dedicado cuando un capítulo las necesite.

El proyecto usa `execute.freeze: auto`. Conserva los resultados de `_freeze/`
que Quarto actualice al ejecutar capítulos; no agregues los notebooks
intermedios `*.quarto_ipynb*` ni carpetas `*_files`, que están ignorados.

## Compilación local

Usa el render más pequeño que cubra el cambio:

```bash
# Un capítulo español
quarto render unidades/.../capitulo.qmd --to html

# Un capítulo inglés
quarto render en/unidades/.../capitulo.qmd

# Libro HTML completo, español e inglés
./build.sh

# Reejecutar el código de ambos libros, sin reutilizar freeze
./build.sh --execute

# Vista previa del sitio español
quarto preview
```

`build.sh` genera únicamente HTML. No necesita TinyTeX/LaTeX. El orden es
inglés primero y español después con `--no-clean`, para que el segundo render
no borre `docs/en/`.

Para presentaciones Quarto:

```bash
quarto render lectures
quarto render lectures/ruta/a/clase.qmd
quarto render presentaciones.qmd --no-execute
```

Lee `lectures/AGENTS.md` antes de tocar presentaciones; las presentaciones de
Manim tienen un flujo adicional obligatorio.

## Publicación remota

Un `push` a `main` dispara `.github/workflows/publish.yml`. El workflow usa
Python 3.12 y Quarto 1.9.38, instala `requirements.txt`, renderiza primero el
HTML inglés y luego el español, sube `docs/` como artefacto y despliega GitHub
Pages.

El workflow **no compila el PDF** y no instala TinyTeX. El HTML generado del
libro (`docs/index.html`, `docs/en/`, `docs/unidades/`, `docs/site_libs/`,
etc.) está ignorado y no debe incluirse en commits normales. Sí permanecen
versionados los recursos que el render no reconstruye por sí solo:
`docs/data/`, `docs/images/` y `docs/lectures/`.

No hagas `push`, despliegues ni cambios de configuración de Pages salvo que el
usuario lo solicite explícitamente. Para un cambio normal, valida localmente y
deja que el usuario decida cuándo publicar.

## PDF del libro

El PDF existe solo para la versión española y se genera localmente:

```bash
./build-pdf.sh
git add docs/*.pdf *.tex
```

Este proceso requiere `lualatex`/TinyTeX o MacTeX, tarda varios minutos y
produce dos artefactos versionados:

- `docs/Almacenes-y-Minería-de-Datos.pdf`, servido por el sitio.
- `Almacenes-y-Minería-de-Datos.tex`, intermedio conservado por `keep-tex`.

No recompiles el PDF como parte de una validación rutinaria. Hazlo cuando el
usuario pida actualizar el PDF o cuando el alcance de una entrega lo incluya
expresamente. Si cambió contenido del libro y no se regeneró, indícalo al
entregar para evitar asumir que HTML y PDF quedaron sincronizados.

La bibliografía global de `_quarto.yml` es necesaria para el documento PDF
completo. Al agregar un archivo `.bib` de capítulo, evalúa si también debe
añadirse allí. Para personalización y limitaciones conocidas del PDF, consulta
`pdf-template/README.md`.

## Archivos generados y limpieza

- No fuerces el agregado de salidas HTML ignoradas de `docs/`.
- No borres ni reemplaces `docs/data/`, `docs/images/` o `docs/lectures/` al
  limpiar un render; son recursos publicados y versionados.
- `./clean.sh` elimina cachés de Quarto, `.DS_Store`, `__pycache__` y `.pyc`,
  pero conserva fuentes y `_freeze/`.
- `lectures/_output/` es salida local ignorada. La copia publicada bajo
  `docs/lectures/` es distinta y está versionada.
- Mantén los entornos `.env/`, `.venv-*` y secretos fuera de Git.

## Criterios de validación

1. Revisa primero `git status` y conserva cambios preexistentes del usuario.
2. Para cambios de contenido, renderiza al menos el capítulo afectado al
   formato correspondiente. Si cambias navegación, configuración global,
   recursos compartidos o ambos idiomas, usa `./build.sh`.
3. Para scripts de shell, ejecuta `bash -n` además de la prueba pertinente.
4. Revisa el diff y confirma que no aparecieron `*.quarto_ipynb*`, carpetas
   `*_files`, cachés o HTML ignorado.
5. Si el cambio afecta una pareja español/inglés, confirma el alcance antes de
   asumir que ambas versiones deben modificarse; son fuentes independientes.
6. Al entregar, informa qué se renderizó, si se reutilizó `freeze` y si el PDF
   quedó actualizado o deliberadamente sin regenerar.

## Limitaciones conocidas

- El PDF no representa algunos diagramas hechos con HTML/CSS crudo.
- El contenido interactivo OJS debe tener una alternativa específica para PDF.
- Hay claves bibliográficas duplicadas entre archivos; el PDF toma la primera
  definición según el orden de la bibliografía global.
- La versión inglesa está configurada actualmente solo para HTML en
  `en/_quarto.yml`; no presupongas que existen artefactos PDF/EPUB ingleses.
