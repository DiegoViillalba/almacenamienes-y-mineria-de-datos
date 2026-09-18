# Plantilla del PDF del libro

Cómo está armado el PDF y cómo tocarlo sin tener que reaprender LaTeX desde cero.

## Cómo funciona

Quarto arma el PDF pasando el libro por Pandoc y luego por LaTeX (`lualatex`).
La plantilla LaTeX que usa por defecto vive dentro de la instalación de
Quarto, dividida en pedazos ("partials"):

```
/Applications/quarto/share/formats/pdf/pandoc/*.tex
```

En vez de copiar y modificar la plantilla completa (que se pisaría en cada
actualización de Quarto), solo sobreescribimos el pedazo que nos interesa.
Eso se declara en `_quarto.yml`:

```yaml
format:
  pdf:
    template-partials:
      - pdf-template/title.tex
```

`pdf-template/title.tex` es una copia del partial original de Quarto
(`.../pandoc/title.tex`) con una línea agregada: mete el logo arriba del
título dentro del mismo `\title{...}`.

## Cambiar el logo de portada

Edita `pdf-template/title.tex`, línea del `\includegraphics`:

```latex
\includegraphics[width=6cm]{images/Logo_FC_Color.png}\\[1.5em]
```

- La ruta es relativa a la raíz del repo (donde vive `_quarto.yml`).
- `width=6cm` controla el tamaño; súbelo o bájalo a gusto.
- `\\[1.5em]` es el salto de línea + espacio antes del título; ajusta el
  espacio cambiando `1.5em`.

## Cambiar los márgenes

En `_quarto.yml`, dentro de `format.pdf`:

```yaml
geometry:
  - margin=2cm
```

Puedes usar un solo valor (como ahora) o separar por lado, por ejemplo:

```yaml
geometry:
  - top=2cm
  - bottom=2cm
  - left=1.5cm
  - right=1.5cm
```

Nota: 2cm es un margen "para pantalla/PDF", no para impresión con
encuadernación (ahí normalmente se necesita más margen interior para el
doblez/engargolado, y a veces un margen distinto en páginas pares/impares).
Si en algún momento se va a imprimir el libro físicamente, probablemente
haya que revisar esto.

## Sobreescribir otras partes de la plantilla

El mismo mecanismo sirve para cualquier otro partial. Por ejemplo, para
tocar la tabla de contenidos, los encabezados de capítulo, la bibliografía,
etc.:

1. Mira qué partials existen:
   ```bash
   ls /Applications/quarto/share/formats/pdf/pandoc/
   ```
2. Copia el que te interese a `pdf-template/`:
   ```bash
   cp /Applications/quarto/share/formats/pdf/pandoc/toc.tex pdf-template/toc.tex
   ```
3. Edítalo, y agrégalo a la lista en `_quarto.yml`:
   ```yaml
   template-partials:
     - pdf-template/title.tex
     - pdf-template/toc.tex
   ```

Los partials más probables de querer tocar:
- `doc-class.tex`: la clase de documento y sus opciones (`scrbook`, tamaño
  de fuente base, etc.)
- `before-body.tex`: qué pasa justo después de `\begin{document}` (aquí
  corre `\maketitle`).
- `toc.tex`: cómo se genera la tabla de contenidos.
- `biblio.tex` / `biblio-config.tex`: formato de bibliografía.

## Probar cambios localmente

```bash
./build-pdf.sh
# equivalente a: quarto render --to pdf --no-clean
```

Genera `docs/Almacenes-y-Minería-de-Datos.pdf`. El libro completo (~1550
páginas) tarda unos minutos incluso con `freeze` activado, porque LaTeX
tiene que tipografiar todo de nuevo aunque el código Python no se
reejecute.

Requiere una instalación de LaTeX con `lualatex` en el PATH (en esta Mac ya
está vía MacTeX/`/Library/TeX/texbin`). Si `quarto render` se queja de un
paquete LaTeX faltante y no tienes permisos de escritura sobre la
instalación del sistema, instálalo en tu árbol de usuario (no necesita
`sudo`):

```bash
tlmgr --usermode install <paquete-faltante>
```

## El PDF ya no se compila en cada push (CI)

Compilar el PDF es lo mas lento del proceso (TinyTeX + paquetes LaTeX +
tipografiar ~1550 paginas), y el contenido del libro no cambia en cada
commit. Por eso el PDF **no** se genera en GitHub Actions: `publish.yml`
solo renderiza HTML (rapido, sin TinyTeX) y publica lo que ya este dentro
de `docs/` — PDF incluido, si lo commiteaste.

Flujo para actualizar el PDF publicado:

```bash
./build.sh           # HTML, rapido, para revisar contenido mientras editas
./build-pdf.sh        # cuando el contenido este listo, recompila el PDF
git add docs/*.pdf *.tex
git commit -m "Actualiza PDF del libro"
git push
```

El link de "Download PDF" del navbar (`downloads: [pdf]` en `_quarto.yml`)
apunta a este archivo fijo en `docs/`; si no lo recompilas y commiteas tras
un cambio de contenido, el PDF publicado queda desactualizado respecto al
HTML (el HTML si se regenera solo, via CI, en cada push).

## Editar el `.tex` a mano

`keep-tex: true` en `_quarto.yml` hace que, ademas del PDF, `quarto render
--to pdf` deje el archivo `.tex` intermedio. A diferencia del PDF, este
**no** queda en `docs/` sino en la **raiz del repo**
(`Almacenes-y-Minería-de-Datos.tex`) — es donde Quarto arma el documento
combinado antes de invocar LaTeX, y las rutas `\includegraphics{...}` que
contiene son relativas a esa raiz (p. ej. `images/Logo_FC_Color.png`), asi
que hay que recompilarlo desde ahi tambien:

```bash
./build-pdf.sh                                # genera/actualiza el .tex
# ... editar Almacenes-y-Minería-de-Datos.tex a mano ...
lualatex Almacenes-y-Minería-de-Datos.tex      # recompila solo ese archivo
cp Almacenes-y-Minería-de-Datos.pdf docs/      # y copia el resultado a docs/
```

Ojo: un `.tex` editado a mano se **pisa** la proxima vez que corras
`./build-pdf.sh`, porque Quarto lo regenera desde los `.qmd` cada vez. Si
el cambio es algo que quieres conservar permanentemente, hazlo en la
plantilla (`pdf-template/*.tex`) o en el `.qmd` fuente, no solo en el
`.tex` generado.

## Limitaciones conocidas de esta primera versión

Estas son cosas que **no** se resolvieron todavía, para que no sea
sorpresa si las ves en el PDF:

- **Diagramas hechos con HTML/CSS crudo** (~29 capítulos usan `<div>`
  personalizados para flowcharts) **desaparecen silenciosamente** en el
  PDF — LaTeX no puede interpretarlos. El texto alrededor sigue fluyendo
  normal, solo falta el diagrama. Arreglarlo implica convertir cada uno a
  imagen estática o a Mermaid (que sí compila a PDF).
- La sección de **exploración interactiva de cubos OLAP** (OJS) se
  reemplazó por una nota en el PDF ("consúltala en la versión web"); ver
  `unidades/OLAP/capitulo_cubos_olap.qmd`, bloques
  `::: {.content-visible when-format="pdf"}` / `when-format="html"`. Ese
  mismo patrón sirve para excluir cualquier otro contenido HTML-only del
  PDF.
- Al menos una tabla (capítulo del Lakehouse, "Vocabulario mínimo") tiene
  una columna con texto monoespaciado (`s3://...`, nombres de catálogo)
  que se desborda sobre la columna siguiente. Es un problema de ancho de
  columna en esa tabla específica, no de la plantilla general.
