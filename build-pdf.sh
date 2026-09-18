#!/bin/bash
# Compila SOLO el PDF del libro (version en espanol). Requiere TinyTeX/LaTeX
# instalado localmente (ver pdf-template/README.md). Es lento (varios
# minutos) por eso esta separado de build.sh y no se ejecuta en CI.
#
# El PDF resultante queda en docs/ (se sube al sitio junto con el HTML y se
# enlaza desde el navbar, `downloads: [pdf]` en _quarto.yml). Como
# keep-tex:true en _quarto.yml, tambien queda el .tex intermedio -- pero en
# la RAIZ del repo, no en docs/: las rutas a imagenes dentro del .tex son
# relativas a la raiz del proyecto, asi que hay que recompilarlo desde ahi
# (ver "Editar el .tex a mano" en pdf-template/README.md).
#
# Flujo tipico tras editar contenido de un capitulo:
#   ./build.sh            # HTML, rapido, para revisar el contenido
#   ./build-pdf.sh         # cuando el contenido este listo, refresca el PDF
#   git add docs/*.pdf *.tex && git commit && git push
set -e

echo "=== Building PDF (Spanish book) ==="
quarto render --to pdf --no-clean

echo
echo "=== PDF build complete! ==="
find docs -maxdepth 1 -name "*.pdf" -exec ls -la {} \;
find . -maxdepth 1 -name "*.tex" -exec ls -la {} \;
