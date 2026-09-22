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
#   ./build.sh             # HTML, rapido, para revisar el contenido
#   ./build-pdf.sh         # cuando el contenido este listo, refresca el PDF
#   git add docs/*.pdf *.tex && git commit && git push
#
# ==============================================================================
# CARACTERÍSTICAS DEL PDF (Formato Libro Académico):
# 1. Supresión de Código: Se eliminan bloques de código, terminales y errores de
#    render interactivo mediante el filtro `pdf-template/hide-code.lua`. El PDF
#    es enteramente teórico y remite a los estudiantes al HTML para la práctica.
# 2. Límite de Imágenes y Mermaid (70%): El script `header.tex` redefine la macro 
#    `\pandocbounded` y el filtro Lua elimina dimensiones estrictas de los diagramas
#    Mermaid para forzar que NINGUNA figura supere el 70% del ancho del texto.
# 3. Estilo KOMA-Script: Se incluyen estilos de encabezado sobrios, portada 
#    institucional y página de derechos (Creative Commons) usando 
#    `title.tex` y `before-body.tex`. Todo configurado en `_quarto.yml`.
# ==============================================================================
set -e

echo "=== Building PDF (Spanish book) ==="
quarto render --to pdf --no-clean

echo
echo "=== PDF build complete! ==="
find docs -maxdepth 1 -name "*.pdf" -exec ls -la {} \;
find . -maxdepth 1 -name "*.tex" -exec ls -la {} \;
