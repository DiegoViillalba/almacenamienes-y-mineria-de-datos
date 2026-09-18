#!/bin/bash
# Compila el sitio HTML completo (ES + EN) usando freeze: los capitulos cuyo
# .qmd no cambio reusan su resultado ya ejecutado (rapido, no requiere
# reinstalar torch/jax). Para forzar reejecucion total: ./build.sh --execute
# Para compilar/actualizar un solo capitulo (mas rapido aun):
#   quarto render unidades/.../archivo.qmd --to html
#   quarto render en/unidades/.../archivo.qmd
#
# Este script SOLO genera HTML (no requiere TinyTeX/LaTeX instalado).
# Para (re)generar el PDF del libro: ./build-pdf.sh
set -e

FLAG="${1:-}"

echo "=== Building English site ==="
quarto render en --to html $FLAG

echo "=== Building Spanish site ==="
quarto render --to html --no-clean $FLAG

echo "=== Build complete! ==="
echo "Spanish site: docs/index.html"
echo "English site: docs/en/index.html"
