#!/bin/bash
# Compila el sitio completo (ES + EN) usando freeze: los capitulos cuyo
# .qmd no cambio reusan su resultado ya ejecutado (rapido, no requiere
# reinstalar torch/jax). Para forzar reejecucion total: ./build.sh --execute
# Para compilar/actualizar un solo capitulo (mas rapido aun):
#   quarto render unidades/.../archivo.qmd
#   quarto render en/unidades/.../archivo.qmd
set -e

FLAG="${1:-}"

echo "=== Building English site ==="
quarto render en $FLAG

echo "=== Building Spanish site ==="
quarto render $FLAG --no-clean

echo "=== Build complete! ==="
echo "Spanish site: docs/index.html"
echo "English site: docs/en/index.html"
