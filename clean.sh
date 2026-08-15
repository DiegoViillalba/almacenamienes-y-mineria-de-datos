#!/bin/bash
set -e

echo "=== Limpiando cachés temporales de Quarto y del sistema ==="

# 1. Eliminar carpeta de caché interno de Quarto
rm -rf .quarto/ en/.quarto/

# 2. Eliminar archivos ocultos del sistema operativo (.DS_Store)
find . -name ".DS_Store" -type f -delete

# 3. Eliminar archivos temporales de Python
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true

echo "=== Limpieza completada exitosamente ==="
echo "Los archivos fuente (.qmd), datos y código frozen (_freeze) permanecen intactos."
