#!/bin/bash
set -e

echo "=== Building English site ==="
quarto render en

echo "=== Building Spanish site ==="
quarto render --no-clean

echo "=== Build complete! ==="
echo "Spanish site: docs/index.html"
echo "English site: docs/en/index.html"
