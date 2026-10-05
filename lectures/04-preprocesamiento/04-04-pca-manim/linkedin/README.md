# Análisis de Componentes Principales · clip para LinkedIn (4:5)

Mismo video que el Reel de [`../instagram/`](../instagram/), solo
**reencuadrado** a 4:5: `PCALinkedIn` hereda de `PCAInstagram` sin cambiar
la animación y mueve el centro de la cámara a y = 0.47, donde está centrado el
contenido del Reel.

- Formato: 4:5, 1080 × 1350, 30 fps, h264 `yuv420p`, sin audio, 109.5 s.
- Marco 8.8 × 11 (≈123 px/unidad frente a 135 en 9:16): los textos quedan
  ≈9 % más chicos que en el Reel.
- Fuente: [`pca_linkedin.py`](pca_linkedin.py), escena `PCALinkedIn`.
- Salida final: `pca-linkedin.mp4`.
- Guion y texto de publicación: ver [`../instagram/README.md`](../instagram/README.md).
  Para LinkedIn, cambiar "link en mi bio" por "enlace en el primer
  comentario" (MANIM-REDES.md §7).

## Compilar

```bash
../../../../.venv-manim/bin/manim -r 544,680 --fps 15 pca_linkedin.py PCALinkedIn   # vista rápida
../../../../.venv-manim/bin/manim pca_linkedin.py PCALinkedIn
cp media/videos/pca_linkedin/1350p30/PCALinkedIn.mp4 pca-linkedin.mp4
```
