# Análisis de Componentes Principales · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en
`unidades/preprocesamiento/4-4-PCA.qmd`. Empieza con una experiencia intuitiva
—una masa unida a un resorte filmada por tres cámaras: seis columnas, un solo
grado de libertad— y construye PCA paso a paso: centrado, dirección de máxima
varianza, Pitágoras (varianza = proyectada + error), Lagrange (`Su = λu`),
autovectores como direcciones que no giran, una nube 3D casi plana y el
regreso al resorte y a la hélice del capítulo. Cierra con límites (escala,
atípicos, clases), un reto en parejas y un ticket de salida.

- [Guion docente](guion-docente.md): tiempos, respuestas, errores frecuentes.
- Notas del presentador en cada estado (tecla `S`).
- Escenas 3D con giro de cámara en bucle: el estado se repite hasta avanzar.
- [`interactivo.html`](interactivo.html): laboratorio autónomo (sin CDN) para
  girar con el ratón el resorte, la nube 3D y el barrido de direcciones; PCA
  se recalcula en el navegador.

## Archivos

- `presentacion.py`: ensambla la escena `ComponentesPrincipales` (`ThreeDSlide`).
- `sections/data.py`: datos y cálculos con NumPy; todos los números en pantalla
  salen de aquí (`python -m sections.data` imprime un resumen).
- `sections/theme.py`, `sections/base.py`: paleta, textos, cintillo, utilidades
  3D (`fixed`, `oriented`, `keep_fixed`) y la convención de notas `beat()`.
- `sections/part01_opening.py` … `part09_practice.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/04-preprocesamiento/04-04-pca-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py ComponentesPrincipales

# Sólo algunos actos (1–10), útil al editar
PCA_PARTS=4,5 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ComponentesPrincipales

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py ComponentesPrincipales

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true ComponentesPrincipales ../../../docs/lectures/_output/04-preprocesamiento/04-04-pca-manim/index.html

# Laboratorio interactivo
cp interactivo.html ../../../docs/lectures/_output/04-preprocesamiento/04-04-pca-manim/
```

Abrir `docs/lectures/_output/04-preprocesamiento/04-04-pca-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados. Los
estados con giro 3D se repiten en bucle hasta que se avanza.
