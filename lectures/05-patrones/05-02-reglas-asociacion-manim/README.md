# Reglas de asociación · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en los capítulos de
`unidades/reglas_asociacion/` (5.0, 5.1 y 5.2) y en las notas de
`lectures/notes/guia-docente-asociacion.md`. Empieza con una experiencia
intuitiva —el gerente que se entusiasma porque 60 % de quienes llevan pan dulce
también llevan leche, cuando la leche ya está en 60 % de todos los tickets— y
construye la teoría en cuatro pasos:

1. **Construir:** cinco tickets → matriz binaria → soporte → Apriori →
   confianza → lift.
2. **Ver:** el cuadrado unitario de probabilidad. Soporte = área, confianza =
   altura del «agua» dentro de la columna del antecedente, lift = altura / nivel
   de independencia; simetría y techo del lift.
3. **Medir:** leverage (tickets de más), conviction (errores esperados /
   observados) y Zhang (dentro contra fuera) en el mismo dibujo.
4. **Decidir:** red de asociación, mapa soporte × lift, fichas de decisión,
   filtro mental y cuándo no usar reglas.

Cierra con un reto en parejas y un ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Archivos

- `presentacion.py`: ensambla la escena `ReglasAsociacion` (`Slide`).
- `sections/data.py`: tickets, reglas y métricas; todos los números en
  pantalla salen de aquí (`python -m sections.data` imprime un resumen).
- `sections/theme.py`, `sections/base.py`: paleta con significado fijo
  (azul = A, naranja = B, verde = positiva, amarillo = independencia, rojo =
  errores o negativa, morado = métricas), cintillo, convención `beat()` y el
  objeto `UnitSquare`.
- `sections/part01_opening.py` … `part06_practice.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/05-patrones/05-02-reglas-asociacion-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py ReglasAsociacion

# Sólo algunos actos (1–7), útil al editar
REGLAS_PARTS=3,4 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ReglasAsociacion

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py ReglasAsociacion

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true ReglasAsociacion ../../../docs/lectures/_output/05-patrones/05-02-reglas-asociacion-manim/index.html
```

Abrir `docs/lectures/_output/05-patrones/05-02-reglas-asociacion-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.
