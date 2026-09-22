# Medidas de concentración · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en `unidades/intro_mineria/2-7-concentracion.qmd`. El ejemplo continuo son cinco cafeterías con ventas `5, 10, 15, 20, 50`. Cada bloque presenta primero una pregunta visual y luego formaliza `CRₖ`, HHI, Lorenz o Gini. El [guion docente](guion-docente.md) contiene tiempos, respuestas y errores frecuentes; las pausas tienen notas para la vista del presentador (`S`).

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/03-eda/03-06-medidas-concentracion-manim/requirements.txt
```

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py MedidasConcentracion

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py MedidasConcentracion

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true MedidasConcentracion ../../../docs/lectures/_output/03-eda/03-06-medidas-concentracion-manim/index.html
```

Abrir `docs/lectures/_output/03-eda/03-06-medidas-concentracion-manim/index.html`. `→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados. El HTML final está integrado en `presentaciones.qmd`.
