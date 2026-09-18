# Medidas de heterogeneidad · repaso animado

Presentación didáctica de Manim Slides para **Almacenes y Minería de Datos**.
Está diseñada como repaso aplicado: los estudiantes ya conocen el tema y
reconstruyen sus medidas mediante predicciones, áreas, distancias y cálculos
breves.

## Resultado de aprendizaje

Al terminar, el estudiante puede:

- distinguir diversidad categórica de separación de una respuesta numérica
  entre grupos;
- convertir conteos a proporciones y leer el reparto antes de calcular;
- construir e interpretar Gini–Simpson y la entropía de Shannon;
- declarar el número de categorías, la base logarítmica y el esquema de
  muestreo;
- traducir índices a categorías efectivas;
- descomponer `SS_T = SS_B + SS_W` e interpretar `η²` sin atribuir causalidad;
- resolver un conjunto nuevo y comunicar número, convención y sentido.

## Archivos y Arquitectura Modular

La presentación está dividida en submódulos dentro de `sections/` para permitir editar diapositivas y fórmulas de manera ágil sin manipular un archivo monolítico:

- `heterogeneidad.py`: ensamblador de la escena `MedidasHeterogeneidad` mediante herencia múltiple de las secciones.
- `sections/`:
  - `theme.py`: paleta de colores, **configuración central de tipografía (`FONT`)**, rutas de logos y funciones auxiliares de dibujo (`safe_text`, `distribution_chart`, `probability_grid`, etc.).
  - `base.py`: clase base `BaseSlide` con utilidades de navegación, pausas (`pause`), limpieza (`clear_content`) y cintillo institucional (`add_branding`).
  - `part01_opening.py`: portada, las dos preguntas de heterogeneidad y datos categóricos iniciales.
  - `part02_gini_simpson.py`: Gini–Simpson, probabilidad de desacuerdo, geometría del cuadrado unitario y cotas.
  - `part03_shannon.py`: entropía de Shannon, función de sorpresa, curva analítica continua $f(p) = -p\ln p$ con punto crítico en $1/e$ y normalización.
  - `part04_comparison.py`: comparación entre escenarios dominantes/uniformes y números de Hill.
  - `part05_anova.py`: respuesta continua en tres grupos, descomposición de áreas euclidianas $SS_T = SS_B + SS_W$, $\eta^2$ y heatmap con soporte muestral.
  - `part06_transfer.py`: reto aplicado con datos nuevos de canales de atención y solución guiada.
  - `part07_closing.py`: cierre metodológico, convenciones y puentes hacia minería de datos (CART, C4.5, $k$-means).
- `interactivo.html`: simulador web interactivo autónomo (Canvas/SVG) para experimentación directa en clase.
- `guion-docente.md`: guía paso a paso para el profesor con tiempos, notas y resultados exactos.
- `manim.cfg`: configuración de render Full HD 1080p a 30 fps.
- `requirements.txt`: dependencias exactas del entorno.

### Cómo cambiar la tipografía (fuente)

En `sections/theme.py`, edita la variable `FONT`:

```python
# sections/theme.py

FONT = "Sans"                # Sans-serif genérica y portable (actual)
# FONT = ""                  # Fuente por defecto de Manim (serif en este equipo)
# FONT = "Helvetica"         # Sans-serif clásica en macOS
# FONT = "Arial"             # Sans-serif estándar multiplataforma
# FONT = "Fira Sans"         # Tipografía técnica moderna
# FONT = "CMU Serif"         # Estilo clásico LaTeX Computer Modern
```

Cualquier cambio en `FONT` afectará inmediatamente a todos los títulos, subtítulos, etiquetas y tarjetas de todas las diapositivas.

## Entorno

Desde la raíz del repositorio:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python \
  -r lectures/03-eda/03-05-medidas-heterogeneidad-manim/requirements.txt
```

Comprobar las herramientas:

```bash
.venv-manim/bin/manim --version
.venv-manim/bin/manim-slides --version
.venv-manim/bin/manim-slides checkhealth
```

## Compilar

Ejecutar desde esta carpeta:

```bash
cd lectures/03-eda/03-05-medidas-heterogeneidad-manim

```

Render rápido para revisión:

```bash
../../../.venv-manim/bin/manim-slides render -ql \
  heterogeneidad.py MedidasHeterogeneidad
```

Render final:

```bash
../../../.venv-manim/bin/manim-slides render \
  --resolution 1920,1080 \
  --fps 30 \
  --max-inflight-encoders 4 \
  heterogeneidad.py MedidasHeterogeneidad
```

Exportar un único HTML sin dependencias externas:

```bash
mkdir -p ../../../docs/lectures/_output/03-eda/03-05-medidas-heterogeneidad-manim
../../../.venv-manim/bin/manim-slides convert \
  --folder slides \
  --offline \
  --one-file \
  -ccontrols=true \
  -cprogress=true \
  -cslide_number=true \
  -chash=true \
  MedidasHeterogeneidad \
  ../../../docs/lectures/_output/03-eda/03-05-medidas-heterogeneidad-manim/index.html
```

Actualizar la galería desde la raíz:

```bash
quarto render presentaciones.qmd --no-execute
```

## Convenciones estadísticas

- `K` es el conjunto de niveles declarado antes de comparar distribuciones;
  puede incluir niveles con frecuencia cero.
- Se usa `ln`, por lo que la entropía está en **nats**.
- Se adopta `0 ln 0 := 0` por continuidad.
- `D = 1 - Σp²` se llama **Gini–Simpson** y corresponde a la probabilidad de
  categorías distintas en dos extracciones independientes/con reemplazo.
- Para `K` fijo, `0 ≤ D ≤ 1 - 1/K`; el máximo no es uno.
- Los números efectivos son `exp(H)` y `1/Σp²`, no `1/D`.
- `η² = SS_B/SS_T` es una descripción de la suma de cuadrados observada; no
  demuestra causalidad ni sustituye la inspección de los datos.

## Navegación

- `→` o `Page Down`: avanzar.
- `←`: volver con la animación inversa.
- `S`: abrir la vista del presentador y las notas.
- `O`: abrir la vista general.
- `B`: oscurecer la pantalla.
- `?`: mostrar la ayuda de Reveal.js.

El reto de transferencia muestra `PAUSA REAL · 3 min`. La solución se construye
en dos estados posteriores: primero proporciones y Gini–Simpson; después
Shannon, normalización y categorías efectivas. Un estado final consolida la
guía de decisión.

## Entrega validada
 
- Resolución: `1920 × 1080`.
- Frecuencia: `30 fps`.
- Estados: `38`.
- Notas docentes: `37`.
- Formato publicable: un solo `index.html` (Reveal.js + video clips embebidos sin CDN) y simulador autónomo `interactivo.html`.
