# Clasificadores lineales · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** que defiende una tesis: separación
lineal, mínimos cuadrados, Fisher, regresión logística (binaria y multinomial)
y kernels producen, todos, **un hiperplano**. Cada idea se ve primero como
geometría y después se formaliza. Usa los ejemplos de
`unidades/clasificadores/capitulo_regresion_clasificadores_lineales.qmd`
(Fisher) y `capitulo_regresion_logistica.qmd` (seis estudiantes):

1. **Separar con una recta** (0–10): dieciséis puntos separables; una recta
   gira y casi siempre separa (infinitas soluciones); w perpendicular gira la
   recta y b la desliza; distancia con signo g(x)/‖w‖; hiperplano en ℝᵈ;
   perceptrón en tres correcciones y la pregunta «¿cuál recta?».
2. **Mínimos cuadrados** (10–19): y = ±1, residuos dibujados como cuadrados
   (SSE 6 → 3.2), frontera en 3.5 h; Sxy = 7, Sxx = 17.5 → β = (−1.4, 0.4);
   el tazón cuadrático: gradiente lineal en β → ecuaciones normales → un solo
   salto; cinco puntos correctos y lejanos llevan a MCO de 0 a 3 errores.
3. **Fisher** (19–25, opcional): sombras sobre una dirección que gira;
   J = 4.22 (medias) → 5.64 (Fisher); w ∝ S_W⁻¹(m_A − m_B), otra fórmula.
4. **Regresión logística** (25–39): la recta se dobla en sigmoide; p = ½ ⇔
   z = 0 (sigue siendo lineal); multas −log p que se encogen por los pasos de
   Newton (0.693 → 0.413); por qué no hay fórmula (β dentro de σ); curvas de
   nivel elípticas contra deformadas, un salto contra Newton (4–5 pasos) y
   gradiente (~3000); separabilidad perfecta (β → ∞, regularizar); los puntos
   lejanos ya no arrastran la frontera.
5. **Muchas variables** (39–48): niveles de p como rectas paralelas
   (wᵀx + b = logit p); proyección sobre w → la logística de una variable;
   un plano en un cubo que gira; softmax con tres clases: fronteras
   (w_k − w_j)ᵀx + (b_k − b_j) = 0.
6. **Kernels** (48–56): una recta con la clase central → (x, x²) y una recta
   horizontal → dos umbrales ±1.41; un anillo → paraboloide y plano →
   círculo r = 1.34; K(x, z) = (xᵀz)² = 25 = φ(x)ᵀφ(z); campanas RBF que sólo
   se escalan (α = (K + λI)⁻¹y): lineal en α.
7. **Comparar, ejercicio y cierre** (56–60): tabla de criterios, ejercicio a
   mano con w = (2, −1), b = −3, y ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Cómo está organizado (qué archivo editar)

```text
presentacion.py            ensambla la escena ClasificadoresLineales y elige los actos
sections/
├── data.py                TODOS los números: puntos, perceptrón, MCO, Fisher,
│                          logística (Newton y gradiente), softmax, kernels
├── plane.py               plano isotrópico, fronteras, semiplanos, recorte de
│                          polígonos y proyección pseudo-3D
├── theme.py               paleta, tipografía y piezas pequeñas (pill, card, table…)
├── base.py                cintillo institucional, encabezados, beat() = pausa + notas
├── part01_separate.py     acto 1
├── part02_ols.py          acto 2 · también define los ejes de parámetros y contornos
├── part03_fisher.py       acto 3
├── part04_logistic.py     acto 4
├── part05_multi.py        acto 5
├── part06_kernels.py      acto 6
└── part07_close.py        acto 7
```

- **Cambiar un número o un ejemplo:** sólo en `sections/data.py`;
  `python -m sections.data` imprime el resumen (coincide con el libro).
- **Cambiar un color:** `sections/theme.py`.
- **Cambiar una lámina:** cada estado empieza con `self.beat("notas…")`.
- **Quitar o reordenar actos:** la lista `acts` de `presentacion.py` (por
  ejemplo, quitar `self.fisher` si no da tiempo).

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-08-clasificadores-lineales-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py ClasificadoresLineales

# Sólo algunos actos (1–7), útil al editar
CLS_PARTS=4,5 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ClasificadoresLineales

# Entrega 4K (3840 × 2160) a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 3840,2160 --fps 30 --max-inflight-encoders 4 presentacion.py ClasificadoresLineales

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true ClasificadoresLineales ../../../docs/lectures/_output/06-clasificacion/06-08-clasificadores-lineales-manim/index.html
```

Abrir `docs/lectures/_output/06-clasificacion/06-08-clasificadores-lineales-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.

Nota: no usar pausas de un solo fotograma (`self.wait(0.1)` a 15 fps); rompen
la concatenación de Manim Slides. Usar al menos `self.wait(0.3)`.
