# Árboles de decisión · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** sobre árboles de clasificación:
ID3, C4.5 y CART. Construye cada idea desde la geometría (rectángulos que
parten el plano, curvas de impureza, áreas, rectas de costo) antes de la
fórmula, siempre con los ejemplos del libro:

1. **Preguntas que parten el plano** (0–10): los diez clientes de la clase
   anterior; la regla «deuda > 44» ya es un árbol; cada pregunta nueva vive
   sólo en su rectángulo; un cliente nuevo cae por el árbol; f(x) = Σ c·1[x∈R].
2. **Entropía e ID3** (10–23): quince días de béisbol; las filas se vuelven
   fichas y se reparten en cubetas; medidor de mezcla H(p); la entropía
   ponderada como suma de áreas y la ganancia como el área que desaparece;
   recursión, reglas SI… ENTONCES y algoritmo de Hunt.
3. **C4.5** (23–31): la trampa del atributo «Día»; SplitInfo como barra
   segmentada; razón de ganancia; barrido de cortes de la humedad numérica.
4. **CART** (31–43): entropía, Gini y error en un plano; concavidad con una
   cuerda; Gini a mano con los compradores de laptop; barrido de cortes en el
   plano de los clientes (reconstruye el árbol del acto 1); invariancia a
   transformaciones monótonas.
5. **Sobreajuste y poda** (43–54): 60 puntos ruidosos; árbol de 18 hojas
   (100 % → 71.7 % en prueba); la poda de costo-complejidad como envolvente de
   rectas R(T) + α|T|; eslabón más débil; ruta de poda.
6. **Límites, ejercicio y cierre** (54–60): inestabilidad (un punto cambia la
   raíz), fronteras en escalera, ejercicio a mano y ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).
- Continúa en [Ensambles y XGBoost](../06-07-ensambles-xgboost-manim/).

## Cómo está organizado (qué archivo editar)

```text
presentacion.py            ensambla la escena ArbolesDecision y elige los actos
sections/
├── data.py                TODOS los números: datos del libro, impurezas, ganancias,
│                          un CART mínimo en NumPy y la ruta de poda
├── theme.py               paleta, tipografía y piezas pequeñas (pill, card, table…)
├── base.py                cintillo institucional, encabezados, beat() = pausa + notas
├── widgets.py             TreeView (diagrama de árbol que crece), Plane (plano con
│                          regiones), fichas, cubetas y barras de área
├── part01_partition.py    acto 1 · también define el plano y el árbol de los clientes
├── part02_entropy.py      acto 2
├── part03_c45.py          acto 3
├── part04_cart.py         acto 4
├── part05_pruning.py      acto 5
└── part06_close.py        acto 6
```

- **Cambiar un número o un ejemplo:** sólo en `sections/data.py`. Las
  láminas leen de ahí; `python -m sections.data` imprime el resumen que se usa
  en el guion para comprobarlo.
- **Cambiar un color o la tipografía:** `sections/theme.py` (la función de cada
  color está documentada al inicio).
- **Cambiar una lámina:** cada acto es una clase con un método por estado; cada
  estado empieza con `self.beat("notas…")`, que crea la pausa y sus notas.
- **Quitar o reordenar actos:** la lista `acts` de `presentacion.py`.

El CART de `data.py` reproduce los resultados de `scikit-learn` del libro
(18 hojas, 71.7 % en prueba, la ruta α = 0.0141 … 0.0902): usa cortes en puntos
medios, valores en `float32` y, en empates, prefiere `x₁`.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-06-arboles-decision-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py ArbolesDecision

# Sólo algunos actos (1–6), útil al editar
CLS_PARTS=4,5 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ArbolesDecision

# Entrega 4K (3840 × 2160) a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 3840,2160 --fps 30 --max-inflight-encoders 4 presentacion.py ArbolesDecision

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true ArbolesDecision ../../../docs/lectures/_output/06-clasificacion/06-06-arboles-decision-manim/index.html
```

Abrir `docs/lectures/_output/06-clasificacion/06-06-arboles-decision-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.

Nota: no usar pausas de un solo fotograma (`self.wait(0.1)` a 15 fps); rompen
la concatenación de Manim Slides. Usar al menos `self.wait(0.3)`.
