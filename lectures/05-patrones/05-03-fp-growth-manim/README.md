# FP-Growth · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en el capítulo
`unidades/reglas_asociacion/5-3-fp_growth.qmd` (mismo ejemplo de cinco
tickets, soporte mínimo 3) y en las 50 canastas de `5-1-apriori.qmd`. Empieza
por la intuición —en qué gasta Apriori su tiempo y cómo un árbol de prefijos
guarda «sol, sola, solar, sopa» sin repetir letras— y construye el algoritmo
en cuatro pasos:

1. **Comprimir:** contar, fijar el orden global, insertar los tickets uno a
   uno (prediciendo nodos nuevos), tabla de encabezados y el contraste con el
   orden inverso (6 contra 10 nodos).
2. **Minar:** partir los 15 itemsets por su último ítem, base condicional de
   Cerveza (el peso es el contador del nodo minado), poda, árbol condicional,
   «tu turno» con Pañales, recursión y pseudocódigo.
3. **Comparar:** lecturas y candidatos frente a Apriori, compresión en las 50
   canastas y en datos densos, cuándo conviene cada uno y `mlxtend`.
4. **Practicar:** reto en parejas con seis tickets de cafetería, cierre y
   ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Archivos

- `presentacion.py`: ensambla la escena `FPGrowth` (`Slide`).
- `sections/data.py`: tickets, FP-tree, bases condicionales, minado recursivo,
  traza de Apriori y curvas de compresión; todos los números en pantalla salen
  de aquí (`python -m sections.data` imprime un resumen y verifica contra
  fuerza bruta).
- `sections/fptree.py`: `TreeView`, dibujo animable del árbol (inserción,
  contadores, caminos y enlaces de nodo).
- `sections/theme.py`, `sections/base.py`: paleta (cada ítem conserva su
  color; verde = frecuente, rojo = podado, amarillo = foco), cintillo,
  encabezados y convención `beat()`.
- `sections/part01_opening.py` … `part05_practice.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/05-patrones/05-03-fp-growth-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py FPGrowth

# Sólo algunos actos (1–6), útil al editar
FP_PARTS=2,3 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py FPGrowth

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py FPGrowth

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true FPGrowth ../../../docs/lectures/_output/05-patrones/05-03-fp-growth-manim/index.html
```

Abrir `docs/lectures/_output/05-patrones/05-03-fp-growth-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.
