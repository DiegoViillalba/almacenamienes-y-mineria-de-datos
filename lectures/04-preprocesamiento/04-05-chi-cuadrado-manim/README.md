# Reducción de variables categóricas con χ² · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en
[`04-05-chi-cuadrado.qmd`](../04-05-chi-cuadrado.qmd). La fórmula no abre la
clase: aparece cuando el grupo ya la necesita. La secuencia es

1. **El problema:** seis columnas categóricas para predecir la satisfacción de
   1,000 pasajeros; con categorías no hay promedio ni Pearson, sólo conteos.
2. **Ver:** 200 pasajeros (círculo = satisfecho, cuadrado = no) se separan por
   retraso y se convierten en barras de perfil. Retraso se separa de la barra
   general; asiento queda pegado. Independencia = mismo perfil en cada fila.
3. **El mundo independiente:** la tabla esperada `E = fila × columna / n`;
   residuos `O − E = ±20`, que siempre suman cero.
4. **Construir χ²:** cuadrado (quita el signo), ÷ E (escala relativa), suma
   (33.33). La fórmula llega como resumen de los tres pasos.
5. **¿Cuánto χ² produce el azar?** Barajar las etiquetas conserva los totales
   y destruye la relación; 400 barajadas forman un histograma que coincide con
   la curva χ² de 1 g. l. Valor crítico 3.84, valor p, grados de libertad como
   celdas libres y curvas para 1, 2 y 4 g. l.
6. **Significancia ≠ fuerza:** misma tabla ×10 → χ² ×10; V de Cramér.
7. **Reducir:** filtro de relevancia (p y V contra el objetivo), filtro de
   redundancia (V entre predictoras), decisión 6 → 2 columnas, receta y
   precauciones.
8. **Reto en parejas** (Wi-Fi a bordo) y cierre con ticket de salida.

- [Guion docente](guion-docente.md): tiempos, respuestas y errores frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Archivos

- `presentacion.py`: ensambla la escena `ReduccionChiCuadrado`.
- `sections/data.py`: datos y cálculos con NumPy (sin SciPy); la encuesta de
  1,000 pasajeros se genera con la misma semilla que el capítulo, así que los
  χ², p y V coinciden con los del `.qmd`. `python -m sections.data` imprime un
  resumen.
- `sections/theme.py`, `sections/base.py`: paleta con significado fijo,
  tablas de contingencia (`Table2`), barras de perfil, cintillo y notas.
- `sections/part01_opening.py` … `part08_practice.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/04-preprocesamiento/04-05-chi-cuadrado-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py ReduccionChiCuadrado

# Sólo algunos actos (1–9), útil al editar
CHI_PARTS=4,5 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ReduccionChiCuadrado

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py ReduccionChiCuadrado

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true ReduccionChiCuadrado ../../../docs/lectures/_output/04-preprocesamiento/04-05-chi-cuadrado-manim/index.html
```

Abrir `docs/lectures/_output/04-preprocesamiento/04-05-chi-cuadrado-manim/index.html`.
Teclas: `→` avanza, `←` retrocede, `S` notas, `O` vista general.
