# Introducción a los clasificadores · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** que abre la parte de Clasificación.
Usa en todo momento el ejemplo de los diez clientes de
`unidades/clasificadores/6-0-introduccion-clasificacion.qmd` (ingreso, deuda,
¿pagó?) y sus cinco clientes de prueba A–E. Cada concepto se ve antes de
formalizarse:

1. **¿Qué es clasificar?** (0–10): decisiones cotidianas → una tabla con
   atributos `x` y etiqueta `y` → puntos en un plano → la predicción `ŷ`, la
   referencia «siempre paga» y una recta que sube contando errores hasta la
   regla «deuda > 44 ⇒ no paga». Al final, `ŷ = f(x)` y frontera de decisión.
2. **Matriz de confusión** (10–22): positivo = «no paga»; los clientes viajan
   del plano a las cuatro celdas; exactitud, precisión y sensibilidad se leen
   como diagonal, columna y fila; mover el corte cambia las barras; la
   paradoja de la exactitud (97 %) y un «tu turno».
3. **k vecinos más cercanos** (22–37): un círculo crece alrededor del cliente
   B; una lupa convierte la distancia en Pitágoras; votación con k = 3;
   regiones para k = 1, 3, 9; la trampa de las unidades; «tu turno» con el
   cliente C; entrenamiento contra prueba.
4. **Clasificadores lineales** (37–52): el corte gira hasta la recta de la
   regresión logística; distancia con signo = puntaje `z`; los clientes suben
   a la sigmoide; bandas de probabilidad; entrenamiento como descenso de la
   «multa» −log p; fórmulas, lectura de pesos y límites de una recta.
5. **Comparación y ejercicio** (52–60): 3-NN (3/5) contra logística (4/5) en
   prueba, matrices lado a lado, umbral 0.2, ejercicio a mano y ticket de
   salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Archivos

- `presentacion.py`: ensambla la escena `IntroClasificadores` (`Slide`).
- `sections/data.py`: clientes, barrido del corte, matrices y métricas, k-NN,
  regresión logística (Newton) y trayectoria de descenso de gradiente. Todos
  los números en pantalla salen de aquí; `python -m sections.data` imprime el
  resumen que usa el guion.
- `sections/plane.py`: plano ingreso × deuda con escala isotrópica (un
  círculo es una distancia euclidiana), fronteras y regiones.
- `sections/confmat.py`: matriz de confusión animable y barras de métricas.
- `sections/theme.py`, `sections/base.py`: paleta (círculo azul = paga,
  cuadrado naranja = no paga, amarillo = cliente nuevo, verde/rojo =
  acierto/error), cintillo, encabezados y convención `beat()`.
- `sections/part01_classify.py` … `part05_compare.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-01-intro-clasificadores-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py IntroClasificadores

# Sólo algunos actos (1–5), útil al editar
CLS_PARTS=3,4 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py IntroClasificadores

# Entrega 1920 × 1080 a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py IntroClasificadores

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true IntroClasificadores ../../../docs/lectures/_output/06-clasificacion/06-01-intro-clasificadores-manim/index.html
```

Abrir `docs/lectures/_output/06-clasificacion/06-01-intro-clasificadores-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.
