# Clasificadores lineales · edición para video

Clase específica de Diego Villalba para **Almacenes y Minería de Datos**,
Facultad de Ciencias, UNAM. La pregunta «Dos rectas aciertan. ¿Cuál elegir?»
conecta geometría, perceptrón, mínimos cuadrados, Fisher, logística, softmax y
kernels. Es una clase posterior a la introducción «¿Pagará el préstamo?».

La duración propuesta del video narrado es **36–42 minutos**, con capítulos;
debe ajustarse después del ensayo. Se conserva el contenido esencial y se
reparten los cálculos y derivaciones en **65 estados**. En aula, la discusión y
las actividades permiten un recorrido de unos 60–70 minutos.

## Entregas

- [HTML autónomo offline](../../../docs/lectures/_output/06-clasificacion/06-08-clasificadores-lineales-manim/index.html), con animaciones y notas.
- [Guion de grabación](guion-youtube.md), narración completa, notas y correspondencia E01–E65.
- [MP4 de animaciones](media/videos/presentacion/1080p30/ClasificadoresLineales.mp4), 1920 × 1080 y 30 fps.
- [Pauta para el aula](guion-docente.md).
- [Revisión editorial y supuestos](revision-youtube.md).

**El HTML y el MP4 no tienen voz. No son un video narrado terminado.** El HTML
permite mantener un fotograma mientras se explica; el MP4 conserva sólo las
animaciones y pausas breves. Para montar un video, grabar la narración y
prolongar los estados estables en edición, o capturar el HTML mientras se habla.
El MP4 es un artefacto local ignorado; el HTML final se versiona.

## Contenido y supuestos

1. Geometría: mismo conjunto de puntos, giros de w, desplazamientos de b,
   distancia con signo y perceptrón. Hiperplano con w≠0 y desempate declarado.
2. MCO: seis estudiantes, SSE 6→3.2, sumas manuales, ajuste (−1.4,0.4) y
   ecuaciones normales. Inversa con rango completo; QR/SVD en implementación.
3. Fisher: sombras y J 4.22→5.64; S_W definida positiva. La dirección no fija
   el umbral; punto medio con previas/costos iguales e interpretación LDA con
   gaussianas de covarianza común.
4. Logística: puntaje, sigmoide, pérdida individual, Newton y gradiente.
   Convexidad, existencia, unicidad y convergencia se distinguen explícitamente.
   En los lejanos se usa pérdida promedio más λ||w||²/2, λ=0.01, sin penalizar b.
5. Varias variables: niveles paralelos, proyección sobre w, plano y softmax.
   Sólo los empates entre puntajes máximos forman frontera activa.
6. Características y kernels: (x,x²), anillo, producto cuadrático 25 y campanas
   RBF. El último ejemplo es kernel ridge con λ=0.3 y sin intercepto, con objetivo
   ||y−Kα||²+λαᵀKα; no representa entrenamiento de una SVM.
7. Comparación y ejercicio con revelación individual de P, Q y R.

Los aciertos son de entrenamiento. No se afirma robustez general frente a
atípicos ni superioridad de generalización. Cambiar umbrales, escalas o
regularización requiere validación independiente de la prueba final.

## Organización

`presentacion.py` ensambla los siete actos. `sections/data.py` calcula los
números; `sections/plane.py` contiene geometría y capas; `sections/theme.py`
usa una paleta oscura inspirada en las infografías de tinta sobre crema: fondo
carbón #14191B, texto crema, círculos azules (periwinkle), cuadrados ocre, pesos
violeta, frontera crema y verde salvia para probabilidades. La terracota marca
el error y el énfasis; turquesa representa proyecciones; rosa, características
y tercera clase (con triángulos). La tipografía es DejaVu Sans y `safe_text`
compone el texto a 4× y lo reduce para evitar espaciado irregular entre letras.
La portada y el cintillo usan el logo blanco institucional sin recuadro.

`narracion.json` es la fuente de texto hablado y notas del presentador. Cada
`self.beat("NN")` señala el comienzo del estado E NN; la pausa anterior tiene
0.4 segundos de fotograma estable. `sincronizar_guion.py` comprueba el orden
contra el manifiesto y genera el Markdown con tiempos del MP4 sin voz.
`verificar_entrega.py` verifica resolución, streams, notas, autonomía del HTML
y que cada video incrustado sea idéntico al segmento renderizado.

## Entorno reproducible

Desde la raíz, si hace falta:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-08-clasificadores-lineales-manim/requirements.txt
```

Las versiones están fijadas: Manim 0.21.0 y Manim Slides 5.6.0. Las fórmulas
necesitan LaTeX; no se requiere Qt para utilizar el HTML. `ffprobe` se usa para
sincronizar el guion. SciPy permite contrastar los cálculos independientemente.

## Compilar

Desde esta carpeta:

```bash
# Comprobación numérica independiente y sintaxis
../../../.venv-manim/bin/python validar_calculos.py
../../../.venv-manim/bin/python -m compileall -q presentacion.py sections

# Prueba visual
../../../.venv-manim/bin/manim-slides render -ql --max-inflight-encoders 4 presentacion.py ClasificadoresLineales

# Iteración parcial; nunca exportar este resultado como entrega
CLS_PARTS=4,5 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py ClasificadoresLineales

# Entrega Full HD explícita a 30 fps, sin CLS_PARTS
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py ClasificadoresLineales

# Exportación de un solo archivo, sin recursos remotos
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true '-ctitle=Clasificadores lineales · Diego Villalba · UNAM' ClasificadoresLineales ../../../docs/lectures/_output/06-clasificacion/06-08-clasificadores-lineales-manim/index.html

# Generar y verificar el guion contra el render completo
../../../.venv-manim/bin/python sincronizar_guion.py
../../../.venv-manim/bin/python verificar_entrega.py
```

La conversión puede necesitar red para descargar e incrustar Reveal.js; el
resultado no necesita conexión. En el HTML, `→` avanza, `←` retrocede, `S` abre
notas y `O` muestra la vista general. Las notas contienen narración y pauta.

Después de modificar la tarjeta de la galería, desde la raíz:

```bash
quarto render presentaciones.qmd --to html --no-execute
```

No ejecutar ese comando sin `--to html`: podría limpiar recursos publicados y
compilar el PDF. Esta entrega no requiere el PDF del libro.

## Verificación

`validar_calculos.py` contrasta MCO con SVD, logística con BFGS y comprueba las
actualizaciones del perceptrón, Fisher, rayos de softmax, matriz kernel y
resultados del ejercicio. La inspección visual cubre todos los estados y
muestras intermedias de las transformaciones importantes, primero en prueba y
después en Full HD. El registro de la entrega está en `revision-youtube.md`.


## Entrega verificada · 20 de noviembre de 2026

65 estados y 65 notas a 1920 × 1080, 30 fps nominales; MP4 H.264 sin audio
(5 min 19 s). El HTML autónomo pesa unos 28.4 MiB. Se comprobaron los bytes de
sus 65 videos, los cálculos, todos los estados y 63 muestras de 21 transiciones.
En navegador se probaron avance, retroceso y vista general; se comprobaron las
65 notas incrustadas. La ventana separada de notas no apareció en el navegador
integrado y queda sin confirmar allí; el guion completo está disponible aparte.

La galería se renderizó sin ejecutar capítulos. No se usó freeze ni se regeneró
el PDF del libro. La [revisión](revision-youtube.md) y [metadatos de verificación](verificacion.json)
documentan el alcance y las comprobaciones. Las capturas locales están en
`media/qa/fullhd/` y `media/qa/transitions/`.
