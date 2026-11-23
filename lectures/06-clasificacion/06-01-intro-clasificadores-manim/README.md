# Introducción a los clasificadores

Presentación animada de Diego Villalba para **Almacenes y Minería de Datos**,
Facultad de Ciencias, UNAM. Diez clientes y cinco casos nuevos conectan
atributos, etiquetas, frontera, matriz de confusión, k-NN y regresión logística.

## Ediciones

- `IntroClasificadores`: edición para grabar un video con narración, con una
  meta editorial de 22–28 minutos que debe ajustarse al ensayar. Abre con la
  paradoja del 97 %, presenta una métrica cada vez, separa entrenamiento,
  validación y prueba y omite F1, el ejercicio de fraude y la formalización
  avanzada. Las actividades invitan a pausar el video.
- `IntroClasificadoresClase`: recorrido de aula con ruta, ejemplos cotidianos,
  F1, ejercicio de fraude, fórmulas, momios y límites de una recta. Conserva
  las correcciones matemáticas y comparte la paleta y las transiciones.

La edición principal está en
`docs/lectures/_output/06-clasificacion/06-01-intro-clasificadores-manim/index.html`.
Es un HTML autónomo con videos y notas. No contiene narración grabada: permite
avanzar mientras el docente explica. El MP4 de Manim contiene sólo animaciones
y pausas breves; no representa por sí solo la duración de la clase narrada.

- [Guion de grabación](guion-youtube.md): narración y correspondencia con los
  estados de la edición compilada.
- [Guion docente](guion-docente.md): pauta presencial de 60 minutos.
- [Revisión editorial inicial](revision-youtube.md): antecedentes y propuestas.

## Entorno

Desde la raíz, si no existe el entorno compartido:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-01-intro-clasificadores-manim/requirements.txt
```

Las fórmulas usan `MathTex`: se necesita MacTeX o TinyTeX. Las versiones están
fijadas en `requirements.txt`; no es necesario actualizar dependencias.

## Compilar

Ejecutar desde esta carpeta:

```bash
# Revisión rápida
../../../.venv-manim/bin/manim-slides render -ql --max-inflight-encoders 4 presentacion.py IntroClasificadores

# Entrega Full HD explícita, 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py IntroClasificadores

# HTML autónomo con controles y notas
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true IntroClasificadores ../../../docs/lectures/_output/06-clasificacion/06-01-intro-clasificadores-manim/index.html

# Versión de aula, si se necesita
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py IntroClasificadoresClase

# Iteración parcial de los actos 3 y 4; no usar ese resultado como entrega
CLS_PARTS=3,4 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py IntroClasificadores
```

El MP4 está en `media/videos/presentacion/1080p30/IntroClasificadores.mp4`.
Para 4K se puede cambiar la resolución a `3840,2160`; el resultado cambia a la
subcarpeta `2160p30`.

Después de actualizar la tarjeta de la galería, desde la raíz:

```bash
quarto render presentaciones.qmd --to html --no-execute
```

En el HTML: `→` avanza, `←` retrocede, `S` abre notas y `O` muestra la vista
general. Abrir las notas en otra ventana y grabar sólo la presentación. Al
llegar al ejercicio, invitar a pausar sin insertar minutos de silencio.

## Convenciones y cálculos

`sections/theme.py` contiene la paleta gris azulada: círculo azul = paga,
cuadrado naranja = no paga, rombo amarillo = consulta. Verde/rojo acompañan
✓/✗. La firma y el logo institucional usan recursos compartidos.

`sections/data.py` calcula distancias, matrices, métricas, regresión logística
y pérdida. `python -m sections.data` imprime el resumen. La logística ajusta
con Newton sobre atributos estandarizados; la animación de entrenamiento
ilustra una trayectoria de descenso de gradiente hasta ese ajuste.

El puntaje es proporcional a la distancia con signo, `d = z / ||w||`. El
ejercicio usa **otro modelo con pesos sencillos**, no un redondeo del ajuste.
El umbral 0.2 ilustra un cambio de decisión y no se presenta como óptimo:
se elige con validación antes de evaluar en prueba.

## Verificación de una entrega

Comprobar Python y el diff; renderizar primero a baja resolución; inspeccionar
estados iniciales, transiciones y cierre; después renderizar a 1920 × 1080.
Verificar los segmentos con `ffprobe`, contar estados y notas en el manifiesto
y revisar que el HTML incluya videos y recursos sin CDN.
`media/`, `slides/` y cachés permanecen ignorados. El HTML final se versiona.


## Entrega verificada el 20 de noviembre de 2026

La edición para video se compiló a 1920 × 1080 y 30 fps, con 48 estados y
48 notas. Se verificaron los streams H.264 y que los 48 videos del HTML
coincidan con los archivos de la compilación. El HTML pesa aproximadamente
26 MiB e incluye sus recursos sin dependencias de CDN.

La revisión visual incluyó los 48 estados, muestras intermedias de seis
transformaciones y un pase posterior de los estados corregidos. Las regiones
de k-NN quedan detrás de los clientes y cada corte conserva un fotograma
estable antes de pausar. Se comprobó la navegación adelante/atrás y la vista
general en navegador; las notas están incrustadas. El guion sigue los 48 estados.

La galería se renderizó con `--to html --no-execute`; no se ejecutaron capítulos,
no se modificó freeze y no se regeneró el PDF del libro. La versión de aula se
conserva como fuente y no se recompiló en esta entrega.
