# Medidas de variabilidad · repaso animado

Presentación de Manim Slides para reconstruir, con ejemplos numéricos, rango,
cuartiles, IQR, cercas de Tukey, boxplot, varianza muestral y desviación
estándar. Está diseñada como **repaso**: el grupo ya conoce las definiciones y
ahora debe anticipar, calcular, visualizar e interpretar.

La portada y el pie persistente usan la firma institucional blanca compartida
del curso: `lectures/assets/Logo_FC_Blanco.png`, **Diego Villalba** y **Almacenes y
Minería de Datos**.

La escena tiene 36 estados navegables y notas del presentador integradas. Su
duración sugerida es de 25–30 minutos, incluyendo dos pausas de predicción.

## Archivos

- `variabilidad.py`: escena `MedidasVariabilidad` y toda la animación.
- `manim.cfg`: formato 16:9, fondo y salida local de Manim.
- `requirements.txt`: Manim y Manim Slides 5.6.0 fijado al tag/commit usado.
- `guion-docente.md`: intención, respuestas y ritmo de cada bloque.

`media/` y `slides/` son cachés reproducibles y no se versionan. La versión
publicada sí queda en:

```text
docs/lectures/_output/03-eda/03-04-medidas-variabilidad-manim/index.html
```

## Instalar

Desde la raíz del repositorio:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python \
  -r lectures/03-eda/03-04-medidas-variabilidad-manim/requirements.txt
```

La instalación utilizada para esta entrega proviene del tag `v5.6.0` de
<https://github.com/jeertmans/manim-slides> y comparte el mismo entorno con
Manim Community 0.21.0. La salida HTML no necesita Qt.

Comprobación:

```bash
.venv-manim/bin/manim-slides --version
.venv-manim/bin/manim-slides checkhealth
```

## Renderizar

Desde la carpeta de esta clase:

```bash
cd lectures/03-eda/03-04-medidas-variabilidad-manim

# Iteración rápida, 854 × 480 y 15 fps
../../../.venv-manim/bin/manim-slides render -ql \
  variabilidad.py MedidasVariabilidad

# Entrega Full HD, 1920 × 1080 y 30 fps
../../../.venv-manim/bin/manim-slides render \
  --resolution 1920,1080 \
  --fps 30 \
  --max-inflight-encoders 4 \
  variabilidad.py MedidasVariabilidad
```

## Exportar a HTML sin conexión

Después del render:

```bash
../../../.venv-manim/bin/manim-slides convert \
  --folder slides \
  --offline \
  --one-file \
  -ccontrols=true \
  -cprogress=true \
  -cslide_number=true \
  -chash=true \
  MedidasVariabilidad \
  ../../../docs/lectures/_output/03-eda/03-04-medidas-variabilidad-manim/index.html
```

El HTML es autocontenido: se puede abrir como archivo local o publicar en
GitHub Pages sin CDN ni carpeta de activos adicional.

## Presentar

- `→` o `Page Down`: avanzar y ejecutar la siguiente construcción.
- `←`: volver; Manim Slides genera las animaciones inversas.
- `espacio`: pausar o reanudar la animación actual.
- `S`: vista del presentador con las notas docentes.
- `O`: vista general.
- `B`: oscurecer temporalmente la pantalla.
- `?`: mostrar la ayuda de teclado de Reveal.js.

La primera pausa pide elegir entre dos rutas con la misma media. La segunda,
marcada `PAUSA · 45 s`, exige resolver un conjunto genérico antes de mostrar la
construcción.

## Convenciones estadísticas

- Se usa varianza **muestral**, con divisor `n - 1`.
- Los cuartiles siguen la convención del material de referencia: mediana de la
  mitad inferior y de la mitad superior.
- Por ello, con `[4, 5, 5, 6, 6, 7, 8, 24]` se obtiene `Q1 = 5` y `Q3 = 7.5`.
  Bibliotecas que interpolan percentiles pueden devolver otro `Q3`; la
  presentación hace visible la convención para evitar ambigüedad.
- Una cerca de Tukey identifica **candidatos** atípicos. No autoriza borrarlos
  sin investigar procedencia y contexto.
