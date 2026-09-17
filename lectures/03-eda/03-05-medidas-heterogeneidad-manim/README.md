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

## Archivos

- `heterogeneidad.py`: escena continua `MedidasHeterogeneidad` con visualizaciones geométricas continuas (curva de Shannon con vértice en 1/e y cuadrados euclidianos de ANOVA).
- `interactivo.html`: simulador web interactivo autónomo (Canvas/SVG) para experimentación directa de proporciones categóricas y carriles ANOVA.
- `guion-docente.md`: ruta de clase, resultados exactos, puentes a Machine Learning y guía del simulador.
- `manim.cfg`: configuración Full HD 16:9 a 30 fps.
- `requirements.txt`: versiones reproducibles.
- `.gitignore`: artefactos locales de Manim y Manim Slides.

La presentación reutiliza
`lectures/assets/Logo_FC_Blanco.png`; no contiene copias locales del logo.

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

La falta de Qt sólo impide la interfaz nativa de `present`; no afecta el render
ni el HTML autónomo.

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
