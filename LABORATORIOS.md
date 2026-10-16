# Guía para crear laboratorios interactivos (HTML/JS)

Esta guía explica cómo están hechos los laboratorios interactivos del libro y cómo
crear otros nuevos con el mismo estilo. Está escrita para docentes, colaboradores y
agentes de IA. Los ejemplos de referencia están en las unidades de preprocesamiento
(`unidades/preprocesamiento/4-*.qmd`), extracción de patrones y clasificación
(`unidades/clasificadores/`).

**Archivos clave**

| Archivo | Para qué sirve |
|---|---|
| [`unidades/preprocesamiento/_labkit.qmd`](unidades/preprocesamiento/_labkit.qmd) | Kit compartido: estilos (claro/oscuro) y `window.LabKit` (utilidades numéricas y graficador SVG). |
| [`unidades/preprocesamiento/_plantilla-laboratorio.qmd`](unidades/preprocesamiento/_plantilla-laboratorio.qmd) | Plantilla mínima y funcional para copiar. |
| [`tools/lab_preview.py`](tools/lab_preview.py) | Vista previa aislada de los laboratorios de un `.qmd`, sin renderizar el libro. |

Los archivos que empiezan con `_` no se renderizan como capítulos.

---

## 1. Idea pedagógica

Cada capítulo sigue la misma ruta, y el laboratorio es una pieza de ella, no un adorno:

1. **Intuición**: un ejemplo pequeño y concreto (tres puntos, cuatro productos, una tabla 2 × 2).
2. **Construcción numérica**: el mismo ejemplo calculado a mano, paso a paso, con números verificables.
3. **Formalización**: la definición general. Los conceptos técnicos de probabilidad o
   estadística se explican en **notas al pie** (`[^nombre]`) para no cortar la lectura.
4. **Laboratorio**: el lector manipula el ejemplo y **comprueba** los números del paso 2;
   después lo rompe (mueve un punto, sube el ruido, agrega un atípico) para ver qué cambia.
5. **Implementación** en Python con `pandas`/`scikit-learn`.

Reglas para el contenido de un laboratorio:

- Usa los **mismos datos** que el ejemplo manual, para que el estado inicial reproduzca sus números.
- Todo se **calcula en el navegador** a partir de los datos. No uses imágenes precalculadas.
- Cada estado debe dar una **retroalimentación** (`.plab-note`) con los números actuales:
  qué pasó y por qué.
- Acompáñalo de una **Guía del laboratorio**: pasos numerados que piden *predecir antes
  de mover*, comprobar un valor concreto y explicar el resultado.
- Incluye siempre una **alternativa para PDF** con los valores de referencia.

---

## 2. Cómo agregar un laboratorio a un capítulo

### Paso 1: incluir el kit (una vez por capítulo)

Justo después del *front matter* (o del primer bloque de *imports*):

```markdown
{{< include _labkit.qmd >}}
```

Desde otra unidad, usa una ruta relativa, por ejemplo
`{{< include ../preprocesamiento/_labkit.qmd >}}`. Si el kit empieza a usarse en muchas
unidades, conviene moverlo a un lugar común y actualizar las rutas.

### Paso 2: copiar la plantilla

Copia de `_plantilla-laboratorio.qmd` el bloque desde `### Laboratorio` hasta el final.
La estructura es siempre la misma:

````markdown
::: {.content-visible when-format="html"}
```{=html}
<div class="plab" id="PREFIJO-lab">
  ...marcado...
</div>
<script>
(() => {
  const K = window.LabKit, root = document.getElementById('PREFIJO-lab');
  const $ = s => root.querySelector(s);
  function render() { /* recalcula todo y redibuja */ }
  // listeners → render
  render();
})();
</script>
```
:::

::: {.content-visible when-format="pdf"}
Texto alternativo con los valores de referencia.
:::
````

### Paso 3: adaptar

1. Elige un **prefijo único** por laboratorio (`pcaS-`, `chi2-`, `mm-`, …) y úsalo en **todos** los `id`.
   Varios laboratorios conviven en la misma página.
2. Escribe una única función `render()` que lea los controles, calcule, dibuje y escriba los textos.
   Cualquier control llama a `render()`. Es un patrón simple, sin estado oculto.
3. Si hay aleatoriedad, usa `K.rng(semilla)` para que los resultados sean reproducibles.

### Paso 4: probar (ver §5) y renderizar el capítulo

```bash
quarto render unidades/.../capitulo.qmd --to html
```

---

## 3. Anatomía visual (clases CSS)

Todas las clases viven dentro de `.plab`. No escribas colores fijos: usa las variables
`var(--lab-…)` para que funcionen en modo claro y oscuro.

| Clase | Uso |
|---|---|
| `.plab` | Contenedor del laboratorio (tarjeta). |
| `.plab-eyebrow`, `.plab-title`, `.plab-sub` | Encabezado: «Laboratorio · Tema», pregunta, instrucción breve. |
| `.plab-controls` | Fila flexible de controles. Cada `<label>` dentro se apila como «texto, valor en `<b>`, control». |
| `.plab-field` | Igual que un `label` de `.plab-controls`, pero para agrupar botones. |
| `.plab-seg` | Botones segmentados (opción única): `<button data-value=… aria-pressed="true">`. Se conecta con `K.segmented`. |
| `.plab-btn`, `.plab-btn.primary` | Botones de acción. |
| `.plab-steps` | Botones tipo «píldora» para recorrer pasos (ver el laboratorio 1 de PCA). |
| `.plab-grid` | Rejilla adaptable de paneles (se apila en pantallas estrechas). Usa `style="grid-template-columns:1fr"` para forzar una columna. |
| `.plab-panel` | Panel con fondo suave y `<h4>` de título. |
| `.plab-stats` > `.plab-stat` | Indicadores: `<span>etiqueta</span><strong id=…></strong>`. |
| `.plab-note` (`.good`, `.warn`, `.bad`) | Retroalimentación. Añade `aria-live="polite"`. |
| `.plab-table` (+ `.hl` en filas o celdas) | Tablas numéricas. **Genéralas desde JS** (ver §6). |
| `.plab-scroll` | Envoltura con desplazamiento horizontal para tablas anchas. |
| `.plab-math` | Bloque monoespaciado para matrices y cálculos que cambian. |
| `.plab-legend` | Leyenda: `<span><i style="background:var(--lab-s1)"></i>texto</span>`; `<i class="dot">` para puntos. |

**Paleta** (variables CSS): `--lab-s1` azul (datos), `--lab-s2` naranja (estimación, dirección
principal), `--lab-s3` verde (referencia, valor real), `--lab-s4` morado (error, residuos),
`--lab-s5` rojo (atípico, valor observado crítico). También `--lab-text`, `--lab-muted`,
`--lab-axis`, `--lab-accent`, `--lab-good`, `--lab-warn` y `--lab-bad`. Mantén estos
significados entre laboratorios y evita juntar naranja y rojo para series que se deben comparar.

---

## 4. Referencia de `window.LabKit`

### Utilidades numéricas

| Función | Descripción |
|---|---|
| `K.sum(a)`, `K.mean(a)` | Suma y media. |
| `K.variance(a, ddof=0)`, `K.sd(a, ddof=0)` | `ddof=0`: poblacional (como `StandardScaler`); `ddof=1`: muestral. |
| `K.quantile(a, q)`, `K.median(a)` | Percentil con interpolación lineal (igual que `numpy.percentile`). |
| `K.clamp(v, lo, hi)` | Recorta al intervalo. |
| `K.fmt(v, d=2)` | Formato numérico `es-MX` con `d` decimales; `—` si es `NaN`. |
| `K.rng(seed)` | Generador reproducible: `r()` da un uniforme en [0, 1) y `r.normal()` una normal estándar. |
| `K.niceTicks(lo, hi, n)` | Marcas de eje «redondas». |
| `K.segmented(el, onChange)` | Conecta un `.plab-seg`. Devuelve `{ get(), set(v) }`. |

### Graficador SVG: `K.chart(svg, opciones)`

```js
const c = K.chart($('#id-svg'), {
  width: 400, height: 300,          // tamaño del viewBox (unidades lógicas)
  x: [xmin, xmax], y: [ymin, ymax], // dominios en unidades de los datos
  margin: { left: 44, right: 14, top: 14, bottom: 38 }
});
c.axes({ xTicks: 5, yTicks: 5, xLabel: 'x', yLabel: 'y', grid: true,
         xFmt: v => K.fmt(v, 0), yFmt: v => …, xCats: [[pos, 'etiqueta'], …] });
c.zeroLines();                              // ejes discontinuos en 0
c.dot(x, y, r, 's1 dot');                   // punto
c.line(x1, y1, x2, y2, 's2 line');          // segmento
c.path([[x, y], …], 's1 line');             // polilínea
c.rect(x1, y1, x2, y2, 's3');               // barra o celda (usa .style.fillOpacity si hace falta)
c.arrow(x1, y1, x2, y2, 's2');              // flecha
c.text(x, y, 'texto', { dx, dy, 'text-anchor': 'middle', class: 'strong' });
c.pointer(evento) → [x, y]                  // coordenadas de datos de un evento del puntero
c.X(v), c.Y(v), c.invX(px), c.invY(py)      // escalas
```

- Cada llamada a `K.chart` **borra y redibuja** el SVG: llámala dentro de `render()`.
- Las clases de estilo se combinan: color (`s1`–`s5`, `muted`) + forma (`line`, `dot`)
  + modificadores (`dash`, `thin`, `soft`, `ghost`).
- Para que ángulos y distancias sean fieles (PCA, KNN), usa **dominios con el mismo rango en
  ambos ejes** y un área útil cuadrada (ver `4-4-PCA.qmd` y el laboratorio de KNN en `4-3-conversion.qmd`).
- Añade `class="wide"` al `<svg>` para gráficas de ancho completo (hasta 640 px); si no, el máximo es 460 px.
- Siempre pon `role="img"` y `aria-label` en el `<svg>`.

**Puntos arrastrables**: marca los círculos con `dataset.i` y usa *pointer events* sobre el
`<svg>` con `setPointerCapture`; en `pointermove`, convierte con `svg.__chart.pointer(e)`
(el kit guarda la última gráfica en `svg.__chart`). Ejemplo completo: laboratorio 1 de PCA.

---

## 5. Probar antes de renderizar

La página completa del libro tarda en cargar y MathJax mueve el desplazamiento. Para iterar
rápido usa la vista previa aislada:

```bash
python tools/lab_preview.py unidades/preprocesamiento/mi-capitulo.qmd
python -m http.server 8766 --directory _lab_preview
```

Abre `http://localhost:8766/mi-capitulo-light.html` y `…-dark.html`. Revisa:

- la consola, que no debe tener errores;
- modo claro y oscuro;
- ancho de móvil (375 px), sin desplazamiento horizontal de la página;
- que el estado inicial reproduzca los números del ejemplo manual;
- los casos límite: tabla vacía, división entre cero, valores extremos del control.

Después renderiza el capítulo y revisa la página real. Al recargar, añade `?v=2` a la URL:
el navegador suele servir versiones en caché.

---

## 6. Reglas aprendidas (evitan errores reales)

1. **No escribas `<table>` estáticas en el HTML crudo.** Quarto intenta convertirlas en tablas
   de Pandoc y puede **perder los `id`** o alterar el contenido. Deja un `<div id="…">` vacío y
   genera la tabla completa desde JS (`innerHTML = '<table class="plab-table">…</table>'`).
   Para cuadrículas con `input`, usa CSS grid (ver `#chi2-lab .chi2-grid`). El aviso
   «Unable to parse table from raw html block» que aparece al renderizar es inofensivo.
2. **Envuelve todo en `content-visible when-format="html"`** y da la alternativa para PDF.
   El PDF no ejecuta JavaScript.
3. **Nada de MathJax dentro de texto dinámico**: no se vuelve a componer. Usa Unicode
   (λ₁, σ², χ², x̄, ⟂) en etiquetas y notas generadas por JS. El `$…$` sí funciona en el HTML estático.
4. **Sin dependencias externas**: solo el kit. No cargues librerías por CDN.
5. **Todo dentro de una IIFE** `(() => { … })();` y todas las búsquedas con `root.querySelector`,
   para no chocar con otros laboratorios.
6. **Convenciones estadísticas explícitas**: si usas varianza poblacional o muestral, o una
   convención de cuartiles, dilo en el texto o en una nota al pie y úsala igual que el código Python del capítulo.
7. **Escritura desde la terminal**: si generas estos bloques con `cat <<EOF`, usa `<<'EOF'`
   (entre comillas). Si no, el shell expande `$…` y destruye las fórmulas y los *template literals*.

---

## 7. Lista de verificación antes de entregar

- [ ] El capítulo incluye `_labkit.qmd` una sola vez.
- [ ] Prefijo de `id` único; sin `<table>` estáticas.
- [ ] El estado inicial reproduce el ejemplo manual; la guía del laboratorio pide predecir, comprobar y explicar.
- [ ] Hay retroalimentación con números en cada estado y leyendas en cada gráfica.
- [ ] Funciona en claro, oscuro y a 375 px, sin errores en consola.
- [ ] Tiene alternativa para PDF con los valores de referencia.
- [ ] `quarto render` del capítulo sin errores; la página real funciona (recargar con `?v=…`).
- [ ] No se agregan `_lab_preview/`, `*.quarto_ipynb*` ni HTML de `docs/` al commit.

---

## Laboratorios de referencia

| Capítulo | Laboratorio | Qué ilustra técnicamente |
|---|---|---|
| `4-4-PCA.qmd` | Dos sensores, seis pasos | Recorrido por pasos (`.plab-steps`), puntos arrastrables, matrices en `.plab-math` |
| `4-4-PCA.qmd` | Cociente de Rayleigh | Dos gráficas enlazadas (dispersión y curva), deslizadores |
| `4-5-chi2.qmd` | Prueba de independencia | Entradas numéricas en CSS grid, mosaico, simulación Monte Carlo, prueba exacta |
| `4-2-datos-faltantes.qmd` | Mecanismos de ausencia | Población simulada con semilla, histogramas superpuestos, tabla de estimadores |
| `4-3-conversion.qmd` | El espacio que ve KNN | Ejes con igual escala, parámetros aprendidos en entrenamiento y aplicados a un punto nuevo |
| `4-6-ruido.qmd` | Filtros | Curva de error frente a un hiperparámetro (compromiso sesgo–varianza) |
| `4-1-introduccion.qmd` | Ajustar y aplicar | Casillas de transformaciones, fuga de datos visible |
| `5-0-introduccion-patrones.qmd` | Retícula de itemsets | Diagrama SVG con nodos clicables (`data-key` + `closest`), poda Apriori sobre la retícula |
| `5-1-introduccion.qmd` | Métricas en una tabla 2 × 2 | Mosaico con `K.chart`, escenarios con `K.segmented`, división entre cero controlada |
| `5-1-apriori.qmd` | Apriori nivel por nivel | Pasos generados dinámicamente, barras apiladas por nivel, generación de reglas |
| `5-2-metricas-extra.qmd` | El techo del lift | Control con `min`/`max` dinámicos (cotas de Fréchet), dos gráficas de curvas enlazadas |
| `5-3-fp_growth.qmd` | Construir y minar el FP-tree | Árbol SVG con distribución por hojas, enlaces de nodo, minado recursivo de bases condicionales |
| `5-3b-cerrados-maximales.qmd` | Frecuentes, cerrados y maximales | Retícula clasificada con flechas de clausura, recuperación de conteos, tickets editables con casillas en CSS grid |
| `5-3c-patrones-secuenciales.qmd` | Base proyectada de PrefixSpan | Prefijo que se extiende con botones, resaltado de la aparición más a la izquierda, lista de patrones clicable, sesiones editables con `input` de texto saneado |
| `5-4-clasificacion-asociativa.qmd` | Clasificador CBA | Validación cruzada LOOCV en el navegador, matriz de confusión, predicción con `select` |
| `6-0-introduccion-clasificacion.qmd` | Complejidad y generalización | Regiones de decisión en rejilla, curva de error frente a k, error de Bayes con la probabilidad real conocida, datos incrustados desde Python |
| `6-0b-vecinos-cercanos.qmd` | Vecindario de k-NN; maldición de la dimensionalidad | Punto consultado arrastrable, «bolas» de Minkowski, voto ponderado; simulación de distancias en alta dimensión |
| `6-2-metricas_clasificacion.qmd` | Umbral, prevalencia y curvas | Modo discreto (20 pacientes) y modo binormal analítico (Φ por aproximación), ROC y PR enlazadas, umbral de mínimo costo |
| `capitulo_evaluacion_modelos_clasificacion.qmd` | Varianza de la estimación | 200 repeticiones de holdout y K pliegues con vecinos precalculados, curva de validación con regla de un error estándar |
| `6-1-clasificacion_arboles.qmd` | Construye el árbol (ID3/C4.5/Gini) | Árbol SVG con nodos seleccionables, profundidad como pre-poda, LOOCV en el navegador, atributo identificador |
| `6-1-clasificacion_cart.qmd` | CART en el plano | Partición rectangular, búsqueda del primer corte, ruta de poda de costo-complejidad (eslabón más débil) |
| `capitulo_arboles_regresion_lineal.qmd` | Bagging; AdaBoost ronda por ronda | Árboles de regresión bootstrap con error fuera de la bolsa; pesos como tamaño de punto, tocones ponderados en 1D y 2D |
| `capitulo_xgboost.qmd` | Gradient boosting paso a paso | Árboles sobre gradientes con pesos −G/(H+λ), residuos y próximo árbol, curva de entrenamiento/prueba por ronda |
| `capitulo_regresion_clasificadores_lineales.qmd` | Mínimos cuadrados; Fisher; perceptrón | Puntos arrastrables y mapa de la SSE; cociente J frente al ángulo; perceptrón por pasos y épocas (XOR) |
| `capitulo_regresion_logistica.qmd` | Ajustar la logística | Pérdida por punto, mapa de log-loss con trayectoria de descenso de gradiente, separabilidad y L2 |
| `capitulo_svm.qmd` | Margen, C y núcleos | SMO simplificado en el navegador, vectores de soporte, núcleo RBF, puntos arrastrables |
| `capitulo_clasificadores_bayesianos.qmd` | Monty Hall; prior y posterior; Naive Bayes; frontera gaussiana; CpG; riesgo mínimo | Simulación con semilla, `select` para un caso nuevo con suavizado de Laplace, regiones con costos asimétricos |
| `capitulo_gmm.qmd` | EM paso a paso | Pasos E y M separados, color por responsabilidad (`color-mix`), log-verosimilitud monótona |
| `capitulo_redes_neuronales.qmd` | Entrena una red en el navegador | MLP de una capa oculta con retropropagación en JS, animación con `setInterval`, cuatro conjuntos 2D |
| `capitulo_pytorch_jax.qmd` | Los cinco pasos de PyTorch | `.plab-steps` con texto y demostración de la acumulación de gradientes sin `zero_grad()` |
