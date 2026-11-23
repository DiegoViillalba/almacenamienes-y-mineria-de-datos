# Similitud y distancias · clase animada (60 min)

Clase de **Almacenes y Minería de Datos** basada en el capítulo
[Similitud y distancias](../../../unidades/clustering/capitulo_clustering_intro.qmd)
y en la sección de UMAP de
[t-SNE y UMAP](../../../unidades/clustering/capitulo_tsne_umap.qmd). Siempre
primero la intuición (un objeto que se mueve, un número a mano) y después la
fórmula. La secuencia es

1. **¿Quién se parece más a Ana?** Cinco clientes con variables mixtas; tres
   reglas razonables dan tres respuestas. Todo algoritmo de la unidad consume
   una distancia.
2. **¿Qué es medir?** Asignar números que respeten la estructura (orden y
   suma de varas). Tres reglas rotas — "sólo edad", el tiempo de subida a un
   cerro y la euclidiana al cuadrado — hacen nacer los axiomas de métrica.
3. **¿Para qué los axiomas?** La desigualdad triangular permite descartar una
   bola de candidatos sin calcular (árboles de bolas y KD). Medida,
   distancia, similitud y desempeño son cuatro objetos distintos; el núcleo
   gaussiano es el puente distancia → similitud.
4. **Geometría 2D.** Taxi, pájaro y rey; la bola unitaria de Minkowski se
   deforma con p y deja de ser convexa con p < 1. Estirar un eje (cambiar
   unidades) cambia al vecino; blanquear una nube correlacionada convierte la
   euclidiana en Mahalanobis, y de regreso la bola es una elipse.
5. **Vistas 3D.** Tres documentos: proyectarlos a la esfera unitaria da el
   coseno; 1 − cos no es métrica (contraejemplo 0°, 45°, 90°), el ángulo sí.
   Dos grupos que sólo se ven al girar: proyectar nunca aleja.
6. **La esfera en alta dimensión.** Dardos en el cuadrado, esfera en el cubo,
   fracción que se desploma con p, casi todo es cáscara y las distancias se
   concentran (Beyer et al.).
7. **UMAP.** Un rollo de papel: por el aire 6.3, sobre la hoja 68.6. Grafo
   k-NN, la regla propia de cada punto (ρᵢ, σᵢ), unión difusa, el dibujo
   (atraer/repeler) y qué no se lee en un mapa de UMAP.
8. **Cierre.** Gower con Ana (cambiar pesos cambia al más parecido), reto en
   parejas para elegir distancia y mapa del capítulo; ticket de salida.

- [Guion docente](guion-docente.md): tiempos, respuestas y errores frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Archivos

- `presentacion.py`: ensambla la escena `SimilitudDistancias` (`ThreeDSlide`).
- `sections/data.py`: todos los números con NumPy y semillas fijas; los de
  Ana, Mahalanobis, coseno, concentración y la recta de UMAP reproducen los
  del libro. `python -m sections.data` imprime un resumen. El rollo se muestrea
  uniformemente en longitud de arco y un `assert` garantiza que el grafo k-NN
  no salta entre capas.
- `sections/theme.py`, `sections/base.py`: paleta con significado fijo
  (turquesa = distancia, morado = similitud, naranja = desempeño, rojo =
  axioma violado, rosa = alta dimensión/UMAP), cintillo, notas y utilidades 3D.
- `sections/part01_opening.py` … `part08_close.py`: un archivo por acto.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/07-no-supervisado/07-01-similitud-distancias-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py SimilitudDistancias

# Sólo algunos actos (1–10), útil al editar
SIM_PARTS=5,6 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py SimilitudDistancias

# Entrega 1920 × 1080 a 30 fps (las escenas 3D tardan varios minutos)
../../../.venv-manim/bin/manim-slides render --resolution 1920,1080 --fps 30 --max-inflight-encoders 4 presentacion.py SimilitudDistancias

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true SimilitudDistancias ../../../docs/lectures/_output/07-no-supervisado/07-01-similitud-distancias-manim/index.html
```

Abrir `docs/lectures/_output/07-no-supervisado/07-01-similitud-distancias-manim/index.html`.
Teclas: `→` avanza, `←` retrocede, `S` notas, `O` vista general. Los giros de
cámara en 3D son estados en bucle: avanzan con `→`.
