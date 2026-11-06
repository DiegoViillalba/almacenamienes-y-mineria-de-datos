# Introducción a las redes neuronales para clasificar · clase animada (60 min)

Continuación de las clases de clasificadores. Parte de una sola neurona y
llega a las redes convolucionales con un hilo conductor: el mapa de **Baarle**,
el pueblo donde Baarle-Hertog (Bélgica) está hecho de enclaves dentro de
Baarle-Nassau (Países Bajos), con contraenclaves neerlandeses dentro de los
enclaves belgas. Ninguna recta puede separar esa frontera; una red ReLU sí.

1. **La neurona** (0–11): de la neurona biológica a la artificial; un cálculo
   a mano (z = 1.25); activaciones; una neurona es un hiperplano w·x + b = 0
   (girar w, deslizar b).
2. **Baarle** (11–19): el mapa estilizado y el problema (x, y) → país; la
   regresión logística acierta 75.2 % y pierde todos los enclaves; apilar capas
   lineales colapsa a W x + b′ (75.2 % otra vez).
3. **ReLU: doblar el plano** (19–29): cuatro ReLU suman un escalón en una
   recta; cada ReLU es una bisagra en el plano y cuatro forman un enclave
   cuadrado; la receta rectas → polígonos → unión con huecos; dos dobleces y un
   corte equivalen a cuatro cortes.
4. **La red en Baarle** (29–38): 2 → 16 → 16 → 1 (337 parámetros); sus 16
   rectas aprendidas; 100 regiones lineales con la capa 1 y 455 con los pliegues
   de la capa 2; la frontera aprendida (99.8 %); la representación interna donde
   una recta basta.
5. **Entrenamiento y retropropagación** (38–48): la frontera época por época
   junto a la pérdida; descenso por gradiente; una red mínima hacia adelante y
   hacia atrás con números (L: 0.83 → 0.03 tras un paso); la regla de la cadena.
6. **De redes densas a CNN** (48–56): aplanar destruye la vecindad (150 M de
   pesos); una convolución 3 × 3 que detecta un borde; la pila conv → pooling →
   densa; línea de tiempo 1958–2020.
7. **Ejercicio y cierre** (56–60): XOR con dos neuronas ReLU y ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

El mapa es **estilizado**: cuatro enclaves y un contraenclave en lugar de los
20 enclaves y 7 contraenclaves reales, pero con la misma topología (la región
belga está partida en pedazos y uno tiene un hueco).

## Cómo está organizado (qué archivo editar)

```text
presentacion.py            ensambla la escena RedesNeuronales y elige los actos
sections/
├── data.py                TODOS los números: neurona, escalón ReLU, enclave
│                          cuadrado, polígonos de Baarle, entrenamiento en NumPy,
│                          retropropagación y convolución
├── baarle_mlp.json        pesos entrenados (logística, red lineal, red ReLU y
│                          sus instantáneas); se regenera si se borra
├── theme.py               paleta, tipografía y piezas pequeñas (pill, card, table…)
├── base.py                cintillo institucional, encabezados, beat() = pausa + notas
├── viz.py                 MapView: mapa, rásteres, rectas recortadas y contornos
├── part01_neuron.py       acto 1
├── part02_baarle.py       acto 2
├── part03_relu.py         acto 3
├── part04_network.py      acto 4
├── part05_training.py     acto 5
├── part06_cnn.py          acto 6
└── part07_close.py        acto 7
```

- **Cambiar un número o el mapa:** sólo en `sections/data.py`. Si cambian los
  polígonos o la arquitectura, borrar `sections/baarle_mlp.json` para reentrenar
  (≈ 4 s). `python -m sections.data` imprime el resumen.
- **Cambiar un color:** `sections/theme.py`.
- **Cambiar una lámina:** cada estado empieza con `self.beat("notas…")`.
- **Quitar o reordenar actos:** la lista `acts` de `presentacion.py`.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-09-redes-neuronales-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX). La
red se entrena con NumPy; no hace falta PyTorch.

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py RedesNeuronales

# Sólo algunos actos (1–7), útil al editar
CLS_PARTS=3,4 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py RedesNeuronales

# Entrega 4K (3840 × 2160) a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 3840,2160 --fps 30 --max-inflight-encoders 4 presentacion.py RedesNeuronales

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true RedesNeuronales ../../../docs/lectures/_output/06-clasificacion/06-09-redes-neuronales-manim/index.html
```

Abrir `docs/lectures/_output/06-clasificacion/06-09-redes-neuronales-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.

Nota: no usar pausas de un solo fotograma (`self.wait(0.1)` a 15 fps); rompen
la concatenación de Manim Slides. Usar al menos `self.wait(0.3)`.
