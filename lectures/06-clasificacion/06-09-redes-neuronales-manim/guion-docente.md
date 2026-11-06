# Guion docente · Introducción a las redes neuronales para clasificar

Duración: 60 minutos. Público: ya vio clasificadores lineales (logística),
árboles y ensambles. Objetivo: que el grupo lea una red ReLU como una
partición del espacio en polígonos y entienda por qué la no linealidad es
indispensable.

## Mapa de tiempos

| Min | Acto | Idea que debe quedar | Pregunta para el grupo |
|---|---|---|---|
| 0–2 | Portada y ruta | Una red corta el espacio con muchas rectas | ¿Qué usaron hoy que tenga una red? |
| 2–7 | Neurona | Recibe, pondera, suma, activa | ¿Cuánto vale z? |
| 7–8 | Activaciones | ReLU es lineal a trozos | ¿Qué tiene de especial ReLU? |
| 8–11 | Hiperplano | Una neurona = una frontera recta | ¿Qué hace b? ¿y w? |
| 11–14 | Baarle | Una frontera partida y con huecos | ¿Cuántas rectas hacen falta? |
| 14–16 | Logística | La mejor recta: 75.2 % | ¿Dónde se equivoca? |
| 16–19 | Capas lineales | Lineal ∘ lineal = lineal | ¿Mejorará con más capas? |
| 19–22 | ReLU en 1D | Sumar codos forma un escalón | ¿Qué forma tendrá la suma? |
| 22–25 | ReLU en 2D | Bisagras → un enclave | ¿Por qué las esquinas se recortan? |
| 25–27 | Receta | Rectas → polígonos → unión − huecos | ¿Cómo se hace un hueco? |
| 27–29 | Papel doblado | La profundidad multiplica regiones | ¿Cuántos cortes vale uno doblado? |
| 29–35 | La red | 16 rectas, 100 → 455 regiones | ¿Qué rectas siguen un borde? |
| 35–38 | Frontera y representación | 99.8 %; en el espacio aprendido basta una recta | ¿Dónde quedaron los enclaves? |
| 38–42 | Entrenamiento | Bajar la pérdida con gradientes | ¿Qué aprende primero? |
| 42–48 | Retropropagación | Regla de la cadena hacia atrás | ¿Subir w sube o baja L? |
| 48–52 | Imágenes | Aplanar destruye la vecindad | ¿Qué se pierde al aplanar? |
| 52–56 | CNN e historia | Filtros locales, pesos compartidos | ¿Por qué tan pocos pesos? |
| 56–60 | XOR y cierre | Dos ReLU resuelven XOR | Completar la tabla |

## Números de la clase (verificados con `python -m sections.data`)

**Neurona.** x = (2, 1, 3), w = (0.5, −1, 0.25), b = 0.5:
z = 1 − 1 + 0.75 + 0.5 = **1.25**; ReLU(z) = 1.25; σ(z) = 0.777.

**Escalón 1D.** f(x) = ReLU(x−3) − ReLU(x−4) − ReLU(x−6) + ReLU(x−7) vale 0
fuera de [3, 7], 1 en [4, 6] y es una rampa entre medio; f > 0.5 en
**[3.5, 6.5]**.

**Enclave cuadrado** [3, 7] × [2, 5]:
s = 1 − 2[ReLU(3−x) + ReLU(x−7) + ReLU(2−y) + ReLU(y−5)]. Dentro s = 1; la
frontera s = 0 está a 0.5 unidades del cuadrado sobre cada lado y corta las
esquinas en diagonal (allí suman dos bisagras: ReLU + ReLU = 0.5).

**Baarle (estilizado).** 4 000 puntos uniformes en un lienzo 10 × 7; 34.5 %
belgas.

| Modelo | Exactitud |
|---|---|
| Siempre «Países Bajos» | 65.5 % |
| Regresión logística | 75.2 % |
| Red lineal 2-16-16-1 (sin ReLU) | 75.2 % |
| Red ReLU 2-16-16-1 (337 parámetros) | 99.8 % |

Entrenamiento de la red ReLU (Adam, η = 0.01, lote completo): época 0 →
59.8 %, 25 → 78.8 %, 100 → 84.2 %, 400 → 97.2 %, 1 500 → 99.2 %,
6 000 → 99.8 %. Pérdida 0.667 → 0.008. Regiones lineales: 100 con la capa 1,
455 con las capas 1 y 2.

**Retropropagación** (x = 2, y = 1, w = 0.5, b = −0.5, v = 1.5, c = −1, η = 0.5):

- Adelante: z = 0.5, a = 0.5, s = −0.25, ŷ = σ(−0.25) = 0.438, L = −log ŷ = 0.826.
- Atrás: ∂L/∂s = ŷ − y = −0.562; ∂L/∂v = a·(−0.562) = −0.281;
  ∂L/∂a = v·(−0.562) = −0.843; ∂L/∂z = −0.843 (ReLU encendida);
  ∂L/∂w = x·(−0.843) = −1.687; ∂L/∂b = −0.843.
- Paso: w = 1.343, b = −0.078, v = 1.641, c = −0.719 → ŷ = 0.972, L = 0.028.

**Convolución.** Imagen 6 × 6 con columnas 0, 0, 0, 1, 1, 1; filtro con filas
(−1, 0, 1). Mapa 4 × 4 con columnas 0, 3, 3, 0 (el borde está entre las
columnas 3 y 4). Densa sobre 224 × 224 × 3 con 1 000 neuronas: 150 528 000
pesos; 64 filtros 3 × 3 × 3: 1 728 pesos.

**XOR.** h₁ = ReLU(x₁ + x₂), h₂ = ReLU(x₁ + x₂ − 1), ŷ = h₁ − 2h₂:
(0,0) → 0; (0,1) → 1; (1,0) → 1; (1,1) → 2 − 2 = 0. ŷ > 0.5 en la franja
0.5 < x₁ + x₂ < 1.5.

## Errores frecuentes

- **«Más capas = más poder» sin activación.** El acto 2 lo desmiente con
  álgebra y con un experimento: la red lineal profunda empata con la logística.
- **Confundir la recta de una neurona con la frontera de la red.** Las rectas
  de la capa 1 son *candidatas*; la frontera final se arma con trozos de ellas
  y de los pliegues de la capa 2.
- **Creer que la red «memoriza» el mapa.** Con 337 parámetros no puede guardar
  4 000 puntos; aprende polígonos.
- **Signo del gradiente.** ∂L/∂w < 0 significa que *subir* w baja la pérdida;
  el paso es w − η·(negativo) = w sube.
- **ReLU muerta.** Si z < 0 para todos los datos, la derivada es 0 y esa
  neurona deja de aprender: menciónelo al hablar de la compuerta ×0.
- **Pensar que la CNN no tiene hiperplanos.** La última capa es densa: separa
  con hiperplanos en el espacio de rasgos aprendido.

## Datos de Baarle para contar

- Baarle-Hertog (Bélgica) está formado por 24 parcelas: la principal
  (Zondereigen), 20 enclaves dentro de Países Bajos y otras parcelas en la
  frontera.
- Dentro de los enclaves belgas hay 7 contraenclaves neerlandeses (y uno más
  cerca de Zondereigen).
- La frontera cruza casas y comercios; la nacionalidad de una casa la decide su
  puerta principal.

## Respuestas del ticket de salida

1. Porque la composición de funciones lineales es lineal: la red completa
   equivale a una sola neurona, y un semiplano no puede tener varias piezas ni
   huecos.
2. Tres: una por lado (un «Y» de tres semiplanos) más la neurona de salida que
   suma y compara con un umbral.
