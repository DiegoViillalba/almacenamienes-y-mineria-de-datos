# Heterogeneidad · Reel para Instagram (83 s)

Reel vertical que resume Gini–Simpson y Shannon en un solo hilo geométrico,
pensado para verse **sin audio**: cada paso trae una explicación de 1–2
líneas que permanece en pantalla el tiempo de lectura (`read_time`:
0.6 s + 0.25 s por palabra, mínimo 1.6 s). Sigue la guía
[`lectures/MANIM-REDES.md`](../../../MANIM-REDES.md) (§8, Reels 9:16).

- Formato: 9:16, 1080 × 1920, 30 fps, h264 `yuv420p`, sin audio, 83.1 s.
- Fuente: [`heterogeneidad_instagram.py`](heterogeneidad_instagram.py), escena `HeterogeneidadInstagram`.
- Salida final: `heterogeneidad-instagram.mp4` (copia de `media/videos/heterogeneidad_instagram/1920p30/`).
- Ejemplo: bolsa de 6 canicas — 3 azules, 1 morada, 2 verdes; p=(1/2, 1/6, 1/3)
  (capítulo `unidades/intro_mineria/2-6-heterogeneidad.qmd`).
- Firma: solo "Diego Villalba · @diego_villalba06" (pie) y, al cierre, nombre
  y usuario en grande. **Sin** nombre de curso ni logo institucional.
- Zona segura: el contenido vive en y ∈ [−4.1, 5.6] del marco 8 × 14.22
  para no quedar bajo la interfaz de Reels (arriba ≈220 px, abajo ≈380 px).

## Guion

| Tiempo aprox. | Acto | Transformación | Explicación en pantalla |
|---|---|---|---|
| 0–9 s | Gancho | 6 canicas | "¿Qué tan mezcladas están las clases?" · 3 azules, 1 morada, 2 verdes |
| 9–15 s | Proporciones | Canicas → franja ½ · 1/6 · 1/3 | "Lo que importa es la fracción de la bolsa" |
| 15–45 s | Gini–Simpson | Franja → ejes "1ª canica × 2ª canica"; celda azul-azul = ¼; diagonal; resto en naranja | Σp² = 0.389 → D = 1 − 0.389 = 0.6111 (61 % de las parejas); tope 2/3 |
| 45–70 s | Shannon | Bloques diagonales → barras (ancho p, alto −log₂p) → áreas | 1, 2.58, 1.58 bits → 0.500 + 0.431 + 0.528 = 1.4591 bits; tope log₂3 |
| 70–83 s | Cierre | Tarjetas por índice | CART usa Gini, ID3/C4.5 Shannon · nombre y usuario |

El "por qué" de Gini–Simpson se construye explícitamente: se saca una
canica, se devuelve y se saca otra; cada celda del cuadrado es una pareja y
su área es su probabilidad; la diagonal son las parejas del mismo color y el
resto (naranja) es D.

## Compilar

Desde esta carpeta (las rutas relativas no funcionan desde la raíz):

```bash
# Vista rápida (dimensiones pares)
../../../../.venv-manim/bin/manim -r 540,960 --fps 15 heterogeneidad_instagram.py HeterogeneidadInstagram

# Final 1080 × 1920
../../../../.venv-manim/bin/manim heterogeneidad_instagram.py HeterogeneidadInstagram
cp media/videos/heterogeneidad_instagram/1920p30/HeterogeneidadInstagram.mp4 heterogeneidad-instagram.mp4
```

## Texto de publicación

Sigue el formato bilingüe de
[`MANIM-REDES.md §7`](../../../MANIM-REDES.md#7-texto-de-publicación-linkedin),
adaptado a Instagram: en IG los enlaces de la descripción y de los
comentarios no son clicables (a diferencia de LinkedIn), así que el CTA
apunta al link de la bio en vez de "primer comentario".

```text
🇲🇽

¿Cómo medimos la heterogeneidad entre clases en ciencia de datos?

Una bolsa con 6 canicas: 3 azules, 1 morada, 2 verdes. Con esa misma distribución respondo dos preguntas distintas — y cada una tiene su propia geometría.

• Gini–Simpson: si extraigo dos canicas al azar, ¿qué tan probable es que sean de clases distintas? Es el área fuera de la diagonal de un cuadrado de parejas. Aquí da 0.6111 — 61% de los pares.

• Shannon: ¿cuánta sorpresa trae, en promedio, una sola extracción? Es el área bajo un histograma de sorpresa ponderada. Aquí da 1.4591 bits.

Ninguno agota la pregunta: Gini–Simpson pesa las clases dominantes; Shannon no ignora a las raras.

Es literalmente lo que usan CART (impureza Gini) e ID3/C4.5 (ganancia de información) para decidir un corte.

¿Cuál usas tú para medir diversidad de clases en tus datos?

Notas completas: link en mi bio.

#CienciaDeDatos #DataScience #Estadística #Statistics #MachineLearning

🇺🇸

How do we measure heterogeneity between classes in data science?

A bag of 6 marbles: 3 blue, 1 purple, 2 green. With that same distribution I answer two different questions — and each one has its own geometry.

• Gini–Simpson: if I draw two marbles at random, how likely are they to be from different classes? It's the off-diagonal area of a square of pairs. Here it's 0.6111 — 61% of the pairs.

• Shannon: how much surprise does a single draw carry, on average? It's the area under a surprise-weighted histogram. Here it's 1.4591 bits.

Neither one tells the whole story: Gini–Simpson weighs dominant classes; Shannon doesn't ignore the rare ones.

This is literally what CART (Gini impurity) and ID3/C4.5 (information gain) use to pick a split.

Which one do you use to measure class diversity in your data?

Full notes: link in my bio.

#DataScience #CienciaDeDatos #Statistics #Estadística #MachineLearning
```
