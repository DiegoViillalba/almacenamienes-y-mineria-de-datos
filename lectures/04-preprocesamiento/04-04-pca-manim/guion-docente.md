# Guion docente · Análisis de Componentes Principales

**Duración:** 60 minutos. **Fuente conceptual:** `unidades/preprocesamiento/4-4-PCA.qmd`.
La presentación (`index.html`) tiene una nota en cada estado: tecla `S` abre la
vista del presentador. Este guion organiza el tiempo, los cálculos de pizarra,
las respuestas y los errores que conviene anticipar. El **laboratorio
interactivo** (`interactivo.html`) reproduce en el navegador las tres escenas
clave (resorte, nube 3D, barrido de direcciones) para que el grupo las
manipule.

Todos los números en pantalla se calculan en `sections/data.py` con NumPy; los
valores de este guion salen de ese mismo archivo.

## Idea que recorre la clase

> Seis columnas medidas no significan seis dimensiones con información. PCA
> gira los ejes para que el primero siga la mayor variación; la varianza en
> cada eje es un autovalor de la matriz de covarianza.

El resorte abre la clase como experiencia intuitiva (una masa, tres cámaras,
seis columnas, un grado de libertad) y regresa en el minuto 43 para comprobar
que la receta formal recupera lo que el grupo ya había intuido.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Explicar PCA como una rotación de ejes que ordena direcciones por varianza.
2. Centrar una tabla pequeña, construir `S = X̃ᵀX̃/(n−1)`, obtener autovalores
   y autovectores de una matriz 2 × 2 y calcular puntuaciones `z = X̃v₁`.
3. Leer PVE (proporción de varianza explicada), un *scree plot* y los loadings.
4. Justificar cuándo estandarizar y reconocer tres límites: escala/atípicos,
   ausencia de etiquetas y estructura no lineal.

**Material:** proyector, pizarra, calculadora opcional, laptop con
`interactivo.html` abierto en una segunda pestaña.

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–3 | Portada y objetivos | Preguntar “¿cuántas columnas necesita realmente una tabla?”. No corregir. | Hipótesis iniciales registradas. |
| 3–10 | **1 · El resorte** (3D) | Masa oscilando; tres cámaras; pantallas; fila de seis números; giro de cámara en bucle. | El grupo afirma: 6 números por instante, 1 grado de libertad + ruido. |
| 10–14 | **2 · La tabla** | Series de tiempo de las 6 columnas; mapa de correlación; notación `X ∈ ℝ^{n×p}`. | Describen “mismo ritmo, distinta escala, a veces en espejo”. |
| 14–20 | **3 · Dos sensores** | `(2,3),(3,4),(4,5)`: media, centrado, PC1/PC2, giro de ejes, puntuaciones. | Calculan `x̄=(3,4)` y `z=(−1.41, 0, 1.41)`. |
| 20–29 | **4 · Mejor dirección** | Barrido de la recta; barra naranja + roja = total; Pitágoras; formalización con Lagrange. | Predicen el ángulo (≈32°); explican por qué la suma es constante; llegan a `Su=λu`. |
| 29–36 | **5 · Autovectores** | `S` aplicada al círculo; direcciones que no giran; cálculo a mano completo; receta. | Obtienen `λ=2,0`, `v₁=(1,1)/√2`, PVE₁=100%. |
| 36–43 | **6 · Nube 3D** | Nube de 70 puntos en 3D; giro; flechas PC; plano PC1–PC2; proyección; puntuaciones. | Reconocen que la nube es “casi una hoja”: PC1+PC2 ≈ 98.9%. |
| 43–50 | **7 · Resorte y hélice** | Scree del resorte; loadings; `z₁(t)` vs. posición real; hélice con tres sensores; hélice recuperada. | PC1 ≈ 98.5% en el resorte; 3 componentes ≈ 99.4% en la hélice. |
| 50–54 | **8 · Límites** | Cambio de unidad (×2) y estandarización; atípico que gira PC1; dos clases que PCA no separa. | Explican cuándo escalar; anticipan el efecto de un atípico; distinguen PCA de LDA. |
| 54–58 | **9 · Reto en parejas** | 4 minutos sin solución; luego solución escalonada. | Resultado completo del reto (abajo). |
| 58–60 | Cierre y ticket | Cuatro verbos: gira, mide, resume, cuidado. Ticket individual. | Respuesta escrita del ticket. |

### Si el tiempo apremia

- Acortar el acto 2 a la matriz de correlación (−2 min).
- En el acto 7, mostrar sólo el scree y la hélice recuperada (−2 min).
- No recortar las predicciones del acto 4 ni el reto: son la evidencia principal.

## Hilo conductor y formalización

**1. El resorte (intuición).** Una masa sobre un eje; tres cámaras en ángulos
desconocidos registran `(x, y)` en sus pantallas. Cada instante produce seis
números `xA, yA, xB, yB, xC, yC`. Pregunta clave: *¿alguna cifra se mueve por su
cuenta?* No: todas dependen de la posición sobre el eje. La cámara A ve el eje
casi de lado, por eso `yA` se mueve poco y su correlación con las demás es más
débil (|r| ≈ 0.67): tiene más ruido relativo. **Pregunta guía:** ¿cómo
descubre una máquina ese único eje sin saber física?

**2. Dos sensores (cálculo mínimo).** `X = [[2,3],[3,4],[4,5]]`. Media `(3,4)`;
centrado `X̃ = [[−1,−1],[0,0],[1,1]]`. Los tres puntos caen en la diagonal:
girar los ejes 45° deja todo sobre PC1 y nada sobre PC2. Puntuaciones
`z = X̃v₁ = (−1.414, 0, 1.414)`. Aquí sólo centramos: ambas columnas tienen las
mismas unidades.

**3. Mejor dirección.** Para una recta con dirección unitaria `u`, cada punto
se descompone en proyección (naranja) y residuo (rojo). Por Pitágoras
`‖x̃ᵢ‖² = (uᵀx̃ᵢ)² + ‖eᵢ‖²`; al promediar, **varianza total = varianza
proyectada + error**. En la nube de 26 puntos: `2.33 = 2.10 + 0.23` en el ángulo
óptimo (32°). Maximizar varianza y minimizar error son el mismo problema.

**4. Lagrange.** `Var(X̃u) = uᵀSu`; restricción `uᵀu = 1` (si no, basta alargar
`u`). `∇(uᵀSu − λ(uᵀu−1)) = 2Su − 2λu = 0 ⇒ Su = λu`. Multiplicando por `uᵀ`:
`uᵀSu = λ`: **la varianza proyectada es el autovalor**.

**5. Autovectores como direcciones que no giran.** Al aplicar
`S = [[1.58, 0.84],[0.84, 0.76]]` al círculo unitario, casi todas las flechas
cambian de dirección; las de PC1 y PC2 sólo se estiran por `λ₁ = 2.10` y
`λ₂ = 0.23`. PVE₁ = 2.10/2.33 = 90.1%.

**6. Nube 3D.** Desviaciones 1.9, 0.95 y 0.22 a lo largo de tres ejes
inclinados. `λ = (3.61, 0.90, 0.05)`, PVE = 79.2%, 19.8%, 1.1%. El plano PC1–PC2
guarda 98.9%; los segmentos rojos al proyectar son el error de reconstrucción.
Las puntuaciones forman una tabla de `n × 2`; los loadings son los pesos de
`x₁, x₂, x₃` en cada componente.

**7. Resorte y hélice.** Tabla de 120 × 6 con ruido de 0.06: PC1 = 98.5%. Los
loadings tienen signos opuestos donde las series iban “en espejo”. Las
puntuaciones de PC1 contra el tiempo coinciden con la posición real
(correlación ≈ 0.999). En la hélice del capítulo (un resorte en el espacio),
tres componentes guardan 99.4% y las tres restantes son ruido; las tres
puntuaciones reconstruyen la hélice **deformada linealmente**. PCA no sabe de
física: encuentra la varianza, y la señal tiene más varianza que el ruido.

**8. Límites.**
- *Escala:* multiplicar una columna por 2 cambia la dirección de PC1.
  Estandarizar cuando las unidades son distintas o arbitrarias; sólo centrar
  si comparten unidades y su varianza importa.
- *Atípicos:* un punto de 27 gira PC1 de 32° a ≈ 50° y baja PVE₁ de 90% a ≈ 66%.
  Revisar atípicos; mencionar PCA robusto.
- *Sin etiquetas:* con dos clases alargadas, PC1 va a lo largo y mezcla las
  clases; la separación está en PC2. Si el objetivo es clasificar, comparar
  con LDA.
- *No linealidad:* para variedades curvas (Swiss roll), Isomap, t-SNE o UMAP.

## Reto en parejas (min 54–58)

`X = [[1,3],[2,6],[4,4],[5,7]]`.

| Paso | Resultado |
|---|---|
| Medias | `x̄ = (3, 5)` |
| Centrado | `X̃ = [[−2,−2],[−1,1],[1,−1],[2,2]]` |
| Covarianza | `S = (1/3)[[10,6],[6,10]] = [[3.33, 2],[2, 3.33]]` |
| Autovalores | `λ₁ = 10/3 + 2 = 5.33`, `λ₂ = 10/3 − 2 = 1.33` |
| Autovector | `v₁ = (1, 1)/√2` |
| PVE₁ | `5.33 / 6.67 = 80%` |
| Puntuaciones | `z = (−2.83, 0, 0, 2.83)` |

**Pista si se atoran:** una matriz `[[a, b],[b, a]]` tiene autovalores `a ± b`
con autovectores `(1, ±1)/√2`. **Pregunta de control:** ¿por qué dos puntos
tienen puntuación 0? Porque están sobre PC2 (perpendiculares a PC1).

## Errores frecuentes a vigilar

- **Olvidar centrar:** sin centrar, la “primera componente” apunta a la media,
  no a la variación.
- **Dividir entre n en lugar de n − 1:** cambia los autovalores por un factor,
  no las direcciones ni las PVE. Declarar la convención.
- **Confundir loadings con puntuaciones:** loadings = pesos de variables
  (columnas de `V`); puntuaciones = coordenadas de observaciones (`Z`).
- **Leer PVE como calidad predictiva:** la varianza explicada describe las
  entradas, no el rendimiento de un modelo.
- **Ajustar PCA con todo el conjunto antes de evaluar un modelo:** escalar y
  ajustar PCA sólo con entrenamiento; transformar prueba con esos parámetros.
- **Signo del autovector:** `v` y `−v` son igualmente válidos; el software
  puede devolver cualquiera.

## Ticket de salida (individual)

> Con `λ = (4, 0.9, 0.1)`, ¿cuántas componentes conservas y qué porcentaje de
> varianza guardas? Bonus: ¿escalarías si las columnas fueran pesos, metros y
> años?

**Respuesta esperada:** dos componentes, `(4 + 0.9)/5 = 98%`. Sí: unidades
distintas ⇒ estandarizar antes de PCA.

## Uso del laboratorio interactivo

- **Resorte y tres cámaras:** subir el ruido (PC1 baja, las demás suben
  parejas); subir el movimiento lateral (aparece un segundo componente real).
  Pregunta: *¿cuántas componentes para 95%?*
- **Nube 3D:** igualar las tres desviaciones (esfera: PVE ≈ 33% cada una, los
  ejes quedan casi arbitrarios); activar el atípico y observar cómo se inclinan
  los ejes.
- **Barrido 2D:** comprobar que naranja + rojo es constante; cambiar la escala
  de x y observar cómo PC1 se “acuesta”.
