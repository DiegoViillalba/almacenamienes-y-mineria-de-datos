# Ensambles de árboles y XGBoost · clase animada (60 min)

Continuación de [Árboles de decisión](../06-06-arboles-decision-manim/). Parte
de la debilidad con que cerró esa clase (un árbol es inestable) y construye
las dos familias de ensambles, de la intuición geométrica a la formalización,
con los ejemplos de `capitulo_arboles_regresion_lineal.qmd` y
`capitulo_xgboost.qmd`:

1. **Árboles de regresión** (0–9): seis laptops; residuos y una barra de SSE que
   se encoge con cada corte (RAM ≤ 12 reduce 90.5 %); un árbol profundo sobre
   40 puntos memoriza el ruido.
2. **La sabiduría de muchos** (9–17): cinco jueces al 70 % (0.837); curvas de
   Condorcet; la varianza de un promedio y el piso ρσ².
3. **Bagging y bosques** (17–27): urna bootstrap {A, A, C, D, F, F}; fuera de
   la bolsa y (1 − 1/n)ⁿ → e⁻¹; árboles tenues y su promedio (0.665 → 0.540);
   bosques aleatorios: variables al azar en cada nodo.
4. **AdaBoost** (27–37): diez puntos, pesos como barras; tres rondas
   (α = 0.424, 0.650, 0.752); la suma ponderada clasifica todo; pérdida
   exponencial.
5. **Gradient boosting** (37–46): cinco casas; los residuos bajan a su panel,
   un tocón los ajusta y se suma η·h (SSE 106 → 43.5 → 16.83 → 8.43); el
   residuo es el gradiente negativo.
6. **XGBoost** (46–55): objetivo regularizado; Taylor de 2º orden sobre la
   log-loss; peso de hoja como vértice de una parábola que λ encoge (−5 → −3.33);
   ganancia de cada corte y poda con γ (29.17); clasificación con g = p − y,
   h = p(1 − p); lo que agrega la librería.
7. **Comparar, ejercicio y cierre** (55–60): bagging frente a boosting,
   hiperparámetros, ejercicio a mano y ticket de salida.

- [Guion docente](guion-docente.md): tiempos, cálculos, respuestas y errores
  frecuentes.
- Notas del presentador en cada estado (tecla `S`).

## Cómo está organizado (qué archivo editar)

```text
presentacion.py            ensambla la escena EnsamblesXGBoost y elige los actos
sections/
├── data.py                TODOS los números: laptops, Condorcet, 40 puntos del
│                          laboratorio de bagging, AdaBoost, gradient boosting y
│                          la primera ronda de XGBoost
├── theme.py               paleta, tipografía y piezas pequeñas (pill, card, table…)
├── base.py                cintillo institucional, encabezados, beat() = pausa + notas
├── part01_regression.py   acto 1 · también define los ejes de las laptops y step_graph
├── part02_crowd.py        acto 2
├── part03_bagging.py      acto 3
├── part04_adaboost.py     acto 4
├── part05_gradient.py     acto 5
├── part06_xgboost.py      acto 6
└── part07_close.py        acto 7
```

- **Cambiar un número o un ejemplo:** sólo en `sections/data.py`;
  `python -m sections.data` imprime el resumen (coincide con el libro).
- **Cambiar un color:** `sections/theme.py`.
- **Cambiar una lámina:** cada estado empieza con `self.beat("notas…")`.
- **Quitar o reordenar actos:** la lista `acts` de `presentacion.py`.

## Entorno

Desde la raíz del repositorio, si aún no existe `.venv-manim`:

```bash
uv venv --python python3 .venv-manim
uv pip install --python .venv-manim/bin/python -r lectures/06-clasificacion/06-07-ensambles-xgboost-manim/requirements.txt
```

Las fórmulas usan `MathTex`, así que se necesita LaTeX (MacTeX o TinyTeX).

## Compilar y presentar

Ejecutar desde esta carpeta:

```bash
# Vista rápida de revisión
../../../.venv-manim/bin/manim-slides render -ql presentacion.py EnsamblesXGBoost

# Sólo algunos actos (1–7), útil al editar
CLS_PARTS=5,6 ../../../.venv-manim/bin/manim-slides render -ql presentacion.py EnsamblesXGBoost

# Entrega 4K (3840 × 2160) a 30 fps
../../../.venv-manim/bin/manim-slides render --resolution 3840,2160 --fps 30 --max-inflight-encoders 4 presentacion.py EnsamblesXGBoost

# HTML autónomo para la galería
../../../.venv-manim/bin/manim-slides convert --folder slides --offline --one-file -ccontrols=true -cprogress=true -cslide_number=true -chash=true EnsamblesXGBoost ../../../docs/lectures/_output/06-clasificacion/06-07-ensambles-xgboost-manim/index.html
```

Abrir `docs/lectures/_output/06-clasificacion/06-07-ensambles-xgboost-manim/index.html`.
`→` avanza, `←` retrocede, `S` abre notas, `O` muestra todos los estados.

Nota: no usar pausas de un solo fotograma (`self.wait(0.1)` a 15 fps); rompen
la concatenación de Manim Slides. Usar al menos `self.wait(0.3)`.
