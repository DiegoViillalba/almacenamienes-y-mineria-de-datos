# Guion docente · repaso animado de variabilidad

## Resultado observable

Al terminar, cada estudiante debe poder tomar una variable numérica genérica y
seguir esta cadena:

```text
pregunta → ordenar y dibujar → mirar forma/extremos → elegir → calcular → interpretar
```

La clase no busca recitar fórmulas. Cada medida nace de una acción geométrica:

- extremos → rango;
- mitad central → IQR;
- distancias al centro → varianza y desviación estándar;
- forma y propósito → elección de la pareja de medidas.

## Ruta sugerida

| Minutos | Bloque | Evidencia esperada |
|---:|---|---|
| 0–3 | Gancho de las rutas | Predicen que A es más constante aunque aún no calculan |
| 3–6 | Rango | Explican por qué un solo extremo lo domina |
| 6–12 | Cuartiles e IQR | Reconstruyen 20.5, 24.5, 29 e IQR = 8.5 min |
| 12–16 | Cercas y boxplot | Detectan 42 y 90; distinguen cerca de bigote |
| 16–23 | Varianza y desviación | Justifican cuadrados, `n-1` y regreso a minutos |
| 23–26 | Forma y robustez | Asignan media+s o mediana+IQR con una salvedad |
| 26–30 | Reto de transferencia | Producen un reporte con número, unidad y contexto |

## Conducción por bloques

### 1. Dos rutas, media 42

No muestre la media antes de la votación. Las dos series son:

```text
Ruta A: 38, 40, 41, 42, 43, 44, 45, 43
Ruta B: 10, 20, 30, 40, 44, 50, 60, 82
```

Ambas suman 336 y tienen media 42 min. La respuesta defendible es A si
“confiable” significa menor variación en el tiempo de llegada.

Frase de enlace:

> La media dice dónde está el grupo; todavía no dice qué tan abierto está.

### 2. El rango escucha sólo a dos datos

Los sueldos habituales, en miles, van de 12 a 15: rango 3. Al incorporar 120,
el rango salta a 108. Pregunte qué ocurrió con los puntos intermedios: nada.

Evite decir que el rango “es malo”. Sirve cuando la extensión total es la
pregunta; es frágil cuando queremos describir el comportamiento típico.

### 3. Ordenar y cortar

Datos de espera:

```text
42, 18, 25, 30, 22, 19, 28, 24, 21, 26, 20, 90
```

Ordenados:

```text
18, 19, 20, 21, 22, 24 | 25, 26, 28, 30, 42, 90
```

Con mediana de mitades:

```text
Q1 = (20 + 21) / 2 = 20.5
Q2 = (24 + 25) / 2 = 24.5
Q3 = (28 + 30) / 2 = 29
IQR = 29 - 20.5 = 8.5 min
```

Interpretación modelo:

> La mitad central de los tiempos ocupa el intervalo de 20.5 a 29 minutos;
> su amplitud es 8.5 minutos.

### 4. Cercas, atípicos y boxplot

```text
LI = 20.5 - 1.5(8.5) = 7.75
LS = 29 + 1.5(8.5) = 41.75
```

42 y 90 son candidatos atípicos altos. Haga una pausa antes de la caja y pida
predecir sus piezas. El boxplot correcto usa:

- caja de 20.5 a 29;
- mediana en 24.5;
- bigotes en 18 y 30, los datos no atípicos más extremos;
- puntos aislados en 42 y 90.

Pregunta de control:

> ¿Las cercas 7.75 y 41.75 son los bigotes?

Respuesta: no. Las cercas son umbrales; los bigotes terminan en observaciones.

### 5. Construir la varianza

Para la ruta A, las desviaciones respecto de 42 son:

```text
-4, -2, -1, 0, 1, 2, 3, 1
```

Suman cero. Esto no significa dispersión cero: los signos se cancelan. Sus
cuadrados son:

```text
16, 4, 1, 0, 1, 4, 9, 1  →  suma = 36
```

Al estimar la media con la misma muestra queda un grado de libertad menos:

```text
s² = 36 / 7 = 5.14 min²
s  = √5.14 = 2.27 min
```

Para la ruta B:

```text
desviaciones: -32, -22, -12, -2, 2, 8, 18, 40
cuadrados:     1024, 484, 144, 4, 4, 64, 324, 1600
suma = 3648; s² = 521.14 min²; s = 22.83 min
```

No presente `s` como “el intervalo que contiene a la mayoría”. Descríbala como
una escala típica de alejamiento respecto de la media.

### 6. Forma antes que fórmula

Regla práctica, no teorema automático:

- forma aproximadamente simétrica y sin extremos influyentes → media + `s`;
- sesgo o extremos → mediana + IQR suele conservar mejor la historia central.

La pregunta de negocio puede cambiar la elección. Para reconstruir un total,
por ejemplo, la media aún puede ser relevante en una distribución sesgada.

### 7. Reto final

Conjunto:

```text
6, 24, 5, 8, 4, 6, 7, 5
```

Dé 45 segundos antes de avanzar. Solución:

```text
orden: 4, 5, 5, 6, 6, 7, 8, 24
Q1 = 5; mediana = 6; Q3 = 7.5; IQR = 2.5
LI = 1.25; LS = 11.25
24 > 11.25 → candidato atípico alto
```

Reporte modelo:

> El valor central es 6 unidades y la mitad central abarca 2.5 unidades. El 24
> es un candidato atípico alto; conviene investigarlo antes de decidir cómo
> tratarlo.

## Correcciones incorporadas respecto del PDF de referencia

- En el ejemplo clínico los atípicos son **42 y 90**, no “40 y 90”.
- La distribución con media ≈ mediana ≈ moda es aproximadamente **simétrica**.
- Las relaciones entre media, mediana y moda bajo sesgo son tendencias, no
  identidades universales.
- El método de cálculo de cuartiles se declara explícitamente.
- Las cercas de Tukey no se dibujan automáticamente como bigotes.

