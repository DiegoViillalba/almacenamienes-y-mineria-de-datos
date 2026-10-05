# Análisis de Componentes Principales · Reel para Instagram (110 s)

Reel vertical que resume la clase de PCA en un solo hilo geométrico, pensado
para verse **sin audio**: cada paso trae una explicación de 1–2 líneas que
permanece en pantalla el tiempo de lectura (`read_time`: 0.6 s + 0.25 s por
palabra, mínimo 1.6 s). Sigue la guía
[`lectures/MANIM-REDES.md`](../../../MANIM-REDES.md) (§8, Reels 9:16).

- Formato: 9:16, 1080 × 1920, 30 fps, h264 `yuv420p`, sin audio, 109.5 s.
- Fuente: [`pca_instagram.py`](pca_instagram.py), escena `PCAInstagram`.
- Salida final: `pca-instagram.mp4` (copia de `media/videos/pca_instagram/1920p30/`).
- Ejemplo: la nube de 26 lecturas de dos sensores del acto 4 de la clase
  (`sections/data.py`, `CLOUD`): PC1 a 32°, λ = 2.10 y 0.23, total 2.33,
  PVE₁ = 90.1 %. Todos los números se calculan desde `sections/data.py`.
- Firma: solo "Diego Villalba · @diego_villalba06" (pie) y, al cierre, nombre
  y usuario en grande. **Sin** nombre de curso ni logo institucional.
- Zona segura: el contenido vive en y ∈ [−4.1, 5.6] del marco 8 × 14.22.

## Guion

| Tiempo aprox. | Acto | Transformación | Explicación en pantalla |
|---|---|---|---|
| 0–14 s | Gancho | Nube sin centrar en ejes "sensor 1 × sensor 2" | "¿Cuántas columnas necesita tu tabla?" · 2 columnas, pero casi una línea |
| 14–25 s | Centrar | Media en amarillo → la nube se traslada al origen | "Ahora cada punto es una desviación" |
| 25–58 s | Barrer | Recta que gira; proyecciones naranjas, residuos rojos; barra conservada + perdida | Máximo a 32°: conserva 2.10 de 2.33; Pitágoras explica por qué la suma es fija |
| 58–80 s | Autovector | Círculo de flechas × S → elipse; dos flechas no giran | Sv₁ = 2.10 v₁, Sv₂ = 0.23 v₂: los λ son las varianzas del barrido |
| 80–100 s | Reducir | Ejes PC1/PC2; la nube gira 32° y cae sobre PC1 | De 2 columnas a 1 guardando el 90.1 % |
| 100–110 s | Cierre | Tarjetas de los 4 pasos | "Muchas columnas, pocas direcciones" · nombre y usuario |

## Compilar

Desde esta carpeta (las rutas relativas no funcionan desde la raíz):

```bash
# Vista rápida (dimensiones pares)
../../../../.venv-manim/bin/manim -r 540,960 --fps 15 pca_instagram.py PCAInstagram

# Final 1080 × 1920
../../../../.venv-manim/bin/manim pca_instagram.py PCAInstagram
cp media/videos/pca_instagram/1920p30/PCAInstagram.mp4 pca-instagram.mp4
```

## Texto de publicación

```text
🇲🇽

Tu tabla tiene 2 columnas, pero tus datos quizá viven en una sola línea.

26 lecturas de dos sensores forman una nube casi recta. PCA encuentra esa recta en cuatro pasos:

• Centrar: resta la media para que la nube quede alrededor del origen.

• Barrer: gira una recta y proyecta cada punto. Lo que conserva más lo que pierde suma siempre 2.33 (Pitágoras). El máximo está a 32° y conserva 2.10.

• Autovector: esa recta es la única dirección que la matriz de covarianza no gira; solo la estira por λ = 2.10, la misma varianza del barrido.

• Reducir: gira los ejes y quédate con PC1. Con 1 columna en vez de 2 guardas el 90.1% de la varianza.

Muchas columnas no significan muchas direcciones con información.

¿En qué datos tuyos sospechas que hay columnas que dicen lo mismo?

Notas completas: link en mi bio.

#CienciaDeDatos #DataScience #MachineLearning #ÁlgebraLineal #Manim

🇺🇸

Your table has 2 columns, but your data may live on a single line.

26 readings from two sensors form an almost straight cloud. PCA finds that line in four steps:

• Center: subtract the mean so the cloud sits around the origin.

• Sweep: rotate a line and project every point. What it keeps plus what it loses always adds up to 2.33 (Pythagoras). The maximum is at 32° and keeps 2.10.

• Eigenvector: that line is the only direction the covariance matrix doesn't rotate; it just stretches it by λ = 2.10, the same variance from the sweep.

• Reduce: rotate the axes and keep PC1. With 1 column instead of 2 you keep 90.1% of the variance.

Many columns don't mean many informative directions.

Where do you suspect your data has columns that say the same thing?

Full notes (in Spanish): link in my bio.

#DataScience #CienciaDeDatos #MachineLearning #LinearAlgebra #Manim
```
