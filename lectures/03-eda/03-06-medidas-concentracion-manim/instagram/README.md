# Medidas de concentración · Reel para Instagram (100 s)

Versión para Reels del clip de LinkedIn
([`../linkedin/`](../linkedin/)), reescrita para verse **sin audio**: cada
paso trae una explicación de 1–3 líneas que permanece en pantalla el tiempo
de lectura (`read_time`: 0.6 s + 0.25 s por palabra, mínimo 1.6 s). Sigue la
guía [`lectures/MANIM-REDES.md`](../../../MANIM-REDES.md) (§8, Reels 9:16).

- Formato: 9:16, 1080 × 1920, 30 fps, h264 `yuv420p`, sin audio, 100.4 s.
- Fuente: [`concentracion_instagram.py`](concentracion_instagram.py), escena `ConcentracionInstagram`.
- Salida final: `concentracion-instagram.mp4` (copia de `media/videos/concentracion_instagram/1920p30/`).
- Ejemplo: cafeterías 5, 10, 15, 20, 50 (capítulo `unidades/intro_mineria/2-7-concentracion.qmd`).
- Firma: solo "Diego Villalba · @diego_villalba06" (pie) y, al cierre, nombre
  y usuario en grande. **Sin** nombre de curso ni logo institucional.

## Guion

| Tiempo aprox. | Acto | Transformación | Explicación en pantalla |
|---|---|---|---|
| 0–13 s | Gancho | Barras + línea de la media | "La media es 20… pero Poniente sola vende 50" |
| 13–31 s | CRₖ | Barras → columna 100 % apilada; llave | CR₁ = 50 %, CR₂ = 70 %; "no mira cómo se reparte el resto" |
| 31–46 s | HHI | Columna → ejes "1ª venta × 2ª venta"; cuadros diagonales | 0.5² = 0.25; HHI = 0.325; ≈ 3 cafeterías iguales |
| 46–58 s | Lorenz | Barras ascendentes → escalera acumulada → curva | 80 % de cafeterías → 50 % de ventas |
| 58–91 s | Gini | Diagonal → brecha → triángulo → G = 0.20/0.50; medidor en vivo | igualdad G = 0, monopolio G = 0.80, caso base G = 0.40 |
| 91–100 s | Cierre | Resumen de las 4 medidas + valores | "Ningún índice lo dice todo" · nombre y usuario |

### Por qué Gini (cambio respecto a LinkedIn)

En el clip de LinkedIn el Gini aparecía como fórmula casi sin contexto. Aquí
se construye en cuatro pasos:

1. La **diagonal** es el reparto igualitario (las 5 venden 20).
2. La **brecha** morada entre la curva y la diagonal mide la desigualdad.
3. El **triángulo** bajo la diagonal es la brecha máxima (área 0.5).
4. G = brecha ÷ triángulo = 0.20 / 0.50 = 0.40, y un **medidor en vivo**
   sigue a la curva mientras se deforma: igualdad → 0; una sola vende todo →
   0.80 (tope con 5 unidades, 1 − 1/5; con muchas se acerca a 1); caso base →
   0.40.

## Compilar

Desde esta carpeta (las rutas relativas no funcionan desde la raíz):

```bash
# Vista rápida (dimensiones pares)
../../../../.venv-manim/bin/manim -r 540,960 --fps 15 concentracion_instagram.py ConcentracionInstagram

# Final 1080 × 1920
../../../../.venv-manim/bin/manim concentracion_instagram.py ConcentracionInstagram
cp media/videos/concentracion_instagram/1920p30/ConcentracionInstagram.mp4 concentracion-instagram.mp4
```

## Texto de publicación

Formato bilingüe de [`MANIM-REDES.md §7`](../../../MANIM-REDES.md#7-texto-de-publicación-linkedin),
adaptado a Instagram: los enlaces no son clicables en la descripción, así
que el CTA apunta al link de la bio.

```text
🇲🇽

Dos mercados pueden tener el mismo promedio y repartos completamente distintos.

Cinco cafeterías venden 100 cafés. La media es 20, pero una sola vende la mitad. El promedio no dice quién acumula el total.

• CRₖ: cuánto reúnen las k más grandes. Aquí las dos primeras suman el 70%. Fácil de leer, pero ignora el resto.

• HHI: la probabilidad de que dos ventas al azar sean de la misma cafetería. Da 0.325, como si hubiera unas 3 cafeterías iguales.

• Curva de Lorenz: ordena de menor a mayor y acumula. Aquí el 80% de las cafeterías reúne solo el 50% de las ventas.

• Gini: qué fracción del triángulo de máxima desigualdad ocupa la brecha entre la curva y la igualdad. 0 si todas venden igual, 0.40 en este caso.

Ningún índice lo dice todo; cada uno responde una pregunta distinta.

¿Cuál usas tú para medir la dependencia de pocos clientes, productos o proveedores?

Notas completas: link en mi bio.

#CienciaDeDatos #DataScience #Estadística #Statistics #Manim

🇺🇸

Two markets can share the same average and still split it completely differently.

Five coffee shops sell 100 coffees. The mean is 20, yet a single shop sells half of them. The average doesn't tell you who holds the total.

• CRₖ: how much the top k hold. Here the top two add up to 70%. Easy to read, but it ignores the rest.

• HHI: the probability that two random sales come from the same shop. It's 0.325, as if there were about 3 equal shops.

• Lorenz curve: sort from smallest to largest and accumulate. Here 80% of the shops hold only 50% of the sales.

• Gini: the share of the maximum-inequality triangle filled by the gap between the curve and the line of equality. 0 if everyone sells the same, 0.40 here.

No single index tells the whole story; each one answers a different question.

Which one do you use to measure dependence on a few customers, products or suppliers?

Full notes (in Spanish): link in my bio.

#DataScience #CienciaDeDatos #Statistics #Estadística #Manim
```
