# Medidas de concentración · clip para LinkedIn (45 s)

Video vertical que resume la clase de concentración en un solo hilo
geométrico. Sigue la guía [`lectures/MANIM-REDES.md`](../../../MANIM-REDES.md).

- Formato: 4:5, 1080 × 1350, 30 fps, sin audio, 44.6 s.
- Fuente: [`concentracion_linkedin.py`](concentracion_linkedin.py), escena `ConcentracionLinkedIn`.
- Salida final: `concentracion-linkedin.mp4` (copia de `media/videos/concentracion_linkedin/1350p30/`).
- Ejemplo: ventas 5, 10, 15, 20, 50 (capítulo `unidades/intro_mineria/2-7-concentracion.qmd`).

## Guion

| Tiempo aprox. | Acto | Transformación | Rótulo / número |
|---|---|---|---|
| 0–5 s | Gancho | Barras + línea de la media | “La media es 20… pero no dice cómo se reparte” |
| 5–12 s | CRₖ | Barras → columna 100 % apilada; llave sobre la cima | CR₁ = 50 %, CR₂ = 70 % |
| 12–21 s | HHI | Columna → franjas de un cuadrado 1 × 1; cuadros de la diagonal | HHI = Σ sᵢ² = 0.325 ≈ 3 unidades efectivas |
| 21–30 s | Lorenz | Barras ascendentes → escalera acumulada → curva | 80 % de sucursales → 50 % de ventas |
| 30–37 s | Deformación | Curva: igualdad → monopolio → caso base; brecha sombreada | “Más arco = más concentración” |
| 37–41 s | Gini | Brecha ÷ triángulo bajo la diagonal | G = 0.20 / 0.50 = 0.40 |
| 41–45 s | Cierre | Resumen de las 4 medidas + curso y autor | “Ningún índice lo dice todo” |

## Compilar

Desde esta carpeta (las rutas relativas no funcionan desde la raíz):

```bash
# Vista rápida (dimensiones pares)
../../../../.venv-manim/bin/manim -r 544,680 --fps 15 concentracion_linkedin.py ConcentracionLinkedIn

# Final 1080 × 1350
../../../../.venv-manim/bin/manim concentracion_linkedin.py ConcentracionLinkedIn
cp media/videos/concentracion_linkedin/1350p30/ConcentracionLinkedIn.mp4 concentracion-linkedin.mp4
```

## Texto publicado

Post:

```text
🇲🇽

Dos mercados pueden tener el mismo promedio y repartos completamente distintos.

Cinco sucursales venden 100 cafés en total. La media es 20, pero una sola vende la mitad. El promedio no dice quién acumula el total.

En este video de 45 segundos muestro cuatro formas de medirlo, cada una con su intuición geométrica:

• CRₖ: cuánto reúnen las k unidades más grandes. Aquí las dos primeras suman el 70%. Es fácil de leer, pero ignora el resto.

• HHI: la probabilidad de que dos ventas elegidas al azar vengan de la misma sucursal. Da 0.325, lo que equivale a unas 3 sucursales "efectivas".

• Curva de Lorenz: ordenas de menor a mayor y acumulas. Aquí el 80% de las sucursales reúne solo el 50% de las ventas.

• Gini: el área entre la curva y la línea de igualdad convertida en un número. Aquí da 0.40.

Ningún índice lo dice todo; cada uno responde una pregunta distinta.

Si quieres saber más, en el primer comentario te dejo el enlace a mis notas del curso Almacenes y Minería de Datos (Facultad de Ciencias, UNAM).

¿Cuál usas tú para medir la dependencia de pocos clientes, productos o proveedores?

🇺🇸

Two markets can share the same average and still have completely different distributions.

Five stores sell 100 coffees in total. The mean is 20, yet a single store sells half of them. The average doesn't tell you who holds the total.

In this 45-second video I show four ways to measure it, each with its geometric intuition:

• CRₖ: how much the top k units hold. Here the top two add up to 70%. Easy to read, but it ignores the rest.

• HHI: the probability that two randomly chosen sales come from the same store. It's 0.325, the equivalent of about 3 "effective" stores.

• Lorenz curve: sort from smallest to largest and accumulate. Here 80% of the stores hold only 50% of the sales.

• Gini: the area between the curve and the line of equality, turned into a single number. Here it's 0.40.

No single index tells the whole story; each one answers a different question.

If you want to learn more, you'll find the link to my course notes for Data Warehousing and Data Mining (Faculty of Sciences, UNAM, in Spanish) in the first comment.

Which one do you use to measure dependence on a few customers, products or suppliers?

#CienciaDeDatos #DataScience #Estadística #Statistics #Manim
```

Primer comentario:

```text
🇲🇽 Notas completas: cálculo paso a paso, laboratorio interactivo y ejercicios.
🇺🇸 Full notes (in Spanish): step-by-step calculations, interactive lab and exercises.

https://diego-villalba.com/almacenamienes-y-mineria-de-datos/unidades/intro_mineria/2-7-concentracion.html
```
