# Reglas de asociación · Reel para Instagram (96 s)

Reel vertical que explica la idea central de las reglas de asociación con un
ejemplo clínico, pensado para verse **sin audio**: cada paso trae una
explicación de 1–3 líneas que permanece en pantalla el tiempo de lectura
(`read_time`: 0.6 s + 0.25 s por palabra, mínimo 1.6 s). Sigue la guía
[`lectures/MANIM-REDES.md`](../../../MANIM-REDES.md) (§8, Reels 9:16).
Las métricas de evaluación (leverage, conviction, Zhang…) quedan para un
video posterior; el cierre lo anuncia.

- Formato: 9:16, 1080 × 1920, 30 fps, h264 `yuv420p`, sin audio, 96.2 s.
- Fuente: [`reglas_instagram.py`](reglas_instagram.py), escena `ReglasInstagram`.
- Salida final: `reglas-instagram.mp4` (copia de `media/videos/reglas_instagram/1920p30/`).
- Ejemplo (**datos ilustrativos**, se rotulan así en pantalla): 20 pacientes
  con diabetes (6), hipertensión (8) y asma (5). Regla Diabetes ⇒ Hipertensión:
  soporte 5/20 = 25 %, confianza 5/6 = 83 %, base P(HTA) = 40 %,
  lift = 2.1. Contraste Asma ⇒ Hipertensión: 2/5 = 40 %, lift = 1.0.
  Los conjuntos y los `assert` están al inicio del script.
- Colores: azul = diabetes (antecedente A), naranja = hipertensión
  (consecuente B), rosa = asma (solo en este clip), verde = soporte,
  morado = lift, amarillo = nivel esperado por independencia (como en la clase).
- Firma: solo "Diego Villalba · @diego_villalba06" (pie) y, al cierre, nombre
  y usuario en grande. **Sin** nombre de curso ni logo institucional.
- Zona segura: el contenido vive en y ∈ [−4.1, 5.6] del marco 8 × 14.22.

## Guion

| Tiempo aprox. | Acto | Transformación | Explicación en pantalla |
|---|---|---|---|
| 0–14 s | Gancho | Cuadrícula de 20 fichas de pacientes con tres puntos (diabetes, hipertensión, asma) | "¿Qué enfermedades aparecen juntas?" · mismo problema que el súper · cada paciente es una canasta |
| 14–30 s | Regla | Aparece "Diabetes ⇒ Hipertensión" (antecedente / consecuente); se resaltan los diabéticos | Si tiene diabetes, ¿suele tener hipertensión? · no implica causa |
| 30–46 s | Soporte | Se apagan todos menos los 5 con ambas; insignias 1…5 | soporte = 5/20 = 25 % · 1 de cada 4 · soporte bajo puede ser casualidad |
| 46–62 s | Confianza | Solo quedan los 6 diabéticos; parpadea el punto naranja de 5 | confianza = 5/6 = 83 % · ¿83 % es mucho? |
| 62–88 s | Lift | Cada ficha se vuelve celda de una barra; línea amarilla en 40 %; barras de diabetes (83 %) y asma (40 %) | lift = 83/40 = 2.1 · asma: lift = 1.0 · lift = 1 sin asociación |
| 88–96 s | Cierre | Tarjetas Regla / Soporte / Confianza / Lift | "Juntas no significa causa" · próximo video · nombre y usuario |

## Compilar

Desde esta carpeta (las rutas relativas no funcionan desde la raíz):

```bash
# Vista rápida (dimensiones pares)
../../../../.venv-manim/bin/manim -r 540,960 --fps 15 reglas_instagram.py ReglasInstagram

# Final 1080 × 1920
../../../../.venv-manim/bin/manim reglas_instagram.py ReglasInstagram
cp media/videos/reglas_instagram/1920p30/ReglasInstagram.mp4 reglas-instagram.mp4
```

## Texto de publicación

```text
🇲🇽

¿La diabetes llega sola? Los datos dicen que casi nunca.

Con 20 pacientes (datos ilustrativos) buscamos qué diagnósticos aparecen juntos. Es el mismo problema que el carrito del súper: cada paciente es una "canasta".

Una regla de asociación dice: si A, entonces B. Por ejemplo, Diabetes ⇒ Hipertensión. Para creerle la medimos así:

• Soporte: ¿qué tan común es la pareja? 5 de 20 pacientes tienen ambas: 25%.

• Confianza: ¿qué tan fiable es la regla? De 6 pacientes con diabetes, 5 tienen hipertensión: 83%.

• Lift: ¿es más de lo esperado? En general, 40% tiene hipertensión; con diabetes, 83%. Lift = 83 / 40 = 2.1. Con asma da 40% → lift = 1: no hay asociación.

Que aparezcan juntas no significa que una cause la otra.

En el próximo video: cómo evaluar reglas y por qué la confianza sola puede engañar.

¿Qué otras enfermedades crees que aparecen juntas?

Notas completas: link en mi bio.

#CienciaDeDatos #DataScience #MineríaDeDatos #DataMining #Manim

🇺🇸

Does diabetes come alone? The data say it rarely does.

With 20 patients (illustrative data) we look for diagnoses that show up together. It's the same problem as a supermarket cart: each patient is a "basket".

An association rule says: if A, then B. For example, Diabetes ⇒ Hypertension. To trust it, we measure it like this:

• Support: how common is the pair? 5 out of 20 patients have both: 25%.

• Confidence: how reliable is the rule? Of 6 patients with diabetes, 5 have hypertension: 83%.

• Lift: is it more than expected? Overall, 40% have hypertension; with diabetes, 83%. Lift = 83 / 40 = 2.1. With asthma it's 40% → lift = 1: no association.

Showing up together doesn't mean one causes the other.

Next video: how to evaluate rules and why confidence alone can mislead.

Which other conditions do you think show up together?

Full notes (in Spanish): link in my bio.

#DataScience #CienciaDeDatos #DataMining #MineríaDeDatos #Manim
```
