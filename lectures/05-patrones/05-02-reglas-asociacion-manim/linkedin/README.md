# Reglas de asociación · clip para LinkedIn (4:5)

Mismo video que el Reel de [`../instagram/`](../instagram/), solo
**reencuadrado** a 4:5: `ReglasLinkedIn` hereda de `ReglasInstagram` sin
cambiar la animación y mueve el centro de la cámara a y = 0.47, donde está
centrado el contenido del Reel.

- Formato: 4:5, 1080 × 1350, 30 fps, h264 `yuv420p`, sin audio, 96.2 s.
- Marco 8.8 × 11 (≈123 px/unidad frente a 135 en 9:16): los textos quedan
  ≈9 % más chicos que en el Reel.
- Fuente: [`reglas_linkedin.py`](reglas_linkedin.py), escena `ReglasLinkedIn`.
- Salida final: `reglas-linkedin.mp4`.
- Guion y ejemplo: ver [`../instagram/README.md`](../instagram/README.md).

## Compilar

```bash
../../../../.venv-manim/bin/manim -r 544,680 --fps 15 reglas_linkedin.py ReglasLinkedIn   # vista rápida
../../../../.venv-manim/bin/manim reglas_linkedin.py ReglasLinkedIn
cp media/videos/reglas_linkedin/1350p30/ReglasLinkedIn.mp4 reglas-linkedin.mp4
```

## Texto de publicación

```text
🇲🇽

¿La diabetes llega sola? En los datos, casi nunca.

Tomé 20 pacientes (datos ilustrativos) y busqué qué diagnósticos aparecen juntos. Es el mismo problema que el análisis de canasta del supermercado, solo que cada "canasta" es un paciente.

Una regla de asociación dice: si A, entonces B. Para Diabetes ⇒ Hipertensión:

• Soporte: ¿qué tan común es la pareja? 5 de 20 pacientes tienen ambas: 25%.

• Confianza: ¿qué tan fiable es la regla? De 6 pacientes con diabetes, 5 tienen hipertensión: 83%.

• Lift: ¿es más de lo esperado al azar? La hipertensión aparece en 40% de todos y en 83% de los diabéticos: lift = 2.1. Con asma da 40%, lift = 1: no hay asociación.

Coocurrencia no es causalidad, pero sí es una buena pista de dónde mirar.

En el próximo video: cómo evaluar reglas y por qué la confianza sola puede engañar.

¿Dónde aplicarías reglas de asociación en tu trabajo?

Las notas completas están en el primer comentario.

#CienciaDeDatos #DataScience #MineríaDeDatos #DataMining #HealthData

🇺🇸

Does diabetes come alone? In the data, it rarely does.

I took 20 patients (illustrative data) and looked for diagnoses that show up together. It's the same problem as supermarket basket analysis, except each "basket" is a patient.

An association rule says: if A, then B. For Diabetes ⇒ Hypertension:

• Support: how common is the pair? 5 out of 20 patients have both: 25%.

• Confidence: how reliable is the rule? Of 6 patients with diabetes, 5 have hypertension: 83%.

• Lift: is it more than chance would predict? Hypertension shows up in 40% of everyone and in 83% of diabetics: lift = 2.1. With asthma it's 40%, lift = 1: no association.

Co-occurrence isn't causation, but it is a good hint of where to look.

Next video: how to evaluate rules and why confidence alone can mislead.

Where would you apply association rules in your work?

Full notes are in the first comment.

#DataScience #CienciaDeDatos #DataMining #MineríaDeDatos #HealthData
```

Primer comentario:

```text
🇲🇽 Notas completas (soporte, confianza, lift y Apriori): https://diego-villalba.com/almacenamienes-y-mineria-de-datos/unidades/reglas_asociacion/5-1-introduccion.html

🇺🇸 Full notes (in Spanish): https://diego-villalba.com/almacenamienes-y-mineria-de-datos/unidades/reglas_asociacion/5-1-introduccion.html
```
