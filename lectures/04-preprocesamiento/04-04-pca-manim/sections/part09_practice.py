"""Acto 9 · Práctica en parejas, solución escalonada y cierre."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import PRACTICE, PRACTICE_VALS, PRACTICE_Z
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, bullet_list, card, pill, safe_text, tex)


class PracticeSection:
    def practice(self) -> None:
        self.beat("54–58 min. Parejas, 4 minutos. No avance hasta que la mayoría tenga S. Pista si "
                  "se atoran: una matriz [[a, b], [b, a]] tiene λ = a ± b y v = (1, ±1)/√2.")
        self.start("Reto en parejas", "Cuatro lecturas, dos sensores", YELLOW)
        X = tex(r"X=\begin{pmatrix}1&3\\2&6\\4&4\\5&7\end{pmatrix}", size=44).move_to([-4.2, -0.4, 0])
        tasks = bullet_list([
            "1. Medias y matriz centrada X̃.",
            "2. Matriz de covarianza S (divisor n − 1).",
            "3. λ₁, λ₂ y el vector v₁.",
            "4. PVE de PC1 y puntuaciones z.",
        ], size=28, width=6.6).move_to([2.0, 0.2, 0])
        timer = pill("4 min · pista: [[a, b], [b, a]] → λ = a ± b", MUTED, size=20).move_to([2.0, -2.4, 0])
        self.play(Write(X), LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tasks], lag_ratio=0.15))
        self.play(FadeIn(timer))

        self.beat("Solución escalonada (1/2). Pida a una pareja dictar medias y S; corrija el divisor "
                  "n − 1 = 3.")
        self.start("Solución", "Paso a paso", GREEN)
        sol = VGroup(
            tex(r"\bar x=(3,\;5),\qquad \tilde X=\begin{pmatrix}-2&-2\\-1&1\\1&-1\\2&2\end{pmatrix}", size=36),
            tex(r"S=\tfrac13\begin{pmatrix}10&6\\6&10\end{pmatrix}=\begin{pmatrix}3.33&2\\2&3.33\end{pmatrix}", size=36),
            tex(rf"\lambda_1=\tfrac{{10}}{{3}}+2={PRACTICE_VALS[0]:.2f},\quad \lambda_2=\tfrac{{10}}{{3}}-2={PRACTICE_VALS[1]:.2f},"
                r"\quad \mathbf v_1=\tfrac1{\sqrt2}(1,\,1)", size=34, color=ORANGE),
            tex(rf"\mathrm{{PVE}}_1=\frac{{{PRACTICE_VALS[0]:.2f}}}{{{PRACTICE_VALS.sum():.2f}}}"
                rf"={100 * PRACTICE_VALS[0] / PRACTICE_VALS.sum():.0f}\%,\qquad z=("
                + r",\;".join(f"{v:.2f}" for v in PRACTICE_Z[:, 0]) + ")", size=34, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([0, -0.45, 0])
        self.play(Write(sol[0]), run_time=1.0)
        self.play(Write(sol[1]), run_time=1.0)
        self.beat("Solución (2/2). Autovalores, varianza explicada y puntuaciones. Pregunta de "
                  "control: ¿por qué dos puntos tienen puntuación 0? (están sobre PC2).")
        self.play(Write(sol[2]), run_time=1.0)
        self.play(Write(sol[3]), run_time=1.0)

    def closing(self) -> None:
        self.beat("58–59 min. Síntesis en tres frases; pida a un estudiante reformular cada una con "
                  "el resorte como ejemplo.")
        self.start("Cierre", "Lo que PCA hace y lo que no", BLUE)
        ideas = VGroup(*[
            VGroup(pill(k, c, width=2.6, size=22), safe_text(t, size=26, max_width=8.8)).arrange(RIGHT, buff=0.4)
            for k, c, t in [
                ("Gira", ORANGE, "Ejes nuevos ortogonales que siguen la mayor variación."),
                ("Mide", YELLOW, "Cada autovalor es la varianza en su eje; PVE = λ / Σλ."),
                ("Resume", GREEN, "Puntuaciones Z: pocas columnas que conservan casi todo."),
                ("Cuidado", RED, "Escala, atípicos, sin etiquetas, sólo estructura lineal."),
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.2, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.2), run_time=1.4)

        self.beat("59–60 min. Ticket individual. Respuesta: 2 componentes; (4 + 0.9) / 5 = 98%. "
                  "Además: si hubiera evaluación predictiva, ajuste escalado y PCA sólo en "
                  "entrenamiento.")
        self.start("Ticket de salida", "Decide y justifica", PURPLE)
        q = VGroup(
            tex(r"\lambda=(4,\;0.9,\;0.1)", size=52, color=INK),
            safe_text("¿Cuántas componentes conservas y qué porcentaje de varianza guardas?",
                      size=28, max_width=11.5),
            safe_text("Bonus: ¿escalarías si las columnas fueran pesos, metros y años?", size=24,
                      color=MUTED, max_width=11.5),
        ).arrange(DOWN, buff=0.45).move_to([0, -0.1, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in q], lag_ratio=0.25))
        final = VGroup(safe_text("Siguiente: t-SNE y UMAP para estructura no lineal", size=22, color=BLUE))
        final.move_to([0, -2.75, 0])
        self.play(FadeIn(final))
