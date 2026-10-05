"""Acto 3 · El ejemplo mínimo del capítulo: tres lecturas de dos sensores."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import TOY, TOY_MEAN, TOY_Z
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED, YELLOW,
                            card, pill, safe_text, tex)


class TwoSensorsSection:
    def two_sensors(self) -> None:
        self.beat("14–15 min. Caso mínimo del capítulo: dos sensores, tres instantes. Pida "
                  "ubicar los puntos antes de mostrarlos y describir el patrón (x₂ = x₁ + 1).")
        head = self.start("Dos sensores", "Tres lecturas: (2, 3), (3, 4), (4, 5)", BLUE)
        ax = Axes(x_range=[-2, 5, 1], y_range=[-2, 6, 1], x_length=4.9, y_length=5.6,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 2,
                                           "include_numbers": True, "font_size": 20})
        ax.move_to([-3.4, -0.62, 0])
        grid = VGroup(*[Line(ax.c2p(x, -2), ax.c2p(x, 6), color=GRID, stroke_width=1)
                        for x in range(-2, 6)],
                      *[Line(ax.c2p(-2, y), ax.c2p(5, y), color=GRID, stroke_width=1)
                        for y in range(-2, 7)])
        names = VGroup(safe_text("sensor 1", size=18, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.35),
                       safe_text("sensor 2", size=18, color=MUTED).rotate(PI / 2)
                       .next_to(ax.y_axis, LEFT, buff=0.35))
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.1, color=BLUE) for p in TOY])
        X = tex(r"X=\begin{pmatrix}2&3\\3&4\\4&5\end{pmatrix}", size=40).move_to([2.2, 1.2, 0])
        self.play(FadeIn(grid), Create(ax), FadeIn(names), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.2), Write(X))

        self.beat("15–16 min. Paso 1: la media. Pregunte las dos medias antes de revelar la cruz "
                  "amarilla (3, 4). PCA estudia variación alrededor del centro.")
        mean = VGroup(Line(UL * 0.14, DR * 0.14), Line(UR * 0.14, DL * 0.14)).set_color(YELLOW)
        mean.set_stroke(width=5).move_to(ax.c2p(*TOY_MEAN))
        mean_txt = tex(r"\bar x=(3,\;4)", size=36, color=YELLOW).next_to(X, DOWN, buff=0.45)
        self.play(GrowFromCenter(mean), Write(mean_txt))

        self.beat("16–17 min. Paso 2: centrar = trasladar la nube para que la media quede en el "
                  "origen. Nada cambia de forma; sólo cambia el punto de referencia.")
        shift = ax.c2p(0, 0) - ax.c2p(*TOY_MEAN)
        arrow = Arrow(ax.c2p(*TOY_MEAN), ax.c2p(0, 0), buff=0.1, color=YELLOW, stroke_width=4)
        self.play(GrowArrow(arrow))
        Xc = tex(r"\tilde X=\begin{pmatrix}-1&-1\\0&0\\1&1\end{pmatrix}", size=40)
        Xc.move_to(X).align_to(X, LEFT)
        self.play(dots.animate.shift(shift), mean.animate.shift(shift),
                  TransformMatchingShapes(X, Xc), FadeOut(arrow), run_time=1.3)

        self.beat("17–18 min. Predicción: ¿qué recta a través del origen describe los tres "
                  "puntos? Luego muestre PC1 (naranja) y su perpendicular PC2 (verde).")
        o = ax.c2p(0, 0)
        d1 = normalize(ax.c2p(1, 1) - o)
        d2 = normalize(ax.c2p(-1, 1) - o)
        pc1 = Line(o - d1 * 2.4, o + d1 * 2.4, color=ORANGE, stroke_width=5)
        pc2 = Line(o - d2 * 1.4, o + d2 * 1.4, color=GREEN, stroke_width=4)
        l1 = safe_text("PC1", size=22, color=ORANGE, weight=BOLD).next_to(pc1.get_end(), UR, buff=0.05)
        l2 = safe_text("PC2", size=22, color=GREEN, weight=BOLD).next_to(pc2.get_end(), UL, buff=0.05)
        self.play(Create(pc1), Create(pc2), FadeIn(l1), FadeIn(l2))
        v1 = tex(r"\mathbf v_1=\tfrac{1}{\sqrt2}(1,\,1)\approx(0.707,\;0.707)", size=34,
                 color=ORANGE).next_to(mean_txt, DOWN, buff=0.5).align_to(Xc, LEFT)
        self.play(Write(v1))

        self.beat("18–19 min. Giramos los ejes (no los datos respecto a ellos): PC1 pasa a ser "
                  "horizontal. Pida leer las nuevas coordenadas; en PC2 todos valen 0. Una sola "
                  "columna reconstruye la tabla completa.")
        self.play(FadeOut(grid), FadeOut(ax), FadeOut(names), FadeOut(mean), run_time=0.5)
        world = VGroup(dots, pc1, pc2)
        self.play(Rotate(world, angle=-PI / 4, about_point=o), FadeOut(l1), FadeOut(l2),
                  run_time=1.6)
        scores = VGroup(*[safe_text(f"{z:.2f}".replace("-0.00", "0.00"), size=22, color=ORANGE, weight=BOLD)
                          .next_to(d, UP, buff=0.22) for d, z in zip(dots, TOY_Z[:, 0])])
        z_eq = tex(r"z=\tilde X\mathbf v_1=(-1.41,\;0,\;1.41)", size=34, color=ORANGE)
        z_eq.next_to(v1, DOWN, buff=0.45).align_to(v1, LEFT)
        self.play(FadeIn(scores), Write(z_eq))
        tag = pill("2 columnas → 1 puntuación, sin perder nada", GREEN, size=20)
        tag.move_to([-3.4, -2.75, 0])
        tag[0].set_fill("#0A1A2E", opacity=1)
        self.play(FadeIn(tag, shift=UP * 0.1))

        self.beat("19–20 min. Pregunta de transición: con datos reales los puntos no caen sobre "
                  "una recta. ¿Qué criterio usaríamos para escoger el eje?")
        q = pill("Con ruido, ¿qué recta escogemos?", YELLOW, size=22).move_to([3.2, -2.65, 0])
        self.play(FadeIn(q, shift=UP * 0.1))
