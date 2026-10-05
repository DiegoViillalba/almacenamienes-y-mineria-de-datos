"""Acto 4 · Barrer direcciones: varianza proyectada contra error perpendicular."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import CLOUD, CLOUD_ANGLE, CLOUD_VALS, cloud_split
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, pill, safe_text, tex)

UNIT = 0.95


class DirectionSection:
    def best_direction(self) -> None:
        self.beat("20–21 min. Nube de 26 lecturas ya centradas. Una recta candidata por el origen; "
                  "cada punto se proyecta sobre ella (pie naranja) y deja un residuo rojo.")
        self.start("La mejor dirección", "¿Qué eje conserva más variación?", ORANGE)
        origin = np.array([-3.1, -0.55, 0])
        grid = VGroup(*[Line(origin + UNIT * np.array([x, -2.9, 0]), origin + UNIT * np.array([x, 2.9, 0]),
                             color=GRID, stroke_width=1) for x in range(-3, 4)],
                      *[Line(origin + UNIT * np.array([-3.4, y, 0]), origin + UNIT * np.array([3.4, y, 0]),
                             color=GRID, stroke_width=1) for y in range(-2, 3)])
        pts = [origin + UNIT * np.array([x, y, 0]) for x, y in CLOUD]
        dots = VGroup(*[Dot(p, radius=0.07, color=BLUE) for p in pts])
        theta = ValueTracker(100 * DEGREES)

        def u() -> np.ndarray:
            a = theta.get_value()
            return np.array([np.cos(a), np.sin(a), 0])

        line = always_redraw(lambda: Line(origin - 3.4 * UNIT * u(), origin + 3.4 * UNIT * u(),
                                          color=ORANGE, stroke_width=4))

        def residuals() -> VGroup:
            g = VGroup()
            for p in pts:
                foot = origin + np.dot(p - origin, u()) * u()
                g.add(Line(p, foot, color=RED, stroke_width=2, stroke_opacity=0.8),
                      Dot(foot, radius=0.045, color=ORANGE))
            return g

        res = always_redraw(residuals)
        self.play(FadeIn(grid), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.03), run_time=1)
        self.play(Create(line), FadeIn(res))

        # Barra apilada: varianza proyectada + residual = total (constante).
        W = 5.0
        bar_left = np.array([1.35, 0.9, 0])

        def split() -> tuple[float, float, float]:
            return cloud_split(theta.get_value())

        def bars() -> VGroup:
            p, r, t = split()
            wp = W * p / t
            b1 = Rectangle(width=max(wp, 0.01), height=0.6, stroke_width=0, fill_color=ORANGE,
                           fill_opacity=0.95).move_to(bar_left + RIGHT * wp / 2)
            b2 = Rectangle(width=max(W - wp, 0.01), height=0.6, stroke_width=0, fill_color=RED,
                           fill_opacity=0.85).move_to(bar_left + RIGHT * (wp + (W - wp) / 2))
            return VGroup(b1, b2)

        outline = Rectangle(width=W, height=0.6, stroke_color=INK, stroke_width=2)
        outline.move_to(bar_left + RIGHT * W / 2)
        bar = always_redraw(bars)
        total = cloud_split(0)[2]
        cap = safe_text(f"varianza total = {total:.2f}  (no depende del eje)", size=22, color=MUTED)
        cap.next_to(outline, UP, buff=0.2).align_to(outline, LEFT)
        lab_p = safe_text("proyectada", size=22, color=ORANGE)
        lab_r = safe_text("residual", size=22, color=RED)
        num_p = DecimalNumber(0, num_decimal_places=2, font_size=34, color=ORANGE)
        num_r = DecimalNumber(0, num_decimal_places=2, font_size=34, color=RED)
        num_a = DecimalNumber(0, num_decimal_places=0, font_size=34, color=INK, unit=r"^\circ")
        num_p.set_value(split()[0])
        num_r.set_value(split()[1])
        num_a.set_value(np.rad2deg(theta.get_value()))
        readout = VGroup(
            VGroup(safe_text("ángulo", size=22, color=MUTED), num_a).arrange(RIGHT, buff=0.25),
            VGroup(lab_p, num_p).arrange(RIGHT, buff=0.25),
            VGroup(lab_r, num_r).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(outline, DOWN, buff=0.4).align_to(outline, LEFT)
        self.play(FadeIn(cap), Create(outline), FadeIn(bar), FadeIn(readout))
        num_p.add_updater(lambda m: m.set_value(split()[0]))
        num_r.add_updater(lambda m: m.set_value(split()[1]))
        num_a.add_updater(lambda m: m.set_value(np.rad2deg(theta.get_value())))

        self.beat("21–22 min. Predicción individual (30 s): ¿en qué ángulo es máxima la barra "
                  "naranja? ¿Y dónde es mínima la roja? Registre dos o tres ángulos propuestos.")
        q = pill("Predice: ¿qué ángulo maximiza la barra naranja?", YELLOW, size=20)
        q.move_to([3.7, -2.55, 0])
        self.play(FadeIn(q, shift=UP * 0.1))

        best = np.rad2deg(CLOUD_ANGLE)
        self.beat("22–24 min. Barra: al girar la recta la suma no cambia; lo que gana el naranja lo "
                  f"pierde el rojo. El máximo ocurre en {best:.0f}°, con varianza proyectada "
                  f"{CLOUD_VALS[0]:.2f}. Ese eje es PC1.")
        self.play(FadeOut(q))
        self.play(theta.animate.set_value(-40 * DEGREES), run_time=4.5, rate_func=smooth)
        self.play(theta.animate.set_value(CLOUD_ANGLE), run_time=2.2)
        star = pill(f"PC1: {best:.0f}°  ·  máx. proyectada = {CLOUD_VALS[0]:.2f}", ORANGE, size=22)
        star.move_to([3.85, -2.55, 0])
        self.play(FadeIn(star, shift=UP * 0.1))

        self.beat("24–25 min. ¿Por qué la suma es constante? Tomemos un punto: su distancia al "
                  "origen es la hipotenusa; la proyección y el residuo son los catetos (Pitágoras).")
        line.clear_updaters()
        res.clear_updaters()
        uu = u()
        along = np.abs(CLOUD @ uu[:2])
        across = np.abs(CLOUD @ np.array([-uu[1], uu[0]]))
        k = int(np.argmax(np.minimum(along, 2.2 * across)))
        p = pts[k]
        foot = origin + np.dot(p - origin, uu) * uu
        hyp = Line(origin, p, color=INK, stroke_width=5)
        leg1 = Line(origin, foot, color=ORANGE, stroke_width=7)
        leg2 = Line(foot, p, color=RED, stroke_width=7)
        corner = RightAngle(Line(foot, origin), Line(foot, p), length=0.18, color=INK, quadrant=(1, 1))
        self.play(res.animate.set_opacity(0.15), dots.animate.set_opacity(0.3), run_time=0.5)
        self.play(Create(hyp), Create(leg1), Create(leg2), Create(corner), Indicate(dots[k]))
        pyth = tex(r"\|\tilde{\mathbf x}_i\|^2=(\mathbf u^{T}\tilde{\mathbf x}_i)^2+\|\mathbf e_i\|^2",
                   size=36).move_to([3.85, -2.0, 0])
        self.play(FadeOut(star), Write(pyth))

        self.beat("25–26 min. Sumando sobre todos los puntos y dividiendo entre n−1 se obtiene la "
                  "identidad de la barra. Conclusión: maximizar la varianza proyectada es lo mismo "
                  "que minimizar el error de reconstrucción.")
        p_best, r_best, t_best = cloud_split(CLOUD_ANGLE)
        ident = tex(rf"\underbrace{{{t_best:.2f}}}_{{\text{{total}}}}=\underbrace{{{p_best:.2f}}}_{{\text{{proyectada}}}}"
                    rf"+\underbrace{{{r_best:.2f}}}_{{\text{{error}}}}", size=34).move_to([3.85, -2.35, 0])
        self.play(ReplacementTransform(pyth, ident))

        self.beat("26–29 min. Formalización. Pida a un estudiante dictar la varianza de z = X̃u. "
                  "Explique la restricción uᵀu = 1 (si no, bastaría alargar u). Derive con "
                  "Lagrange y concluya que la varianza del óptimo es el autovalor.")
        self.start("Formalización", "Del barrido a un problema de optimización", PURPLE)
        forms = [
            tex(r"\operatorname{Var}(\tilde X\mathbf u)=\frac{1}{n-1}\mathbf u^{T}\tilde X^{T}\tilde X\,\mathbf u"
                r"=\mathbf u^{T}S\,\mathbf u", size=36),
            tex(r"\max_{\mathbf u}\;\mathbf u^{T}S\,\mathbf u\quad\text{sujeto a}\quad \mathbf u^{T}\mathbf u=1",
                size=36, color=ORANGE),
            tex(r"\mathcal L=\mathbf u^{T}S\mathbf u-\lambda(\mathbf u^{T}\mathbf u-1)\;\Rightarrow\;"
                r"\nabla_{\mathbf u}\mathcal L=2S\mathbf u-2\lambda\mathbf u=\mathbf 0", size=36),
            tex(r"S\mathbf u=\lambda\mathbf u\qquad\text{y}\qquad \mathbf u^{T}S\mathbf u=\lambda", size=44,
                color=YELLOW),
        ]
        tags = [
            safe_text("varianza de las puntuaciones", size=19, color=MUTED),
            safe_text("el mejor eje, sin hacer trampa con la longitud", size=19, color=MUTED),
            safe_text("multiplicador de Lagrange", size=19, color=MUTED),
            safe_text("autovector · la varianza ES el autovalor", size=19, color=YELLOW),
        ]
        blocks = VGroup(*[VGroup(f, t).arrange(DOWN, buff=0.1) for f, t in zip(forms, tags)])
        blocks.arrange(DOWN, buff=0.36).move_to([0, -0.55, 0])
        for b in blocks:
            self.play(Write(b[0]), FadeIn(b[1]), run_time=1.0)
        box = SurroundingRectangle(blocks[-1], color=YELLOW, buff=0.12, corner_radius=0.1)
        self.play(Create(box))
