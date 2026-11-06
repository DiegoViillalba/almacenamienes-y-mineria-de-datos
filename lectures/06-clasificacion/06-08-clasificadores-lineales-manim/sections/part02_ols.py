"""Acto 2 · Mínimos cuadrados: una fórmula cerrada (10–19 min).

Seis estudiantes con y = ±1. Una recta horizontal deja residuos que se dibujan
como cuadrados reales; al girar hacia la solución, los cuadrados se encogen.
Donde la recta cruza 0 está la frontera (3.5 h). La pérdida es un tazón
cuadrático: su gradiente es lineal en β, se iguala a cero y se despeja.
Al final, el defecto: puntos correctos y lejanos arrastran la frontera.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (HOURS, HX, HYPM, LOSS_B0, LOSS_B1, MASK_ALL, MASK_BASE, MASK_BOX,
                           MASK_FAR, MASK_OLS, MASK_OLS_FAR, OLS0, OLS1, OLS_CUT, OLS_LEVELS,
                           OLS_MIN, OLS_SXX, OLS_SXY, level_curve, n_errors, ols_loss, ols_sse)
from sections.plane import Plane, legend
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, YELLOW, card,
                            marker, safe_text, table, tex)


def hours_axes(center=(-2.45, -0.4, 0), unit: float = 0.8, y_range=(-1.8, 1.8)):
    ax = Axes(x_range=[0, 7, 1], y_range=[y_range[0], y_range[1], 1], x_length=7 * unit,
              y_length=(y_range[1] - y_range[0]) * unit, tips=False,
              axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
    ax.move_to(center)
    labs = VGroup(
        *[safe_text(str(h), size=16, color=MUTED).next_to(ax.c2p(h, y_range[0]), DOWN, buff=0.1)
          for h in range(1, 7)],
        safe_text("horas de estudio", size=18, color=MUTED).next_to(ax.c2p(3.5, y_range[0]), DOWN,
                                                                     buff=0.45),
    )
    zero = DashedLine(ax.c2p(0, 0), ax.c2p(7, 0), color=GRID, stroke_width=2, dash_length=0.1)
    return ax, labs, zero


def param_axes(center=(3.3, -0.5, 0), width=5.6, height=4.2):
    ax = Axes(x_range=[LOSS_B0[0], LOSS_B0[1], 1], y_range=[LOSS_B1[0], LOSS_B1[1], 0.5],
              x_length=width, y_length=height, tips=False,
              axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
    ax.move_to(center)
    labs = VGroup(tex(r"\beta_0", size=28, color=MUTED).next_to(ax.x_axis, RIGHT, buff=0.1),
                  tex(r"\beta_1", size=28, color=MUTED).next_to(ax.y_axis, UP, buff=0.1))
    frame = Rectangle(width=width, height=height, stroke_color=GRID, stroke_width=1.5).move_to(ax)
    for v in (-6, -4, -2, 0, 2):
        labs.add(safe_text(f"{v}", size=13, color=MUTED).next_to(ax.c2p(v, LOSS_B1[0]), DOWN,
                                                                 buff=0.06))
    for v in (0, 1, 2):
        labs.add(safe_text(f"{v}", size=13, color=MUTED).next_to(ax.c2p(LOSS_B0[0], v), LEFT,
                                                                 buff=0.06))
    return ax, VGroup(frame, labs)


def contours(ax: Axes, f, center, levels, color) -> VGroup:
    out = VGroup()
    b0lo, b0hi = LOSS_B0
    b1lo, b1hi = LOSS_B1
    for k, lv in enumerate(levels):
        pts = level_curve(f, center, lv)
        inside = [(a, b) for a, b in pts]
        poly = [ax.c2p(np.clip(a, b0lo, b0hi), np.clip(b, b1lo, b1hi)) for a, b in inside]
        c = VMobject(stroke_color=color, stroke_width=2.4,
                     stroke_opacity=0.95 - 0.1 * k).set_points_smoothly(poly + [poly[0]])
        out.add(c)
    return out


class OLSSection:
    def _student_marks(self, ax, y_of=lambda y: y) -> VGroup:
        out = VGroup()
        for (h, y01), y in zip(HOURS, HYPM):
            m = marker(int(y), r=0.13).move_to(ax.c2p(h, y_of(y)))
            m.h, m.y = h, y
            out.add(m)
        return out

    def ols(self) -> None:
        self.beat("10–11 min. Primer criterio: tratar la etiqueta como número (−1 reprobó, +1 "
                  "aprobó) y ajustar una recta por mínimos cuadrados. Los seis estudiantes del "
                  "libro. Arranque con la recta horizontal ŷ = 0: cada residuo se dibuja como "
                  "un CUADRADO real. Pregunte: ¿cómo giraría la recta para encoger el área roja?")
        head = self.start("Mínimos cuadrados con y = ±1", "La recta que minimiza los cuadrados",
                          BLUE)
        ax, labs, zero = hours_axes()
        ylabs = VGroup(safe_text("+1 aprobó", size=17, color=ORANGE).next_to(ax.c2p(0, 1), LEFT,
                                                                             buff=0.1),
                       safe_text("−1 reprobó", size=17, color=BLUE).next_to(ax.c2p(0, -1), LEFT,
                                                                            buff=0.1))
        marks = self._student_marks(ax)
        self.play(Create(ax), FadeIn(labs), FadeIn(zero), FadeIn(ylabs))
        self.play(LaggedStart(*[GrowFromCenter(m) for m in marks], lag_ratio=0.1))
        b0 = ValueTracker(0.0)
        b1 = ValueTracker(0.0)

        def fitted():
            return Line(ax.c2p(0, b0.get_value()), ax.c2p(7, b0.get_value() + 7 * b1.get_value()),
                        color=PURPLE, stroke_width=5)

        def squares():
            g = VGroup()
            unit = ax.c2p(0, 1)[1] - ax.c2p(0, 0)[1]
            for m in marks:
                yh = b0.get_value() + b1.get_value() * m.h
                e = m.y - yh
                if abs(e) < 1e-3:
                    continue
                side = abs(e) * unit
                sq = Square(side, stroke_color=RED, stroke_width=1.5, fill_color=RED,
                            fill_opacity=0.28)
                top = max(m.y, yh)
                sq.move_to(ax.c2p(m.h, top)).shift(DOWN * side / 2 + RIGHT * side / 2)
                g.add(sq)
            return g

        line = always_redraw(fitted)
        sqs = always_redraw(squares)
        sse = always_redraw(lambda: safe_text(
            f"SSE = {ols_sse(b0.get_value(), b1.get_value()):.2f}", size=30, color=RED,
            weight=BOLD).move_to([3.9, 1.5, 0]))
        self.add(sqs, line)
        self.bring_to_front(marks)
        self.play(FadeIn(sse))

        self.beat("11–12 min. La recta gira hacia la de mínimos cuadrados: el área roja total "
                  "baja de 6 a 3.2. La frontera es donde la recta cruza el cero: 3.5 horas. A la "
                  "derecha se predice «aprueba». Pregunte: ¿a qué estudiantes clasifica mal? "
                  "(al de 3 h que aprobó y al de 4 h que reprobó.)")
        self.play(b1.animate.set_value(0.25), run_time=1.0)
        self.play(b0.animate.set_value(OLS0), b1.animate.set_value(OLS1), run_time=1.6)
        for m in (line, sqs, sse):
            m.clear_updaters()
        cut = Dot(ax.c2p(OLS_CUT, 0), radius=0.1, color=YELLOW)
        cut_l = DashedLine(ax.c2p(OLS_CUT, -1.8), ax.c2p(OLS_CUT, 1.8), color=YELLOW,
                           stroke_width=3, dash_length=0.1)
        cut_t = safe_text("frontera: 3.5 h", size=22, color=YELLOW, weight=BOLD).next_to(
            ax.c2p(OLS_CUT, 1.8), UP, buff=0.08)
        self.play(Create(cut_l), GrowFromCenter(cut), FadeIn(cut_t))
        eq = tex(r"\hat{y} = -1.4 + 0.4\,x", size=36, color=PURPLE).next_to(sse, DOWN, buff=0.4)
        rule = tex(r"\hat{y} > 0 \Rightarrow \text{aprueba}", size=30, color=YELLOW).next_to(
            eq, DOWN, buff=0.3)
        self.play(Write(eq), Write(rule))
        self.ols_scene = VGroup(ax, labs, zero, ylabs, marks, line, sqs, cut, cut_l, cut_t)

        self.beat("12–13 min. El cálculo, a mano. Centrar, calcular la covariación Sxy y la "
                  "variación Sxx, dividir. Pida que verifiquen Sxy = 7 con la tabla. Sin "
                  "iteraciones, sin elegir tasas: una división. Esa es la marca de un problema "
                  "con fórmula cerrada.")
        self.play(FadeOut(eq), FadeOut(rule), FadeOut(sse))
        dx = HX - HX.mean()
        rows = [["x", *[f"{h:g}" for h in HX]],
                ["x − x̄", *[f"{v:+.1f}" for v in dx]],
                ["y", *[f"{v:+.0f}" for v in HYPM]],
                ["(x − x̄)·y", *[f"{v:+.1f}" for v in dx * HYPM]]]
        tbl = table(rows, col_widths=[1.5] + [0.68] * 6, size=18, row_h=0.42)
        tbl.move_to([3.6, 1.3, 0])
        calc = VGroup(
            tex(rf"\bar{{x}} = 3.5,\quad \bar{{y}} = 0", size=30),
            tex(rf"S_{{xy}} = {OLS_SXY:g},\quad S_{{xx}} = {OLS_SXX:g}", size=30),
            tex(rf"\hat\beta_1 = \frac{{{OLS_SXY:g}}}{{{OLS_SXX:g}}} = {OLS1:.1f},\quad "
                rf"\hat\beta_0 = \bar{{y}} - \hat\beta_1\bar{{x}} = {OLS0:.1f}", size=30,
                color=PURPLE, max_width=6.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).next_to(tbl, DOWN, buff=0.35)
        calc.align_to(tbl, LEFT)
        self.play(FadeIn(tbl))
        self.play(LaggedStart(*[Write(c) for c in calc], lag_ratio=0.4))
        self.ols_calc = VGroup(tbl, calc)
        self.bowl()
        self.masking()

    # ------------------------------------------------------------------
    def bowl(self) -> None:
        self.beat("13–15 min. ¿Por qué hay fórmula? Cada par (β₀, β₁) es un punto de este plano "
                  "de parámetros; las curvas son niveles de la pérdida. Son ELIPSES: la pérdida "
                  "es un tazón cuadrático. Derivar un cuadrado da algo LINEAL en β; igualar a "
                  "cero da un sistema lineal de 2 × 2 que se resuelve de un golpe, desde "
                  "cualquier punto de partida.")
        head = self.start("¿Por qué hay fórmula?", "La pérdida es un tazón cuadrático", BLUE)
        ax, deco = param_axes()
        ells = contours(ax, ols_loss, OLS_MIN, OLS_LEVELS, BLUE)
        star = Star(n=5, outer_radius=0.14, color=YELLOW, fill_opacity=1).move_to(
            ax.c2p(*OLS_MIN))
        self.play(FadeIn(deco), Create(ax))
        self.play(LaggedStart(*[Create(e) for e in ells], lag_ratio=0.12), run_time=1.6)
        self.play(GrowFromCenter(star))
        steps = VGroup(
            tex(r"\text{SSE}(\boldsymbol\beta) = \lVert \mathbf{y} - X\boldsymbol\beta\rVert^2",
                size=32),
            tex(r"\nabla = 2X^\top(X\boldsymbol\beta - \mathbf{y}) = \mathbf{0}", size=32),
            tex(r"X^\top X\,\boldsymbol\beta = X^\top\mathbf{y}", size=34, color=YELLOW),
            tex(r"\begin{pmatrix} 6 & 21 \\ 21 & 91 \end{pmatrix}\boldsymbol\beta = "
                r"\begin{pmatrix} 0 \\ 7 \end{pmatrix}", size=30),
            tex(r"\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\mathbf{y} = (-1.4,\ 0.4)",
                size=30, color=PURPLE, max_width=6.0),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.4, -0.4, 0])
        lin = safe_text("lineal en β", size=20, color=YELLOW, weight=BOLD).next_to(steps[1], RIGHT,
                                                                                buff=0.2)
        self.play(Write(steps[0]))
        self.play(Write(steps[1]), FadeIn(lin))
        self.play(Write(steps[2]))
        self.play(Write(steps[3]))
        self.play(Write(steps[4]))

        self.beat("15–16 min. Desde cualquier punto de partida, la solución está a UN salto: no "
                  "hay que caminar. Recuerde esta imagen; en la logística las curvas de nivel "
                  "dejarán de ser elipses y tendremos que dar pasos.")
        starts = [(1.0, 1.7), (-6.3, 1.6), (-5.5, -0.4), (1.2, -0.4)]
        jumps = VGroup()
        for s in starts:
            d0 = Dot(ax.c2p(*s), radius=0.07, color=INK)
            arr = CurvedArrow(ax.c2p(*s), ax.c2p(*OLS_MIN), color=YELLOW, stroke_width=3,
                              angle=-0.5, tip_length=0.15)
            jumps.add(VGroup(d0, arr))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(j[0]), Create(j[1])) for j in jumps],
                              lag_ratio=0.25), run_time=2.0)
        one = safe_text("un solo paso: resolver un sistema lineal", size=22, color=YELLOW,
                        weight=BOLD).next_to(steps, DOWN, buff=0.45).align_to(steps, LEFT)
        self.play(FadeIn(one))

    # ------------------------------------------------------------------
    def masking(self) -> None:
        self.beat("16–17 min. Lo mismo en el plano, con dos variables: la recta de MCO (z = 0 "
                  "del plano ajustado) separa los 13 puntos sin errores. Pregunte: si llegan "
                  "cinco cuadrados MÁS, todos lejos y bien clasificados, ¿debería moverse la "
                  "frontera? Pida votación a mano alzada.")
        head = self.start("El defecto de los cuadrados", "Ser «demasiado correcto» también cuesta",
                          RED)
        plane = Plane(*MASK_BOX, height=4.6, center=(-2.6, -0.55, 0))
        base = plane.dots(MASK_BASE)
        far = plane.dots([(x, y, 1) for x, y in MASK_FAR])
        leg = legend().next_to(plane.axes, UP, buff=0.1).align_to(plane.axes, RIGHT)
        b, w1, w2 = MASK_OLS
        line = plane.boundary(b, w1, w2)
        regions = plane.half_planes(b, w1, w2)
        self.play(FadeIn(plane), FadeIn(leg), FadeIn(regions), FadeIn(base))
        self.play(Create(line))
        ok = safe_text(f"MCO: {n_errors(MASK_OLS, MASK_BASE)} errores", size=26, color=GREEN,
                       weight=BOLD).move_to([1.9, 1.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(ok))

        self.beat("17–18 min. Llegan los cinco lejanos. La recta de MCO gira y ahora comete 3 "
                  "errores. ¿Por qué? Con la recta anterior, un punto lejano tenía ŷ ≈ 5.2 "
                  "cuando su etiqueta es +1: residuo (5.2 − 1)² ≈ 17.6. MCO castiga a quien está "
                  "«demasiado bien»: para él, +1 es una meta exacta, no un lado.")
        yh = float(MASK_OLS[0] + MASK_OLS[1] * MASK_FAR[0][0] + MASK_OLS[2] * MASK_FAR[0][1])
        self.play(LaggedStart(*[FadeIn(d, shift=LEFT * 0.5) for d in far], lag_ratio=0.12))
        why = VGroup(tex(rf"\hat{{y}} = {yh:.1f},\ \ y = +1", size=30, color=INK),
                     tex(rf"(\hat{{y}} - y)^2 = {(yh - 1) ** 2:.1f}", size=30, color=RED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ok, DOWN, buff=0.4)
        why.align_to(ok, LEFT)
        arrow = Arrow(why.get_left() + LEFT * 0.1, far[0].get_center() + RIGHT * 0.2 + UP * 0.1,
                      color=RED, stroke_width=3, buff=0.1)
        self.play(FadeIn(why), GrowArrow(arrow))
        b, w1, w2 = MASK_OLS_FAR
        nl = plane.boundary(b, w1, w2)
        nr = plane.half_planes(b, w1, w2)
        self.play(Transform(line, nl), Transform(regions, nr), run_time=1.8)
        self.bring_to_front(base, far)
        allp = base.submobjects + far.submobjects
        rings = VGroup(*[Circle(radius=0.24, color=RED, stroke_width=4).move_to(d) for d in allp
                         if np.sign(b + w1 * d.xy[0] + w2 * d.xy[1]) != d.cls])
        bad = safe_text(f"MCO: {n_errors(MASK_OLS_FAR, MASK_ALL)} errores", size=26, color=RED,
                        weight=BOLD).move_to(ok, aligned_edge=LEFT)
        self.play(Create(rings), Transform(ok, bad))
        moral = VGroup(safe_text("La fórmula es exacta,", size=22, weight=BOLD),
                       safe_text("pero el criterio no es", size=22, weight=BOLD),
                       safe_text("el adecuado para clasificar.", size=22, weight=BOLD),
                       safe_text("→ la logística (acto 4) lo arregla", size=20, color=GREEN)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(why, DOWN, buff=0.45)
        moral.align_to(why, LEFT)
        self.play(FadeIn(moral))
        self.mask_plane = plane
