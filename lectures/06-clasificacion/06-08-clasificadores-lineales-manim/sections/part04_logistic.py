"""Acto 4 · Regresión logística: sin fórmula, hay que iterar (25–39 min).

La recta de MCO con y ∈ {0, 1} se sale de [0, 1]; se dobla en una sigmoide.
La frontera p = ½ es z = 0: sigue siendo lineal. La pérdida −log p se ve como
barras que se encogen mientras β recorre los pasos de Newton. Comparación de
gradientes: en MCO β aparece lineal; en la logística, dentro de σ → no se
puede despejar. Curvas de nivel: elipses contra curvas deformadas, un salto
contra pasos. Separabilidad perfecta y, al final, los puntos lejanos ya no
arrastran la frontera.
"""
from __future__ import annotations

from math import exp, log

import numpy as np
from manim import *

from sections.data import (GRAD0, HOURS, HX, HY01, LOG0, LOG1, LOG_BETA, LOG_CUT, LOG_GD,
                           LOG_LEVELS, LOG_LOSS, LOG_NEWTON, MASK_ALL, MASK_BASE, MASK_BOX,
                           MASK_FAR, MASK_LOG, MASK_LOG_FAR, MASK_OLS_FAR, OLS0, OLS1,
                           OLS_LEVELS, OLS_MIN, SEP_LOSSES, SEP_SLOPES, SEP_Y01, log_loss_h,
                           n_errors, ols_loss)
from sections.part02_ols import contours, param_axes
from sections.plane import Plane, legend
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, YELLOW, card,
                            marker, pill, safe_text, tex)


def sig(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def prob_axes(center=(-3.0, -0.35, 0), x_len=6.0, y_len=3.6, y_range=(-0.3, 1.3)):
    ax = Axes(x_range=[0, 7, 1], y_range=[y_range[0], y_range[1], 0.5], x_length=x_len,
              y_length=y_len, tips=False,
              axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
    ax.move_to(center)
    labs = VGroup(*[safe_text(str(h), size=16, color=MUTED).next_to(ax.c2p(h, y_range[0]), DOWN,
                                                                   buff=0.1) for h in range(1, 7)],
                  safe_text("horas de estudio", size=18, color=MUTED).next_to(
                      ax.c2p(3.5, y_range[0]), DOWN, buff=0.45))
    for v, t in [(0, "0"), (0.5, "0.5"), (1, "1")]:
        labs.add(safe_text(t, size=16, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1))
    guides = VGroup(*[DashedLine(ax.c2p(0, v), ax.c2p(7, v), color=GRID, stroke_width=1.5,
                                 dash_length=0.08) for v in (0, 1)])
    return ax, labs, guides


class LogisticSection:
    def _marks01(self, ax) -> VGroup:
        out = VGroup()
        for h, y in HOURS:
            m = marker(1 if y else -1, r=0.13).move_to(ax.c2p(h, y))
            m.h, m.y = h, y
            out.add(m)
        return out

    def logistic(self) -> None:
        self.beat("25–26 min. Volvemos a los seis estudiantes, ahora con y ∈ {0, 1} para leer "
                  "probabilidades. La recta de MCO da −0.2 y 1.2 en los extremos: no son "
                  "probabilidades. Pregunte: ¿cómo deformarían la recta para que nunca salga "
                  "de [0, 1] y conserve el orden?")
        head = self.start("Regresión logística", "Doblar la recta para leer probabilidades",
                          GREEN)
        ax, labs, guides = prob_axes()
        marks = self._marks01(ax)
        self.play(Create(ax), FadeIn(labs), FadeIn(guides))
        self.play(LaggedStart(*[GrowFromCenter(m) for m in marks], lag_ratio=0.1))
        a0, a1 = (OLS0 + 1) / 2, OLS1 / 2      # MCO sobre y ∈ {0, 1}
        line = ax.plot(lambda x: a0 + a1 * x, x_range=[0, 7], color=PURPLE, stroke_width=5)
        self.play(Create(line))
        outs = VGroup(
            safe_text(f"{a0:.1f}", size=22, color=RED, weight=BOLD).next_to(ax.c2p(0, a0), DR,
                                                                         buff=0.12),
            safe_text(f"{a0 + 7 * a1:.1f}", size=22, color=RED, weight=BOLD).next_to(
                ax.c2p(7, a0 + 7 * a1), LEFT, buff=0.15),
        )
        self.play(FadeIn(outs), Flash(ax.c2p(7, a0 + 7 * a1), color=RED))
        q = VGroup(safe_text("¿«probabilidad» 1.2?", size=28, color=RED, weight=BOLD),
                   safe_text("Necesitamos una curva que", size=24, color=MUTED),
                   safe_text("viva entre 0 y 1.", size=24, color=MUTED)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([3.7, 1.2, 0])
        self.play(FadeIn(q))

        self.beat("26–28 min. La recta se dobla en una sigmoide: p = σ(β₀ + β₁x). Lo que entra "
                  "a σ, z = β₀ + β₁x, ¡es la misma expresión lineal de antes! σ sólo aplasta el "
                  "puntaje. La frontera p = 0.5 ocurre en z = 0, es decir en x = 3.5 h: un "
                  "punto, el hiperplano de dimensión 0. La logística es un clasificador lineal.")
        curve = ax.plot(lambda x: sig(LOG0 + LOG1 * x), x_range=[0, 7], color=GREEN,
                        stroke_width=5)
        self.play(FadeOut(outs), FadeOut(q), Transform(line, curve), run_time=1.8)
        half = DashedLine(ax.c2p(0, 0.5), ax.c2p(7, 0.5), color=YELLOW, stroke_width=3,
                          dash_length=0.1)
        cut = DashedLine(ax.c2p(LOG_CUT, -0.3), ax.c2p(LOG_CUT, 1.3), color=YELLOW,
                         stroke_width=3, dash_length=0.1)
        cut_t = safe_text(f"frontera: {LOG_CUT:.1f} h", size=22, color=YELLOW,
                          weight=BOLD).next_to(ax.c2p(LOG_CUT, 1.3), UP, buff=0.08)
        self.play(Create(half), Create(cut), FadeIn(cut_t))
        form = VGroup(
            tex(r"p = \sigma(z) = \frac{1}{1 + e^{-z}}", size=36, color=GREEN),
            tex(r"z = \beta_0 + \beta_1 x", size=36, color=PURPLE),
            tex(r"p = \tfrac12 \iff z = 0", size=34, color=YELLOW),
            tex(rf"\hat\beta = ({LOG0:.2f},\ {LOG1:.2f})", size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([1.6, 0.6, 0], aligned_edge=LEFT)
        self.play(LaggedStart(*[Write(f) for f in form], lag_ratio=0.35), run_time=2.4)
        lin = safe_text("dentro de σ: lo mismo de siempre", size=22, color=PURPLE,
                        weight=BOLD).next_to(form, DOWN, buff=0.35).align_to(form, LEFT)
        self.play(FadeIn(lin))
        self.log_ax, self.log_marks, self.log_curve = ax, marks, line
        self.log_static = VGroup(labs, guides, half, cut, cut_t)
        self.log_head = head
        self.log_text = VGroup(form, lin)
        self.loss_bars()
        self.no_formula()
        self.landscape()
        self.separable()
        self.mask_logistic()

    # ------------------------------------------------------------------
    def loss_bars(self) -> None:
        ax, marks = self.log_ax, self.log_marks
        self.beat("28–30 min. ¿Cómo se eligen β₀ y β₁? Cada estudiante paga una multa −log p de "
                  "lo que realmente pasó: si aprobó, −log p; si no, −log(1 − p). Con β = 0 todos "
                  "reciben p = 0.5 y pagan 0.693. Ahora β avanza por los pasos de Newton: la "
                  "curva se ajusta y las barras rojas se encogen hasta 0.413 de promedio.")
        head = self.swap_heading(self.log_head, "Máxima verosimilitud",
                                 "Cada punto paga −log de su probabilidad", GREEN)
        self.log_head = head
        self.play(FadeOut(self.log_text), FadeOut(self.log_static[2:]))
        b0 = ValueTracker(0.0)
        b1 = ValueTracker(0.0)
        self.remove(self.log_curve)
        curve = always_redraw(lambda: ax.plot(
            lambda x: sig(b0.get_value() + b1.get_value() * x), x_range=[0, 7], color=GREEN,
            stroke_width=5))
        self.add(curve)
        bx = Axes(x_range=[0, 7, 1], y_range=[0, 3.2, 1], x_length=5.0, y_length=3.0, tips=False,
                  axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False}
                  ).move_to([3.9, -0.6, 0])
        bx_t = safe_text("multa de cada estudiante: −log p(lo observado)", size=18,
                         color=MUTED).next_to(bx, UP, buff=0.15)

        def losses():
            out = []
            for h, y in HOURS:
                p = sig(b0.get_value() + b1.get_value() * h)
                out.append(-log(p if y else 1 - p))
            return out

        def bars():
            g = VGroup()
            for (h, y), L in zip(HOURS, losses()):
                hgt = max(L, 0.01) * (bx.c2p(0, 1)[1] - bx.c2p(0, 0)[1])
                r = Rectangle(width=0.42, height=hgt, stroke_width=0, fill_color=RED,
                              fill_opacity=0.75)
                r.move_to(bx.c2p(h, 0), aligned_edge=DOWN)
                lab = safe_text(f"{L:.2f}", size=14, color=INK).next_to(r, UP, buff=0.05)
                g.add(r, lab)
            return g

        hl = VGroup(*[marker(1 if y else -1, r=0.09).next_to(bx.c2p(h, 0), DOWN, buff=0.1)
                      for h, y in HOURS])
        br = always_redraw(bars)
        tot = always_redraw(lambda: safe_text(
            f"pérdida promedio = {np.mean(losses()):.3f}", size=26, color=RED,
            weight=BOLD).next_to(bx, DOWN, buff=0.45))
        beta_t = always_redraw(lambda: safe_text(
            f"β = ({b0.get_value():.2f}, {b1.get_value():.2f})", size=28, color=PURPLE,
            weight=BOLD).move_to([3.9, 2.0, 0]))
        self.play(Create(bx), FadeIn(bx_t), FadeIn(hl), FadeIn(br), FadeIn(tot), FadeIn(beta_t))
        for beta in LOG_NEWTON[1:5]:
            self.play(b0.animate.set_value(float(beta[0])), b1.animate.set_value(float(beta[1])),
                      run_time=1.3)
            self.wait(0.3)
        for m in (curve, br, tot, beta_t):
            m.clear_updaters()
        self.play(Indicate(tot, color=RED))
        self.log_curve = curve

    # ------------------------------------------------------------------
    def no_formula(self) -> None:
        self.beat("30–32 min. Momento clave. Ponga lado a lado las dos condiciones de óptimo. "
                  "En MCO, β aparece multiplicando: es un sistema lineal y se despeja. En la "
                  "logística, β está DENTRO de σ, sumado en seis exponenciales distintas: no hay "
                  "álgebra que lo despeje. Pregunte: ¿cómo resolverían x + eˣ = 3? (Probando: "
                  "iterar.)")
        self.start("¿Por qué no hay fórmula?", "β queda atrapado dentro de σ", GREEN)
        left = VGroup(
            safe_text("Mínimos cuadrados", size=28, color=BLUE, weight=BOLD),
            tex(r"X^\top\big(X\boldsymbol\beta - \mathbf{y}\big) = \mathbf{0}", size=38),
            tex(r"\Downarrow", size=36, color=MUTED),
            tex(r"\boldsymbol\beta = (X^\top X)^{-1}X^\top\mathbf{y}", size=36, color=BLUE),
            safe_text("lineal en β: se despeja", size=24, color=GREEN, weight=BOLD),
        ).arrange(DOWN, buff=0.32)
        right = VGroup(
            safe_text("Regresión logística", size=28, color=GREEN, weight=BOLD),
            tex(r"X^\top\big(\sigma(X\boldsymbol\beta) - \mathbf{y}\big) = \mathbf{0}", size=38),
            tex(r"\Downarrow", size=36, color=MUTED),
            tex(r"\sum_i \Big(\frac{1}{1+e^{-\boldsymbol\beta^\top\mathbf{x}_i}} - y_i\Big)"
                r"\mathbf{x}_i = \mathbf{0}", size=32),
            safe_text("no lineal en β: no se despeja", size=24, color=RED, weight=BOLD),
        ).arrange(DOWN, buff=0.32)
        cards = VGroup(card(6.2, 4.6, BLUE), card(6.2, 4.6, GREEN)).arrange(RIGHT, buff=0.4)
        cards.move_to([0, -0.55, 0])
        left.move_to(cards[0])
        right.move_to(cards[1])
        self.play(FadeIn(cards[0]), FadeIn(left[0]), Write(left[1]))
        self.play(FadeIn(cards[1]), FadeIn(right[0]), Write(right[1]))
        box = SurroundingRectangle(right[1][0][3:8], color=RED, buff=0.06)
        self.play(Create(box))
        self.play(FadeIn(left[2]), Write(left[3]), FadeIn(left[4]))
        self.play(FadeIn(right[2]), Write(right[3]), FadeIn(right[4]))

    # ------------------------------------------------------------------
    def landscape(self) -> None:
        self.beat("32–34 min. Las dos pérdidas en el plano de parámetros. Izquierda, MCO: "
                  "elipses perfectas, un salto. Derecha, logística: curvas de nivel deformadas, "
                  "pero un solo valle (la pérdida es CONVEXA): hay un único mínimo y cualquier "
                  "método razonable llega. Newton llega en 4–5 pasos; el descenso de gradiente "
                  "con η = 0.5 necesita miles.")
        self.start("Buscar el fondo del valle", "Un salto contra muchos pasos", GREEN)
        axL, decoL = param_axes(center=(-3.4, -0.55, 0), width=5.4, height=3.6)
        axR, decoR = param_axes(center=(3.4, -0.55, 0), width=5.4, height=3.6)
        tL = safe_text("MCO: tazón cuadrático", size=24, color=BLUE, weight=BOLD).next_to(
            axL, UP, buff=0.5)
        tR = safe_text("Logística: convexa, no cuadrática", size=24, color=GREEN,
                       weight=BOLD).next_to(axR, UP, buff=0.5)
        ellL = contours(axL, ols_loss, OLS_MIN, OLS_LEVELS, BLUE)
        ellR = contours(axR, log_loss_h, LOG_BETA, LOG_LEVELS, GREEN)
        self.play(FadeIn(decoL), FadeIn(decoR), Create(axL), Create(axR), FadeIn(tL), FadeIn(tR))
        self.play(LaggedStart(*[Create(e) for e in ellL], lag_ratio=0.1),
                  LaggedStart(*[Create(e) for e in ellR], lag_ratio=0.1), run_time=1.8)
        start = (0.0, 0.0)
        jump = CurvedArrow(axL.c2p(*start), axL.c2p(*OLS_MIN), color=YELLOW, stroke_width=3,
                           angle=0.6, tip_length=0.15)
        self.play(FadeIn(Dot(axL.c2p(*start), radius=0.07, color=INK)), Create(jump))
        sL = safe_text("1 paso (fórmula)", size=22, color=YELLOW, weight=BOLD).next_to(
            axL, DOWN, buff=0.4)
        self.play(FadeIn(sL))

        self.beat("34–35 min. A la derecha, primero el descenso de gradiente: pasos pequeños "
                  "que zigzaguean por el valle angosto (la traza gris resume 3000 pasos). "
                  "Después Newton, que usa también la curvatura: cuatro saltos amarillos y "
                  "llega. Las dos rutas terminan en el mismo punto porque el valle es único.")
        gd = np.array(LOG_GD)
        sub = gd[np.unique(np.r_[np.arange(0, 60), np.geomspace(60, len(gd) - 1, 80).astype(int)])]
        gd_path = VMobject(stroke_color=MUTED, stroke_width=2.5).set_points_as_corners(
            [axR.c2p(*b) for b in sub])
        self.play(Create(gd_path), run_time=2.6)
        sR1 = safe_text("gradiente: ~3000 pasos", size=20, color=MUTED).next_to(
            axR, DOWN, buff=0.4).align_to(axR, LEFT)
        self.play(FadeIn(sR1))
        nw = VGroup()
        for a, b in zip(LOG_NEWTON[:4], LOG_NEWTON[1:5]):
            nw.add(Arrow(axR.c2p(*a), axR.c2p(*b), buff=0.02, color=YELLOW, stroke_width=4,
                         max_tip_length_to_length_ratio=0.15))
        dots = VGroup(*[Dot(axR.c2p(*b), radius=0.06, color=YELLOW) for b in LOG_NEWTON[:5]])
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(d)) for a, d in
                                zip(nw, dots[1:])], lag_ratio=0.6), FadeIn(dots[0]),
                  run_time=2.4)
        sR2 = safe_text("Newton: 4–5 pasos", size=20, color=YELLOW, weight=BOLD).next_to(
            sR1, RIGHT, buff=0.4)
        self.play(FadeIn(sR2))

    # ------------------------------------------------------------------
    def separable(self) -> None:
        self.beat("35–37 min. Una trampa. Si los datos son PERFECTAMENTE separables (aquí el "
                  "de 3 h reprobó y el de 4 h aprobó), la pérdida siempre puede bajar más "
                  "haciendo la sigmoide más empinada: β crece sin fin y el mínimo no existe. "
                  "Pregunte qué harían. La solución práctica: penalizar λ‖w‖² (regularización).")
        self.start("Una trampa: separabilidad perfecta", "La sigmoide se vuelve un escalón", RED)
        ax, labs, guides = prob_axes()
        marks = VGroup(*[marker(1 if y else -1, r=0.13).move_to(ax.c2p(h, y))
                         for h, y in zip(HX, SEP_Y01)])
        self.play(Create(ax), FadeIn(labs), FadeIn(guides), FadeIn(marks))
        s = ValueTracker(SEP_SLOPES[0])
        curve = always_redraw(lambda: ax.plot(
            lambda x: sig(s.get_value() * (x - 3.5)), x_range=[0, 7], color=GREEN,
            stroke_width=5))
        X = np.column_stack([np.ones(6), HX])

        def loss_now():
            v = s.get_value()
            p = 1 / (1 + np.exp(-v * (HX - 3.5)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(SEP_Y01 * np.log(p) + (1 - SEP_Y01) * np.log(1 - p)))

        read = always_redraw(lambda: VGroup(
            safe_text(f"β₁ = {s.get_value():.1f}", size=30, color=PURPLE, weight=BOLD),
            safe_text(f"pérdida = {loss_now():.3f}", size=30, color=RED, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([2.2, 1.2, 0], aligned_edge=LEFT))
        self.play(Create(curve), FadeIn(read))
        self.play(s.animate.set_value(SEP_SLOPES[-1]), run_time=3.0, rate_func=linear)
        self.play(s.animate.set_value(30.0), run_time=1.5)
        for m in (curve, read):
            m.clear_updaters()
        fix = VGroup(
            safe_text("β → ∞,  𝓛 → 0: no hay mínimo", size=24, color=RED, weight=BOLD),
            tex(r"\mathcal{L}(\boldsymbol\beta) + \lambda\lVert\mathbf{w}\rVert^2", size=34,
                color=GREEN),
            safe_text("la penalización detiene el crecimiento", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.6, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(f) for f in fix], lag_ratio=0.3))

    # ------------------------------------------------------------------
    def mask_logistic(self) -> None:
        self.beat("37–39 min. Regresamos al defecto de MCO. Con la logística, los cinco "
                  "lejanos ya tienen p ≈ 1: su multa −log p es casi cero y su gradiente "
                  "(p − y)x también. La frontera casi no se mueve y no comete errores. Compare "
                  "con la recta gris punteada: la de MCO con los lejanos.")
        self.start("La logística no se deja arrastrar", "Lo que está bien lejos ya no pesa",
                   GREEN)
        plane = Plane(*MASK_BOX, height=4.6, center=(-2.6, -0.55, 0))
        base = plane.dots(MASK_BASE)
        far = plane.dots([(x, y, 1) for x, y in MASK_FAR])
        leg = legend().next_to(plane.axes, UP, buff=0.1).align_to(plane.axes, RIGHT)
        b, w1, w2 = MASK_LOG
        line = plane.boundary(b, w1, w2, color=GREEN)
        regions = plane.half_planes(b, w1, w2)
        self.play(FadeIn(plane), FadeIn(leg), FadeIn(regions), FadeIn(base), Create(line))
        self.play(LaggedStart(*[FadeIn(d, shift=LEFT * 0.5) for d in far], lag_ratio=0.12))
        b2, v1, v2 = MASK_LOG_FAR
        self.play(Transform(line, plane.boundary(b2, v1, v2, color=GREEN)),
                  Transform(regions, plane.half_planes(b2, v1, v2)), run_time=1.5)
        self.bring_to_front(base, far)
        ob, o1, o2 = MASK_OLS_FAR
        ols_line = plane.boundary(ob, o1, o2, color=MUTED, width=3, dashed=True)
        self.play(Create(ols_line))
        z = float(b2 + v1 * MASK_FAR[0][0] + v2 * MASK_FAR[0][1])
        info = VGroup(
            safe_text(f"logística: {n_errors(MASK_LOG_FAR, MASK_ALL)} errores", size=26,
                      color=GREEN, weight=BOLD),
            safe_text(f"MCO: {n_errors(MASK_OLS_FAR, MASK_ALL)} errores", size=24, color=MUTED),
            tex(rf"\text{{lejano: }} p = \sigma({z:.0f}) \approx 1", size=30, color=INK),
            tex(r"-\log p \approx 0,\quad (p - y)\,\mathbf{x} \approx \mathbf{0}", size=30,
                color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([1.9, 0.9, 0], aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(i) for i in info], lag_ratio=0.3))
