"""Acto 3 · Fisher: la mejor sombra (19–25 min, opcional).

Seis puntos del libro. Una recta de proyección gira; cada punto deja su
sombra. A la derecha, las sombras como recta numérica y el cociente J(θ).
La diferencia de medias da J = 4.22; Fisher, J = 5.64. Después la fórmula
cerrada w ∝ S_W⁻¹(m_A − m_B) y la frontera perpendicular.
"""
from __future__ import annotations

from math import cos, radians, sin

import numpy as np
from manim import *

from sections.data import (ANGLES, FISHER_A, FISHER_ANGLE, FISHER_B, J_CURVE, J_FISHER, J_MEANS,
                           M_A, M_B, MEANS_ANGLE, S_W, W_FISHER, W_FISHER_U, fisher_J)
from sections.plane import Plane, legend
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL, YELLOW,
                            marker, safe_text, tex)

CENTER = (M_A + M_B) / 2
THETA_MEANS = MEANS_ANGLE + 180.0      # dirección de m_B − m_A: 41.6°
THETA_FISHER = FISHER_ANGLE + 180.0    # 80.1°


class FisherSection:
    def fisher(self) -> None:
        self.beat("19–20 min (opcional). Fisher cambia la pregunta: en vez de una recta que "
                  "separe, busca una DIRECCIÓN donde, al proyectar, las sombras de las dos "
                  "clases queden lejos entre sí y apretadas dentro de cada clase. Pregunte: "
                  "¿desde qué dirección proyectarían? La intuición dice: la que une las medias.")
        head = self.start("Discriminante de Fisher", "Buscar la mejor sombra", TEAL)
        plane = Plane((0, 7), (0, 7), height=4.6, center=(-3.6, -0.5, 0), tick_step=1)
        pts = [(x, y, -1) for x, y in FISHER_A] + [(x, y, 1) for x, y in FISHER_B]
        dots = plane.dots(pts, r=0.13)
        leg = legend("clase B", "clase A").next_to(plane.axes, UP, buff=0.1).align_to(plane.axes,
                                                                                     RIGHT)
        self.play(FadeIn(plane), FadeIn(leg), LaggedStart(*[GrowFromCenter(d) for d in dots],
                                                         lag_ratio=0.1))
        th = ValueTracker(THETA_MEANS)

        def u():
            a = radians(th.get_value())
            return np.array([cos(a), sin(a)])

        def proj_line():
            d = u()
            a = CENTER - 3.0 * d
            b = CENTER + 3.0 * d
            return Line(plane.p(*a), plane.p(*b), color=TEAL, stroke_width=4)

        def shadows():
            d = u()
            g = VGroup()
            for (x, y, c) in pts:
                t = float((np.array([x, y]) - CENTER) @ d)
                f = CENTER + t * d
                g.add(DashedLine(plane.p(x, y), plane.p(*f), color=GRID, stroke_width=2,
                                 dash_length=0.06))
                g.add(marker(c, r=0.08).move_to(plane.p(*f)))
            return g

        # recta numérica de sombras
        nl = NumberLine(x_range=[-4, 4, 1], length=5.6, color=GRID, include_ticks=False,
                        stroke_width=2).move_to([3.6, 1.35, 0])
        nl_t = safe_text("sombras  z = uᵀx", size=20, color=TEAL).next_to(nl, UP, buff=0.2)

        def shadow_line():
            d = u()
            g = VGroup()
            for (x, y, c) in pts:
                t = float((np.array([x, y]) - CENTER) @ d)
                g.add(marker(c, r=0.11).move_to(nl.n2p(t)))
            za = (FISHER_A - CENTER) @ d
            zb = (FISHER_B - CENTER) @ d
            for z, col in [(za, BLUE), (zb, ORANGE)]:
                g.add(Line(nl.n2p(z.min()), nl.n2p(z.max()), color=col, stroke_width=8,
                           stroke_opacity=0.45).shift(DOWN * 0.3))
            return g

        # curva J(θ)
        jax = Axes(x_range=[0, 180, 30], y_range=[0, 6, 2], x_length=5.6, y_length=2.3, tips=False,
                   axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False}
                   ).move_to([3.6, -1.75, 0])
        jcurve = VMobject(color=PURPLE, stroke_width=3).set_points_smoothly(
            [jax.c2p(a, j) for a, j in zip(ANGLES[::4], J_CURVE[::4])])
        jlabs = VGroup(safe_text("ángulo θ de la dirección", size=16, color=MUTED).next_to(
            jax, DOWN, buff=0.32),
            tex("J(\\theta)", size=26, color=PURPLE).next_to(jax.y_axis, UP, buff=0.05))
        for a in (0, 90, 180):
            jlabs.add(safe_text(f"{a}°", size=13, color=MUTED).next_to(jax.c2p(a, 0), DOWN,
                                                                      buff=0.05))
        jdot = always_redraw(lambda: Dot(jax.c2p(th.get_value() % 180, fisher_J(u())),
                                         radius=0.08, color=YELLOW))
        jval = always_redraw(lambda: safe_text(f"J = {fisher_J(u()):.2f}", size=26, color=YELLOW,
                                               weight=BOLD).move_to([5.6, 0.1, 0]))
        pl = always_redraw(proj_line)
        sh = always_redraw(shadows)
        sl = always_redraw(shadow_line)
        self.play(Create(pl), FadeIn(nl), FadeIn(nl_t))
        self.add(sh)
        self.bring_to_front(dots)
        self.play(FadeIn(sl))
        means = safe_text("dirección de las medias", size=20, color=MUTED).move_to([2.6, 0.1, 0])
        self.play(Create(jax), FadeIn(jlabs), Create(jcurve), FadeIn(jdot), FadeIn(jval),
                  FadeIn(means))

        self.beat("20–22 min. Las sombras azules se dispersan: la clase A está estirada en "
                  "diagonal. J = separación² / dispersión. Ahora giramos la dirección y miramos "
                  "J. Pregunte cuándo creen que J será máximo. Se llega a 80°: las sombras de A "
                  "se apretan y J sube a 5.64 (34 % más).")
        self.play(th.animate.set_value(10), run_time=1.6)
        self.play(th.animate.set_value(150), run_time=2.4)
        fis = safe_text("dirección de Fisher", size=20, color=TEAL, weight=BOLD).move_to(means)
        self.play(th.animate.set_value(THETA_FISHER), Transform(means, fis), run_time=1.8)
        for m in (pl, sh, sl, jdot, jval):
            m.clear_updaters()
        self.play(Indicate(jval, color=YELLOW))

        self.beat("22–24 min. Formalización. J es un cociente de dos formas cuadráticas "
                  "(Rayleigh). Derivar e igualar a cero da otra vez un sistema LINEAL: "
                  "w ∝ S_W⁻¹(m_A − m_B). Fórmula cerrada, como en MCO. La frontera es "
                  "perpendicular a w, en el punto medio de las medias proyectadas. Comente: si "
                  "las clases son normales con igual covarianza, esto es LDA.")
        self.play(FadeOut(VGroup(nl, nl_t, sl, jax, jlabs, jcurve, jdot, jval, means)))
        form = VGroup(
            tex(r"J(\mathbf{w}) = \frac{\mathbf{w}^\top S_B\,\mathbf{w}}{\mathbf{w}^\top S_W\,"
                r"\mathbf{w}}", size=36),
            tex(r"S_W = \begin{pmatrix} 4 & 1 \\ 1 & 1.33\end{pmatrix},\ "
                r"\mathbf{m}_A - \mathbf{m}_B = \begin{pmatrix} -3 \\ -2.67\end{pmatrix}", size=28,
                max_width=6.6),
            tex(r"\mathbf{w}^{*} \propto S_W^{-1}(\mathbf{m}_A - \mathbf{m}_B)", size=36,
                color=TEAL),
            tex(rf"\mathbf{{w}}^{{*}} = ({W_FISHER[0]:.2f},\ {W_FISHER[1]:.2f})", size=30,
                color=TEAL),
            safe_text("fórmula cerrada: un sistema lineal", size=22, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, 0.0, 0])
        self.play(LaggedStart(*[Write(f) for f in form], lag_ratio=0.35), run_time=3.0)
        # frontera perpendicular a w en el punto medio de las medias proyectadas
        wv = -W_FISHER_U
        thr = 0.5 * (float(M_A @ wv) + float(M_B @ wv))
        b = -thr
        border = plane.boundary(b, wv[0], wv[1])
        reg = plane.half_planes(b, wv[0], wv[1])
        self.add(reg)
        self.bring_to_front(dots)
        self.play(FadeIn(reg), Create(border))
        lab = safe_text("frontera ⟂ w, en el punto medio", size=22, color=YELLOW,
                        weight=BOLD).next_to(form, DOWN, buff=0.3).align_to(form, LEFT)
        self.play(FadeIn(lab))
        assert abs(fisher_J(W_FISHER_U) - J_FISHER) < 1e-9 and J_MEANS < J_FISHER
