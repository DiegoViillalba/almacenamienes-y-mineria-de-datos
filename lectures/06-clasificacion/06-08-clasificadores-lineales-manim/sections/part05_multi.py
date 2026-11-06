"""Acto 5 · Muchas variables: siempre un hiperplano (39–48 min).

Dieciocho puntos (dos cruzados). Las curvas de nivel de p son rectas paralelas
porque σ(wᵀx + b) = p ⇔ wᵀx + b = logit(p). Los puntos se proyectan sobre la
dirección de w y la nube se vuelve la logística de una variable. En tres
variables la frontera es un plano dentro de un cubo que gira. Con tres clases
(softmax) cada par de clases se separa por un hiperplano y las regiones son
convexas.
"""
from __future__ import annotations

from math import log

import numpy as np
from manim import *

from sections.data import (CUBE_B, CUBE_PTS, CUBE_W, LOG2_B, LOG2_LOSS, LOG2_W1, LOG2_W2, NOISY,
                           P_LEVELS, SOFT_RAYS, SOFT_W, THREE, TRIPLE, X_BOX, Y_BOX, logit,
                           sigma)
from sections.plane import Plane, View3D, box_edges, clip_poly, legend, plane_cube_polygon
from sections.theme import (BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, YELLOW,
                            card, marker, safe_text, tex)

SOFT_COL = {0: BLUE, 1: ORANGE, 2: PINK}


def soft_marker(c: int, r: float = 0.12) -> VMobject:
    if c < 2:
        return marker(-1 if c == 0 else 1, r=r)
    t = Triangle().scale_to_fit_height(2.1 * r).set_stroke(PINK, 3).set_fill(PINK, 0.9)
    return t


class MultiSection:
    def multi(self) -> None:
        b, w1, w2 = LOG2_B, LOG2_W1, LOG2_W2
        self.beat("39–41 min. Dos variables, dieciocho puntos; dos de ellos están cruzados, así "
                  "que ya no hay separación perfecta y la logística tiene un óptimo finito. Las "
                  "franjas son niveles de probabilidad 0.1, 0.3, 0.5, 0.7, 0.9. Pregunte: ¿por "
                  "qué todas son RECTAS y además PARALELAS?")
        head = self.start("Logística con dos variables", "Los niveles de p son rectas paralelas",
                          PURPLE)
        plane = Plane(X_BOX, Y_BOX, height=4.7, center=(-3.25, -0.55, 0))
        dots = plane.dots(NOISY)
        leg = legend().move_to([1.4, 2.25, 0], aligned_edge=LEFT)
        box = [(X_BOX[0], Y_BOX[0]), (X_BOX[1], Y_BOX[0]), (X_BOX[1], Y_BOX[1]),
               (X_BOX[0], Y_BOX[1])]
        cuts = [-np.inf] + [logit(p) for p in P_LEVELS] + [np.inf]
        stripes = VGroup()
        for lo, hi in zip(cuts, cuts[1:]):
            poly = box
            if np.isfinite(lo):
                poly = clip_poly(poly, b - lo, w1, w2)
            if np.isfinite(hi):
                poly = clip_poly(poly, hi - b, -w1, -w2)
            zm = (0 if not np.isfinite(lo) else lo) * 0.5 + (0 if not np.isfinite(hi) else hi) * 0.5
            if not np.isfinite(lo):
                zm = hi - 1.0
            if not np.isfinite(hi):
                zm = lo + 1.0
            pm = float(sigma(zm))
            col = interpolate_color(ManimColor(BLUE), ManimColor(ORANGE), pm)
            if len(poly) >= 3:
                stripes.add(Polygon(*[plane.p(*q) for q in poly], stroke_width=0, fill_color=col,
                                    fill_opacity=0.22))
        levels = VGroup()
        for p in P_LEVELS:
            col = YELLOW if p == 0.5 else GREEN
            ln = plane.boundary(b - logit(p), w1, w2, color=col, width=5 if p == 0.5 else 2.5,
                                dashed=p != 0.5)
            end = ln.get_end() if ln.get_end()[1] > ln.get_start()[1] else ln.get_start()
            lab = safe_text(f"{p:g}", size=16, color=col, weight=BOLD).next_to(end, UP, buff=0.05)
            levels.add(VGroup(ln, lab))
        self.play(FadeIn(plane), FadeIn(leg))
        self.play(FadeIn(stripes), LaggedStart(*[GrowFromCenter(d) for d in dots],
                                               lag_ratio=0.03))
        self.play(LaggedStart(*[Create(l[0]) for l in levels], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(l[1]) for l in levels], lag_ratio=0.15))
        self.bring_to_front(dots)
        rings = VGroup(*[Circle(radius=0.23, color=RED, stroke_width=4).move_to(d)
                         for d in dots[-2:]])
        self.play(Create(rings))

        self.beat("41–42 min. La respuesta en una línea: fijar p es fijar z = logit(p). Todas "
                  "las curvas comparten el MISMO w y sólo cambian la constante: rectas paralelas, "
                  "perpendiculares a w. La de p = 0.5 es z = 0. La probabilidad sólo depende de "
                  "qué tan lejos estás de la frontera.")
        eqs = VGroup(
            tex(r"\sigma(\mathbf{w}^\top\mathbf{x} + b) = p", size=36, color=GREEN),
            tex(r"\iff \mathbf{w}^\top\mathbf{x} + b = \log\frac{p}{1-p}", size=36, color=PURPLE),
            safe_text("mismo w, distinta constante → paralelas", size=22, color=YELLOW,
                      weight=BOLD),
            tex(rf"\hat b = {b:.2f},\ \ \hat{{\mathbf{{w}}}} = ({w1:.2f},\ {w2:.2f})", size=28,
                color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.7, 0.6, 0])
        self.play(LaggedStart(*[Write(e) for e in eqs], lag_ratio=0.35), run_time=2.6)
        self.multi_scene = (plane, dots, stripes, levels, rings, eqs, head)
        self.project_to_w()
        self.cube()
        self.softmax()

    # ------------------------------------------------------------------
    def project_to_w(self) -> None:
        plane, dots, stripes, levels, rings, eqs, head = self.multi_scene
        b, w1, w2 = LOG2_B, LOG2_W1, LOG2_W2
        self.beat("42–44 min. Truco geométrico: proyectemos cada punto sobre la dirección de w "
                  "(la flecha morada). Cada punto se desliza por su recta de nivel hasta el eje. "
                  "Después, el eje se lleva a la derecha y se le pone la probabilidad encima: "
                  "aparece la MISMA sigmoide de una variable. Mensaje: en d variables, la "
                  "logística mira un solo número, z = wᵀx + b.")
        head = self.swap_heading(head, "Logística con dos variables",
                                 "Mirada desde w: una logística de una variable", PURPLE)
        self.play(FadeOut(eqs), FadeOut(rings), FadeOut(levels[0]), FadeOut(levels[1]),
                  FadeOut(levels[3]), FadeOut(levels[4]))
        wn = np.array([w1, w2]) / np.hypot(w1, w2)
        anchor = np.array(plane.foot(b, w1, w2, 5.0, 4.0))
        a0, a1 = anchor - 4.6 * wn, anchor + 4.4 * wn
        # recorte simple al rectángulo
        axis = Line(plane.p(*a0), plane.p(*a1), color=PURPLE, stroke_width=3)
        arrow = plane.w_arrow(b, w1, w2, length=1.2, anchor=tuple(anchor))
        self.play(Create(axis), GrowArrow(arrow))
        moves = []
        for d in dots:
            x = np.array(d.xy)
            t = float((x - anchor) @ wn)
            moves.append(d.animate.move_to(plane.p(*(anchor + t * wn))).scale(0.8))
        self.play(*moves, run_time=1.8)

        zax = Axes(x_range=[-7, 7, 1], y_range=[-0.15, 1.15, 0.5], x_length=5.8, y_length=3.4,
                   tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                            "include_ticks": False}).move_to([3.7, -0.4, 0])
        zl = VGroup(tex(r"z = \mathbf{w}^\top\mathbf{x} + b", size=26, color=PURPLE).next_to(
            zax, DOWN, buff=0.15),
            safe_text("p", size=20, color=MUTED).next_to(zax.c2p(-7, 1), LEFT, buff=0.1),
)
        guides = VGroup(*[DashedLine(zax.c2p(-7, v), zax.c2p(7, v), color=GRID, stroke_width=1.5,
                                     dash_length=0.08) for v in (0, 1)])
        self.play(Create(zax), FadeIn(zl), FadeIn(guides))
        copies = VGroup()
        targets = []
        for (x, y, c), d in zip(NOISY, dots):
            z = b + w1 * x + w2 * y
            cp = d.copy()
            copies.add(cp)
            targets.append(cp.animate.move_to(zax.c2p(z, 1 if c > 0 else 0)))
        self.add(copies)
        self.play(*targets, run_time=2.0)
        sg = zax.plot(lambda z: float(sigma(z)), x_range=[-7, 7], color=GREEN, stroke_width=5)
        zero = DashedLine(zax.c2p(0, -0.15), zax.c2p(0, 1.15), color=YELLOW, stroke_width=3,
                          dash_length=0.08)
        self.play(Create(sg), Create(zero))
        msg = safe_text("2 variables → 1 número z → σ(z)", size=24, color=YELLOW,
                        weight=BOLD).next_to(zax, UP, buff=0.25)
        self.play(FadeIn(msg))

    # ------------------------------------------------------------------
    def cube(self) -> None:
        self.beat("44–46 min. Tres variables: la frontera z = 0 es un PLANO dentro del cubo. El "
                  "cubo gira para que vean que es plano de verdad, no una superficie curva. "
                  "Pregunte: con 50 variables, ¿qué forma tiene la frontera de la logística? "
                  "(Un hiperplano de 49 dimensiones, que no podemos dibujar pero sí calcular.)")
        head = self.start("Tres variables y más", "La frontera es un plano; en general, un "
                          "hiperplano", PURPLE)
        az = ValueTracker(-0.75)
        view = View3D(origin=(-5.3, -2.55, 0), scale=0.8, az=-0.75, el=0.42)

        def scene3d():
            a = az.get_value()
            g = VGroup()
            g.add(box_edges(view, az=a))
            poly = plane_cube_polygon(CUBE_W, CUBE_B)
            g.add(Polygon(*[view(*p, az=a) for p in poly], stroke_color=YELLOW, stroke_width=3,
                          fill_color=YELLOW, fill_opacity=0.22))
            pts = sorted(CUBE_PTS, key=lambda q: view.depth(q[0], q[1], q[2], az=a))
            for x, y, z, c in pts:
                g.add(marker(c, r=0.085).move_to(view(x, y, z, az=a)))
            for lab, p in [("x₁", (4.6, 0, 0)), ("x₂", (0, 4.7, 0)), ("x₃", (0, 0, 4.6))]:
                g.add(safe_text(lab, size=18, color=MUTED).move_to(view(*p, az=a)))
            return g

        sc = always_redraw(scene3d)
        self.play(FadeIn(sc))
        self.play(az.animate.set_value(0.35), run_time=3.5, rate_func=smooth)
        self.play(az.animate.set_value(-0.35), run_time=2.0)
        sc.clear_updaters()
        rows = VGroup(
            tex(r"p = \sigma(w_1x_1 + w_2x_2 + w_3x_3 + b)", size=32, color=GREEN),
            tex(r"\text{frontera: } \mathbf{w}^\top\mathbf{x} + b = 0", size=32, color=YELLOW),
            safe_text("2 variables → recta", size=22, color=MUTED),
            safe_text("3 variables → plano", size=22, color=MUTED),
            safe_text("d variables → hiperplano (d − 1)", size=22, color=YELLOW, weight=BOLD),
            safe_text("Sin importar d, la cuenta es la misma:", size=22, color=MUTED),
            safe_text("un producto punto y una sigmoide.", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.5, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2), run_time=2.2)

    # ------------------------------------------------------------------
    def softmax(self) -> None:
        self.beat("46–48 min. ¿Y con más de dos clases? La logística multinomial (softmax) da un "
                  "puntaje lineal z_k a cada clase y elige el mayor. La frontera entre dos "
                  "clases es donde empatan: z_k = z_j, que se reescribe como (w_k − w_j)ᵀx + "
                  "(b_k − b_j) = 0: ¡otro hiperplano! Las regiones son intersecciones de "
                  "semiplanos, por eso son convexas: tres rayos que se cruzan en un punto.")
        head = self.start("Más de dos clases", "Softmax: cada frontera es un hiperplano", PURPLE)
        plane = Plane(X_BOX, Y_BOX, height=4.7, center=(-3.25, -0.55, 0))
        box = [(X_BOX[0], Y_BOX[0]), (X_BOX[1], Y_BOX[0]), (X_BOX[1], Y_BOX[1]),
               (X_BOX[0], Y_BOX[1])]
        W = SOFT_W
        regions = VGroup()
        for k in range(3):
            poly = box
            for j in range(3):
                if j != k:
                    poly = clip_poly(poly, W[0, k] - W[0, j], W[1, k] - W[1, j], W[2, k] - W[2, j])
            regions.add(Polygon(*[plane.p(*q) for q in poly], stroke_width=0,
                                fill_color=SOFT_COL[k], fill_opacity=0.16))
        rays = VGroup()
        for (k, j), d in SOFT_RAYS:
            ends = plane.clip(W[0, k] - W[0, j], W[1, k] - W[1, j], W[2, k] - W[2, j])
            e = max(ends, key=lambda q: (np.array(q) - TRIPLE) @ d)
            rays.add(Line(plane.p(*TRIPLE), plane.p(*e), color=YELLOW, stroke_width=4))
        pts = VGroup(*[soft_marker(c).move_to(plane.p(x, y)) for x, y, c in THREE])
        legd = VGroup(*[VGroup(soft_marker(c, r=0.1), safe_text(f"clase {c + 1}", size=18,
                                                                color=MUTED)).arrange(RIGHT,
                                                                                      buff=0.12)
                        for c in range(3)]).arrange(RIGHT, buff=0.35)
        legd.next_to(plane.axes, UP, buff=0.12).align_to(plane.axes, RIGHT)
        self.play(FadeIn(plane), FadeIn(legd), LaggedStart(*[GrowFromCenter(p) for p in pts],
                                                          lag_ratio=0.05))
        self.play(FadeIn(regions), LaggedStart(*[Create(r) for r in rays], lag_ratio=0.3))
        self.bring_to_front(pts)
        eqs = VGroup(
            tex(r"z_k = \mathbf{w}_k^\top\mathbf{x} + b_k", size=34, color=PURPLE),
            tex(r"p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}", size=34, color=GREEN),
            tex(r"\text{frontera entre } k \text{ y } j:\ z_k = z_j", size=30, color=MUTED),
            tex(r"(\mathbf{w}_k - \mathbf{w}_j)^\top\mathbf{x} + (b_k - b_j) = 0", size=30,
                color=YELLOW),
            safe_text("→ hiperplano; regiones convexas", size=22, color=YELLOW, weight=BOLD,
                      max_width=5.0),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26).move_to([3.4, -0.2, 0])
        self.play(LaggedStart(*[Write(e) for e in eqs], lag_ratio=0.3), run_time=3.0)
