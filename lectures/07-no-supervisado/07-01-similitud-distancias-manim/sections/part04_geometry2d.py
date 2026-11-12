"""Acto 4 · La forma de una distancia en 2D (21–31 min).

Taxi, pájaro y rey (Manhattan, euclidiana, Chebyshev); la bola unitaria se
deforma con p y, para p < 1, deja de ser convexa (falla la triangular).
Después dos transformaciones del plano: cambiar unidades estira el espacio y
cambia al vecino más cercano; "blanquear" una nube correlacionada convierte
la euclidiana en Mahalanobis.
"""
from __future__ import annotations

from manim import *

from sections.data import (ANA_DANIEL, MAHA_A, MAHA_B, MAHA_CLOUD, MAHA_DA, MAHA_DB,
                           SCALE_D0, SCALE_D1, SCALE_NN0, SCALE_NN1, SCALE_PTS, SIGMA,
                           SIGMA_INV_HALF, STRETCH)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL,
                            YELLOW, card, fmt, safe_text, tex)


def p_ball_point(theta: float, p: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    if p > 30:
        n = max(abs(c), abs(s))
    else:
        n = (abs(c) ** p + abs(s) ** p) ** (1 / p)
    return np.array([c / n, s / n, 0])


def light_plane(x_range, y_range, unit: float) -> NumberPlane:
    return NumberPlane(
        x_range=x_range, y_range=y_range,
        x_length=(x_range[1] - x_range[0]) * unit, y_length=(y_range[1] - y_range[0]) * unit,
        background_line_style={"stroke_color": GRID, "stroke_width": 1.2, "stroke_opacity": 0.7},
        axis_config={"stroke_color": MUTED, "stroke_width": 1.6})


class Geometry2DSection:
    def geometry_2d(self) -> None:
        # ── taxi, pájaro, rey ───────────────────────────────────────────────
        self.beat("21–22 min. Volvamos a la geometría. Para ir de A a B, avanzando 4 cuadras al "
                  "este y 3 al norte: el taxi recorre 7, el pájaro vuela 5 y el rey del ajedrez, "
                  "que mueve en diagonal, necesita 4 movimientos. Las tres son métricas y las tres "
                  "dicen cosas distintas sobre el mismo par.")
        head = self.start("Geometría", "Tres maneras honestas de medir el mismo par", BLUE)
        unit = 0.95
        plane = light_plane([0, 6, 1], [0, 4, 1], unit).move_to([-3.0, -0.7, 0])
        A = plane.c2p(1, 0.5)
        B = plane.c2p(5, 3.5)
        da = Dot(A, radius=0.12, color=YELLOW)
        db = Dot(B, radius=0.12, color=BLUE)
        la = tex("A", size=34, color=YELLOW).next_to(da, DL, buff=0.08)
        lb = tex("B", size=34, color=BLUE).next_to(db, UR, buff=0.08)
        self.play(Create(plane), run_time=0.8)
        self.play(GrowFromCenter(da), GrowFromCenter(db), FadeIn(la), FadeIn(lb))
        taxi = VMobject(color=ORANGE, stroke_width=6).set_points_as_corners(
            [plane.c2p(*xy) for xy in [(1, .5), (2, .5), (2, 1.5), (3, 1.5), (3, 2.5), (4, 2.5),
                                       (4, 3.5), (5, 3.5)]])
        bird = Line(A, B, color=TEAL, stroke_width=6)
        king = VMobject(color=PURPLE, stroke_width=6).set_points_as_corners(
            [plane.c2p(*xy) for xy in [(1, .5), (2, 1.5), (3, 2.5), (4, 3.5), (5, 3.5)]])
        legend = VGroup(
            VGroup(safe_text("taxi", size=26, color=ORANGE, weight=BOLD),
                   tex(r"|4|+|3| = 7", size=34, color=ORANGE)).arrange(RIGHT, buff=0.3),
            VGroup(safe_text("pájaro", size=26, color=TEAL, weight=BOLD),
                   tex(r"\sqrt{4^2+3^2} = 5", size=34, color=TEAL)).arrange(RIGHT, buff=0.3),
            VGroup(safe_text("rey", size=26, color=PURPLE, weight=BOLD),
                   tex(r"\max(4,3) = 4", size=34, color=PURPLE)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.9, -0.3, 0])
        for path, row in zip((taxi, bird, king), legend):
            self.play(Create(path), FadeIn(row, shift=LEFT * 0.2), run_time=1.0)

        # ── la bola unitaria ────────────────────────────────────────────────
        self.beat("22–24 min. Dibujemos todos los puntos a distancia 1 del centro: la 'bola'. Con "
                  "p = 2 es un círculo; con p = 1, un rombo (taxi); con p grande, un cuadrado "
                  "(rey). Pregunte antes de mover p: ¿qué forma tendrá con p = 1?")
        head = self.swap_heading(head, "La bola unitaria", "La forma cambia con p", BLUE)
        self.clear_except(head)
        unit = 1.85
        plane = light_plane([-1.5, 1.5, 0.5], [-1.5, 1.5, 0.5], unit).move_to([-3.0, -0.55, 0])
        o = plane.c2p(0, 0)
        p = ValueTracker(2.0)

        def ball():
            pv = p.get_value()
            return ParametricFunction(lambda t: o + unit * p_ball_point(t, pv), t_range=[0, TAU, TAU / 400],
                                      color=TEAL if pv >= 1 else RED, stroke_width=6,
                                      fill_color=TEAL if pv >= 1 else RED, fill_opacity=0.12)

        shape = always_redraw(ball)
        p_eq = tex("p =", size=44, color=INK).move_to([2.9, 1.5, 0])
        p_num = DecimalNumber(2.0, num_decimal_places=2, font_size=44, color=TEAL)
        p_num.next_to(p_eq, RIGHT, buff=0.18)
        p_inf = tex(r"\infty", size=44, color=TEAL).next_to(p_eq, RIGHT, buff=0.18)
        p_inf.set_opacity(0)

        def upd_num(m):
            pv = p.get_value()
            m.set_value(min(pv, 99.0))
            m.set_color(TEAL if pv >= 1 else RED)
            m.next_to(p_eq, RIGHT, buff=0.18)
            m.set_opacity(0 if pv > 30 else 1)

        p_num.add_updater(upd_num)
        p_inf.add_updater(lambda m: m.set_opacity(1 if p.get_value() > 30 else 0))
        plab = VGroup(p_eq, p_num, p_inf)
        center = Dot(o, radius=0.07, color=YELLOW)
        self.play(Create(plane), FadeIn(center), run_time=0.8)
        self.play(Create(shape), FadeIn(plab))
        self.wait(0.3)
        self.play(p.animate.set_value(1.0), run_time=2.0)
        self.wait(0.4)
        self.play(p.animate.set_value(6.0), run_time=2.0)
        self.play(p.animate.set_value(40.0), run_time=1.0)
        names = VGroup(safe_text("p = 1   Manhattan (rombo)", size=24, color=ORANGE),
                       safe_text("p = 2   euclidiana (círculo)", size=24, color=TEAL),
                       safe_text("p = ∞   Chebyshev (cuadrado)", size=24, color=PURPLE))
        names.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.4, 0.0, 0])
        self.play(FadeIn(names))

        self.beat("24–25 min. ¿Y con p menor que 1? La bola se vuelve una estrella: deja de ser "
                  "convexa. Tome u = (1, 0) y v = (0, 1), ambos en la bola. Su suma mide 4 con "
                  "p = ½, más que 1 + 1. Falla la desigualdad triangular: la convexidad de la bola "
                  "y el axioma 4 son la misma propiedad vista desde dos lados.")
        self.play(FadeOut(names), p.animate.set_value(2.0), run_time=1.0)
        self.play(p.animate.set_value(0.5), run_time=2.0)
        u = Arrow(o, plane.c2p(1, 0), buff=0, color=YELLOW, stroke_width=5)
        v = Arrow(o, plane.c2p(0, 1), buff=0, color=YELLOW, stroke_width=5)
        mid = DashedLine(plane.c2p(1, 0), plane.c2p(0, 1), color=RED, stroke_width=3)
        mdot = Dot(plane.c2p(0.5, 0.5), radius=0.09, color=RED)
        self.play(GrowArrow(u), GrowArrow(v))
        self.play(Create(mid), FadeIn(mdot))
        fail = VGroup(
            tex(r"\lVert u \rVert_{1/2} = \lVert v \rVert_{1/2} = 1", size=34, color=YELLOW),
            tex(r"\lVert u+v \rVert_{1/2} = (1+1)^{2} = 4 \;>\; 1+1", size=34, color=RED),
            safe_text("bola no convexa  ⇔  falla la triangular", size=24, color=RED, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.4, -0.5, 0])
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in fail], lag_ratio=0.35))

        self.beat("25–26 min. Formalización: la familia de Minkowski. Vuelva a Ana y Daniel con "
                  "edad e ingreso normalizados por rango: las diferencias son 0.111 y 0.041. "
                  "Manhattan suma, la euclidiana compone, Chebyshev se queda con la mayor. Al "
                  "crecer p, manda la variable con más diferencia.")
        head = self.swap_heading(head, "Formalización", "La familia de Minkowski", BLUE)
        self.clear_except(head)
        shape.clear_updaters()
        plab.clear_updaters()
        mink = tex(r"d_p(\mathbf{x},\mathbf{y}) = \Big(\sum_{j=1}^{m} |x_j-y_j|^p\Big)^{1/p},"
                   r"\qquad p \ge 1", size=46, color=INK).move_to([0, 1.5, 0])
        dx, dy = ANA_DANIEL["diffs"]
        rows = VGroup(
            tex(rf"d_1 = {dx:.3f} + {dy:.3f} = {ANA_DANIEL['L1']:.3f}", size=38, color=ORANGE),
            tex(rf"d_2 = \sqrt{{{dx:.3f}^2 + {dy:.3f}^2}} = {ANA_DANIEL['L2']:.3f}", size=38,
                color=TEAL),
            tex(rf"d_\infty = \max({dx:.3f},\,{dy:.3f}) = {ANA_DANIEL['Linf']:.3f}", size=38,
                color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, -1.05, 0])
        cap = safe_text("Ana ↔ Daniel, edad e ingreso normalizados por rango", size=22, color=MUTED)
        cap.next_to(rows, UP, buff=0.3)
        self.play(Write(mink))
        self.play(FadeIn(cap), LaggedStart(*[Write(r) for r in rows], lag_ratio=0.3))
        order = tex(r"d_\infty \le d_2 \le d_1", size=38, color=YELLOW).next_to(rows, DOWN, buff=0.4)
        self.play(Write(order))

        # ── cambiar unidades estira el plano ────────────────────────────────
        self.beat("26–27 min. Primera transformación: cambiar UNIDADES. Seis clientes y una "
                  "consulta q. Con estas unidades, el más cercano es el de la derecha (verde). "
                  "Pregunte: si medimos el eje horizontal en una unidad 2.5 veces más chica, ¿sigue "
                  "siendo el más cercano?")
        head = self.swap_heading(head, "Transformar el plano", "Cambiar unidades estira el espacio",
                                 ORANGE)
        self.clear_except(head)
        unit = 1.0
        plane = light_plane([-2, 2, 1], [-2, 2, 1], unit).move_to([-1.3, -0.55, 0])
        q = plane.c2p(0, 0)
        pts = VGroup(*[Dot(plane.c2p(*xy), radius=0.1, color=BLUE) for xy in SCALE_PTS])
        qd = Dot(q, radius=0.12, color=YELLOW)
        ql = tex("q", size=34, color=YELLOW).next_to(qd, DL, buff=0.05)
        circ0 = Circle(radius=SCALE_D0[SCALE_NN0] * unit, color=GREEN, stroke_width=4).move_to(q)
        self.play(Create(plane), FadeIn(pts), FadeIn(qd), FadeIn(ql))
        self.play(Create(circ0), pts[SCALE_NN0].animate.set_color(GREEN).scale(1.4))
        info = VGroup(
            safe_text("antes", size=24, color=GREEN, weight=BOLD),
            tex(rf"d = {SCALE_D0[SCALE_NN0]:.2f}", size=32, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([5.0, 1.2, 0])
        self.play(FadeIn(info))

        self.beat("27–28 min. Estiramos el eje x por 2.5: es una transformación lineal del plano. "
                  "El círculo verde se vuelve elipse, el vecino anterior se aleja y el nuevo más "
                  "cercano es el de arriba. Misma fórmula euclidiana, otra respuesta: la ESCALA es "
                  "un peso implícito.")
        S = np.diag([STRETCH, 1.0, 1.0])
        self.play(ApplyMatrix(S, VGroup(plane, circ0), about_point=q),
                  *[d.animate.move_to(plane.c2p(*xy) + (S[:2, :2] @ xy - xy)[0] * unit * RIGHT)
                    for d, xy in zip(pts, SCALE_PTS)], run_time=2.2)
        circ1 = Circle(radius=SCALE_D1[SCALE_NN1] * unit, color=YELLOW, stroke_width=4).move_to(q)
        self.play(pts[SCALE_NN0].animate.set_color(BLUE).scale(1 / 1.4),
                  pts[SCALE_NN1].animate.set_color(YELLOW).scale(1.4), Create(circ1))
        info2 = VGroup(
            safe_text("después", size=24, color=YELLOW, weight=BOLD),
            tex(rf"d = {SCALE_D1[SCALE_NN1]:.2f}", size=32, color=YELLOW),
            safe_text("otro vecino", size=22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(info, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(info2))
        wformula = tex(r"d_w(\mathbf{x},\mathbf{y}) = \sqrt{\textstyle\sum_j w_j (x_j-y_j)^2}",
                       size=32, color=INK).move_to([3.6, -2.55, 0])
        wnote = tex(r"w_x = 2.5^2,\quad \text{estandarizar: } w_j = 1/s_j^2", size=28,
                    color=MUTED).next_to(wformula, DOWN, buff=0.15)
        wbg = card(wformula.width + 0.5, 1.25, ORANGE).move_to(VGroup(wformula, wnote))
        self.play(FadeIn(wbg), Write(wformula), FadeIn(wnote))

        # ── Mahalanobis como blanqueo ───────────────────────────────────────
        self.beat("28–29 min. Segunda transformación. Una nube con correlación 0.8. Los puntos a = "
                  "(1, 1) y b = (1, −1) están a la misma distancia euclidiana del centro, √2: el "
                  "círculo pasa por ambos. Pregunte: mirando la nube, ¿cuál de los dos es más "
                  "raro?")
        head = self.swap_heading(head, "Transformar el plano", "Mirar la nube desde su propia forma",
                                 PURPLE)
        self.clear_except(head)
        unit = 0.85
        plane = light_plane([-2, 2, 1], [-2, 2, 1], unit).move_to([-2.2, -0.45, 0])
        o = plane.c2p(0, 0)
        cloud = VGroup(*[Dot(plane.c2p(*xy), radius=0.045, color=BLUE, fill_opacity=0.75)
                         for xy in MAHA_CLOUD])
        a = Dot(plane.c2p(*MAHA_A), radius=0.12, color=GREEN)
        b = Dot(plane.c2p(*MAHA_B), radius=0.12, color=RED)
        la = tex("a", size=34, color=GREEN).add_updater(lambda m: m.next_to(a, UR, buff=0.04))
        lb = tex("b", size=34, color=RED).add_updater(lambda m: m.next_to(b, DR, buff=0.04))
        circ = Circle(radius=np.sqrt(2) * unit, color=TEAL, stroke_width=4).move_to(o)
        self.play(Create(plane), LaggedStart(*[FadeIn(d) for d in cloud], lag_ratio=0.004),
                  run_time=1.2)
        self.play(FadeIn(a), FadeIn(b), FadeIn(la), FadeIn(lb), Create(circ))
        eu = tex(r"\lVert a \rVert = \lVert b \rVert = \sqrt{2} = 1.41", size=34, color=TEAL)
        eu.move_to([3.7, 1.4, 0])
        self.play(Write(eu))

        self.beat("29–30 min. Aplicamos la transformación Σ^(−1/2): comprime la dirección larga de "
                  "la nube y estira la corta, hasta dejarla redonda. Ahora a queda cerca (1.05) y b "
                  "muy lejos (3.16). Esa distancia euclidiana DESPUÉS de transformar es la "
                  "distancia de Mahalanobis.")
        M = np.eye(3)
        M[:2, :2] = SIGMA_INV_HALF
        W = SIGMA_INV_HALF

        def moved(dots, data, mat):
            return [d.animate.move_to(o + unit * np.array([*(mat @ xy), 0]))
                    for d, xy in zip(dots, data)]

        self.play(ApplyMatrix(M, VGroup(plane, circ), about_point=o),
                  *moved(cloud, MAHA_CLOUD, W), *moved([a, b], [MAHA_A, MAHA_B], W), run_time=2.4)
        cm = Circle(radius=MAHA_DA * unit, color=GREEN, stroke_width=4).move_to(o)
        self.play(Create(cm))
        res = VGroup(tex(rf"d_M(a) = {MAHA_DA:.2f}", size=36, color=GREEN),
                     tex(rf"d_M(b) = {MAHA_DB:.2f}", size=36, color=RED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eu, DOWN, buff=0.5)
        self.play(FadeIn(res, shift=LEFT * 0.2))

        self.beat("30–31 min. Regresamos al espacio original aplicando la transformación inversa. "
                  "El círculo verde se convierte en una ELIPSE alineada con la nube: esa es la bola "
                  "de Mahalanobis. Formalice: d_M es la euclidiana de Σ^(−1/2)(x − y). La misma "
                  "idea de 'cambiar cómo miramos' explica estandarizar, PCA y Mahalanobis.")
        Mi = np.eye(3)
        vals, vecs = np.linalg.eigh(SIGMA)
        Mi[:2, :2] = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        I2 = np.eye(2)
        self.play(ApplyMatrix(Mi, VGroup(plane, circ, cm), about_point=o),
                  *moved(cloud, MAHA_CLOUD, I2), *moved([a, b], [MAHA_A, MAHA_B], I2), run_time=2.2)
        la.clear_updaters()
        lb.clear_updaters()
        form = VGroup(
            tex(r"d_M(\mathbf{x},\mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^\top \Sigma^{-1}"
                r"(\mathbf{x}-\mathbf{y})}", size=32, color=INK),
            tex(r"= \big\lVert \Sigma^{-1/2}(\mathbf{x}-\mathbf{y}) \big\rVert_2", size=32,
                color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fbg = card(form.width + 0.5, form.height + 0.5, PURPLE)
        fb = VGroup(fbg, form.move_to(fbg)).move_to([3.55, -1.95, 0])
        self.play(FadeIn(fb, shift=UP * 0.15))
