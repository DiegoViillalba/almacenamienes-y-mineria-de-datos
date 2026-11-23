"""Acto 1 · Separar con una recta (0–10 min).

Dieciséis puntos separables. Una recta gira y varias posiciones
separan esta muestra con margen estricto: hay infinitas soluciones. La geometría de g(x) = wᵀx + b: w es
perpendicular y apunta a la clase +1, b desliza la recta, g/‖w‖ es la distancia
con signo. Después, el perceptrón: cada error empuja la recta. Converge, pero
la recta final depende del inicio: hace falta un criterio.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (EYE_B, EYE_W, PERC, PERC_ETA, SEP_ANGLES, SEP_CENTER, TWO, X_BOX,
                           Y_BOX, line_through, separates)
from sections.plane import Plane, legend
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, tex)

RIGHT_X = 3.6


class SeparateSection:
    def _plane(self):
        plane = Plane(X_BOX, Y_BOX, height=4.7, center=(-3.25, -0.55, 0))
        dots = plane.dots(TWO)
        leg = legend().next_to(plane.axes, UP, buff=0.12).align_to(plane.axes, RIGHT)
        return plane, dots, leg

    def opening(self) -> None:
        self.beat("01")
        L = -6.25  # margen izquierdo de la portada
        logo = self.logo_lockup(width=3.3).move_to([L + 1.65, 3.05, 0])
        top_rule = Line([L, 2.45, 0], [L + 1.1, 2.45, 0], color=RED, stroke_width=5)
        eyebrow = safe_text("CLASIFICADORES LINEALES", size=22, color=TEAL, weight=BOLD)
        title = VGroup(
            safe_text("Dos rectas aciertan.", size=58, weight=BOLD),
            safe_text("¿Cuál elegir?", size=58, weight=BOLD, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        subtitle = safe_text("Una frontera parecida; distintos criterios para aprenderla",
                             size=24, color=MUTED, max_width=7.4)
        block = VGroup(eyebrow, title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        block.move_to([0, 0.15, 0]).align_to(Line([L, 0, 0], [L, 1, 0]), LEFT)
        author = VGroup(
            safe_text("Almacenes y Minería de Datos", size=24, color=INK, weight=BOLD),
            safe_text("Diego Villalba  ·  Facultad de Ciencias  ·  UNAM", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        author.align_to(block, LEFT).to_edge(DOWN, buff=0.55)
        # Motivo: dos nubes y una frontera, tenue, a la derecha.
        art = VGroup()
        sx, sy, cx, cy = 0.50, 0.55, 4.35, -0.15
        mp = lambda x, y: [cx + (x - 5) * sx, cy + (y - 4) * sy, 0]
        for x, y, c in TWO:
            if c < 0:
                art.add(Dot(mp(x, y), radius=0.11, color=BLUE, fill_opacity=0.85))
            else:
                art.add(Square(0.2, color=ORANGE, fill_opacity=0.85, stroke_width=0).move_to(mp(x, y)))
        art.add(Line(mp(2.2, 7.4), mp(8.2, 0.6), color=INK, stroke_width=4))
        art.add(Arrow(mp(5.2, 4.3), mp(6.4, 5.5), buff=0, color=RED, stroke_width=6,
                      max_tip_length_to_length_ratio=0.35))
        cover = Group(top_rule, eyebrow, title, subtitle, author, logo, art)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)
        self.beat("02")
        self.clear()
        self.add_branding()
        self.start("La pregunta del video", "La forma no determina cómo se aprende", YELLOW)
        route = VGroup(*[pill(t, c, size=28, width=10.5) for t,c in [
            ("1 · Geometría: ¿qué decide una recta?", PURPLE),
            ("2 · Aprender: cuadrados, sombras y probabilidades", GREEN),
            ("3 · Ampliar: más variables, clases y kernels", PINK),
        ]]).arrange(DOWN, buff=0.45).move_to([0,-0.3,0])
        self.play(LaggedStart(*[FadeIn(r) for r in route], lag_ratio=0.2))
        self.many_lines()
        self.geometry()
        self.perceptron()

    # ------------------------------------------------------------------
    def many_lines(self) -> None:
        self.beat("03")
        head = self.start("Separación lineal", "¿Cabe una recta entre las dos nubes?", YELLOW)
        plane, dots, leg = self._plane()
        self.play(FadeIn(plane), FadeIn(leg))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.05))
        q = VGroup(safe_text("Una recta parte el plano", size=28, weight=BOLD),
                   safe_text("en dos semiplanos.", size=28, weight=BOLD),
                   safe_text("¿Existe una que deje cada clase", size=24, color=MUTED),
                   safe_text("de su propio lado?", size=24, color=MUTED)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([RIGHT_X, 0.6, 0])
        self.play(FadeIn(q))
        self.plane, self.dots, self.leg = plane, dots, leg

        self.beat("04")
        ang = ValueTracker(SEP_ANGLES[0])

        def coefs():
            (w1, w2), b = line_through(ang.get_value())
            return b, w1, w2

        line = always_redraw(lambda: plane.boundary(*coefs()))
        regions = always_redraw(lambda: plane.half_planes(*coefs()))

        def bad_marks():
            b, w1, w2 = coefs()
            out = VGroup()
            for d in dots:
                x, y = d.xy
                if np.sign(b + w1 * x + w2 * y) != d.cls:
                    out.add(Circle(radius=0.22, color=RED, stroke_width=4).move_to(d))
            return out

        bad = always_redraw(bad_marks)
        counter = always_redraw(lambda: safe_text(
            f"errores: {len(bad_marks())}", size=26, color=RED if len(bad_marks()) else GREEN,
            weight=BOLD).move_to([RIGHT_X - 1.2, -1.4, 0]))
        self.add(regions)
        self.bring_to_front(dots)
        self.play(FadeOut(q), Create(line), FadeIn(counter))
        self.add(bad)
        for a in SEP_ANGLES[1:]:
            self.play(ang.animate.set_value(a), run_time=1.0)
        self.play(ang.animate.set_value(85), run_time=1.0)
        self.wait(0.4)
        self.play(ang.animate.set_value(45), run_time=1.2)
        for m in (line, regions, bad, counter):
            m.clear_updaters()
        self.remove(bad)
        many = VGroup(*[plane.boundary(*(lambda wb: (wb[1], *wb[0]))(line_through(a)),
                                       color=YELLOW, width=2) for a in (22, 30, 38, 52, 60)])
        many.set_stroke(opacity=0.45)
        msg = VGroup(safe_text("Con margen estricto,", size=28, weight=BOLD),
                     safe_text("hay infinitas que separan.", size=28, weight=BOLD, color=YELLOW),
                     safe_text("¿Cuál es la mejor?", size=26, color=MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([RIGHT_X, 0.7, 0])
        self.play(FadeOut(counter), FadeIn(many), FadeIn(msg))
        self.sep_line, self.sep_regions, self.sep_many, self.sep_msg = line, regions, many, msg
        self.sep_head = head

    # ------------------------------------------------------------------
    def geometry(self) -> None:
        plane, dots = self.plane, self.dots
        self.beat("05")
        head = self.swap_heading(self.sep_head, "Geometría",
                                 "w gira la recta; b la desliza", PURPLE)
        self.sep_head = head
        self.play(FadeOut(self.sep_many), FadeOut(self.sep_msg))
        ang = ValueTracker(45.0)
        shift = ValueTracker(0.0)

        def coefs():
            (w1, w2), b = line_through(ang.get_value())
            return b - shift.get_value(), w1, w2

        self.remove(self.sep_line, self.sep_regions)
        line = always_redraw(lambda: plane.boundary(*coefs()))
        regions = always_redraw(lambda: plane.half_planes(*coefs()))
        arrow = always_redraw(lambda: plane.w_arrow(*coefs(), length=1.5))
        # The purple arrow is identified in the equation; no moving label over data.
        wlab = VectorizedPoint()
        self.add(regions, line)
        self.bring_to_front(dots)
        self.play(GrowArrow(arrow), FadeIn(wlab))
        eq = tex(r"g(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b", size=44).move_to(
            [RIGHT_X, 1.7, 0])
        facts = VGroup(
            safe_text("w ⟂ frontera, apunta a +1", size=24, color=PURPLE),
            safe_text("con w fijo, b desplaza", size=24, color=YELLOW),
            safe_text("frontera:  g(x) = 0", size=24, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(eq, DOWN, buff=0.45)
        self.play(Write(eq), FadeIn(facts[0]))
        self.play(ang.animate.set_value(15), run_time=1.3)
        self.play(ang.animate.set_value(70), run_time=1.5)
        self.play(ang.animate.set_value(45), run_time=1.0)

        self.beat("06")
        self.play(FadeIn(facts[1]))
        self.play(shift.animate.set_value(-1.6), run_time=1.3)
        self.play(shift.animate.set_value(1.6), run_time=1.6)
        self.play(shift.animate.set_value(0.0), run_time=1.0)
        self.play(FadeIn(facts[2]))
        for m in (line, regions, arrow, wlab):
            m.clear_updaters()

        self.beat("07")
        b, w1, w2 = coefs()
        samples = [(8.8, 4.0, ORANGE), (1.5, 2.0, BLUE)]
        segs = VGroup()
        for x, y, col in samples:
            fx, fy = plane.foot(b, w1, w2, x, y)
            dist = (b + w1 * x + w2 * y) / np.hypot(w1, w2)
            seg = DashedLine(plane.p(fx, fy), plane.p(x, y), color=col, stroke_width=4,
                             dash_length=0.08)
            lab = safe_text(f"{dist:+.1f}", size=24, color=col, weight=BOLD).next_to(
                seg.get_center(), DR if col == ORANGE else UL, buff=0.08)
            segs.add(VGroup(seg, lab))
        dist_eq = tex(r"\text{distancia con signo} = \frac{g(\mathbf{x})}{\lVert\mathbf{w}\rVert}",
                      size=34, color=INK).next_to(facts, DOWN, buff=0.45).align_to(facts, LEFT)
        self.play(LaggedStart(*[Create(s[0]) for s in segs], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(s[1]) for s in segs], lag_ratio=0.3))
        self.play(Write(dist_eq))

        self.beat("08")
        self.play(FadeOut(segs), FadeOut(dist_eq), FadeOut(facts), FadeOut(eq))
        rule = tex(r"\hat{y} = \operatorname{signo}\!\left(\mathbf{w}^\top\mathbf{x} + b\right)",
                   size=44, color=INK).move_to([RIGHT_X, 1.8, 0])
        hyp = tex(r"\{\mathbf{x}\in\mathbb{R}^d : \mathbf{w}^\top\mathbf{x} + b = 0\}", size=34,
                  color=YELLOW).next_to(rule, DOWN, buff=0.35)
        dims = VGroup()
        for d, txt in [(1, "d = 1: un punto"), (2, "d = 2: una recta"), (3, "d = 3: un plano")]:
            ic = self._dim_icon(d)
            lab = safe_text(txt, size=22, color=MUTED)
            dims.add(VGroup(ic, lab).arrange(RIGHT, buff=0.3))
        dims.arrange(DOWN, aligned_edge=LEFT, buff=0.26).next_to(hyp, DOWN, buff=0.4)
        gen = safe_text("d variables: hiperplano de dimensión d − 1", size=22, color=YELLOW,
                        weight=BOLD, max_width=5.6).next_to(dims, DOWN, buff=0.3).align_to(dims, LEFT)
        self.play(Write(rule))
        self.beat("09")
        self.play(Write(hyp))
        self.play(LaggedStart(*[FadeIn(r) for r in dims], lag_ratio=0.2))
        self.play(FadeIn(gen))
        assumption = safe_text("w ≠ 0 · en g = 0, fijar el desempate", size=20, color=MUTED,
                               max_width=5.8).next_to(gen, DOWN, buff=0.22)
        self.play(FadeIn(assumption))
        # Keep the right column inside the safe frame without scaling the graph.
        text_group = VGroup(rule, hyp, dims, gen, assumption)
        if text_group.get_right()[0] > 6.65:
            text_group.shift(LEFT * (text_group.get_right()[0] - 6.65))
        self.geo_objs = VGroup(line, regions, arrow, wlab)
        self.geo_text = VGroup(rule, hyp, dims, gen, assumption)

    @staticmethod
    def _dim_icon(d: int) -> VGroup:
        if d == 1:
            ax = Line(LEFT * 0.6, RIGHT * 0.6, color=GRID, stroke_width=3)
            return VGroup(ax, Dot(ORIGIN, radius=0.07, color=YELLOW))
        if d == 2:
            sq = Square(1.0, color=GRID, stroke_width=2)
            return VGroup(sq, Line(sq.get_corner(DL), sq.get_corner(UR), color=YELLOW,
                                   stroke_width=4))
        cube = VGroup(Square(0.8, color=GRID, stroke_width=2),
                      Square(0.8, color=GRID, stroke_width=2).shift(UR * 0.25))
        pl = Polygon([-0.4, -0.4, 0], [0.4, -0.15, 0], [0.65, 0.45, 0], [-0.15, 0.2, 0],
                     stroke_color=YELLOW, stroke_width=2, fill_color=YELLOW, fill_opacity=0.35)
        return VGroup(cube, pl)

    # ------------------------------------------------------------------
    def perceptron(self) -> None:
        plane, dots = self.plane, self.dots
        self.beat("10")
        head = self.swap_heading(self.sep_head, "El perceptrón",
                                 "Cada error empuja la recta", ORANGE)
        self.play(FadeOut(self.geo_objs), FadeOut(self.geo_text))
        rule = VGroup(
            safe_text("Si y·g(x) ≤ 0: corregir", size=24, color=MUTED),
            tex(r"\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\, \mathbf{x}", size=38,
                color=PURPLE),
            tex(r"b \leftarrow b + \eta\, y", size=38, color=YELLOW),
            safe_text(f"η = {PERC_ETA}", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([RIGHT_X, 1.2, 0])
        _, w, b = PERC[0]
        bt, wt1, wt2 = ValueTracker(b), ValueTracker(w[0]), ValueTracker(w[1])
        def current():
            return bt.get_value(), wt1.get_value(), wt2.get_value()
        line = always_redraw(lambda: plane.boundary(*current()))
        regions = always_redraw(lambda: plane.half_planes(*current()))
        arrow = always_redraw(lambda: plane.w_arrow(*current(), length=1.2))
        self.bring_to_front(dots)
        self.play(FadeIn(regions), Create(line), GrowArrow(arrow), FadeIn(rule))
        step_lab = safe_text("inicio", size=24, color=INK, weight=BOLD).move_to([RIGHT_X - 1.1,
                                                                               -1.0, 0])
        self.play(FadeIn(step_lab))
        for k, (i, w, b) in enumerate(PERC[1:], start=1):
            if k > 1:
                self.beat(f"{9+k:02}")
            d = dots[i]
            ring = Circle(radius=0.24, color=RED, stroke_width=5).move_to(d)
            self.play(Create(ring), Flash(d, color=RED, flash_radius=0.35))
            new_lab = safe_text(f"corrección {k}", size=24, color=INK, weight=BOLD).move_to(step_lab)
            self.play(FadeOut(step_lab),run_time=0.15)
            step_lab.become(new_lab)
            self.play(bt.animate.set_value(b),wt1.animate.set_value(w[0]),wt2.animate.set_value(w[1]),
                      FadeIn(step_lab),run_time=1.2)
            self.play(FadeOut(ring), run_time=0.3)
        for obj in (line, regions, arrow):
            obj.clear_updaters()
        _, w, b = PERC[-1]
        ok = safe_text("separa: 0 errores", size=26, color=GREEN, weight=BOLD).next_to(
            step_lab, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(FadeIn(ok))

        self.beat("13")
        eb, ew = EYE_B, EYE_W
        other = plane.boundary(eb, *ew, color=MUTED, width=3, dashed=True)
        crit = VGroup(*[pill(t, c, size=20, width=5.6) for t, c in [
            ("MCO: errores al cuadrado", BLUE),
            ("Fisher: separar las sombras", TEAL),
            ("Logística: verosimilitud", GREEN),
            ("SVM: margen máximo", PINK)]]).arrange(DOWN, buff=0.14)
        crit.move_to([RIGHT_X, -0.6, 0])
        self.play(Create(other), FadeOut(rule), FadeOut(step_lab), FadeOut(ok))
        q = safe_text("¿Cuál recta? Depende del criterio:", size=26, weight=BOLD).next_to(
            crit, UP, buff=0.3)
        self.play(FadeIn(q), LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in crit],
                                         lag_ratio=0.15))
        assert separates(w, b)
