"""Acto 1 · Separar con una recta (0–10 min).

Dieciséis puntos separables. Una recta gira y casi todas las posiciones
separan: hay infinitas soluciones. La geometría de g(x) = wᵀx + b: w es
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
        self.beat("0–1 min. Portada. Pregunte: ¿qué tienen en común una regresión, una "
                  "regresión logística y una SVM con kernel? Hoy vamos a defender una tesis: "
                  "todas son, en el fondo, un hiperplano. Lo veremos primero con geometría y "
                  "después lo escribiremos.")
        eyebrow = safe_text("CLASIFICACIÓN · CLASIFICADORES LINEALES", size=22, color=PURPLE,
                            weight=BOLD)
        title = safe_text("Todo es un hiperplano", size=60, weight=BOLD, max_width=12.6)
        subtitle = safe_text("Separación lineal, Fisher, regresión logística y kernels",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Ruta. Cada método responde de manera distinta a la misma pregunta: "
                  "¿cuál recta? Mínimos cuadrados y Fisher tienen fórmula; la logística no, hay "
                  "que iterar. Los kernels parecen no lineales, pero son lineales en otro "
                  "espacio. Fisher es opcional si el tiempo apremia.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "Una misma pregunta: ¿cuál recta?", YELLOW)
        route = VGroup(*[pill(t, c, size=23, width=6.0) for t, c in [
            ("1 · Separar con una recta", YELLOW),
            ("2 · Mínimos cuadrados: una fórmula", BLUE),
            ("3 · Fisher: la mejor sombra", TEAL),
            ("4 · Logística: sin fórmula, iterar", GREEN),
            ("5 · Muchas variables: hiperplanos", PURPLE),
            ("6 · Kernels: lineal en otro espacio", PINK),
            ("7 · Comparar y ejercicio", ORANGE)]]).arrange(DOWN, buff=0.17)
        route.move_to([-3.3, -0.45, 0])
        promise = VGroup(
            safe_text("Al terminar sabrás", size=26, color=MUTED),
            safe_text("· leer w y b como geometría", size=23),
            safe_text("· por qué MCO tiene fórmula cerrada", size=23),
            safe_text("· por qué la logística hay que iterarla", size=23),
            safe_text("· que su frontera sigue siendo un hiperplano", size=23),
            safe_text("· en qué sentido un kernel es lineal", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([0.35, -0.4, 0], aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in route], lag_ratio=0.1))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in promise], lag_ratio=0.12))
        self.many_lines()
        self.geometry()
        self.perceptron()

    # ------------------------------------------------------------------
    def many_lines(self) -> None:
        self.beat("2–3 min. Dieciséis puntos de dos clases (forma y color). Pregunte: ¿pueden "
                  "trazar UNA recta que deje a cada clase de un lado? Pida a alguien que la "
                  "describa con las manos antes de avanzar.")
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

        self.beat("3–4 min. La recta gira alrededor del centro de las nubes. Casi todas las "
                  "posiciones separan perfectamente; sólo al inclinarla demasiado aparecen "
                  "errores (en rojo). Mensaje: cuando los datos son separables hay INFINITAS "
                  "rectas que separan. ¿Cuál elegir? Esa pregunta organiza toda la clase.")
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
        msg = VGroup(safe_text("Si una recta separa,", size=28, weight=BOLD),
                     safe_text("hay infinitas que separan.", size=28, weight=BOLD, color=YELLOW),
                     safe_text("¿Cuál es la mejor?", size=26, color=MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([RIGHT_X, 0.7, 0])
        self.play(FadeOut(counter), FadeIn(many), FadeIn(msg))
        self.sep_line, self.sep_regions, self.sep_many, self.sep_msg = line, regions, many, msg
        self.sep_head = head

    # ------------------------------------------------------------------
    def geometry(self) -> None:
        plane, dots = self.plane, self.dots
        self.beat("4–6 min. Toda recta se describe con un vector w y un número b: "
                  "g(x) = w·x + b. La flecha morada es w. Hágales notar que es PERPENDICULAR a "
                  "la frontera y apunta hacia la clase +1. Al girar w, gira la recta. Pregunte: "
                  "¿qué dirección de w haría que todos los naranjas quedaran del lado azul?")
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
        wlab = always_redraw(lambda: tex(r"\mathbf{w}", size=40, color=PURPLE).next_to(
            arrow.get_end(), UR, buff=0.04))
        self.add(regions, line)
        self.bring_to_front(dots)
        self.play(GrowArrow(arrow), FadeIn(wlab))
        eq = tex(r"g(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b", size=44).move_to(
            [RIGHT_X, 1.7, 0])
        facts = VGroup(
            safe_text("w ⟂ frontera, apunta a +1", size=24, color=PURPLE),
            safe_text("b desplaza sin girar", size=24, color=YELLOW),
            safe_text("frontera:  g(x) = 0", size=24, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(eq, DOWN, buff=0.45)
        self.play(Write(eq), FadeIn(facts[0]))
        self.play(ang.animate.set_value(15), run_time=1.3)
        self.play(ang.animate.set_value(70), run_time=1.5)
        self.play(ang.animate.set_value(45), run_time=1.0)

        self.beat("6–7 min. Ahora sólo cambia b: la recta se desliza paralela a sí misma, sin "
                  "girar. Pregunte: si b crece mucho, ¿qué clase gana terreno? (El lado de w "
                  "crece, más puntos se predicen +1.) Regrese al centro.")
        self.play(FadeIn(facts[1]))
        self.play(shift.animate.set_value(-1.6), run_time=1.3)
        self.play(shift.animate.set_value(1.6), run_time=1.6)
        self.play(shift.animate.set_value(0.0), run_time=1.0)
        self.play(FadeIn(facts[2]))
        for m in (line, regions, arrow, wlab):
            m.clear_updaters()

        self.beat("7–8 min. El número g(x) no sólo da el lado (signo), también la distancia: "
                  "g(x)/‖w‖ es la distancia con signo a la frontera. Con ‖w‖ = 1 se lee directo. "
                  "El cuadrado naranja de la derecha está a +2.3 unidades; el círculo azul de abajo a −4.2. Este puntaje es lo "
                  "que la logística convertirá en probabilidad.")
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

        self.beat("8–9 min. Formalización. Con d variables nada cambia: ŷ = signo(wᵀx + b). "
                  "En una variable la frontera es un punto, en dos una recta, en tres un plano; "
                  "en general un HIPERPLANO de dimensión d − 1. Pregunte: ¿cuántos números "
                  "definen un hiperplano en 10 variables? (11: diez pesos y b.)")
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
        self.play(Write(hyp))
        self.play(LaggedStart(*[FadeIn(r) for r in dims], lag_ratio=0.2))
        self.play(FadeIn(gen))
        self.geo_objs = VGroup(line, regions, arrow, wlab)
        self.geo_text = VGroup(rule, hyp, dims, gen)

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
        self.beat("9–10 min. El algoritmo clásico (Rosenblatt, 1958): empezar con cualquier "
                  "recta. Si un punto está del lado equivocado, sumarle a w el punto "
                  "multiplicado por su etiqueta: la recta se mueve hacia corregirlo. Prediga: "
                  "¿hacia dónde se moverá la recta con el cuadrado marcado?")
        head = self.swap_heading(self.sep_head, "El perceptrón",
                                 "Cada error empuja la recta", ORANGE)
        self.play(FadeOut(self.geo_objs), FadeOut(self.geo_text))
        rule = VGroup(
            safe_text("Si y·g(x) ≤ 0  (error):", size=24, color=MUTED),
            tex(r"\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\, \mathbf{x}", size=38,
                color=PURPLE),
            tex(r"b \leftarrow b + \eta\, y", size=38, color=YELLOW),
            safe_text(f"η = {PERC_ETA}", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([RIGHT_X, 1.2, 0])
        _, w, b = PERC[0]
        line = plane.boundary(b, *w)
        regions = plane.half_planes(b, *w)
        arrow = plane.w_arrow(b, *w, length=1.2)
        self.add(regions, line, arrow)
        self.bring_to_front(dots)
        self.play(FadeIn(regions), Create(line), GrowArrow(arrow), FadeIn(rule))
        step_lab = safe_text("inicio", size=24, color=INK, weight=BOLD).move_to([RIGHT_X - 1.1,
                                                                               -1.0, 0])
        self.play(FadeIn(step_lab))
        for k, (i, w, b) in enumerate(PERC[1:], start=1):
            if k > 1:
                self.beat(f"Corrección {k}. El nuevo error está marcado; la recta vuelve a "
                          "moverse. Note que cada corrección puede descomponer otros puntos: "
                          "el perceptrón sólo mira un error a la vez.")
            d = dots[i]
            ring = Circle(radius=0.24, color=RED, stroke_width=5).move_to(d)
            self.play(Create(ring), Flash(d, color=RED, flash_radius=0.35))
            new_line = plane.boundary(b, *w)
            new_regions = plane.half_planes(b, *w)
            new_arrow = plane.w_arrow(b, *w, length=1.2)
            new_lab = safe_text(f"corrección {k}", size=24, color=INK, weight=BOLD).move_to(
                step_lab)
            self.play(Transform(line, new_line), Transform(regions, new_regions),
                      Transform(arrow, new_arrow), Transform(step_lab, new_lab), run_time=1.2)
            self.play(FadeOut(ring), run_time=0.3)
        _, w, b = PERC[-1]
        ok = safe_text("separa: 0 errores", size=26, color=GREEN, weight=BOLD).next_to(
            step_lab, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(FadeIn(ok))

        self.beat("Cierre del acto. El perceptrón converge si los datos son separables, pero "
                  "la recta final depende de dónde empezó y del orden de los puntos. La recta "
                  "punteada separa igual de bien. Necesitamos un CRITERIO que diga cuál recta "
                  "es mejor: cada método de hoy es un criterio distinto.")
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
