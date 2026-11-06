"""Acto 3 · ReLU: doblar el plano (19–29 min).

Una recta: cuatro ReLU suman un escalón que separa un intervalo. Un plano: cada
ReLU es una bisagra a lo largo de su recta; cuatro bisagras forman un enclave
cuadrado. Capas: rectas → polígonos → unión de polígonos con huecos. Y la
intuición del papel doblado: cada capa dobla el espacio.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (BUMP_KNOTS, BUMP_NEG, BUMP_POS, BUMP_SIGNS, COUNTER, ENCLAVES,
                           BE_SOUTH, SQUARE, SQUARE_NEURONS, bump, grid, relu, square_score)
from sections.theme import (BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED,
                            TEAL, YELLOW, card, marker, pill, safe_text, tex)
from sections.viz import BG_RGB, MapView, _rgb

PURPLE_RGB = _rgb(PURPLE)


def hinge_image(mv: MapView, F: np.ndarray, vmax: float) -> ImageMobject:
    t = np.clip(F / vmax, 0, 1)[..., None] ** 0.8
    rgb = BG_RGB * (1 - 0.75 * t) + PURPLE_RGB * 0.75 * t
    return mv.image(np.dstack([rgb, np.full(F.shape, 255.0)]))


class ReluSection:
    def relu_act(self) -> None:
        self.relu_1d()
        self.relu_hinges()
        self.relu_square()
        self.relu_layers()
        self.relu_folds()

    # ------------------------------------------------------------------
    def relu_1d(self) -> None:
        self.beat("19–21 min. Primero en UNA dimensión. Bélgica está en el centro de la recta: "
                  "ningún umbral único la separa. Construimos una función con ReLU, una a la "
                  "vez: cada ReLU agrega un «codo» en su umbral. Pida que predigan la forma de "
                  "la suma antes de la cuarta.")
        self.start("ReLU en una recta", "Sumar codos para construir un escalón", PURPLE)
        ax = Axes(x_range=[0, 10, 1], y_range=[-0.6, 1.6, 0.5], x_length=10.5, y_length=3.4,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2})
        ax.move_to([0, 0.15, 0])
        nl = NumberLine(x_range=[0, 10, 1], length=10.5, color=GRID, include_numbers=False)
        nl.move_to([0, -2.55, 0])
        pts = VGroup()
        for x in BUMP_POS:
            pts.add(marker(1, r=0.12).move_to(nl.n2p(x)))
        for x in BUMP_NEG:
            pts.add(marker(0, r=0.12).move_to(nl.n2p(x)))
        self.play(Create(nl), LaggedStart(*[FadeIn(p, scale=0.5) for p in pts], lag_ratio=0.06))
        q = safe_text("¿un solo umbral?", size=24, color=YELLOW).next_to(nl, DOWN, buff=0.2)
        self.play(FadeIn(q), Create(ax))

        colors = [TEAL, RED, RED, TEAL]
        names = [r"+\,\text{ReLU}(x-3)", r"-\,\text{ReLU}(x-4)", r"-\,\text{ReLU}(x-6)",
                 r"+\,\text{ReLU}(x-7)"]
        total = None
        legend = VGroup()
        for k, (knot, sgn, col, nm) in enumerate(zip(BUMP_KNOTS, BUMP_SIGNS, colors, names)):
            term = ax.plot(lambda x, kn=knot, s=sgn: float(np.clip(s * relu(x - kn), -0.6, 1.6)),
                           x_range=[0, 10, 0.01], color=col, stroke_width=3,
                           use_smoothing=False).set_stroke(opacity=0.55)
            lab = MathTex(nm, font_size=30, color=col)
            legend.add(lab)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.45).shift(
                DOWN * 0.15)
            kk = k
            new_total = ax.plot(lambda x, n=kk: float(np.clip(sum(
                s * relu(x - kn) for kn, s in list(zip(BUMP_KNOTS, BUMP_SIGNS))[:n + 1]),
                -0.6, 1.6)), x_range=[0, 10, 0.01], color=YELLOW, stroke_width=6,
                use_smoothing=False)
            if total is None:
                self.play(Create(term), FadeIn(lab))
                self.play(Transform(term, new_total))
                total = term
            else:
                self.play(Create(term), FadeIn(lab))
                self.play(Transform(total, new_total), FadeOut(term))
        self.play(FadeOut(q))

        self.beat("21–22 min. Umbral en 0.5: la suma supera 0.5 exactamente en [3.5, 6.5]. "
                  "Cuatro neuronas ReLU y una suma lograron lo que ninguna recta podía: un "
                  "intervalo. La función es lineal a trozos; los codos están donde cada neurona "
                  "se enciende.")
        th = DashedLine(ax.c2p(0, 0.5), ax.c2p(10, 0.5), color=MUTED, dash_length=0.12)
        thl = MathTex("0.5", font_size=28, color=MUTED).next_to(th, LEFT, buff=0.1)
        band = Rectangle(width=nl.n2p(6.5)[0] - nl.n2p(3.5)[0], height=0.5, stroke_width=0,
                         fill_color=BLUE, fill_opacity=0.25).move_to(
            (nl.n2p(3.5) + nl.n2p(6.5)) / 2)
        self.play(Create(th), FadeIn(thl))
        self.play(FadeIn(band))
        res = safe_text("4 neuronas ReLU + 1 suma = un intervalo", size=26, color=YELLOW,
                        weight=BOLD).next_to(nl, DOWN, buff=0.22)
        self.play(FadeIn(res, shift=UP * 0.1))

    # ------------------------------------------------------------------
    def relu_hinges(self) -> None:
        self.beat("22–24 min. En el plano, una neurona ReLU es una BISAGRA: vale cero de un lado "
                  "de su recta y crece linealmente del otro. Cuatro neuronas, una por lado del "
                  "cuadrado, cada una mide «qué tanto me salí por este lado». Señale que las "
                  "normales apuntan hacia afuera.")
        self.start("ReLU en el plano", "Cada neurona es una bisagra sobre su recta", PURPLE)
        X, Y = grid(300, 210)
        x0, x1, y0, y1 = SQUARE
        labels = [r"\text{ReLU}(3 - x)", r"\text{ReLU}(x - 7)", r"\text{ReLU}(2 - y)",
                  r"\text{ReLU}(y - 5)"]
        panels = Group()
        for k, ((n, c), lab) in enumerate(zip(SQUARE_NEURONS, labels)):
            mv = MapView(3.05, (-5.0 + k * 3.33, 0.05, 0))
            F = relu(n[0] * X + n[1] * Y - c)
            img = hinge_image(mv, F, 4.0)
            if n[0] != 0:
                xl = c / n[0]
                ln = Line(mv.p(xl, 0), mv.p(xl, 7), color=YELLOW, stroke_width=3)
            else:
                yl = c / n[1]
                ln = Line(mv.p(0, yl), mv.p(10, yl), color=YELLOW, stroke_width=3)
            sq = mv.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], INK, width=1.5)
            sq = DashedVMobject(sq, num_dashes=24)
            t = MathTex(lab, font_size=28, color=PURPLE).next_to(mv.frame(), DOWN, buff=0.2)
            panels.add(Group(img, mv.frame(), sq, ln, t))
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.1) for p in panels], lag_ratio=0.25),
                  run_time=2.2)
        cap = VGroup(
            safe_text("0 de un lado de la recta", size=22, color=MUTED),
            safe_text("·", size=22, color=MUTED),
            safe_text("crece en línea recta del otro", size=22, color=PURPLE),
        ).arrange(RIGHT, buff=0.25).move_to([0, -2.55, 0])
        self.play(FadeIn(cap))
        self.hinge_panels = panels

    # ------------------------------------------------------------------
    def relu_square(self) -> None:
        self.beat("24–25 min. Sumamos las cuatro bisagras y restamos de 1: s = 1 − 2·Σ ReLU. "
                  "Dentro del cuadrado todas valen 0 y s = 1 (Bélgica); al salir por cualquier "
                  "lado s baja y se vuelve negativa (Países Bajos). Una capa de ReLU + una suma "
                  "= un polígono: es un «Y» de cuatro semiplanos. Las esquinas quedan recortadas "
                  "en diagonal: ¿por qué? (ahí se suman dos bisagras).")
        X, Y = grid(500, 350)
        S = square_score(X, Y)
        mv = MapView(6.3, (-3.2, -0.45, 0))
        img = mv.field_image(S, alpha=0.6, scale=3.0)
        x0, x1, y0, y1 = SQUARE
        sq = DashedVMobject(mv.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], INK, width=2),
                            num_dashes=30)
        cont = mv.contour(S, YELLOW, 6)
        head = self.heading("ReLU en el plano", "Cuatro bisagras = un enclave", PURPLE)
        old = [m for m in self.mobjects if m is not self.branding]
        self.play(*[FadeOut(m) for m in old], FadeIn(head))
        self.play(FadeIn(img), Create(mv.frame()), Create(sq))
        self.play(Create(cont), run_time=1.5)
        eq = VGroup(
            tex(r"s(x, y) = 1 - 2\big[\,\text{ReLU}(3-x) + \text{ReLU}(x-7)", size=30,
                max_width=6.4),
            tex(r"+\, \text{ReLU}(2-y) + \text{ReLU}(y-5)\,\big]", size=30, max_width=6.4),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.12).move_to([3.6, 1.3, 0])
        reading = VGroup(
            VGroup(tex(r"s = 1", size=30, color=BLUE), safe_text("dentro: Bélgica", size=22)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(tex(r"s < 0", size=30, color=ORANGE), safe_text("fuera: Países Bajos",
                                                                   size=22)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(tex(r"s = 0", size=30, color=YELLOW), safe_text("la frontera", size=22)
                   ).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.4, -0.4, 0])
        key = pill("1 capa ReLU + suma = 1 polígono", PURPLE, size=23).move_to([3.5, -2.25, 0])
        self.play(FadeIn(eq))
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in reading], lag_ratio=0.2))
        self.play(FadeIn(key, shift=UP * 0.1))

    # ------------------------------------------------------------------
    def relu_layers(self) -> None:
        self.beat("25–27 min. Con eso ya se ve la receta para Baarle. Capa 1: muchas rectas "
                  "(semiplanos). Capa 2: cada neurona combina algunas y forma un polígono (un "
                  "enclave). Salida: suma los polígonos belgas y RESTA el contraenclave. Es una "
                  "construcción a mano para entender; la red entrenada encontrará la suya.")
        self.start("La receta para Baarle", "Rectas → polígonos → unión con huecos", YELLOW)
        centers = [(-4.65, 0.1, 0), (0.0, 0.1, 0), (4.65, 0.1, 0)]
        mvs = [MapView(4.1, c) for c in centers]
        # capa 1: rectas
        rng = np.random.default_rng(5)
        lines = VGroup()
        for pts in list(ENCLAVES.values()):
            n = len(pts)
            for i in range(n):
                (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
                a, c = yb - ya, -(xb - xa)
                d = a * xa + c * ya
                ln = mvs[0].line(a, c, d, TEAL, 1.6)
                if ln is not None:
                    lines.add(ln.set_stroke(opacity=0.7))
        p1 = VGroup(mvs[0].frame(), lines)
        # capa 2: polígonos
        polys = VGroup(*[mvs[1].polygon(p, BLUE, width=3, fill=0.35) for p in ENCLAVES.values()],
                       *[mvs[1].polygon(p, ORANGE, width=3, fill=0.35) for p in
                         COUNTER.values()])
        p2 = VGroup(mvs[1].frame(), polys)
        # salida
        img = mvs[2].truth_image(alpha=0.45)
        p3 = Group(img, mvs[2].frame(), mvs[2].borders(width=1.8))
        titles = VGroup(
            safe_text("capa 1: rectas", size=24, color=TEAL, weight=BOLD),
            safe_text("capa 2: polígonos", size=24, color=BLUE, weight=BOLD),
            safe_text("salida: unión − huecos", size=24, color=YELLOW, weight=BOLD),
        )
        for t, mv in zip(titles, mvs):
            t.next_to(mv.frame(), UP, buff=0.2)
        subs = VGroup(
            safe_text("semiplanos w·x + b > 0", size=19, color=MUTED),
            safe_text("«Y»: estar dentro de todos", size=19, color=MUTED),
            safe_text("«O» de piezas, «menos» el hueco", size=19, color=MUTED),
        )
        for s, mv in zip(subs, mvs):
            s.next_to(mv.frame(), DOWN, buff=0.2)
        arrows = VGroup(*[Arrow(mvs[i].frame().get_right(), mvs[i + 1].frame().get_left(),
                                buff=0.05, color=GRID, stroke_width=4) for i in range(2)])
        self.play(FadeIn(titles[0]), Create(p1[0]))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.05), FadeIn(subs[0]))
        self.play(GrowArrow(arrows[0]), FadeIn(titles[1]), Create(p2[0]))
        self.play(LaggedStart(*[DrawBorderThenFill(p) for p in polys], lag_ratio=0.2),
                  FadeIn(subs[1]))
        self.play(GrowArrow(arrows[1]), FadeIn(titles[2]))
        self.play(FadeIn(p3), FadeIn(subs[2]))
        formula = tex(r"\hat s = \sum_{\text{piezas}} \text{ReLU}(s_{\text{pieza}}) \;-\; "
                      r"2\,\text{ReLU}(s_{\text{hueco}}) \;-\; \tfrac12", size=32,
                      color=INK).move_to([0, -2.75, 0])
        self.play(Write(formula))

    # ------------------------------------------------------------------
    def relu_folds(self) -> None:
        self.beat("27–29 min. Otra forma de verlo: cada capa DOBLA el espacio como una hoja de "
                  "papel. |x| = ReLU(x) + ReLU(−x) dobla la recta en el origen; los puntos de "
                  "los dos lados quedan encimados. Pida que observen los azules: después de dos "
                  "dobleces caen juntos y UN corte basta. Un corte en el papel doblado son "
                  "varios cortes en el original: por eso la profundidad multiplica las regiones.")
        self.start("La intuición del papel doblado", "Cada capa dobla el espacio", PURPLE)
        xs = np.array([-3.7, -3.2, -2.4, -2.0, -1.6, -0.9, -0.4, 0.3, 0.8, 1.5, 1.9, 2.5, 3.1,
                       3.6])
        cls = np.where(np.abs(np.abs(xs) - 2) < 0.7, 0, 1)
        rows_y = [1.3, -0.3, -1.9]
        nls = [NumberLine(x_range=[-4, 4, 1], length=9.0, color=GRID, include_numbers=True,
                          font_size=22).move_to([0.9, y, 0]) for y in rows_y]
        tags = VGroup(
            tex(r"x", size=32, color=INK),
            tex(r"u = |x|", size=30, color=PURPLE),
            tex(r"v = |u - 2|", size=30, color=PURPLE),
        )
        for t, nl in zip(tags, nls):
            t.next_to(nl, LEFT, buff=0.3)
            if t.get_left()[0] < -6.9:
                t.scale_to_fit_width(3.0).next_to(nl, LEFT, buff=0.25)
        ident = tex(r"|x| = \text{ReLU}(x) + \text{ReLU}(-x)", size=30, color=PURPLE
                    ).move_to([3.6, 2.55, 0])
        dots = VGroup(*[marker(int(c), r=0.12).move_to(nls[0].n2p(x)) for x, c in zip(xs, cls)])
        self.play(Create(nls[0]), FadeIn(tags[0]))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.05))
        need = safe_text("4 cortes en x", size=22, color=RED).next_to(nls[0], UP, buff=0.35)
        cuts = VGroup(*[DashedLine(nls[0].n2p(c) + UP * 0.3, nls[0].n2p(c) + DOWN * 0.3,
                                   color=RED) for c in (-2.7, -1.3, 1.3, 2.7)])
        self.play(FadeIn(need), Create(cuts))
        # primer doblez
        d1 = dots.copy()
        self.play(FadeIn(nls[1]), FadeIn(tags[1]), FadeIn(ident), d1.animate.shift(DOWN * (rows_y[0] - rows_y[1])))
        self.play(*[d.animate.move_to(nls[1].n2p(abs(x))) for d, x in zip(d1, xs)],
                  run_time=1.8, rate_func=smooth)
        cuts1 = VGroup(*[DashedLine(nls[1].n2p(c) + UP * 0.3, nls[1].n2p(c) + DOWN * 0.3,
                                    color=RED) for c in (1.3, 2.7)])
        self.play(Create(cuts1))
        # segundo doblez
        d2 = d1.copy()
        self.play(FadeIn(nls[2]), FadeIn(tags[2]), d2.animate.shift(DOWN * (rows_y[1] - rows_y[2])))
        self.play(*[d.animate.move_to(nls[2].n2p(abs(abs(x) - 2))) for d, x in zip(d2, xs)],
                  run_time=1.8, rate_func=smooth)
        cut2 = DashedLine(nls[2].n2p(0.7) + UP * 0.35, nls[2].n2p(0.7) + DOWN * 0.35,
                          color=YELLOW, stroke_width=5)
        self.play(Create(cut2))
        msg = safe_text("2 dobleces + 1 corte = 4 cortes en el original", size=25, color=YELLOW,
                        weight=BOLD).move_to([0.6, -2.95, 0])
        self.play(FadeIn(msg, shift=UP * 0.1))
