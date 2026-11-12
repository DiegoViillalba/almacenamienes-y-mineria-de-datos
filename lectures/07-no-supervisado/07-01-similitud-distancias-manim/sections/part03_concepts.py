"""Acto 3 · ¿Por qué exigir axiomas? Distancia, similitud y desempeño (14–21 min).

La desigualdad triangular permite descartar una bola entera de candidatos al
buscar vecinos (árboles de bolas, KD). Después: cuatro conceptos que en la
práctica se llaman "métrica" y no lo son — medida de conjuntos, distancia,
similitud y medida de desempeño — y el puente núcleo gaussiano s = e^{−d²/2σ²}.
"""
from __future__ import annotations

from manim import *

from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL,
                            YELLOW, card, safe_text, tex)


class ConceptsSection:
    def concepts(self) -> None:
        self.beat("14–15 min. ¿Para qué tanto rigor? Buscamos el vecino más cercano de q entre "
                  "miles de puntos. Ya encontramos uno a distancia 1.2. A la derecha hay una bola "
                  "con 30 puntos, centro c y radio 1.6. Pregunte: ¿hace falta calcular las 30 "
                  "distancias?")
        head = self.start("¿Por qué exigirlos?", "Descartar sin calcular", GREEN)
        rng = np.random.default_rng(2)
        q = np.array([-4.6, -0.9, 0])
        best = q + np.array([0.95, 0.73, 0])
        c = np.array([1.6, -0.9, 0])
        r = 1.6
        ball_pts = []
        while len(ball_pts) < 30:
            v = rng.uniform(-r, r, 2)
            if np.linalg.norm(v) < r * 0.93:
                ball_pts.append(c + np.array([v[0], v[1], 0]))
        noise = [q + np.array([*rng.uniform(-1.6, 1.6, 2), 0]) for _ in range(6)]
        noise = [p for p in noise if np.linalg.norm(p - q) > 1.4]
        dots = VGroup(*[Dot(p, radius=0.06, color=BLUE) for p in ball_pts])
        extra = VGroup(*[Dot(p, radius=0.06, color=BLUE) for p in noise])
        ball = Circle(radius=r, color=TEAL, stroke_width=3).move_to(c)
        cdot = Dot(c, radius=0.07, color=TEAL)
        cl = tex("c", size=32, color=TEAL).next_to(cdot, DOWN, buff=0.08)
        qdot = Dot(q, radius=0.12, color=YELLOW)
        ql = tex("q", size=36, color=YELLOW).next_to(qdot, LEFT, buff=0.12)
        bdot = Dot(best, radius=0.08, color=GREEN)
        bline = Line(q, best, color=GREEN, stroke_width=4)
        bl = tex(r"\delta = 1.2", size=30, color=GREEN).next_to(bline, UP, buff=0.05).shift(LEFT * 0.4)
        self.play(FadeIn(dots), FadeIn(extra), Create(ball), FadeIn(cdot), FadeIn(cl))
        self.play(GrowFromCenter(qdot), FadeIn(ql), FadeIn(bdot), Create(bline), FadeIn(bl))

        self.beat("15–16 min. Una sola distancia, d(q, c) = 6.2, basta. Para cualquier x de la bola, "
                  "la desigualdad triangular dice d(q,x) ≥ d(q,c) − d(c,x) ≥ 6.2 − 1.6 = 4.6, que "
                  "ya es mayor que 1.2. Se descartan 30 puntos sin tocarlos. Así funcionan los "
                  "árboles de bolas y KD de scikit-learn; sin el axioma 4 darían respuestas "
                  "equivocadas.")
        dqc = np.linalg.norm(c - q)
        qc = DashedLine(q, c, color=INK, stroke_width=3)
        qcl = tex(rf"d(q,c)={dqc:.1f}", size=30, color=INK).next_to(qc.get_center(), DOWN, buff=0.25)
        edge = c + r * normalize(q - c)
        gap = Line(q, edge, color=RED, stroke_width=7).set_opacity(0.85)
        self.play(Create(qc), FadeIn(qcl))
        self.play(Create(gap))
        bound = tex(r"d(q,x)\;\ge\; d(q,c)-d(c,x)\;\ge\;" + f"{dqc:.1f}" + r"-1.6=" +
                    f"{dqc - r:.1f}" + r"\;>\;\delta", size=34, color=INK).move_to([0, 1.75, 0])
        self.play(Write(bound))
        self.play(ball.animate.set_stroke(color=MUTED, opacity=0.4), dots.animate.set_opacity(0.2),
                  run_time=0.8)
        saved = safe_text("30 distancias ahorradas", size=28, color=GREEN, weight=BOLD)
        saved.next_to(ball, DOWN, buff=0.25)
        self.play(FadeIn(saved, shift=UP * 0.15))

        self.beat("16–18 min. En el software todo se llama 'metric', pero son cuatro objetos "
                  "matemáticos distintos. Una MEDIDA asigna tamaño a conjuntos (área, "
                  "probabilidad) y es aditiva. Una DISTANCIA asigna separación a pares y cumple los "
                  "axiomas. Una SIMILITUD asigna parecido a pares: más alto = más parecido, sin "
                  "axiomas obligatorios. Una medida de DESEMPEÑO evalúa un modelo completo. Pida "
                  "que clasifiquen: silueta, coseno, F1, Jaccard, área.")
        head = self.swap_heading(head, "No todo es una métrica", "Cuatro conceptos que se confunden",
                                 TEAL)
        self.clear_except(head)
        cols = [
            ("Medida", r"\mu:\ \text{conjuntos}\to[0,\infty]", "tamaño · aditiva",
             "área · probabilidad", BLUE),
            ("Distancia", r"d:\ \text{pares}\to[0,\infty)", "separación · 4 axiomas",
             "euclidiana · Manhattan · Gower", TEAL),
            ("Similitud", r"s:\ \text{pares}\to\mathbb{R}", "parecido · sin axiomas",
             "coseno · Jaccard · núcleo", PURPLE),
            ("Desempeño", r"F:\ (\hat{y},\,y)\to\mathbb{R}", "calidad de un modelo",
             "exactitud · F1 · silueta", ORANGE),
        ]
        cards = VGroup()
        for name, sig, what, ex, col in cols:
            box = card(3.05, 3.6, col)
            t = safe_text(name, size=30, color=col, weight=BOLD)
            s = tex(sig, size=26, color=INK, max_width=2.75)
            w = safe_text(what, size=19, color=MUTED, max_width=2.75)
            e = safe_text(ex, size=18, color=col, max_width=2.75)
            VGroup(t, s, w, e).arrange(DOWN, buff=0.32).move_to(box)
            cards.add(VGroup(box, t, s, w, e))
        cards.arrange(RIGHT, buff=0.22).move_to([0, -0.35, 0])
        self.play(LaggedStart(*[FadeIn(cd, shift=UP * 0.2) for cd in cards], lag_ratio=0.25))
        warn = safe_text("scikit-learn llama metric= a las tres últimas", size=22, color=RED)
        warn.next_to(cards, DOWN, buff=0.3)
        self.play(FadeIn(warn))

        self.beat("18–19 min. Desempeño no es distancia: la silueta USA distancias entre casos, "
                  "pero califica una partición entera; el F1 compara predicciones con la verdad. "
                  "Nunca responden '¿qué tan lejos están estos dos casos?'. Y de similitud a "
                  "distancia hay puentes; el más importante es el núcleo gaussiano.")
        self.play(FadeOut(warn), cards[0].animate.set_opacity(0.25), cards[3].animate.set_opacity(0.25))
        bridge = Arrow(cards[1].get_bottom() + DOWN * 0.05, cards[2].get_bottom() + DOWN * 0.05,
                       path_arc=PI / 2.4, color=INK, stroke_width=3, buff=0.1)
        bl = safe_text("¿cómo pasar de una a otra?", size=22, color=INK).next_to(bridge, DOWN,
                                                                                buff=0.05)
        self.play(Create(bridge), FadeIn(bl))

        self.beat("19–21 min. El núcleo gaussiano convierte distancia en similitud: vale 1 en "
                  "d = 0 y decae a 0. σ es el 'alcance' de la vecindad: mueva σ y pregunte qué "
                  "pasa con un punto a distancia 2. Esta campana reaparece en t-SNE y, con otra "
                  "forma, en UMAP. El camino inverso, d = 1 − s, NO garantiza la triangular: lo "
                  "veremos con el coseno.")
        head = self.swap_heading(head, "El puente", "De distancia a similitud", PURPLE)
        self.clear_except(head)
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 1.1, 0.5], x_length=7.0, y_length=3.8, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 1.6, "include_numbers": True,
                               "font_size": 24}).move_to([-2.3, -0.55, 0])
        xl = tex(r"d(\mathbf{x},\mathbf{y})", size=30, color=TEAL).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = tex(r"s", size=34, color=PURPLE).next_to(ax.y_axis, UP, buff=0.12)
        sig = ValueTracker(1.0)
        curve = always_redraw(lambda: ax.plot(
            lambda d: np.exp(-d * d / (2 * sig.get_value() ** 2)), x_range=[0, 4, 0.02],
            color=PURPLE, stroke_width=5))
        probe = always_redraw(lambda: Dot(ax.c2p(2, np.exp(-4 / (2 * sig.get_value() ** 2))),
                                          radius=0.09, color=YELLOW))
        s_eq = tex(r"s(2) =", size=30, color=YELLOW)
        s_num = DecimalNumber(0.14, num_decimal_places=2, font_size=30, color=YELLOW)

        def upd_probe(m):
            s_num.set_value(np.exp(-4 / (2 * sig.get_value() ** 2)))
            s_num.next_to(s_eq, RIGHT, buff=0.12)
            m.next_to(probe, UR, buff=0.08)

        probe_l = VGroup(s_eq, s_num)
        s_num.next_to(s_eq, RIGHT, buff=0.12)
        probe_l.add_updater(upd_probe)
        sig_eq = tex(r"\sigma =", size=36, color=PURPLE).move_to([3.3, -1.6, 0])
        sig_num = DecimalNumber(1.0, num_decimal_places=2, font_size=36, color=PURPLE)
        sig_num.next_to(sig_eq, RIGHT, buff=0.15)
        sig_num.add_updater(lambda m: m.set_value(sig.get_value()).next_to(sig_eq, RIGHT, buff=0.15))
        sig_l = VGroup(sig_eq, sig_num)
        formula = tex(r"s(\mathbf{x},\mathbf{y}) = \exp\!\left(-\frac{d(\mathbf{x},\mathbf{y})^2}"
                      r"{2\sigma^2}\right)", size=40, color=INK).move_to([3.4, 1.3, 0])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), Write(formula), FadeIn(sig_l))
        self.play(FadeIn(probe), FadeIn(probe_l))
        self.play(sig.animate.set_value(0.5), run_time=1.5)
        self.play(sig.animate.set_value(2.0), run_time=2.0)
        self.play(sig.animate.set_value(1.0), run_time=1.0)
        inv = tex(r"d = 1 - s\ \ \text{(no siempre m\'etrica)}", size=32, color=RED)
        inv.next_to(formula, DOWN, buff=0.5)
        self.play(Write(inv))
        for m in (curve, probe, probe_l, sig_num):
            m.clear_updaters()
