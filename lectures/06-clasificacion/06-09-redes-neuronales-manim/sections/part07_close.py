"""Acto 7 · Ejercicio y cierre (56–60 min).

XOR con dos neuronas ReLU: el problema que detuvo al perceptrón en 1969,
resuelto a mano. Cierre con cuatro ideas y ticket de salida.
"""
from __future__ import annotations

from manim import *

from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, marker, pill, safe_text, table, tex)

XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def xor_net(x1, x2):
    h1 = max(0, x1 + x2)
    h2 = max(0, x1 + x2 - 1)
    return h1, h2, h1 - 2 * h2


class CloseSection:
    def close(self) -> None:
        self.exercise()
        self.closing()

    # ------------------------------------------------------------------
    def exercise(self) -> None:
        self.beat("56–58 min. Ejercicio individual (2 min). XOR: ninguna recta separa (0,1) y "
                  "(1,0) de (0,0) y (1,1); es el ejemplo con el que Minsky y Papert frenaron al "
                  "perceptrón. Con dos neuronas ReLU: h₁ = ReLU(x₁ + x₂), "
                  "h₂ = ReLU(x₁ + x₂ − 1), ŷ = h₁ − 2h₂. Que completen la tabla y digan qué "
                  "recta representa cada neurona.")
        self.start("Tu turno", "XOR con dos neuronas ReLU", YELLOW)
        ax = NumberPlane(x_range=[-0.5, 1.5, 0.5], y_range=[-0.5, 1.5, 0.5], x_length=3.6,
                         y_length=3.6, background_line_style={"stroke_color": GRID,
                                                              "stroke_width": 1,
                                                              "stroke_opacity": 0.4},
                         axis_config={"stroke_color": GRID}).move_to([-4.6, -0.4, 0])
        pts = VGroup(*[marker(1 if y else 0, r=0.16).move_to(ax.c2p(*x)) for x, y in XOR])
        labs = VGroup(*[MathTex(f"({x[0]},{x[1]})", font_size=24, color=MUTED).next_to(
            p, DL if x[0] == 0 else DR, buff=0.05) for (x, _), p in zip(XOR, pts)])
        self.play(Create(ax), FadeIn(pts), FadeIn(labs))
        net = VGroup(
            tex(r"h_1 = \text{ReLU}(x_1 + x_2)", size=32, color=TEAL),
            tex(r"h_2 = \text{ReLU}(x_1 + x_2 - 1)", size=32, color=TEAL),
            tex(r"\hat y = h_1 - 2h_2", size=32, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([1.0, 1.55, 0])
        rows = [["x₁", "x₂", "h₁", "h₂", "ŷ"]] + [[str(x[0]), str(x[1]), "?", "?", "?"]
                                                 for x, _ in XOR]
        tb = table(rows, col_widths=[1.0] * 5, size=24, row_h=0.5,
                   colors=[[INK, INK, MUTED, MUTED, MUTED]] * 4).move_to([1.3, -1.25, 0])
        timer = pill("2 minutos", YELLOW, size=24).move_to([5.3, 1.55, 0])
        self.play(FadeIn(net, shift=LEFT * 0.1))
        self.play(FadeIn(tb), FadeIn(timer))

        self.beat("58 min. Solución: (0,0) → 0, 0, 0; (0,1) y (1,0) → 1, 0, 1; (1,1) → 2, 1, 0. "
                  "h₁ y h₂ son dos rectas paralelas (x₁ + x₂ = 0 y x₁ + x₂ = 1, en teal); la "
                  "segunda neurona «dobla» la función y le quita a (1,1) lo que h₁ le dio. Con "
                  "umbral 0.5 la región naranja es la franja 0.5 < x₁ + x₂ < 1.5.")
        sol_rows = [["x₁", "x₂", "h₁", "h₂", "ŷ"]]
        cols = []
        for x, y in XOR:
            h1, h2, yh = xor_net(*x)
            sol_rows.append([str(x[0]), str(x[1]), str(h1), str(h2), str(yh)])
            cols.append([INK, INK, TEAL, TEAL, ORANGE if yh else BLUE])
        sol = table(sol_rows, col_widths=[1.0] * 5, size=24, row_h=0.5,
                    colors=cols).move_to(tb)
        # ŷ > 0.5  ⇔  0.5 < x₁ + x₂ < 1.5 (franja diagonal)
        strip = Polygon(ax.c2p(-0.5, 1.0), ax.c2p(1.0, -0.5), ax.c2p(1.5, -0.5),
                        ax.c2p(1.5, 0.0), ax.c2p(0.0, 1.5), ax.c2p(-0.5, 1.5),
                        stroke_width=0, fill_color=ORANGE, fill_opacity=0.22)
        # rectas de las neuronas: x₁ + x₂ = 0 y x₁ + x₂ = 1
        l1 = Line(ax.c2p(-0.5, 0.5), ax.c2p(0.5, -0.5), color=TEAL, stroke_width=4)
        l2 = Line(ax.c2p(-0.5, 1.5), ax.c2p(1.5, -0.5), color=TEAL, stroke_width=4)
        self.play(FadeOut(timer), ReplacementTransform(tb, sol))
        self.play(FadeIn(strip), Create(l1), Create(l2))
        self.bring_to_front(pts)

    # ------------------------------------------------------------------
    def closing(self) -> None:
        self.beat("59–60 min. Cierre con cuatro ideas y el ticket de salida.")
        self.start("Cierre", "Lo que nos llevamos", GREEN)
        ideas = VGroup(*[VGroup(safe_text(n, size=30, color=c, weight=BOLD),
                                safe_text(t, size=23)).arrange(RIGHT, buff=0.3)
                         for n, t, c in [
                             ("neurona", "suma ponderada + activación = un hiperplano", TEAL),
                             ("ReLU", "sin no linealidad, cualquier red es una recta", PURPLE),
                             ("red", "muchas rectas y pliegues → polígonos → Baarle", YELLOW),
                             ("backprop", "regla de la cadena hacia atrás + descenso", RED),
                             ("CNN", "filtros locales con pesos compartidos", PINK),
                         ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = VGroup(
            safe_text("Ticket de salida", size=24, color=YELLOW, weight=BOLD),
            safe_text("¿Por qué una red de 100 capas sin activación no puede aprender Baarle?",
                      size=21, max_width=11.5),
            safe_text("¿Cuántas neuronas ReLU necesitarías, como mínimo, para un enclave "
                      "triangular?", size=21, max_width=11.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = card(ticket.width + 0.6, ticket.height + 0.4, YELLOW)
        tg = VGroup(box, ticket.move_to(box)).move_to([0, -2.3, 0])
        self.play(FadeIn(tg, shift=UP * 0.1))
