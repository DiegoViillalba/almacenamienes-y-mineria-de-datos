"""Acto 2 · La sabiduría de muchos (9–17 min).

Cinco jueces que aciertan 70 %: el histograma binomial muestra por qué la
mayoría acierta 83.7 %. La curva «exactitud de la mayoría contra número de
jueces» sube hacia 1 sólo si cada juez es mejor que el azar. Si los jueces se
copian, no hay ganancia: la varianza de un promedio tiene un piso ρσ².
"""
from __future__ import annotations

from manim import *

from sections.data import binom_terms, majority, var_mean
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, safe_text, tex)


def judge(color=BLUE, label: str = "70 %") -> VGroup:
    head = Circle(radius=0.22, color=color, fill_color=color, fill_opacity=0.25, stroke_width=3)
    body = Arc(radius=0.36, start_angle=PI * 0.15, angle=PI * 0.7, color=color, stroke_width=4)
    body.next_to(head, DOWN, buff=0.04)
    t = safe_text(label, size=16, color=MUTED).next_to(body, DOWN, buff=0.08)
    return VGroup(head, body, t)


class CrowdSection:
    def crowd(self) -> None:
        self.condorcet()
        self.more_judges()
        self.correlation()

    # ------------------------------------------------------------------
    def condorcet(self) -> None:
        self.beat("9–11 min. Cinco jueces independientes; cada uno acierta 70 % de las veces. "
                  "Decidimos por mayoría (3 de 5). Pregunte: ¿la mayoría acierta más, menos o "
                  "igual que un juez? Las barras son la probabilidad de que acierten exactamente "
                  "k jueces; la mayoría acierta cuando k ≥ 3 (barras verdes): 0.837.")
        self.start("La sabiduría de muchos", "Cinco jueces que aciertan 70 %", PURPLE)
        judges = VGroup(*[judge() for _ in range(5)]).arrange(RIGHT, buff=0.35)
        judges.move_to([-3.6, 1.2, 0])
        vote = safe_text("decisión = lo que diga la mayoría", size=21, color=MUTED
                         ).next_to(judges, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(j, shift=UP * 0.1) for j in judges], lag_ratio=0.1))
        self.play(FadeIn(vote))
        probs = binom_terms(5, 0.7)
        ax = Axes(x_range=[-0.5, 5.5, 1], y_range=[0, 0.4, 0.1], x_length=5.6, y_length=3.0,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False})
        ax.move_to([3.4, -0.2, 0])
        bars = VGroup()
        for k, pk in enumerate(probs):
            col = GREEN if k >= 3 else MUTED
            b = Rectangle(width=0.62, height=max(pk * 3.0 / 0.4, 0.01), stroke_width=0,
                          fill_color=col, fill_opacity=0.85)
            b.move_to(ax.c2p(k, 0), aligned_edge=DOWN)
            v = safe_text(f"{pk:.3f}", size=16, color=INK).next_to(b, UP, buff=0.06)
            n = safe_text(str(k), size=18, color=MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.1)
            bars.add(VGroup(b, v, n))
        xl = safe_text("k = jueces que aciertan", size=18, color=MUTED).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(xl))
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(VGroup(b[1], b[2])) for b in bars], lag_ratio=0.1))
        f = tex(r"P(\text{mayor\'ia acierta}) = \sum_{k=3}^{5}\binom{5}{k}0.7^k\,0.3^{5-k}",
                size=30).move_to([-3.4, -1.0, 0])
        g = tex(r"= 0.309 + 0.360 + 0.168 = 0.837", size=32, color=GREEN
                ).next_to(f, DOWN, buff=0.3).align_to(f, LEFT)
        self.play(Write(f))
        self.play(Write(g))

    # ------------------------------------------------------------------
    def more_judges(self) -> None:
        self.beat("11–13 min. Más jueces. Con p = 0.70, 25 jueces aciertan 98 %; con p = 0.55 "
                  "también sube, más despacio; con p = 0.45 (peor que una moneda) la mayoría "
                  "se HUNDE. Teorema del jurado de Condorcet: dos condiciones hacen el trabajo, "
                  "que cada juez sea mejor que el azar y que sus errores sean independientes.")
        self.start("Jurado de Condorcet", "¿Y si son 25 jueces?", PURPLE)
        ax = Axes(x_range=[0, 52, 10], y_range=[0, 1, 0.25], x_length=8.2, y_length=4.2,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": True, "tick_size": 0.04})
        ax.move_to([-1.6, -0.4, 0])
        nums = VGroup(
            *[safe_text(str(v), size=15, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
              for v in (1, 11, 21, 31, 41, 51)],
            *[safe_text(f"{int(v * 100)} %", size=15, color=MUTED).next_to(ax.c2p(0, v), LEFT,
                                                                           buff=0.1)
              for v in (0.25, 0.5, 0.75, 1.0)])
        xl = safe_text("número de jueces (impar)", size=18, color=MUTED).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(nums), FadeIn(xl))
        half = DashedLine(ax.c2p(0, 0.5), ax.c2p(52, 0.5), color=GRID, stroke_width=2)
        self.add(half)
        legend = VGroup()
        for p, col in [(0.7, GREEN), (0.55, YELLOW), (0.45, RED)]:
            pts = [ax.c2p(n, majority(n, p)) for n in range(1, 52, 2)]
            curve = VMobject(color=col, stroke_width=4).set_points_smoothly(pts)
            entry = safe_text(f"cada juez {int(p * 100)} %", size=20, color=col, weight=BOLD)
            legend.add(entry)
            self.play(Create(curve), FadeIn(entry), run_time=1.0)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.6, 1.0, 0])
        for n, col in [(5, GREEN), (25, GREEN)]:
            d = Dot(ax.c2p(n, majority(n, 0.7)), radius=0.08, color=YELLOW)
            t = safe_text(f"{n}: {majority(n, 0.7):.3f}", size=17, color=YELLOW
                          ).next_to(d, DOWN + RIGHT * 0.3, buff=0.08)
            self.play(FadeIn(d), FadeIn(t), run_time=0.4)
        cond = VGroup(safe_text("Dos condiciones", size=22, color=INK, weight=BOLD),
                      safe_text("1 · mejor que el azar", size=20),
                      safe_text("2 · errores independientes", size=20, color=YELLOW)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.6, -1.3, 0])
        self.play(FadeIn(cond))

    # ------------------------------------------------------------------
    def correlation(self) -> None:
        self.beat("13–17 min. ¿Y si los cinco jueces copian al mismo? La mayoría acierta 70 %: "
                  "votar no sirve. Para números: B modelos con varianza σ² y correlación ρ. "
                  "Promediar borra el segundo término, pero NO el primero: la correlación pone un "
                  "piso. Toda la estrategia de los bosques aleatorios es bajar ρ. Y como "
                  "promediar no cambia el valor esperado, el SESGO se queda igual: se promedian "
                  "árboles profundos (poco sesgo, mucha varianza).")
        self.start("Diversidad", "La correlación pone un piso", PURPLE)
        f = tex(r"\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B}\hat f_b\Big) = \rho\,\sigma^2"
                r" + \frac{1-\rho}{B}\,\sigma^2", size=36).move_to([0, 1.95, 0])
        self.play(Write(f))
        ax = Axes(x_range=[0, 50, 10], y_range=[0, 1, 0.25], x_length=7.6, y_length=3.6,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": True, "tick_size": 0.04})
        ax.move_to([-1.9, -0.95, 0])
        nums = VGroup(
            *[safe_text(str(v), size=15, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
              for v in (1, 10, 20, 30, 40, 50)],
            *[safe_text(f"{v:g}σ²", size=15, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1)
              for v in (0.5, 1.0)])
        xl = safe_text("B = número de modelos", size=18, color=MUTED).next_to(ax, DOWN, buff=0.42)
        self.play(Create(ax), FadeIn(nums), FadeIn(xl))
        legend = VGroup()
        for rho, col in [(0.6, RED), (0.3, YELLOW), (0.0, GREEN)]:
            curve = ax.plot(lambda b, r=rho: var_mean(b, r), x_range=[1, 50, 0.25], color=col,
                            stroke_width=4)
            floor = DashedLine(ax.c2p(0, rho), ax.c2p(50, rho), color=col, stroke_width=2) \
                if rho > 0 else VMobject()
            entry = safe_text(f"ρ = {rho:g}" + (f"  → piso {rho:g}σ²" if rho else "  → 0"),
                              size=20, color=col, weight=BOLD)
            legend.add(entry)
            self.play(Create(curve), Create(floor), FadeIn(entry), run_time=1.0)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, 0.2, 0]).align_to(
            [2.45, 0, 0], LEFT)
        notes = VGroup(safe_text("promediar no cambia el sesgo:", size=20, color=MUTED),
                       safe_text("promediar árboles PROFUNDOS", size=21, color=PURPLE,
                                 weight=BOLD)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(legend, DOWN,
                                                                           buff=0.5
                                                                           ).align_to(legend, LEFT)
        self.play(FadeIn(notes))
