"""Estados 20–23: Comparación entre escenarios con mismo K, números de Hill y transición hacia datos numéricos."""

from __future__ import annotations

import numpy as np
from manim import *
from manim_slides import Slide

from sections.theme import (
    BG, INK, MUTED, GRID, BLUE, ORANGE, PURPLE, YELLOW, GREEN, RED,
    CAT_COLORS, FONT, LOGO_PATH, safe_text, pill, metric_card,
    distribution_chart, token_row, probability_grid
)
from sections.base import BaseSlide



class ComparisonSection:
    """Estados 20–23: Comparación entre escenarios con mismo K, números de Hill y transición hacia datos numéricos."""

    def comparison_and_transition(self) -> None:
        header = self.heading("Comparación", "Mismo K; movamos la masa", YELLOW)
        labels = ["A", "B", "C", "D"]
        dominant = [0.82, 0.08, 0.06, 0.04]
        chart = distribution_chart(dominant, labels, width=7.1, height=3.15)
        chart.move_to([-2.6, -0.25, 0])
        hidden_metrics = VGroup(
            metric_card("GINI–SIMPSON", "?", ORANGE, 3.0),
            metric_card("SHANNON NORM.", "?", PURPLE, 3.0),
            metric_card("EFECTIVAS · SHANNON", "?", GREEN, 3.0),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        prompt = safe_text(
            "Predice: ¿cómo cambiará cada medidor?",
            size=26,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), FadeIn(chart), FadeIn(hidden_metrics), Write(prompt))
        self.pause(
            "Antes de revelar cifras, pida decidir si cada medidor sube, baja o queda "
            "igual cuando el reparto se equilibre. Compare cada medida consigo misma."
        )

        intermediate = [0.50, 0.25, 0.15, 0.10]
        intermediate_chart = distribution_chart(intermediate, labels, width=7.1, height=3.15)
        intermediate_chart.move_to([-2.6, -0.25, 0])
        metrics_mid = VGroup(
            metric_card("GINI–SIMPSON", "0.316  →  0.655", ORANGE, 3.1),
            metric_card("SHANNON NORM.", "0.478  →  0.871", PURPLE, 3.1),
            metric_card("EFECTIVAS · SHANNON", "1.94  →  3.35", GREEN, 3.1),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        self.play(Transform(chart, intermediate_chart), Transform(hidden_metrics, metrics_mid), FadeOut(prompt), run_time=1.35)
        moved = safe_text(
            "masa: dominante → categorías minoritarias",
            size=25,
            color=INK,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(Write(moved))
        self.pause(
            "Pida señalar la redistribución dentro del dibujo: salió masa de A "
            "y llegó a categorías minoritarias. Los tres resúmenes aumentan."
        )

        uniform = [0.25, 0.25, 0.25, 0.25]
        uniform_chart = distribution_chart(uniform, labels, width=7.1, height=3.15)
        uniform_chart.move_to([-2.6, -0.25, 0])
        metrics_uniform = VGroup(
            metric_card("GINI–SIMPSON", "D = 0.750", ORANGE, 3.1),
            metric_card("SHANNON NORM.", "Hnorm = 1.000", PURPLE, 3.1),
            metric_card("EFECTIVAS · SHANNON", "exp(H) = 4.00", GREEN, 3.1),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        conclusion = safe_text(
            "equilibrio máximo para K = 4",
            size=27,
            color=GREEN,
            weight=BOLD,
        ).move_to(moved)
        self.play(Transform(chart, uniform_chart), Transform(hidden_metrics, metrics_uniform), Transform(moved, conclusion), run_time=1.30)
        self.pause(
            "Concluya: con el mismo K, uniformidad maximiza ambos índices y ambos "
            "números efectivos valen cuatro. Un índice alto no significa mejor desempeño."
        )

        self.clear_content()
        header = self.heading("Cambio de pregunta", "De composición a respuesta numérica", GREEN)
        categorical = distribution_chart(uniform, labels, width=6.2, height=2.8)
        categorical.move_to([-3.25, -0.25, 0])
        divider = Arrow([-0.10, -0.15, 0], [1.30, -0.15, 0], color=MUTED, stroke_width=4)
        lanes = VGroup()
        for idx, (label, vals, color) in enumerate(zip(["A", "B", "C"], [[4, 5], [8, 9], [5, 5]], CAT_COLORS[:3])):
            y = 0.85 - idx * 0.95
            axis = NumberLine(x_range=[3, 10, 1], length=4.0, include_numbers=False, color=GRID)
            axis.move_to([4.0, y, 0])
            lane_label = safe_text(label, size=22, color=color, weight=BOLD).next_to(axis, LEFT, buff=0.18)
            dots = VGroup(*[Dot(axis.n2p(v), radius=0.10, color=color) for v in vals])
            lanes.add(VGroup(axis, lane_label, dots))
        left_q = safe_text("reparto de categorías", size=24, color=ORANGE, weight=BOLD).move_to([-3.25, -2.10, 0])
        right_q = safe_text("separación entre grupos", size=24, color=GREEN, weight=BOLD).move_to([4.0, -2.10, 0])
        self.play(FadeIn(header), FadeIn(categorical), FadeIn(left_q))
        self.play(GrowArrow(divider), LaggedStart(*[FadeIn(lane) for lane in lanes], lag_ratio=0.13), FadeIn(right_q))
        self.pause(
            "Nombre el cambio: ahora las categorías definen grupos y la respuesta es "
            "numérica. Ya no preguntamos cómo se reparte la masa, sino cuánto se separan."
        )
        self.clear_content()

