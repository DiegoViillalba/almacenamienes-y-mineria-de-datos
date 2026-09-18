"""Estados 6–12: Gini–Simpson como probabilidad de desacuerdo y área complementaria del cuadrado unitario."""

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



class GiniSimpsonSection:
    """Estados 6–12: Gini–Simpson como probabilidad de desacuerdo y área complementaria del cuadrado unitario."""

    def gini_simpson_story(self) -> None:
        header = self.heading("Gini–Simpson", "Dos extracciones: ¿coinciden o difieren?", ORANGE)
        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], width=5.2, height=2.6)
        chart.move_to([-3.3, -0.35, 0])
        urn_one = VGroup(
            Circle(radius=0.62, color=BLUE, fill_color=BLUE, fill_opacity=0.09),
            safe_text("1", size=29, color=INK, weight=BOLD),
        ).move_to([2.0, 0.25, 0])
        urn_two = VGroup(
            Circle(radius=0.62, color=PURPLE, fill_color=PURPLE, fill_opacity=0.09),
            safe_text("2", size=29, color=INK, weight=BOLD),
        ).move_to([4.05, 0.25, 0])
        urn_one[1].move_to(urn_one[0])
        urn_two[1].move_to(urn_two[0])
        independence = safe_text(
            "independientes · con reemplazo",
            size=21,
            color=MUTED,
        ).move_to([3.02, -0.82, 0])
        choice = VGroup(
            pill("MISMA", RED, 1.9),
            safe_text("o", size=24, color=MUTED),
            pill("DISTINTA", ORANGE, 2.15),
        ).arrange(RIGHT, buff=0.20).move_to([3.0, -1.72, 0])
        self.play(FadeIn(header), FadeIn(chart))
        self.play(GrowFromCenter(urn_one), GrowFromCenter(urn_two), FadeIn(independence), FadeIn(choice))
        self.pause(
            "Pregunte qué evento conviene calcular primero: misma categoría o distinta. "
            "Declare desde ahora dos extracciones independientes, equivalentes a reemplazo."
        )

        self.play(FadeOut(chart), FadeOut(urn_one), FadeOut(urn_two), FadeOut(independence), FadeOut(choice))
        grid, diagonal = probability_grid([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], CAT_COLORS[:3])
        grid.move_to([-2.45, -0.35, 0])
        axes_labels = VGroup(
            safe_text("extracción 1", size=20, color=BLUE, weight=BOLD).next_to(grid, DOWN, buff=0.18),
            safe_text("extracción 2", size=20, color=PURPLE, weight=BOLD).rotate(PI / 2).next_to(grid, LEFT, buff=0.18),
        )
        same = safe_text("diagonal = misma categoría", size=29, color=RED, weight=BOLD)
        same.move_to([3.1, 0.65, 0])
        different = safe_text("resto = categorías distintas", size=27, color=ORANGE, weight=BOLD)
        different.move_to([3.1, -0.25, 0])
        self.play(FadeIn(grid[0]), FadeIn(grid[1]), FadeIn(grid[3]), FadeIn(axes_labels))
        self.play(LaggedStart(*[GrowFromCenter(cell) for cell in diagonal], lag_ratio=0.16), Write(same))
        self.play(FadeIn(different))
        self.pause(
            "Lea el cuadrado como todos los pares posibles. Pida estimar si el área "
            "diagonal —los pares iguales— supera la mitad antes de poner cifras."
        )

        area_labels = VGroup()
        for cell, tex in zip(diagonal, [r"\frac14", r"\frac1{36}", r"\frac19"]):
            label = MathTex(tex, font_size=28, color=INK).move_to(cell)
            if label.width > cell.width * 0.78:
                label.scale_to_fit_width(cell.width * 0.78)
            area_labels.add(label)
        sum_formula = MathTex(
            r"\sum_{k=1}^{3}p_k^2=\frac14+\frac1{36}+\frac19=\frac7{18}=0.3889",
            font_size=36,
            color=INK,
        ).move_to([2.85, -1.45, 0])
        if sum_formula.width > 6.0:
            sum_formula.scale_to_fit_width(6.0)
        self.play(FadeOut(same), FadeOut(different), FadeIn(area_labels), Write(sum_formula))
        self.pause(
            "Nombre los sucesos AA, BB y CC. Como son excluyentes, sus áreas se "
            "suman: 1/4 + 1/36 + 1/9 = 7/18."
        )

        d_formula = MathTex(
            r"D=1-\sum_{k=1}^{K}p_k^2=1-\frac7{18}=\frac{11}{18}=0.6111",
            font_size=42,
            color=ORANGE,
        ).move_to([2.75, 0.55, 0])
        if d_formula.width > 6.2:
            d_formula.scale_to_fit_width(6.2)
        interpretation = safe_text(
            "61.11% de probabilidad de categorías distintas",
            size=26,
            color=GREEN,
            weight=BOLD,
            max_width=6.1,
        ).move_to([2.75, -0.35, 0])
        orange_frame = SurroundingRectangle(grid[0], color=ORANGE, buff=0.03, stroke_width=4)
        self.play(Create(orange_frame), Write(d_formula), FadeIn(interpretation))
        self.pause(
            "Modele la frase completa: D=0.6111 es la probabilidad de que dos "
            "extracciones independientes pertenezcan a categorías distintas. "
            "No diga simplemente ‘61.11% de diversidad’."
        )

        self.clear_content()
        header = self.heading("Extremo 1", "Toda la masa en una categoría", RED)
        chart = distribution_chart([1.0, 0.0, 0.0], ["A", "B", "C"], width=5.2, height=2.8)
        chart.move_to([-3.2, -0.25, 0])
        full_square = Square(
            side_length=3.0,
            color=RED,
            fill_color=RED,
            fill_opacity=0.32,
        ).move_to([3.1, -0.35, 0])
        full_label = MathTex(r"\sum p_k^2=1", font_size=38, color=INK).move_to(full_square)
        zero = MathTex(r"D=1-1=0", font_size=45, color=RED).next_to(full_square, DOWN, buff=0.28)
        self.play(FadeIn(header), FadeIn(chart), GrowFromCenter(full_square), Write(full_label))
        self.play(Write(zero))
        self.pause(
            "Pregunte por qué tres nombres de categoría no garantizan diversidad "
            "observada. Mantenga K=3: B y C existen, pero tienen frecuencia cero."
        )

        uniform_chart = distribution_chart([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"], width=5.2, height=2.8)
        uniform_chart.move_to([-3.2, -0.25, 0])
        uniform_grid, _ = probability_grid([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"], CAT_COLORS[:3], side=3.0)
        uniform_grid.move_to([3.1, -0.35, 0])
        uniform_header = self.heading("Extremo 2", "Misma masa en cada categoría", ORANGE)
        max_formula = MathTex(
            r"D_{\max}=1-\frac1K=1-\frac13=\frac23",
            font_size=43,
            color=ORANGE,
        ).next_to(uniform_grid, DOWN, buff=0.25)
        self.play(
            Transform(header, uniform_header),
            Transform(chart, uniform_chart),
            ReplacementTransform(full_square, uniform_grid),
            FadeOut(full_label),
            FadeOut(zero),
        )
        self.play(Write(max_formula))
        self.pause(
            "Aclare que el máximo ocurre bajo uniformidad y depende de K. Para "
            "K finito D no llega a 1; con K=3, el máximo es 2/3."
        )

        self.clear_content()
        header = self.heading("Escala interpretable", "¿Cuántas categorías igualmente frecuentes?", GREEN)
        diagonal_card = metric_card("PROBABILIDAD DE COINCIDIR", "Σp² = 7/18", ORANGE, 3.5)
        diagonal_card.move_to([-3.5, -0.10, 0])
        arrow = Arrow([-1.35, -0.10, 0], [0.65, -0.10, 0], color=MUTED, stroke_width=4)
        inverse = metric_card("NÚMERO EFECTIVO · SIMPSON", "1 / Σp² = 18/7 = 2.57", GREEN, 4.4)
        inverse.move_to([3.25, -0.10, 0])
        warning = safe_text(
            "No es 1/D · tampoco es el Gini de desigualdad",
            size=23,
            color=RED,
            weight=BOLD,
            max_width=9.2,
        ).move_to([0, -1.55, 0])
        self.play(FadeIn(header), FadeIn(diagonal_card), GrowArrow(arrow), FadeIn(inverse))
        self.play(FadeIn(warning))
        self.pause(
            "Traduzca el índice: esta composición equivale a 2.57 categorías "
            "igualmente frecuentes según Simpson. Prevenga el error 1/D: se "
            "invierte la probabilidad de coincidencia, suma de p al cuadrado."
        )
        self.clear_content()

