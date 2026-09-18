"""Estados 1–5: Portada, las dos preguntas de heterogeneidad y reparto inicial de clase."""

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



class OpeningSection:
    """Estados 1–5: Portada, las dos preguntas de heterogeneidad y reparto inicial de clase."""

    def opening_and_distribution(self) -> None:
        eyebrow = safe_text(
            "ANÁLISIS EXPLORATORIO · REPASO APLICADO",
            size=23,
            color=BLUE,
            weight=BOLD,
        )
        title = safe_text(
            "Medidas de heterogeneidad",
            size=60,
            color=INK,
            weight=BOLD,
            max_width=12.5,
        )
        subtitle = safe_text(
            "del reparto de la masa a la separación entre grupos",
            size=30,
            color=MUTED,
            max_width=11.8,
        )
        visual = MathTex(
            r"p_1,\ldots,p_K\quad\longrightarrow\quad D,\ H,\ \eta^2",
            font_size=48,
            color=INK,
        )
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text(
            "Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED
        )
        identity = VGroup(course, author).arrange(DOWN, buff=0.08)
        logo = self.logo_lockup(4.0, 1.15)
        cover = Group(eyebrow, title, subtitle, visual, identity, logo).arrange(DOWN, buff=0.19)

        self.play(FadeIn(eyebrow, shift=UP * 0.15), Write(title), run_time=0.95)
        self.play(FadeIn(subtitle), Write(visual), FadeIn(identity), FadeIn(logo), run_time=0.85)
        self.pause(
            "Portada. Recuerde que el grupo ya estudió el tema: hoy justificará "
            "cada medida mediante objetos y decisiones, no mediante memorización."
        )

        self.clear_content()
        self.add_branding()
        header = self.heading("Dos preguntas", "¿Reparto o separación?", YELLOW)
        left_box = RoundedRectangle(
            width=5.6, height=3.7, corner_radius=0.18, color=BLUE, fill_color=BLUE, fill_opacity=0.06
        ).move_to([-3.25, -0.15, 0])
        right_box = RoundedRectangle(
            width=5.6, height=3.7, corner_radius=0.18, color=GREEN, fill_color=GREEN, fill_opacity=0.06
        ).move_to([3.25, -0.15, 0])
        left_title = safe_text("Variable categórica", size=26, color=BLUE, weight=BOLD)
        left_title.next_to(left_box, UP, buff=-0.48)
        mini_bars = distribution_chart([0.55, 0.30, 0.15], ["A", "B", "C"], width=3.7, height=2.0)
        mini_bars.move_to(left_box).shift(DOWN * 0.20)
        left_q = safe_text("¿cómo se reparte?", size=27, color=INK, weight=BOLD)
        left_q.next_to(left_box, DOWN, buff=-0.55)
        right_title = safe_text("Respuesta numérica", size=26, color=GREEN, weight=BOLD)
        right_title.next_to(right_box, UP, buff=-0.48)
        lanes = VGroup()
        for idx, (ys, color) in enumerate(zip([0.75, -0.05, -0.85], CAT_COLORS[:3])):
            line = NumberLine(x_range=[0, 10, 2], length=3.9, include_numbers=False, color=GRID)
            line.move_to([3.25, ys, 0])
            dots = VGroup(*[Dot(line.n2p(v), radius=0.07, color=color) for v in ([2, 3], [5, 7], [6, 8])[idx]])
            lanes.add(VGroup(line, dots))
        right_q = safe_text("¿cuánto se separan?", size=27, color=INK, weight=BOLD)
        right_q.next_to(right_box, DOWN, buff=-0.55)
        self.play(FadeIn(header), Create(left_box), Create(right_box))
        self.play(FadeIn(left_title), FadeIn(mini_bars), FadeIn(left_q))
        self.play(FadeIn(right_title), LaggedStart(*[FadeIn(lane) for lane in lanes], lag_ratio=0.15), FadeIn(right_q))
        self.pause(
            "Pida un ejemplo de cada pregunta. No nombre aún Gini–Simpson, "
            "Shannon ni eta cuadrada; haga que distingan composición de respuesta numérica."
        )

        self.clear_content()
        header = self.heading("Predicción", "¿Qué tan fácil es adivinar la categoría?", YELLOW)
        color_map = {"A": BLUE, "B": PURPLE, "C": GREEN}
        labels = ["A", "A", "A", "B", "C", "C"]
        tokens = token_row(labels, color_map, y=-0.05)
        cover_token = RoundedRectangle(
            width=1.05, height=1.05, corner_radius=0.15, color=YELLOW, fill_color=YELLOW, fill_opacity=0.12
        ).move_to([0, -1.55, 0])
        question = safe_text("?", size=42, color=YELLOW, weight=BOLD).move_to(cover_token)
        prompt = safe_text("baja · media · alta", size=29, color=MUTED).next_to(cover_token, DOWN, buff=0.30)
        self.play(FadeIn(header), LaggedStart(*[GrowFromCenter(t) for t in tokens], lag_ratio=0.10))
        self.play(Create(cover_token), Write(question), FadeIn(prompt))
        self.pause(
            "Solicite una predicción cualitativa: diversidad baja, media o alta. "
            "Pida justificarla sin calcular y aclare que adivinar es sólo una intuición."
        )

        bins = VGroup()
        target_positions: list[np.ndarray] = []
        counts = [3, 1, 2]
        for idx, (label, count, color) in enumerate(zip(["A", "B", "C"], counts, CAT_COLORS)):
            x = [-3.2, 0, 3.2][idx]
            box = RoundedRectangle(
                width=2.2, height=2.7, corner_radius=0.15, color=color, fill_color=color, fill_opacity=0.08
            ).move_to([x, -0.25, 0])
            label_mob = safe_text(label, size=28, color=color, weight=BOLD).next_to(box, DOWN, buff=0.16)
            count_mob = safe_text(f"f = {count}", size=24, color=INK).next_to(box, UP, buff=0.12)
            bins.add(VGroup(box, label_mob, count_mob))
            for level in range(count):
                target_positions.append(np.array([x, -1.05 + level * 0.58, 0]))
        ordered_targets = target_positions[:3] + target_positions[3:4] + target_positions[4:]
        self.play(FadeOut(cover_token), FadeOut(question), FadeOut(prompt), FadeIn(bins))
        self.play(*[token.animate.move_to(pos) for token, pos in zip(tokens, ordered_targets)], run_time=1.2)
        self.pause(
            "Lea los conteos 3, 1 y 2. Pida convertirlos mentalmente a proporciones "
            "y comprobar que los conteos suman seis."
        )

        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"])
        chart.shift(DOWN * 0.35)
        proportions = MathTex(
            r"p=\left(\frac12,\frac16,\frac13\right)", font_size=43, color=INK
        ).next_to(header, DOWN, buff=0.30)
        self.play(FadeOut(tokens), FadeOut(bins), FadeIn(chart), Write(proportions))
        uniform_chart = distribution_chart([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"])
        uniform_chart.shift(DOWN * 0.35)
        uniform_formula = MathTex(
            r"(3,1,2)\ \longrightarrow\ (2,2,2)", font_size=38, color=GREEN
        ).move_to(proportions)
        prediction = safe_text(
            "mismo n · mismo K · ¿más o menos diversidad?",
            size=27,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(Transform(chart, uniform_chart), TransformMatchingTex(proportions, uniform_formula), run_time=1.25)
        self.play(Write(prediction))
        self.pause(
            "Pregunte qué cambió si n=6 y K=3 permanecieron fijos. Espere que "
            "identifiquen el mayor equilibrio antes de introducir cualquier índice."
        )
        self.clear_content()

