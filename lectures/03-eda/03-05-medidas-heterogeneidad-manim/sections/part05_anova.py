"""Estados 24–32: Respuesta continua entre grupos, medias grupales, descomposición de áreas euclidianas SS_T = SS_B + SS_W, eta cuadrada y heatmap."""

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



class AnovaSection:
    """Estados 24–32: Respuesta continua entre grupos, medias grupales, descomposición de áreas euclidianas SS_T = SS_B + SS_W, eta cuadrada y heatmap."""

    def group_separation_story(self) -> None:
        header = self.heading("Tres grupos", "La misma respuesta, tres carriles", GREEN)
        group_data = {"A": [4, 5], "B": [8, 9], "C": [5, 5]}
        axes = VGroup()
        all_dots = VGroup()
        group_dots: dict[str, VGroup] = {}
        for idx, (label, vals) in enumerate(group_data.items()):
            y = 1.20 - idx * 1.25
            color = CAT_COLORS[idx]
            axis = NumberLine(
                x_range=[3, 10, 1],
                length=9.6,
                include_numbers=True,
                font_size=18,
                include_tip=False,
                color=GRID,
            ).move_to([0.55, y, 0])
            lane_label = pill(f"GRUPO {label}", color, 1.65).next_to(axis, LEFT, buff=0.25)
            dots = VGroup()
            duplicate: dict[int, int] = {}
            for value in vals:
                level = duplicate.get(value, 0)
                duplicate[value] = level + 1
                dot = Dot(axis.n2p(value) + UP * (0.15 + 0.18 * level), radius=0.11, color=color)
                dots.add(dot)
                all_dots.add(dot)
            group_dots[label] = dots
            axes.add(VGroup(axis, lane_label))
        data_formula = MathTex(
            r"A=(4,5)\qquad B=(8,9)\qquad C=(5,5)",
            font_size=39,
            color=INK,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(axis) for axis in axes], lag_ratio=0.13))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in all_dots], lag_ratio=0.10), Write(data_formula))
        self.pause(
            "Pida estimar la media global sin calcular. Haga que señalen visualmente "
            "qué grupo está más lejos del centro conjunto."
        )

        mean_lines = VGroup()
        mean_labels = VGroup()
        for lane in axes:
            axis = lane[0]
            x = axis.n2p(6)[0]
            line = DashedLine([x, axis.get_y() - 0.42, 0], [x, axis.get_y() + 0.42, 0], color=YELLOW, dash_length=0.08)
            mean_lines.add(line)
        global_mean = MathTex(r"\bar y=\frac{36}{6}=6", font_size=45, color=YELLOW)
        global_mean.next_to(header, DOWN, buff=0.27)
        distance_lines = VGroup()
        for label, vals in group_data.items():
            axis = axes[["A", "B", "C"].index(label)][0]
            for value in vals:
                distance_lines.add(
                    Line(axis.n2p(value), axis.n2p(6), color=YELLOW, stroke_width=5, stroke_opacity=0.55)
                )
        self.play(LaggedStart(*[Create(line) for line in mean_lines], lag_ratio=0.12), Write(global_mean))
        self.play(LaggedStart(*[GrowFromPoint(line, line.get_start()) for line in distance_lines], lag_ratio=0.08))
        self.pause(
            "Compruebe 36/6=6. Lea las seis distancias respecto de una sola media "
            "y pida localizar los puntos que aportarán los cuadrados mayores."
        )

        squared_values = [4, 1, 4, 9, 1, 1]
        squares_visual = VGroup()
        for label, vals in group_data.items():
            axis_idx = ["A", "B", "C"].index(label)
            axis = axes[axis_idx][0]
            col = CAT_COLORS[axis_idx]
            for val in vals:
                p_val = axis.n2p(val)
                p_mean = axis.n2p(6)
                dist = abs(p_val[0] - p_mean[0])
                if dist > 0.05:
                    sq = Square(
                        side_length=dist,
                        stroke_color=col,
                        stroke_width=1.6,
                        fill_color=col,
                        fill_opacity=0.18,
                    )
                    sq_center_x = (p_val[0] + p_mean[0]) / 2
                    sq_center_y = axis.get_y() + dist / 2
                    sq.move_to([sq_center_x, sq_center_y, 0])
                    squares_visual.add(sq)

        cards = VGroup(*[
            metric_card(f"({value}−6)²", str(square), color, 1.65)
            for value, square, color in zip([4, 5, 8, 9, 5, 5], squared_values, [BLUE, BLUE, PURPLE, PURPLE, GREEN, GREEN])
        ]).arrange(RIGHT, buff=0.16).scale(0.88).move_to([0, -2.35, 0])
        total = MathTex(r"SS_T=4+1+4+9+1+1=20", font_size=41, color=YELLOW)
        total.move_to([0, 2.02, 0])
        self.play(
            FadeOut(data_formula),
            FadeIn(cards, shift=UP * 0.18),
            LaggedStart(*[FadeIn(sq) for sq in squares_visual], lag_ratio=0.10),
            Transform(global_mean, total)
        )
        self.pause(
            "Convierta cada distancia en su cuadrado y sume 20. Insista: SS_T mide "
            "la variación total respecto de la media global, representada por el área "
            "acumulada de los cuadrados geométricos."
        )

        self.play(FadeOut(cards), FadeOut(squares_visual), FadeOut(distance_lines), FadeOut(mean_lines), FadeOut(global_mean))
        hidden_means = VGroup()
        for idx, label in enumerate(["A", "B", "C"]):
            axis = axes[idx][0]
            marker = Triangle(color=CAT_COLORS[idx], fill_color=CAT_COLORS[idx], fill_opacity=0.75).scale(0.14)
            marker.next_to(axis, UP, buff=0.28)
            q = safe_text("?", size=23, color=YELLOW, weight=BOLD).next_to(marker, UP, buff=0.05)
            hidden_means.add(VGroup(marker, q))
        question = safe_text(
            "Predice las medias y la contribución mayor",
            size=28,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(hidden_means), Write(question))
        self.pause(
            "Pida predecir 4.5, 8.5 y 5, y decidir qué grupo contribuirá más a "
            "la separación entre medias. No revele todavía el factor n_g."
        )

        group_means = [4.5, 8.5, 5.0]
        revealed_means = VGroup()
        for idx, (label, mean) in enumerate(zip(["A", "B", "C"], group_means)):
            axis = axes[idx][0]
            marker = Triangle(color=CAT_COLORS[idx], fill_color=CAT_COLORS[idx], fill_opacity=0.82).scale(0.16)
            direction = DOWN if label == "C" else UP
            marker.next_to(axis.n2p(mean), direction, buff=0.27)
            if label == "C":
                marker.rotate(PI)
            value = MathTex(rf"\bar y_{label}={mean:g}", font_size=29, color=CAT_COLORS[idx]).next_to(marker, direction, buff=0.05)
            revealed_means.add(VGroup(marker, value))
        preserved = safe_text(
            "cada grupo se resume por centro + tamaño",
            size=27,
            color=INK,
            weight=BOLD,
        ).move_to(question)
        self.play(Transform(hidden_means, revealed_means), Transform(question, preserved), run_time=1.1)
        self.pause(
            "Revele las medias. Pregunte qué se conserva al sustituir cada dato por "
            "su media grupal: la posición del centro y cuántas observaciones representa."
        )

        self.clear_content()
        header = self.heading("Entre grupos", "Mover los centros hasta la media global", GREEN)
        mean_axis = NumberLine(
            x_range=[3, 10, 1], length=9.5, include_numbers=True, font_size=20, include_tip=False, color=GRID
        ).move_to([0, 0.65, 0])
        global_mark = DashedLine(
            mean_axis.n2p(6) + DOWN * 0.75,
            mean_axis.n2p(6) + UP * 0.75,
            color=YELLOW,
            dash_length=0.09,
        )
        global_label = pill("media global = 6", YELLOW, 2.65).next_to(global_mark, UP, buff=0.14)
        center_dots = VGroup()
        center_labels = VGroup()
        arrows = VGroup()
        for label, mean, color in zip(["A", "B", "C"], group_means, CAT_COLORS[:3]):
            dot = Dot(mean_axis.n2p(mean), radius=0.12, color=color)
            dot.shift(UP * ({"A": 0.18, "B": 0.18, "C": -0.18}[label]))
            center_dots.add(dot)
            center_labels.add(safe_text(label, size=20, color=color, weight=BOLD).next_to(dot, UP if label != "C" else DOWN, buff=0.10))
            arrows.add(Arrow(dot.get_center(), mean_axis.n2p(6), color=color, buff=0.12, stroke_width=4))
        general_ssb = MathTex(
            r"SS_B=\sum_g n_g(\bar y_g-\bar y)^2",
            font_size=29,
            color=MUTED,
        ).move_to([3.35, 1.72, 0])
        ssb = MathTex(
            r"SS_B=2(4.5-6)^2+2(8.5-6)^2+2(5-6)^2",
            font_size=37,
            color=GREEN,
        ).move_to([0, -0.90, 0])
        ssb_result = MathTex(r"SS_B=4.5+12.5+2=19", font_size=43, color=GREEN).move_to([0, -1.85, 0])
        self.play(FadeIn(header), FadeIn(general_ssb), Create(mean_axis), Create(global_mark), FadeIn(global_label))
        self.play(FadeIn(center_dots), FadeIn(center_labels), LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.16))
        self.play(Write(ssb), Write(ssb_result))
        self.pause(
            "Señale el factor n_g=2: cada centro representa dos observaciones. "
            "No permita sumar sólo tres distancias; el componente entre grupos es 19."
        )

        self.clear_content()
        header = self.heading("Dentro de grupos", "Lo que las medias grupales no capturan", PURPLE)
        residual_cards = VGroup(
            metric_card("GRUPO A", "0.25 + 0.25 = 0.5", BLUE, 3.2),
            metric_card("GRUPO B", "0.25 + 0.25 = 0.5", PURPLE, 3.2),
            metric_card("GRUPO C", "0 + 0 = 0", GREEN, 3.2),
        ).arrange(RIGHT, buff=0.35).move_to([0, 0.05, 0])
        ssw = MathTex(r"SS_W=0.5+0.5+0=1", font_size=48, color=PURPLE).move_to([0, -1.50, 0])
        cue = safe_text(
            "distancias cortas: dato → media de su grupo",
            size=25,
            color=MUTED,
        ).move_to([0, 1.62, 0])
        residual_visual = VGroup()
        for center_x, color, coincident in zip([-3.35, 0, 3.35], CAT_COLORS[:3], [False, False, True]):
            center = Dot([center_x, 0.86, 0], radius=0.07, color=YELLOW)
            if coincident:
                points = VGroup(
                    Dot([center_x, 0.86, 0], radius=0.10, color=color),
                    Dot([center_x, 1.05, 0], radius=0.10, color=color),
                )
                segments = VGroup(Line([center_x, 1.05, 0], [center_x, 0.86, 0], color=color, stroke_width=4))
            else:
                points = VGroup(
                    Dot([center_x - 0.42, 0.86, 0], radius=0.10, color=color),
                    Dot([center_x + 0.42, 0.86, 0], radius=0.10, color=color),
                )
                segments = VGroup(
                    Line([center_x - 0.42, 0.86, 0], [center_x, 0.86, 0], color=color, stroke_width=4),
                    Line([center_x + 0.42, 0.86, 0], [center_x, 0.86, 0], color=color, stroke_width=4),
                )
            residual_visual.add(VGroup(segments, center, points))
        self.play(FadeIn(header), FadeIn(cue))
        self.play(LaggedStart(*[FadeIn(item) for item in residual_visual], lag_ratio=0.16))
        self.play(LaggedStart(*[FadeIn(card) for card in residual_cards], lag_ratio=0.14))
        self.play(Write(ssw))
        self.pause(
            "Verifique directamente los seis residuos dentro de grupos. A y B aportan "
            "0.5 cada uno; C no aporta porque sus dos valores coinciden con su media."
        )

        self.clear_content()
        header = self.heading("Partición", "El total se divide sin perder nada", YELLOW)
        total_bar = RoundedRectangle(
            width=10.5, height=1.0, corner_radius=0.17, color=INK, fill_color=GRID, fill_opacity=0.22
        ).move_to([0, 0.20, 0])
        between_bar = Rectangle(
            width=10.5 * 19 / 20,
            height=0.96,
            stroke_width=0,
            fill_color=GREEN,
            fill_opacity=0.80,
        )
        between_bar.align_to(total_bar, LEFT).move_to([total_bar.get_left()[0] + between_bar.width / 2, total_bar.get_y(), 0])
        within_bar = Rectangle(
            width=10.5 / 20,
            height=0.96,
            stroke_width=0,
            fill_color=PURPLE,
            fill_opacity=0.88,
        )
        within_bar.align_to(total_bar, RIGHT).move_to([total_bar.get_right()[0] - within_bar.width / 2, total_bar.get_y(), 0])
        between_label = safe_text("ENTRE = 19", size=24, color=BG, weight=BOLD).move_to(between_bar)
        within_label = safe_text("1", size=20, color=INK, weight=BOLD).move_to(within_bar)
        identity = MathTex(r"SS_T=SS_B+SS_W\qquad 20=19+1", font_size=43, color=INK).move_to([0, -1.05, 0])
        eta = MathTex(r"\eta^2=\frac{SS_B}{SS_T}=\frac{19}{20}=0.95", font_size=52, color=GREEN).move_to([0, -2.05, 0])
        self.play(FadeIn(header), FadeIn(total_bar))
        self.play(GrowFromEdge(between_bar, LEFT), GrowFromEdge(within_bar, LEFT), FadeIn(between_label), FadeIn(within_label))
        self.play(Write(identity), Write(eta))
        self.pause(
            "Interprete primero la barra: 19 partes entre y una dentro. Después lea "
            "eta cuadrada como 95% de la suma de cuadrados total observada asociada a medias grupales."
        )

        self.clear_content()
        header = self.heading("Límites", "Un cociente no sustituye la inspección", RED)
        cautions = VGroup(
            metric_card("CENTRO", "medias y medianas", YELLOW, 2.8),
            metric_card("FORMA", "dispersión y colas", PURPLE, 2.8),
            metric_card("TAMAÑO", "n por grupo", BLUE, 2.8),
            metric_card("ALCANCE", "descriptivo ≠ causal", RED, 3.0),
        ).arrange(RIGHT, buff=0.25).scale(0.92).move_to([0, -0.05, 0])
        footer = safe_text(
            "η² no prueba causalidad ni sustituye la inspección",
            size=28,
            color=RED,
            weight=BOLD,
        ).move_to([0, -1.55, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in cautions], lag_ratio=0.13))
        self.play(Write(footer))
        self.pause(
            "Modele la cautela: eta cuadrada es descriptiva; no demuestra causalidad, "
            "y no reemplaza forma, dispersión ni tamaños grupales."
        )
        self.clear_content()

        header = self.heading("Dos agrupadores", "Cruzar variables exige mirar cada celda", BLUE)
        row_labels = ["REGIÓN 1", "REGIÓN 2"]
        col_labels = ["TIPO A", "TIPO B", "TIPO C"]
        means = [[52, 60, 68], [48, 57, 73]]
        counts = [[40, 38, 3], [35, 32, 2]]
        cells = VGroup()
        for row in range(2):
            for col in range(3):
                x = -2.7 + col * 2.7
                y = 0.75 - row * 1.45
                small = counts[row][col] <= 3
                color = RED if small else CAT_COLORS[col]
                opacity = 0.10 + (means[row][col] - 45) / 55
                cell = RoundedRectangle(
                    width=2.25,
                    height=1.15,
                    corner_radius=0.12,
                    color=color,
                    fill_color=color,
                    fill_opacity=min(0.60, opacity),
                    stroke_width=3 if small else 1.8,
                )
                value = safe_text(f"media = {means[row][col]}", size=21, color=INK, weight=BOLD)
                n_label = safe_text(f"n = {counts[row][col]}", size=18, color=RED if small else MUTED, weight=BOLD)
                content = VGroup(value, n_label).arrange(DOWN, buff=0.10).move_to(cell)
                cells.add(VGroup(cell, content).move_to([x, y, 0]))
        rows = VGroup(*[
            safe_text(label, size=20, color=MUTED, weight=BOLD).move_to([-5.25, 0.75 - idx * 1.45, 0])
            for idx, label in enumerate(row_labels)
        ])
        cols = VGroup(*[
            safe_text(label, size=20, color=CAT_COLORS[idx], weight=BOLD).move_to([-2.7 + idx * 2.7, 1.65, 0])
            for idx, label in enumerate(col_labels)
        ])
        warning = safe_text(
            "heatmap + n por celda · una celda pequeña exige cautela",
            size=27,
            color=RED,
            weight=BOLD,
        ).move_to([0, -2.10, 0])
        self.play(FadeIn(header), FadeIn(rows), FadeIn(cols))
        self.play(LaggedStart(*[FadeIn(cell, scale=0.95) for cell in cells], lag_ratio=0.10))
        self.play(Write(warning), Circumscribe(cells[-1], color=RED))
        self.pause(
            "Presente el heatmap como heterogeneidad cruzada. Haga comparar medias, "
            "pero pida leer también n: las celdas con 3 y 2 observaciones requieren cautela."
        )
        self.clear_content()

