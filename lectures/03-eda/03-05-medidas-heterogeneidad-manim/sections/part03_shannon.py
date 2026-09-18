"""Estados 13–19: Entropía de Shannon, función de sorpresa, curva analítica f(p)=-p ln p con punto crítico en 1/e, bloques acumulados y normalización."""

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



class ShannonSection:
    """Estados 13–19: Entropía de Shannon, función de sorpresa, curva analítica f(p)=-p ln p con punto crítico en 1/e, bloques acumulados y normalización."""

    def shannon_story(self) -> None:
        header = self.heading("Shannon", "¿Qué revelación produciría mayor sorpresa?", PURPLE)
        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], width=6.6, height=2.7)
        chart.move_to([-2.8, -0.35, 0])
        hidden = VGroup()
        for idx, color in enumerate(CAT_COLORS[:3]):
            card = RoundedRectangle(
                width=1.25,
                height=1.25,
                corner_radius=0.16,
                color=color,
                fill_color=color,
                fill_opacity=0.10,
            )
            qmark = safe_text("?", size=38, color=color, weight=BOLD).move_to(card)
            hidden.add(VGroup(card, qmark))
        hidden.arrange(RIGHT, buff=0.35).move_to([3.65, -0.25, 0])
        intuition = safe_text(
            "intuición: menos probable → más sorpresa",
            size=25,
            color=MUTED,
        ).move_to([3.65, -1.35, 0])
        self.play(FadeIn(header), FadeIn(chart), LaggedStart(*[FadeIn(card) for card in hidden], lag_ratio=0.13))
        self.play(FadeIn(intuition))
        self.pause(
            "Pregunte cuál revelación sorprendería más: A, B o C. Espere B porque "
            "es la categoría menos probable. Presente la adivinanza sólo como intuición."
        )

        self.clear_content()
        header = self.heading("Sorpresa", "Lo raro aporta más información por evento", PURPLE)
        probs = [1 / 2, 1 / 6, 1 / 3]
        surprises = [0.6931, 1.7918, 1.0986]
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2.1, 0.5],
            x_length=8.1,
            y_length=4.1,
            axis_config={"color": GRID, "include_tip": False},
        ).move_to([-0.7, -0.20, 0])
        bars = VGroup()
        labels = VGroup()
        values = VGroup()
        for idx, (label, value, color) in enumerate(zip(["A", "B", "C"], surprises, CAT_COLORS[:3]), start=1):
            bar = Rectangle(
                width=1.05,
                height=axes.y_axis.unit_size * value,
                color=color,
                fill_color=color,
                fill_opacity=0.70,
            )
            bar.move_to([axes.c2p(idx, 0)[0], axes.c2p(idx, value / 2)[1], 0])
            bars.add(bar)
            labels.add(safe_text(label, size=23, color=color, weight=BOLD).next_to(bar, DOWN, buff=0.15))
            values.add(MathTex(rf"-\ln p={value:.3f}", font_size=28, color=INK).next_to(bar, UP, buff=0.10))
        side_formula = MathTex(r"-\ln(p_k)", font_size=43, color=PURPLE).move_to([4.55, 0.50, 0])
        rare = pill("B es rara", PURPLE, 2.5).move_to([4.55, -0.60, 0])
        self.play(FadeIn(header), Create(axes), LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.15))
        self.play(FadeIn(labels), FadeIn(values), Write(side_formula), FadeIn(rare))
        self.pause(
            "Compare el factor -ln(p) que ya vive dentro de la fórmula de Shannon: "
            "-ln(1/6) es mayor que -ln(1/2). Pida explicar por qué B sorprende más."
        )

        contribution_values = [0.3466, 0.2986, 0.3662]
        contribution_bars = VGroup()
        contribution_labels = VGroup()
        for idx, (value, color) in enumerate(zip(contribution_values, CAT_COLORS[:3]), start=1):
            bar = Rectangle(
                width=1.05,
                height=axes.y_axis.unit_size * value,
                color=color,
                fill_color=color,
                fill_opacity=0.75,
            )
            bar.move_to([axes.c2p(idx, 0)[0], axes.c2p(idx, value / 2)[1], 0])
            contribution_bars.add(bar)
            contribution_labels.add(
                MathTex(rf"-p\ln p={value:.4f}", font_size=26, color=INK).next_to(bar, UP, buff=0.10)
            )
        balance = safe_text(
            "sorpresa × frecuencia",
            size=27,
            color=YELLOW,
            weight=BOLD,
        ).move_to([4.55, -0.55, 0])
        self.play(Transform(bars, contribution_bars), Transform(values, contribution_labels), FadeOut(rare), Transform(side_formula, MathTex(r"p_k[-\ln(p_k)]", font_size=41, color=PURPLE).move_to(side_formula)), run_time=1.2)
        self.play(FadeIn(balance))
        self.pause(
            "Antes de elegir el término mayor, haga comparar rareza y frecuencia. "
            "B sorprende mucho, pero ocurre poco; C produce la mayor contribución ponderada."
        )

        self.clear_content()
        header = self.heading("Paisaje continuo", "El punto dulce de la sorpresa: 1/e", PURPLE)
        curve_axes = Axes(
            x_range=[0, 1.05, 0.2],
            y_range=[0, 0.45, 0.1],
            x_length=7.4,
            y_length=3.7,
            axis_config={"color": GRID, "include_tip": False},
        ).move_to([-1.2, -0.30, 0])
        x_lbl = safe_text("probabilidad p", size=18, color=MUTED).next_to(curve_axes.x_axis, DOWN, buff=0.15)
        y_lbl = MathTex(r"-p\ln p", font_size=24, color=PURPLE).next_to(curve_axes.y_axis, UP, buff=0.12)

        curve = curve_axes.plot(
            lambda p: -p * np.log(p) if p > 1e-5 else 0.0,
            x_range=[0.001, 1.0, 0.005],
            color=PURPLE,
            stroke_width=4,
        )

        inv_e = 1.0 / np.e
        peak_dot = Dot(curve_axes.c2p(inv_e, inv_e), radius=0.09, color=YELLOW)
        peak_v_line = DashedLine(curve_axes.c2p(inv_e, 0), curve_axes.c2p(inv_e, inv_e), color=YELLOW, dash_length=0.06)
        peak_label = MathTex(r"p=\frac{1}{e}\approx 0.368", font_size=25, color=YELLOW).next_to(peak_dot, UP, buff=0.12)

        dots_on_curve = VGroup()
        dot_labels = VGroup()
        for label, prob, col in zip(["A", "B", "C"], [0.5, 1 / 6, 1 / 3], CAT_COLORS[:3]):
            val = -prob * np.log(prob)
            pt = curve_axes.c2p(prob, val)
            d = Dot(pt, radius=0.10, color=col)
            d_line = DashedLine(curve_axes.c2p(prob, 0), pt, color=col, stroke_opacity=0.6, dash_length=0.05)
            lbl = safe_text(label, size=20, color=col, weight=BOLD).next_to(d, UP if label != "C" else RIGHT, buff=0.10)
            dots_on_curve.add(VGroup(d_line, d))
            dot_labels.add(lbl)

        card_peak = metric_card("PICO DE INFORMACIÓN", "p = 1/e ≈ 0.368", YELLOW, 3.5).move_to([4.55, 0.70, 0])
        card_zero = metric_card("LÍMITE NATURAL", "0 ln 0 := 0", BLUE, 3.5).move_to([4.55, -0.65, 0])

        self.play(FadeIn(header), Create(curve_axes), FadeIn(x_lbl), FadeIn(y_lbl), Create(curve), run_time=1.1)
        self.play(Create(peak_v_line), GrowFromCenter(peak_dot), Write(peak_label), FadeIn(card_peak))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_on_curve], lag_ratio=0.14), FadeIn(dot_labels), FadeIn(card_zero))
        self.pause(
            "Revele la colina continua f(p) = -p ln(p). Señale el punto crítico en 1/e ≈ 0.368. "
            "C (1/3) está casi en la cima, por eso aporta más información que A o B. "
            "Observe que la curva aterriza suavemente en el origen: 0 ln 0 = 0 es continuidad pura."
        )

        self.clear_content()
        header = self.heading("Entropía", "Apilar las contribuciones", PURPLE)
        block_width = 5.7
        x_left = -block_width / 2
        blocks = VGroup()
        block_labels = VGroup()
        cumulative = 0.0
        total = sum(contribution_values)
        for label, value, color in zip(["A  0.3466", "B  0.2986", "C  0.3662"], contribution_values, CAT_COLORS[:3]):
            width = block_width * value / total
            block = Rectangle(
                width=width,
                height=1.05,
                color=color,
                fill_color=color,
                fill_opacity=0.68,
            ).move_to([x_left + cumulative + width / 2, 0.45, 0])
            blocks.add(block)
            label_mob = safe_text(label, size=18, color=INK, weight=BOLD, max_width=width - 0.10)
            label_mob.move_to(block)
            block_labels.add(label_mob)
            cumulative += width
        formula = MathTex(
            r"H=-\sum_{k=1}^{K}p_k\ln(p_k)", font_size=48, color=PURPLE
        ).move_to([0, -0.95, 0])
        result = MathTex(r"H=1.0114\ \text{nats}", font_size=48, color=GREEN).move_to([0, -2.05, 0])
        self.play(FadeIn(header), LaggedStart(*[GrowFromEdge(block, LEFT) for block in blocks], lag_ratio=0.15), FadeIn(block_labels))
        self.play(Write(formula), Write(result))
        self.pause(
            "Reconstruya 0.3466 + 0.2986 + 0.3662 de izquierda a derecha. "
            "Exija la unidad: con logaritmo natural, H=1.0114 nats."
        )

        self.clear_content()
        header = self.heading("Convenciones", "Declare la escala antes de comparar", YELLOW)
        chips = VGroup(
            metric_card("BASE", "ln  →  nats", PURPLE, 3.2),
            metric_card("CATEGORÍA VACÍA", "0 ln 0 := 0", BLUE, 3.2),
            metric_card("UNIVERSO", "K = niveles declarados", ORANGE, 3.6),
        ).arrange(RIGHT, buff=0.35).move_to([0, -0.20, 0])
        note = safe_text(
            "Cambiar la base cambia la unidad, no el orden.",
            size=29,
            color=INK,
            weight=BOLD,
        ).move_to([0, -1.65, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(chip, shift=UP * 0.15) for chip in chips], lag_ratio=0.15))
        self.play(Write(note))
        self.pause(
            "Declare que toda la clase usa ln y nats. Verbalice 0 ln 0 = 0 por "
            "continuidad. Defina K como los niveles fijados antes del análisis."
        )

        self.clear_content()
        header = self.heading("Normalización", "¿Qué fracción del máximo posible?", PURPLE)
        max_h = np.log(3)
        observed_h = 1.0114042647
        bar_bg = RoundedRectangle(
            width=9.0, height=0.85, corner_radius=0.18, color=GRID, fill_color=GRID, fill_opacity=0.25
        ).move_to([0, -0.10, 0])
        bar_fill = RoundedRectangle(
            width=9.0 * observed_h / max_h,
            height=0.85,
            corner_radius=0.18,
            color=PURPLE,
            fill_color=PURPLE,
            fill_opacity=0.78,
        )
        bar_fill.align_to(bar_bg, LEFT).move_to([
            bar_bg.get_left()[0] + bar_fill.width / 2,
            bar_bg.get_y(),
            0,
        ])
        min_label = safe_text("0", size=21, color=MUTED).next_to(bar_bg, DOWN, aligned_edge=LEFT)
        max_label = MathTex(r"H_{\max}=\ln3=1.0986", font_size=29, color=MUTED).next_to(bar_bg, DOWN, aligned_edge=RIGHT)
        norm = MathTex(
            r"H_{\mathrm{norm}}=\frac{1.0114}{\ln3}=0.9206",
            font_size=48,
            color=GREEN,
        ).move_to([0, -1.62, 0])
        self.play(FadeIn(header), FadeIn(bar_bg), FadeIn(min_label), FadeIn(max_label))
        self.play(GrowFromEdge(bar_fill, LEFT), Write(norm))
        self.pause(
            "Pregunte qué permite comparar la normalización. Precise que numerador "
            "y denominador deben usar la misma base y que K debe estar fijado."
        )

        self.clear_content()
        header = self.heading("Número efectivo", "Volver a una escala de categorías", GREEN)
        source = metric_card("SHANNON", "H = 1.0114 nats", PURPLE, 3.5).move_to([-3.7, 0.15, 0])
        arrow = Arrow([-1.55, 0.15, 0], [0.15, 0.15, 0], color=MUTED, stroke_width=4)
        target = metric_card("CATEGORÍAS EFECTIVAS", "exp(H) = 2.75", GREEN, 4.2).move_to([2.65, 0.15, 0])
        full_one = Circle(radius=0.34, color=BLUE, fill_color=BLUE, fill_opacity=0.25)
        full_two = Circle(radius=0.34, color=PURPLE, fill_color=PURPLE, fill_opacity=0.25)
        partial_outline = Circle(radius=0.34, color=GREEN)
        partial_fill = Sector(
            radius=0.31,
            angle=3 * PI / 2,
            start_angle=PI / 2,
            stroke_width=0,
            fill_color=GREEN,
            fill_opacity=0.28,
        )
        partial = VGroup(partial_outline, partial_fill)
        fraction = safe_text("¾", size=18, color=GREEN, weight=BOLD).move_to(partial)
        partial.add(fraction)
        equal_tokens = VGroup(full_one, full_two, partial).arrange(RIGHT, buff=0.28).move_to([2.65, -1.25, 0])
        comparison = safe_text(
            "Simpson: 2.57  ·  Shannon: 2.75",
            size=27,
            color=INK,
            weight=BOLD,
        ).move_to([-1.5, -1.25, 0])
        self.play(FadeIn(header), FadeIn(source), GrowArrow(arrow), FadeIn(target))
        self.play(FadeIn(equal_tokens), FadeIn(comparison))
        self.pause(
            "Interprete 2.75 como el número de categorías igualmente frecuentes "
            "con la misma entropía. Compare con 2.57 de Simpson: ponderan de forma distinta."
        )
        self.clear_content()

