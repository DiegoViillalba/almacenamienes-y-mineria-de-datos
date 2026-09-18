"""Estados 33–35: Reto de transferencia de canales de atención con datos nuevos, cálculo y números de Hill."""

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



class TransferSection:
    """Estados 33–35: Reto de transferencia de canales de atención con datos nuevos, cálculo y números de Hill."""

    def transfer_challenge(self) -> None:
        header = self.heading("Reto de transferencia", "Canales de atención · datos nuevos", YELLOW)
        data = [
            ("WEB", 9, BLUE),
            ("APP", 6, PURPLE),
            ("TELÉFONO", 3, GREEN),
            ("TIENDA", 2, ORANGE),
        ]
        cards = VGroup(*[
            metric_card(label, f"f = {count}", color, 2.6) for label, count, color in data
        ]).arrange(RIGHT, buff=0.30).move_to([0, 0.65, 0])
        tasks = VGroup(
            safe_text("1  D de Gini–Simpson", size=25, color=ORANGE, weight=BOLD),
            safe_text("2  H normalizada", size=25, color=PURPLE, weight=BOLD),
            safe_text("3  categorías efectivas de Shannon e interpretación", size=25, color=GREEN, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -0.85, 0])
        timer = pill("PAUSA REAL · 3 min", YELLOW, 3.35, 0.68).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.16) for card in cards], lag_ratio=0.13))
        self.play(FadeIn(tasks), FadeIn(timer))
        self.pause(
            "Deje tres minutos reales. No revele proporciones ni fórmulas resueltas. "
            "Pida primero dibujar el reparto, fijar K=4 y anotar que se usa ln."
        )

        self.clear_content()
        header = self.heading("Solución · 1/2", "Conteos → proporciones → pares", ORANGE)
        probs = [0.45, 0.30, 0.15, 0.10]
        labels = ["WEB", "APP", "TEL", "TIENDA"]
        chart = distribution_chart(probs, labels, width=6.7, height=3.0)
        chart.move_to([-3.15, -0.25, 0])
        p_formula = MathTex(
            r"p=(0.45,\ 0.30,\ 0.15,\ 0.10)", font_size=37, color=INK
        ).move_to([3.45, 1.05, 0])
        check = MathTex(r"\sum p_k=1", font_size=34, color=GREEN).move_to([3.45, 0.35, 0])
        squares = MathTex(
            r"\sum p_k^2=0.2025+0.09+0.0225+0.01=0.325",
            font_size=33,
            color=INK,
        ).move_to([3.45, -0.55, 0])
        if squares.width > 6.0:
            squares.scale_to_fit_width(6.0)
        d_result = MathTex(r"D=1-0.325=0.675", font_size=47, color=ORANGE).move_to([3.45, -1.55, 0])
        interp = safe_text(
            "67.5%: dos extracciones independientes difieren",
            size=23,
            color=GREEN,
            weight=BOLD,
            max_width=6.0,
        ).move_to([3.45, -2.35, 0])
        self.play(FadeIn(header), FadeIn(chart), Write(p_formula))
        self.play(Write(check), Write(squares))
        self.play(Write(d_result), FadeIn(interp))
        self.pause(
            "Compruebe que las proporciones suman uno. Construya 0.325 término a "
            "término y solicite interpretar D=0.675 como probabilidad, con el esquema de muestreo."
        )

        self.clear_content()
        header = self.heading("Solución · 2/2", "Sorpresa → escala comparable → significado", PURPLE)
        contributions = VGroup(
            metric_card("WEB", "0.35933", BLUE, 2.35),
            metric_card("APP", "0.36119", PURPLE, 2.35),
            metric_card("TELÉFONO", "0.28457", GREEN, 2.35),
            metric_card("TIENDA", "0.23026", ORANGE, 2.35),
        ).arrange(RIGHT, buff=0.28).move_to([0, 1.10, 0])
        h_formula = MathTex(
            r"H=\sum[-p_k\ln(p_k)]\approx1.2353\ \text{nats}",
            font_size=42,
            color=PURPLE,
        ).move_to([0, -0.15, 0])
        summary = VGroup(
            metric_card("H NORMALIZADA", "1.2353 / ln 4 = 0.8911", PURPLE, 3.8),
            metric_card("EFECTIVAS · SHANNON", "exp(H) = 3.44", GREEN, 3.8),
            metric_card("EFECTIVAS · SIMPSON", "1 / 0.325 = 3.08", ORANGE, 3.8),
        ).arrange(RIGHT, buff=0.25).move_to([0, -1.25, 0])
        workflow = VGroup(
            pill("dibujar", BLUE, 1.65),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("calcular", ORANGE, 1.75),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("declarar", YELLOW, 1.75),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("interpretar", GREEN, 2.05),
        ).arrange(RIGHT, buff=0.18).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.12) for card in contributions], lag_ratio=0.12))
        self.play(Write(h_formula), FadeIn(summary))
        self.play(FadeIn(workflow))
        self.pause(
            "Sume con valores no redondeados: H≈1.2353 nats. Pida una frase final: "
            "el reparto equivale a 3.44 categorías igualmente frecuentes según Shannon "
            "y 3.08 según Simpson. Observe cómo Simpson penaliza más a las categorías minoritarias."
        )
        self.clear_content()

