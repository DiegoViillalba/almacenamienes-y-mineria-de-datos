"""Estados 36–37: Cierre metodológico, síntesis de convenciones y puentes hacia algoritmos de Minería de Datos."""

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



class ClosingSection:
    """Estados 36–37: Cierre metodológico, síntesis de convenciones y puentes hacia algoritmos de Minería de Datos."""

    def closing(self) -> None:
        header = self.heading("Cierre", "Una pregunta decide qué objeto mirar", BLUE)
        cards = VGroup(
            metric_card("UNA CATEGORÍA", "barras → D, H → efectivos", ORANGE, 3.75),
            metric_card("RESPUESTA POR GRUPO", "puntos → SS → η²", GREEN, 3.75),
            metric_card("DOS AGRUPADORES", "heatmap + n por celda", PURPLE, 3.75),
        ).arrange(RIGHT, buff=0.35).scale(0.94).move_to([0, 0.55, 0])
        conventions = VGroup(
            pill("fijar K", ORANGE, 1.75),
            pill("declarar ln", PURPLE, 2.0),
            pill("nombrar el evento", BLUE, 2.65),
            pill("acotar η²", RED, 1.85),
            pill("SIMULADOR EN VIVO", YELLOW, 2.75),
        ).arrange(RIGHT, buff=0.18).move_to([0, -0.85, 0])
        takeaway = safe_text(
            "Primero mire el reparto; después resúmalo.",
            size=34,
            color=INK,
            weight=BOLD,
        ).move_to([0, -1.95, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in cards], lag_ratio=0.14))
        self.play(LaggedStart(*[FadeIn(chip) for chip in conventions], lag_ratio=0.12))
        self.play(Write(takeaway))
        self.pause(
            "Cierre metodológico. Recuerde conectar estas medidas con Minería de Datos: "
            "D es la impureza de Gini en CART; H es la ganancia de información en C4.5; "
            "y SS_T = SS_B + SS_W es el criterio de agrupamiento en k-means. "
            "Invite al grupo a explorar el simulador interactivo en interactivo.html."
        )
        self.wait(0.8)
