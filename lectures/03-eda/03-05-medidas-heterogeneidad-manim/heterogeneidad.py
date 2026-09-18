"""Repaso aplicado de medidas de heterogeneidad con Manim Slides.

La presentación está dividida en submódulos dentro de 'sections/' para permitir
editar, agregar o ajustar cualquier diapositiva o bloque temático de forma ágil.
"""

from __future__ import annotations

from manim import ManimColor
from manim_slides import Slide

from sections.theme import (
    BG, INK, MUTED, GRID, BLUE, ORANGE, PURPLE, YELLOW, GREEN, RED,
    CAT_COLORS, FONT, LOGO_PATH, safe_text, pill, metric_card,
    distribution_chart, token_row, probability_grid
)
from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_gini_simpson import GiniSimpsonSection
from sections.part03_shannon import ShannonSection
from sections.part04_comparison import ComparisonSection
from sections.part05_anova import AnovaSection
from sections.part06_transfer import TransferSection
from sections.part07_closing import ClosingSection


class MedidasHeterogeneidad(
    BaseSlide,
    OpeningSection,
    GiniSimpsonSection,
    ShannonSection,
    ComparisonSection,
    AnovaSection,
    TransferSection,
    ClosingSection,
    Slide,
):
    """Una narrativa continua: reparto categórico, separación y transferencia."""

    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.branding = None
        self.opening_and_distribution()
        self.gini_simpson_story()
        self.shannon_story()
        self.comparison_and_transition()
        self.group_separation_story()
        self.transfer_challenge()
        self.closing()


if __name__ == "__main__":
    pass
