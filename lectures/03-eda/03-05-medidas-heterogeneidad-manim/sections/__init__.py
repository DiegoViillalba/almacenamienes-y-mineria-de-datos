"""Módulos de diapositivas para la presentación de Medidas de Heterogeneidad."""

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

__all__ = [
    "BG", "INK", "MUTED", "GRID", "BLUE", "ORANGE", "PURPLE", "YELLOW", "GREEN", "RED",
    "CAT_COLORS", "FONT", "LOGO_PATH", "safe_text", "pill", "metric_card",
    "distribution_chart", "token_row", "probability_grid",
    "BaseSlide",
    "OpeningSection",
    "GiniSimpsonSection",
    "ShannonSection",
    "ComparisonSection",
    "AnovaSection",
    "TransferSection",
    "ClosingSection",
]
