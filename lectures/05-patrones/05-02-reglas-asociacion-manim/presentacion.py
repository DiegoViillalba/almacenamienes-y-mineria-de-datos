"""Clase de una hora: reglas de asociación con Manim Slides.

Secuencia: el dilema del gerente con 100 tickets (intuición) → construcción
con cinco tickets (soporte, Apriori, confianza, lift) → probabilidad como área
(cuadrado unitario) → leverage, conviction y Zhang → red, mapa y decisiones →
reto en parejas y cierre.

Para iterar sobre un solo acto: ``REGLAS_PARTS=3,4 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_construction import ConstructionSection
from sections.part03_square import SquareSection
from sections.part04_metrics import MetricsSection
from sections.part05_decisions import DecisionsSection
from sections.part06_practice import PracticeSection
from sections.theme import BG


class ReglasAsociacion(
    BaseSlide,
    OpeningSection,
    ConstructionSection,
    SquareSection,
    MetricsSection,
    DecisionsSection,
    PracticeSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,        # 1 · 0–10 min
            self.construction,   # 2 · 10–20
            self.square_view,    # 3 · 20–30
            self.metrics,        # 4 · 30–40
            self.decisions,      # 5 · 40–50
            self.practice,       # 6 · 50–57
            self.closing,        # 7 · 57–60
        ]
        wanted = os.environ.get("REGLAS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
