"""Clase de una hora: FP-Growth con Manim Slides.

Secuencia: el costo de Apriori y la analogía de las palabras (intuición) →
construcción del FP-tree a mano (contar, ordenar, insertar) → minado con bases
y árboles condicionales, con un «tu turno» → comparación con Apriori, escala y
Python → reto en parejas y cierre.

Para iterar sobre un solo acto: ``FP_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_build import BuildSection
from sections.part03_mine import MineSection
from sections.part04_compare import CompareSection
from sections.part05_practice import PracticeSection
from sections.theme import BG


class FPGrowth(
    BaseSlide,
    OpeningSection,
    BuildSection,
    MineSection,
    CompareSection,
    PracticeSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,        # 1 · 0–10 min
            self.build,          # 2 · 10–24
            self.mine_tree,      # 3 · 24–42
            self.compare,        # 4 · 42–50
            self.practice,       # 5 · 50–57
            self.closing,        # 6 · 57–60
        ]
        wanted = os.environ.get("FP_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
