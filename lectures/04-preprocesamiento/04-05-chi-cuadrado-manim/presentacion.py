"""Clase de una hora: reducción de variables categóricas con la prueba χ².

Secuencia: el problema (seis columnas categóricas) → ver la dependencia con
barras de perfil → la tabla esperada del mundo independiente → construir χ²
paso a paso → barajar etiquetas para ver cuánto χ² produce el azar (valor p,
grados de libertad) → significancia ≠ fuerza (V de Cramér) → filtros de
relevancia y redundancia → reto en parejas y cierre.

Para iterar sobre un solo acto: ``CHI_PARTS=3,4 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_profiles import ProfilesSection
from sections.part03_expected import ExpectedSection
from sections.part04_statistic import StatisticSection
from sections.part05_null import NullSection
from sections.part06_strength import StrengthSection
from sections.part07_reduction import ReductionSection
from sections.part08_practice import PracticeSection
from sections.theme import BG


class ReduccionChiCuadrado(
    BaseSlide,
    OpeningSection,
    ProfilesSection,
    ExpectedSection,
    StatisticSection,
    NullSection,
    StrengthSection,
    ReductionSection,
    PracticeSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–6 min
            self.profiles,     # 2 · 6–14
            self.expected,     # 3 · 14–21
            self.statistic,    # 4 · 21–27
            self.null_world,   # 5 · 27–38
            self.strength,     # 6 · 38–43
            self.reduction,    # 7 · 43–53
            self.practice,     # 8 · 53–59
            self.closing,      # 9 · 59–60
        ]
        wanted = os.environ.get("CHI_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
