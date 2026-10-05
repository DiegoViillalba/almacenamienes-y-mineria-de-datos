"""Clase de una hora: Análisis de Componentes Principales con Manim Slides.

Secuencia: resorte observado por tres cámaras (intuición) → dos sensores
(cálculo a mano) → dirección de máxima varianza → autovectores → nube 3D →
regreso al resorte y a la hélice → límites → práctica y cierre.

Para iterar sobre un solo acto: ``PCA_PARTS=3,4 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import ThreeDSlide

from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_redundancy import RedundancySection
from sections.part03_two_sensors import TwoSensorsSection
from sections.part04_direction import DirectionSection
from sections.part05_eigen import EigenSection
from sections.part06_cloud3d import Cloud3DSection
from sections.part07_spring_pca import SpringPCASection
from sections.part08_limits import LimitsSection
from sections.part09_practice import PracticeSection
from sections.theme import BG


class ComponentesPrincipales(
    BaseSlide,
    OpeningSection,
    RedundancySection,
    TwoSensorsSection,
    DirectionSection,
    EigenSection,
    Cloud3DSection,
    SpringPCASection,
    LimitsSection,
    PracticeSection,
    ThreeDSlide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,          # 1 · 0–10 min
            self.redundancy,       # 2 · 10–14
            self.two_sensors,      # 3 · 14–20
            self.best_direction,   # 4 · 20–29
            self.eigen_story,      # 5 · 29–36
            self.cloud_3d,         # 6 · 36–43
            self.spring_pca,       # 7 · 43–50
            self.limits,           # 8 · 50–54
            self.practice,         # 9 · 54–58
            self.closing,          # 10 · 58–60
        ]
        wanted = os.environ.get("PCA_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
