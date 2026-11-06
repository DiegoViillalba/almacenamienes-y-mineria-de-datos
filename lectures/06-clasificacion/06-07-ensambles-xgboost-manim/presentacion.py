"""Clase de una hora: ensambles de árboles y XGBoost con Manim Slides.

Secuencia: árboles de regresión y su varianza → la sabiduría de muchos
(Condorcet, varianza de un promedio) → bagging, error fuera de la bolsa y
bosques aleatorios → AdaBoost ronda por ronda → gradient boosting a mano →
XGBoost (Taylor de segundo orden, pesos de hoja, ganancia, λ y γ) →
comparación, ejercicio y cierre.

Para iterar sobre un solo acto: ``CLS_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_regression import RegressionSection
from sections.part02_crowd import CrowdSection
from sections.part03_bagging import BaggingSection
from sections.part04_adaboost import AdaBoostSection
from sections.part05_gradient import GradientSection
from sections.part06_xgboost import XGBoostSection
from sections.part07_close import CloseSection
from sections.theme import BG


class EnsamblesXGBoost(
    BaseSlide,
    RegressionSection,
    CrowdSection,
    BaggingSection,
    AdaBoostSection,
    GradientSection,
    XGBoostSection,
    CloseSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–9 min
            self.crowd,        # 2 · 9–17
            self.bagging,      # 3 · 17–27
            self.adaboost,     # 4 · 27–37
            self.gradient,     # 5 · 37–46
            self.xgboost,      # 6 · 46–55
            self.close,        # 7 · 55–60
        ]
        wanted = os.environ.get("CLS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
