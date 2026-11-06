"""Clase de una hora: clasificadores lineales con Manim Slides.

Secuencia: separar con una recta (geometría de w y b, perceptrón) → mínimos
cuadrados y su fórmula cerrada → Fisher (opcional) → regresión logística: la
sigmoide, la verosimilitud y por qué no hay fórmula → muchas variables y
softmax: siempre hiperplanos → kernels: curvos en x, lineales en φ(x) →
comparación, ejercicio y cierre.

Para iterar sobre un solo acto: ``CLS_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_separate import SeparateSection
from sections.part02_ols import OLSSection
from sections.part03_fisher import FisherSection
from sections.part04_logistic import LogisticSection
from sections.part05_multi import MultiSection
from sections.part06_kernels import KernelSection
from sections.part07_close import CloseSection
from sections.theme import BG


class ClasificadoresLineales(
    BaseSlide,
    SeparateSection,
    OLSSection,
    FisherSection,
    LogisticSection,
    MultiSection,
    KernelSection,
    CloseSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–10 min
            self.ols,          # 2 · 10–19
            self.fisher,       # 3 · 19–25 (opcional)
            self.logistic,     # 4 · 25–39
            self.multi,        # 5 · 39–48
            self.kernels,      # 6 · 48–56
            self.close,        # 7 · 56–60
        ]
        wanted = os.environ.get("CLS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
