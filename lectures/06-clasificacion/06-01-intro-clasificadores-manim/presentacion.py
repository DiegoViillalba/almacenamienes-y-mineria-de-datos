"""Clase de una hora: introducción a los clasificadores con Manim Slides.

Un solo ejemplo recorre la clase: diez clientes de una financiera (ingreso,
deuda, ¿pagó?) del capítulo 6.0 del libro. Secuencia: qué es clasificar
(atributos, etiqueta, predicción, frontera) → matriz de confusión y métricas →
k vecinos más cercanos → clasificadores lineales y regresión logística →
comparación en clientes nuevos y ejercicio.

Para iterar sobre un solo acto: ``CLS_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_classify import ClassifySection
from sections.part02_confusion import ConfusionSection
from sections.part03_knn import KnnSection
from sections.part04_linear import LinearSection
from sections.part05_compare import CompareSection
from sections.theme import BG


class IntroClasificadores(
    BaseSlide,
    ClassifySection,
    ConfusionSection,
    KnnSection,
    LinearSection,
    CompareSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–10 min
            self.confusion,    # 2 · 10–22
            self.knn,          # 3 · 22–37
            self.linear,       # 4 · 37–52
            self.compare,      # 5 · 52–60
        ]
        wanted = os.environ.get("CLS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
