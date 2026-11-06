"""Clase de una hora: árboles de decisión con Manim Slides.

Secuencia: preguntas que parten el plano (los diez clientes de la clase
anterior) → entropía y ganancia de información (ID3, béisbol) → C4.5: razón de
ganancia y atributos numéricos → Gini y cortes binarios (CART, laptops y
clientes) → sobreajuste y poda de costo-complejidad (60 puntos) → ejercicio y
cierre.

Para iterar sobre un solo acto: ``CLS_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_partition import PartitionSection
from sections.part02_entropy import EntropySection
from sections.part03_c45 import C45Section
from sections.part04_cart import CartSection
from sections.part05_pruning import PruningSection
from sections.part06_close import CloseSection
from sections.theme import BG


class ArbolesDecision(
    BaseSlide,
    PartitionSection,
    EntropySection,
    C45Section,
    CartSection,
    PruningSection,
    CloseSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–10 min
            self.entropy,      # 2 · 10–23
            self.c45,          # 3 · 23–31
            self.cart,         # 4 · 31–43
            self.pruning,      # 5 · 43–54
            self.close,        # 6 · 54–60
        ]
        wanted = os.environ.get("CLS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
