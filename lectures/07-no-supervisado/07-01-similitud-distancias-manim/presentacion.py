"""Clase de una hora: similitud y distancias con Manim Slides.

Secuencia: ¿quién se parece más a Ana? → qué significa medir y tres reglas
rotas que hacen nacer los axiomas → por qué exigirlos (descartar sin calcular)
y la diferencia entre medida, distancia, similitud y desempeño → la forma de
una distancia en 2D (Minkowski, escala, Mahalanobis como blanqueo) → cambiar
el punto de vista en 3D (coseno sobre la esfera, proyecciones) → la esfera en
alta dimensión → UMAP → volver con Ana, reto y cierre.

Para iterar sobre un solo acto: ``SIM_PARTS=4,5 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import ThreeDSlide

from sections.base import BaseSlide
from sections.part01_opening import OpeningSection
from sections.part02_measure import MeasureSection
from sections.part03_concepts import ConceptsSection
from sections.part04_geometry2d import Geometry2DSection
from sections.part05_views3d import Views3DSection
from sections.part06_curse import CurseSection
from sections.part07_umap import UmapSection
from sections.part08_close import CloseSection
from sections.theme import BG


class SimilitudDistancias(
    BaseSlide,
    OpeningSection,
    MeasureSection,
    ConceptsSection,
    Geometry2DSection,
    Views3DSection,
    CurseSection,
    UmapSection,
    CloseSection,
    ThreeDSlide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,       # 1 · 0–5 min
            self.measure,       # 2 · 5–14
            self.concepts,      # 3 · 14–21
            self.geometry_2d,   # 4 · 21–31
            self.views_3d,      # 5 · 31–37
            self.curse,         # 6 · 37–45
            self.umap,          # 7 · 45–54
            self.close_gower,   # 8 · 54–56
            self.challenge,     # 9 · 56–58
            self.closing,       # 10 · 58–60
        ]
        wanted = os.environ.get("SIM_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
