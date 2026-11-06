"""Clase de una hora: introducción a las redes neuronales para clasificar, con Manim Slides.

Secuencia: la neurona (suma ponderada + activación = un hiperplano) → Baarle,
una frontera partida en enclaves que ninguna recta separa y que apilar capas
lineales tampoco resuelve → ReLU como bisagra: escalones, enclaves y papel
doblado → la red 2-16-16-1 entrenada: sus rectas, sus regiones lineales, su
frontera y su representación interna → entrenamiento y retropropagación con
números → de redes densas a CNN → ejercicio (XOR) y cierre.

Para iterar sobre un solo acto: ``CLS_PARTS=2,3 manim-slides render -ql ...``.
"""
from __future__ import annotations

import os

from manim import ManimColor
from manim_slides import Slide

from sections.base import BaseSlide
from sections.part01_neuron import NeuronSection
from sections.part02_baarle import BaarleSection
from sections.part03_relu import ReluSection
from sections.part04_network import NetworkSection
from sections.part05_training import TrainingSection
from sections.part06_cnn import CnnSection
from sections.part07_close import CloseSection
from sections.theme import BG


class RedesNeuronales(
    BaseSlide,
    NeuronSection,
    BaarleSection,
    ReluSection,
    NetworkSection,
    TrainingSection,
    CnnSection,
    CloseSection,
    Slide,
):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        acts = [
            self.opening,      # 1 · 0–11 min
            self.baarle,       # 2 · 11–19
            self.relu_act,     # 3 · 19–29
            self.network,      # 4 · 29–38
            self.training,     # 5 · 38–48
            self.cnn,          # 6 · 48–56
            self.close,        # 7 · 56–60
        ]
        wanted = os.environ.get("CLS_PARTS")
        chosen = {int(x) for x in wanted.split(",")} if wanted else None
        for i, act in enumerate(acts, start=1):
            if chosen is None or i in chosen:
                if self.branding is None and i > 1:
                    self.add_branding()
                act()
