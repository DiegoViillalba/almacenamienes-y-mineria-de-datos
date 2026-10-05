"""Versión 4:5 para LinkedIn del Reel de PCA: misma escena, solo reencuadrada.

Reutiliza ``PCAInstagram`` sin cambios. El marco 8.8 × 11 (1080 × 1350) se
centra en la banda que ocupa el contenido del Reel (navegación en y ≈ 5.7,
firma en y ≈ −4.76), así que los textos quedan ≈ 9 % más chicos que en 9:16.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instagram"))
from pca_instagram import PCAInstagram  # noqa: E402

CONTENT_CENTER_Y = 0.47                 # (5.70 + (−4.76)) / 2


class PCALinkedIn(PCAInstagram):
    def setup(self) -> None:
        super().setup()
        self.camera.frame_center = np.array([0.0, CONTENT_CENTER_Y, 0.0])
