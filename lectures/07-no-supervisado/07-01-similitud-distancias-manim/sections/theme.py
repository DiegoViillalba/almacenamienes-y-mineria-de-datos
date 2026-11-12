"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase:
- azul = casos / datos; amarillo = el punto de consulta (Ana, q);
- turquesa = distancia (una regla que cumple los axiomas);
- morado = similitud; naranja = medida de desempeño;
- rojo = axioma violado o advertencia; verde = axioma cumplido / decisión válida;
- rosa = alta dimensión y vecindarios de UMAP.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"     # datos
YELLOW = "#F7C948"   # consulta / Ana
TEAL = "#4FD1C5"     # distancia
PURPLE = "#B28DFF"   # similitud
ORANGE = "#FFB86B"   # desempeño
RED = "#FF6B6B"      # violación / advertencia
GREEN = "#6FCF97"    # cumple
PINK = "#F78FB3"     # alta dimensión / UMAP

FONT = "Sans"
LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


def safe_text(content: str, *, size: float = 32, color: str = INK,
              weight: str = NORMAL, max_width: float | None = None) -> Text:
    mob = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def tex(content, *, size: float = 40, color: str = INK,
        max_width: float | None = 12.0, **kw) -> MathTex:
    parts = content if isinstance(content, (list, tuple)) else [content]
    mob = MathTex(*parts, font_size=size, color=color, **kw)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def pill(content: str, color: str, *, width: float | None = None,
         size: float = 24) -> VGroup:
    label = safe_text(content, size=size, color=color, weight=BOLD)
    w = width if width is not None else label.width + 0.55
    if label.width > w - 0.3:
        label.scale_to_fit_width(w - 0.3)
    box = RoundedRectangle(corner_radius=0.2, width=w, height=label.height + 0.34,
                           stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.12)
    label.move_to(box)
    return VGroup(box, label)


def card(width: float, height: float, color: str = GRID) -> RoundedRectangle:
    return RoundedRectangle(corner_radius=0.14, width=width, height=height,
                            stroke_color=color, stroke_width=1.6,
                            fill_color=PANEL, fill_opacity=0.92)


def boxed(content: Mobject, color: str = GRID, *, pad: float = 0.3) -> VGroup:
    box = card(content.width + 2 * pad, content.height + 2 * pad, color).move_to(content)
    return VGroup(box, content)


def fmt(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def check(color: str = GREEN, size: float = 0.32) -> VMobject:
    return VMobject(stroke_color=color, stroke_width=6).set_points_as_corners(
        [LEFT * size * 0.5 + UP * size * 0.05, DOWN * size * 0.35, RIGHT * size * 0.6 + UP * size * 0.5])


def cross(color: str = RED, size: float = 0.28) -> VGroup:
    a = Line(UL * size / 2, DR * size / 2, color=color, stroke_width=6)
    b = Line(UR * size / 2, DL * size / 2, color=color, stroke_width=6)
    return VGroup(a, b)


def gradient_color(t: float) -> ManimColor:
    """Color a lo largo del rollo: azul → rosa → amarillo, t ∈ [0, 1]."""
    if t < 0.5:
        return interpolate_color(ManimColor(BLUE), ManimColor(PINK), t * 2)
    return interpolate_color(ManimColor(PINK), ManimColor(YELLOW), (t - 0.5) * 2)
