"""Paleta, tipografía y piezas visuales reutilizables."""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"     # datos observados
YELLOW = "#F7C948"   # centro / media
ORANGE = "#FFB86B"   # PC1
GREEN = "#6FCF97"    # PC2
PURPLE = "#B28DFF"   # PC3
RED = "#FF6B6B"      # residuo / advertencia
CAM_COLORS = {"A": BLUE, "B": GREEN, "C": PURPLE}

FONT = "Sans"
LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


def safe_text(content: str, *, size: float = 32, color: str = INK,
              weight: str = NORMAL, max_width: float | None = None) -> Text:
    mob = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def tex(content: str, *, size: float = 40, color: str = INK,
        max_width: float | None = 12.0) -> MathTex:
    mob = MathTex(content, font_size=size, color=color)
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


def bullet_list(items: list[str], *, size: float = 28, color: str = INK,
                width: float = 11.5, buff: float = 0.28) -> VGroup:
    rows = VGroup(*[safe_text(s, size=size, color=color, max_width=width) for s in items])
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)


def pve_bars(pve, *, labels=None, height: float = 2.4, width: float = 0.62,
             gap: float = 0.28, colors=None, show_pct: bool = True,
             label_size: float = 20) -> VGroup:
    """Barras de proporción de varianza explicada (escala 0–100 %)."""
    labels = labels or [f"PC{i + 1}" for i in range(len(pve))]
    colors = colors or [ORANGE, GREEN, PURPLE] + [MUTED] * 10
    group = VGroup()
    base = Line(LEFT * 0.1, RIGHT * (len(pve) * (width + gap)), color=MUTED, stroke_width=2)
    group.add(base)
    for i, (p, lab) in enumerate(zip(pve, labels)):
        h = max(0.02, float(p) * height)
        bar = Rectangle(width=width, height=h, stroke_width=0,
                        fill_color=colors[i], fill_opacity=0.92)
        bar.move_to(base.get_left() + RIGHT * (0.1 + width / 2 + i * (width + gap)) + UP * h / 2)
        name = safe_text(lab, size=label_size, color=MUTED).next_to(bar, DOWN, buff=0.12)
        name.set_y(base.get_y() - 0.22)
        group.add(bar, name)
        if show_pct:
            pct = safe_text(f"{100 * p:.1f}%", size=label_size, color=INK, weight=BOLD)
            pct.next_to(bar, UP, buff=0.08)
            group.add(pct)
    return group
