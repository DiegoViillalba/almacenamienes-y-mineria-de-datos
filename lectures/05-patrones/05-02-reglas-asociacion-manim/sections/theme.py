"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase:
azul = antecedente A, naranja = consecuente B, verde = asociación positiva,
amarillo = referencia de independencia, rojo = errores / asociación negativa,
morado = métricas complementarias.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"     # antecedente A
ORANGE = "#FFB86B"   # consecuente B
GREEN = "#6FCF97"    # asociación positiva / decisión favorable
YELLOW = "#F7C948"   # nivel de independencia
RED = "#FF6B6B"      # errores, asociación negativa, advertencias
PURPLE = "#B28DFF"   # métricas complementarias

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


def value_row(label: str, value: str, color: str = INK, *, size: float = 26,
              label_width: float = 2.6) -> VGroup:
    """Etiqueta a la izquierda y valor en negritas, alineados como en una ficha."""
    lab = safe_text(label, size=size, color=MUTED)
    val = safe_text(value, size=size, color=color, weight=BOLD)
    val.next_to(lab, RIGHT, buff=0.25).set_x(lab.get_left()[0] + label_width + val.width / 2)
    return VGroup(lab, val)


def table(rows: list[list[str]], *, col_widths: list[float], size: float = 24,
          row_h: float = 0.52, header_color: str = MUTED,
          colors: list[list[str]] | None = None) -> VGroup:
    """Tabla simple de texto. Devuelve VGroup(filas), cada fila VGroup(celdas)."""
    out = VGroup()
    for i, r in enumerate(rows):
        row = VGroup()
        x = 0.0
        for j, (content, w) in enumerate(zip(r, col_widths)):
            if i == 0:
                c = header_color
            elif colors is not None:
                c = colors[i - 1][j]
            else:
                c = INK
            cell = safe_text(content, size=size, color=c,
                             weight=BOLD if i == 0 else NORMAL, max_width=w - 0.15)
            cell.move_to([x + w / 2, -i * row_h, 0])
            row.add(cell)
            x += w
        out.add(row)
    rule = Line([0, -row_h / 2, 0], [sum(col_widths), -row_h / 2, 0], color=GRID,
                stroke_width=2)
    out.add(rule)
    return out
