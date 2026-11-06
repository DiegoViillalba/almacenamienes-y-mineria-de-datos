"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase (cambiar aquí cambia toda la clase):
- azul + círculo = clase −1 (y = 0); naranja + cuadrado = clase +1 (y = 1);
- amarillo = frontera de decisión (g = 0) y foco;
- morado = vector de pesos w y puntaje z; verde = probabilidad / sigmoide;
- rojo = error y pérdida; turquesa = proyecciones (sombras de Fisher);
- rosa = espacio de características φ(x) y kernels; gris = estructura.
La forma (círculo/cuadrado) acompaña siempre al color.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"     # clase −1
ORANGE = "#FFB86B"   # clase +1
PURPLE = "#B28DFF"   # w y puntaje z
TEAL = "#4FD1C5"     # sombras / proyecciones
PINK = "#F78FB3"     # espacio de características
GREEN = "#6FCF97"    # probabilidad, acierto
YELLOW = "#F7C948"   # frontera, foco
RED = "#FF6B6B"      # error

CLASS_COLOR = {0: BLUE, 1: ORANGE, -1: BLUE}
CELL_COLOR = {"TP": GREEN, "TN": GREEN, "FP": RED, "FN": RED}

FONT = "Sans"
MONO = "Menlo"
LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


def safe_text(content: str, *, size: float = 32, color: str = INK,
              weight: str = NORMAL, max_width: float | None = None,
              font: str = FONT) -> Text:
    mob = Text(content, font=font, font_size=size, color=color, weight=weight)
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
    rows = VGroup(*[safe_text(t, size=size, color=color, max_width=width) for t in items])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return rows


def table(rows: list[list[str]], *, col_widths: list[float], size: float = 24,
          row_h: float = 0.52, header_color: str = MUTED,
          colors: list[list[str]] | None = None) -> VGroup:
    """Tabla simple de texto. Devuelve VGroup(filas…, regla)."""
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


def marker(cls: int | None, *, r: float = 0.13, filled: bool = True) -> VMobject:
    """Círculo azul (clase −1 ó 0), cuadrado naranja (+1) o rombo amarillo (desconocido)."""
    if cls is None:
        m = Square(side_length=2 * r * 0.95).rotate(PI / 4)
        m.set_stroke(YELLOW, 3).set_fill(YELLOW, 0.9 if filled else 0.0)
        return m
    col = CLASS_COLOR[cls]
    m = Circle(radius=r) if cls <= 0 else Square(side_length=2 * r * 0.88)
    m.set_stroke(col, 3).set_fill(col, 0.9 if filled else 0.0)
    return m


def legend_item(cls: int, text: str, size: float = 20) -> VGroup:
    return VGroup(marker(cls, r=0.11), safe_text(text, size=size, color=MUTED)
                  ).arrange(RIGHT, buff=0.15)


def fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"
