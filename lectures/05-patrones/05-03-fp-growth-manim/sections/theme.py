"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase:
- cada ítem conserva su color (y siempre su nombre o abreviatura);
- verde = frecuente / se conserva, rojo = infrecuente / podado,
  amarillo = foco, pregunta o umbral, gris = estructura.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"
ORANGE = "#FFB86B"
PINK = "#F78FB3"
PURPLE = "#B28DFF"
GREEN = "#6FCF97"    # frecuente / se conserva
YELLOW = "#F7C948"   # foco, pregunta, umbral
RED = "#FF6B6B"      # infrecuente, podado, advertencia

# Ítems: el color sigue la posición en el orden global (1.º azul … 4.º morado).
ITEM_COLOR = {
    "Leche": BLUE, "Pan": ORANGE, "Pañales": PINK, "Cerveza": PURPLE,
    "café": BLUE, "galletas": ORANGE, "chocolate": PINK, "azúcar": PURPLE, "té": MUTED,
}
ABBR = {"Leche": "Le", "Pan": "Pa", "Pañales": "Pñ", "Cerveza": "Ce"}

FONT = "Sans"
MONO = "Menlo"
LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


def color_of(item: str) -> str:
    return ITEM_COLOR.get(item, INK)


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


def text_lines(lines: list[str], *, size: float, color_of_line, font: str = FONT,
               buff: float = 0.14, max_width: float | None = None) -> VGroup:
    """Líneas alineadas a la izquierda que respetan sangría y líneas vacías."""
    em = safe_text("n", size=size, font=font).width
    rows = VGroup()
    for ln in lines:
        body = ln.lstrip(" ")
        mob = safe_text(body or ".", size=size, color=color_of_line(ln), font=font,
                        max_width=max_width)
        if not body:
            mob.set_opacity(0)
        mob.indent = (len(ln) - len(body)) * em
        rows.add(mob)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    for mob in rows:
        mob.shift(RIGHT * mob.indent)
    return rows


def bullet_list(items: list[str], *, size: float = 28, color: str = INK,
                width: float = 11.5, buff: float = 0.28) -> VGroup:
    return text_lines(items, size=size, color_of_line=lambda _: color, buff=buff,
                      max_width=width)


def chip(item: str, *, size: float = 22, label: str | None = None,
         width: float | None = None, height: float = 0.46) -> VGroup:
    """Ficha de un ítem: caja de su color con el nombre (o abreviatura) dentro."""
    col = color_of(item)
    txt = safe_text(label if label is not None else item, size=size, color=col, weight=BOLD)
    w = width if width is not None else txt.width + 0.32
    if txt.width > w - 0.12:
        txt.scale_to_fit_width(w - 0.12)
    box = RoundedRectangle(corner_radius=0.10, width=w, height=height, stroke_color=col,
                           stroke_width=2, fill_color=col, fill_opacity=0.16)
    txt.move_to(box)
    g = VGroup(box, txt)
    g.item = item
    return g


def chip_row(items, *, size: float = 22, buff: float = 0.12, **kw) -> VGroup:
    return VGroup(*[chip(i, size=size, **kw) for i in items]).arrange(RIGHT, buff=buff)


def slot_row(items, universe: list[str], *, slot: float = 0.52, gap: float = 0.06,
             size: float = 17) -> VGroup:
    """Fila tipo matriz binaria: una casilla por ítem del universo, llena si está."""
    row = VGroup()
    for k, it in enumerate(universe):
        if it in items:
            cell = chip(it, label=ABBR.get(it, it[:2]), width=slot, height=0.40, size=size)
        else:
            cell = RoundedRectangle(corner_radius=0.08, width=slot, height=0.40,
                                    stroke_color=GRID, stroke_width=1.2, fill_color=PANEL,
                                    fill_opacity=0.6)
        cell.move_to(RIGHT * k * (slot + gap))
        row.add(cell)
    return row


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


def code_block(lines: list[str], *, size: float = 22, width: float | None = None) -> VGroup:
    """Bloque de código en monoespaciada; ``#`` al inicio se pinta como comentario."""
    rows = text_lines(lines, size=size, font=MONO,
                      color_of_line=lambda ln: MUTED if ln.strip().startswith("#") else INK)
    if width is not None and rows.width > width:
        rows.scale_to_fit_width(width)
    box = card(rows.width + 0.6, rows.height + 0.5)
    box.move_to(rows)
    return VGroup(box, rows)
