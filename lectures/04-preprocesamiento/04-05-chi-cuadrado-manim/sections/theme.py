"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase:
- verde (círculo) = satisfacción alta; naranja (cuadrado) = satisfacción baja;
- azul = frecuencia observada O; amarillo = frecuencia esperada E (mundo
  independiente, bordes punteados); rosa = residuo O − E;
- morado = estadístico χ² y su distribución; rojo = advertencia o descarte;
- turquesa = fuerza de asociación (V de Cramér).
La forma acompaña siempre al color en las personas (círculo / cuadrado).
"""
from __future__ import annotations

from pathlib import Path

from manim import *

BG = "#07111F"
PANEL = "#0D1E34"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"     # observado
YELLOW = "#F7C948"   # esperado
PINK = "#F78FB3"     # residuo
PURPLE = "#B28DFF"   # χ²
TEAL = "#4FD1C5"     # V de Cramér
GREEN = "#6FCF97"    # satisfacción alta
ORANGE = "#FFB86B"   # satisfacción baja
RED = "#FF6B6B"      # advertencia / descarte

SAT_COLOR = {1: GREEN, 0: ORANGE}

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


def person(sat: int, size: float = 0.13) -> VMobject:
    """Pasajero: círculo verde (alta) o cuadrado naranja (baja)."""
    if sat == 1:
        return Circle(radius=size / 2, stroke_width=0, fill_color=GREEN, fill_opacity=1)
    return Square(side_length=size * 0.9, stroke_width=0, fill_color=ORANGE, fill_opacity=1)


def people_grid(labels, *, cols: int, size: float = 0.13, gap: float = 0.07) -> VGroup:
    group = VGroup(*[person(int(s), size) for s in labels])
    step = size + gap
    for i, m in enumerate(group):
        r, c = divmod(i, cols)
        m.move_to(RIGHT * c * step + DOWN * r * step)
    return group.center()


def fmt(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


class Table2(VGroup):
    """Tabla de contingencia con celdas accesibles por ``cell(i, j)``."""

    def __init__(self, values, rows, cols, *, cw: float = 1.7, ch: float = 0.8,
                 color: str = BLUE, size: float = 30, row_w: float = 2.3,
                 dashed: bool = False, totals: bool = False, nd: int = 0,
                 label_size: float = 22):
        super().__init__()
        values = [[v for v in row] for row in values]
        self.rows_n, self.cols_n = len(rows), len(cols)
        self.boxes, self.nums = {}, {}
        for i in range(self.rows_n):
            for j in range(self.cols_n):
                box = Rectangle(width=cw, height=ch, stroke_color=color, stroke_width=2,
                                fill_color=color, fill_opacity=0.08)
                if dashed:
                    box = DashedVMobject(box, num_dashes=24)
                box.move_to(RIGHT * (row_w / 2 + cw * (j + 0.5)) + DOWN * ch * (i + 0.5))
                v = values[i][j]
                num = safe_text(fmt(v, nd) if isinstance(v, (int, float)) else str(v),
                                size=size, color=color, weight=BOLD).move_to(box)
                self.boxes[i, j], self.nums[i, j] = box, num
                self.add(box, num)
        self.row_labels = VGroup(*[
            safe_text(r, size=label_size, color=MUTED, max_width=row_w - 0.15)
            .move_to(RIGHT * 0 + DOWN * ch * (i + 0.5)) for i, r in enumerate(rows)])
        for lab in self.row_labels:
            lab.align_to(RIGHT * (row_w / 2 - 0.12), RIGHT)
        self.col_labels = VGroup(*[
            safe_text(c, size=label_size, color=MUTED, max_width=cw - 0.1)
            .move_to(RIGHT * (row_w / 2 + cw * (j + 0.5)) + UP * 0.32)
            for j, c in enumerate(cols)])
        self.add(self.row_labels, self.col_labels)
        self.row_tot = VGroup()
        self.col_tot = VGroup()
        if totals:
            arr = values
            for i in range(self.rows_n):
                t = safe_text(fmt(sum(arr[i]), nd), size=size - 4, color=INK)
                t.next_to(self.boxes[i, self.cols_n - 1], RIGHT, buff=0.35)
                self.row_tot.add(t)
            for j in range(self.cols_n):
                t = safe_text(fmt(sum(arr[i][j] for i in range(self.rows_n)), nd),
                              size=size - 4, color=INK)
                t.next_to(self.boxes[self.rows_n - 1, j], DOWN, buff=0.25)
                self.col_tot.add(t)
            self.add(self.row_tot, self.col_tot)
        self.center()

    def cell(self, i: int, j: int) -> VMobject:
        return self.boxes[i, j]

    def num(self, i: int, j: int) -> Text:
        return self.nums[i, j]

    def grid_part(self) -> VGroup:
        return VGroup(*self.boxes.values(), *self.nums.values())


def stacked_bar(p_high: float, *, width: float = 1.1, height: float = 3.2,
                show_pct: bool = True, size: float = 22) -> VGroup:
    """Barra al 100 %: verde abajo (alta), naranja arriba (baja)."""
    hi = Rectangle(width=width, height=max(0.001, height * p_high), stroke_width=0,
                   fill_color=GREEN, fill_opacity=0.9)
    lo = Rectangle(width=width, height=max(0.001, height * (1 - p_high)), stroke_width=0,
                   fill_color=ORANGE, fill_opacity=0.9)
    hi.move_to(DOWN * height / 2 + UP * hi.height / 2)
    lo.next_to(hi, UP, buff=0)
    group = VGroup(hi, lo)
    if show_pct:
        t1 = safe_text(f"{round(100 * p_high)}%", size=size, color=BG, weight=BOLD).move_to(hi)
        t2 = safe_text(f"{round(100 * (1 - p_high))}%", size=size, color=BG, weight=BOLD).move_to(lo)
        if hi.height < 0.35:
            t1.set_opacity(0)
        if lo.height < 0.35:
            t2.set_opacity(0)
        group.add(t1, t2)
    return group


def sat_legend(size: float = 20) -> VGroup:
    a = VGroup(person(1, 0.2), safe_text("satisfacción alta", size=size, color=GREEN)).arrange(RIGHT, buff=0.12)
    b = VGroup(person(0, 0.2), safe_text("satisfacción baja", size=size, color=ORANGE)).arrange(RIGHT, buff=0.12)
    return VGroup(a, b).arrange(RIGHT, buff=0.5)
