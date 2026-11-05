"""Matriz de confusión animable: los clientes viajan del plano a su celda."""
from __future__ import annotations

from manim import *

from sections.theme import (BLUE, CELL_COLOR, GRID, INK, MUTED, ORANGE, PANEL, safe_text)

# (fila, columna) de cada celda. Filas: real no paga, real paga.
# Columnas: predijo no paga, predijo paga. Positivo = no paga.
CELL_POS = {"TP": (0, 0), "FN": (0, 1), "FP": (1, 0), "TN": (1, 1)}
CELL_NAME = {"TP": "verdadero positivo", "FN": "falso negativo",
             "FP": "falso positivo", "TN": "verdadero negativo"}


class ConfMatrix(VGroup):
    def __init__(self, center=(3.4, 0.4, 0), cw: float = 1.8, ch: float = 1.25,
                 header_size: float = 19, codes: bool = True):
        super().__init__()
        self.cw, self.ch = cw, ch
        self.cells = {}
        self.slots_used = {k: 0 for k in CELL_POS}
        origin = np.array(center, float)
        boxes = VGroup()
        for code, (r, c) in CELL_POS.items():
            box = Rectangle(width=cw, height=ch, stroke_color=GRID, stroke_width=2,
                            fill_color=PANEL, fill_opacity=0.9)
            box.move_to(origin + np.array([(c - 0.5) * cw, (0.5 - r) * ch, 0]))
            self.cells[code] = box
            boxes.add(box)
        self.boxes = boxes
        self.add(boxes)
        self.code_labels = VGroup()
        if codes:
            for code, box in self.cells.items():
                lab = safe_text(code, size=18, color=MUTED, weight=BOLD)
                lab.move_to(box.get_corner(UL) + np.array([0.3, -0.2, 0]))
                self.code_labels.add(lab)
        # encabezados
        col_h = VGroup(
            safe_text("no paga", size=header_size, color=ORANGE, weight=BOLD)
            .next_to(self.cells["TP"], UP, buff=0.1),
            safe_text("paga", size=header_size, color=BLUE, weight=BOLD)
            .next_to(self.cells["FN"], UP, buff=0.1),
        )
        pred_t = safe_text("predicción  ŷ", size=header_size, color=MUTED).next_to(
            col_h, UP, buff=0.1)
        row_h = VGroup(
            safe_text("no paga", size=header_size, color=ORANGE, weight=BOLD)
            .next_to(self.cells["TP"], LEFT, buff=0.12),
            safe_text("paga", size=header_size, color=BLUE, weight=BOLD)
            .next_to(self.cells["FP"], LEFT, buff=0.12),
        )
        real_t = safe_text("real  y", size=header_size, color=MUTED).rotate(PI / 2)
        real_t.next_to(row_h, LEFT, buff=0.12)
        self.headers = VGroup(col_h, pred_t, row_h, real_t)
        self.add(self.headers)
        self.counts = {}

    def frame(self) -> VGroup:
        return VGroup(self.boxes, self.headers)

    def slot(self, code: str, k: int, per_row: int = 6):
        box = self.cells[code]
        step = min(0.3, (self.cw - 0.3) / per_row)
        row, col = divmod(k, per_row)
        x0 = box.get_left()[0] + 0.15 + step / 2
        y0 = box.get_bottom()[1] + 0.25 + row * 0.3
        return np.array([x0 + col * step, y0, 0])

    def count_mob(self, code: str, n: int, size: float = 36) -> Text:
        col = CELL_COLOR[code] if n else MUTED
        mob = safe_text(str(n), size=size, color=col, weight=BOLD)
        box = self.cells[code]
        mob.move_to(box.get_corner(UR) + np.array([-0.32, -0.32, 0]))
        return mob

    def highlight(self, codes, color, buff: float = 0.04) -> SurroundingRectangle:
        grp = VGroup(*[self.cells[c] for c in codes])
        return SurroundingRectangle(grp, color=color, buff=buff, stroke_width=5)


def metric_bars(values: dict, *, origin=(1.2, -1.6, 0), length: float = 3.6,
                size: float = 22) -> tuple[VGroup, dict]:
    """Barras horizontales con ValueTrackers. ``values`` = {nombre: (valor, color)}."""
    trackers = {}
    group = VGroup()
    o = np.array(origin, float)
    for i, (name, (v, col)) in enumerate(values.items()):
        y = -i * 0.5
        tr = ValueTracker(v)
        trackers[name] = tr
        lab = safe_text(name, size=size, color=MUTED).move_to(o + [0, y, 0], aligned_edge=LEFT)
        base_x = o[0] + 1.9
        track = Rectangle(width=length, height=0.24, stroke_color=GRID, stroke_width=1.5,
                          fill_opacity=0).move_to([base_x + length / 2, o[1] + y, 0])

        def bar(tr=tr, col=col, y=y, base_x=base_x):
            w = max(tr.get_value(), 1e-3) * length
            return Rectangle(width=w, height=0.24, stroke_width=0, fill_color=col,
                             fill_opacity=0.85).move_to([base_x + w / 2, o[1] + y, 0])

        def val(tr=tr, col=col, y=y, base_x=base_x):
            return safe_text(f"{tr.get_value():.2f}", size=size, color=col, weight=BOLD
                             ).move_to([base_x + length + 0.45, o[1] + y, 0])

        group.add(VGroup(lab, track, always_redraw(bar), always_redraw(val)))
    return group, trackers
