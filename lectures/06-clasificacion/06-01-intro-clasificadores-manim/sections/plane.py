"""El plano ingreso × endeudamiento, con escala isotrópica.

La misma unidad en ambos ejes hace que una distancia euclidiana se vea como un
círculo: indispensable para la intuición de k-NN.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import GRID_NX, GRID_NY, TEST, TRAIN, X_RANGE, Y_RANGE
from sections.theme import (BG, BLUE, CLASS_COLOR, GRID, INK, MUTED, ORANGE, YELLOW, marker,
                            safe_text)


class ClientPlane(VGroup):
    def __init__(self, height: float = 4.6, center=(-3.5, -0.2, 0), labels: bool = True):
        super().__init__()
        unit = height / (Y_RANGE[1] - Y_RANGE[0])
        width = unit * (X_RANGE[1] - X_RANGE[0])
        self.unit = unit
        self.axes = Axes(
            x_range=[X_RANGE[0], X_RANGE[1], 10], y_range=[Y_RANGE[0], Y_RANGE[1], 10],
            x_length=width, y_length=height, tips=False,
            axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False},
        )
        self.axes.move_to(center)
        self.add(self.axes)
        ticks = VGroup()
        for v in (10, 20, 30, 40):
            ticks.add(safe_text(str(v), size=15, color=MUTED).next_to(
                self.axes.c2p(v, Y_RANGE[0]), DOWN, buff=0.1))
        for v in (20, 30, 40, 50, 60):
            ticks.add(safe_text(f"{v}", size=15, color=MUTED).next_to(
                self.axes.c2p(X_RANGE[0], v), LEFT, buff=0.1))
        self.ticks = ticks
        self.add(ticks)
        if labels:
            self.xlab = safe_text("ingreso (miles $/mes)", size=17, color=MUTED)
            self.xlab.next_to(self.axes.x_axis, DOWN, buff=0.38)
            self.ylab = safe_text("deuda (% del ingreso)", size=17, color=MUTED).rotate(PI / 2)
            self.ylab.next_to(self.axes.y_axis, LEFT, buff=0.42)
            self.add(self.xlab, self.ylab)

    def p(self, x: float, y: float):
        return self.axes.c2p(x, y)

    # -- clientes ---------------------------------------------------------
    def client(self, row, *, show_label: bool = True, unknown: bool = False,
               r: float = 0.13) -> VGroup:
        cid, x, y, cls = row
        m = marker(None if unknown else cls, r=r).move_to(self.p(x, y))
        g = VGroup(m)
        if show_label:
            lab = safe_text(str(cid), size=15, color=INK)
            lab.next_to(m, UR, buff=0.02)
            g.add(lab)
        g.row = row
        g.cls = cls
        return g

    def clients(self, rows=TRAIN, **kw) -> VGroup:
        return VGroup(*[self.client(r, **kw) for r in rows])

    def test_clients(self, **kw) -> VGroup:
        return self.clients(TEST, **kw)

    # -- líneas y regiones -------------------------------------------------
    def hline(self, y: float, color=YELLOW, dashed: bool = True) -> VMobject:
        ln = Line(self.p(X_RANGE[0], y), self.p(X_RANGE[1], y), color=color, stroke_width=4)
        return DashedLine(ln.get_start(), ln.get_end(), color=color, stroke_width=4,
                          dash_length=0.12) if dashed else ln

    def clip_line(self, b: float, w1: float, w2: float):
        """Extremos (en coordenadas de datos) de b + w1·x + w2·y = 0 dentro del rectángulo."""
        x0, x1 = X_RANGE
        y0, y1 = Y_RANGE
        pts = []
        if abs(w2) > 1e-12:
            for x in (x0, x1):
                y = -(b + w1 * x) / w2
                if y0 - 1e-9 <= y <= y1 + 1e-9:
                    pts.append((x, y))
        if abs(w1) > 1e-12:
            for y in (y0, y1):
                x = -(b + w2 * y) / w1
                if x0 - 1e-9 <= x <= x1 + 1e-9:
                    pts.append((x, y))
        pts = sorted(set((round(a, 6), round(c, 6)) for a, c in pts))
        if len(pts) < 2:
            return None
        return pts[0], pts[-1]

    def boundary(self, b, w1, w2, color=YELLOW, width: float = 5) -> Line:
        ends = self.clip_line(b, w1, w2)
        return Line(self.p(*ends[0]), self.p(*ends[1]), color=color, stroke_width=width)

    def half_planes(self, b, w1, w2, opacity: float = 0.16) -> VGroup:
        """Dos polígonos: región «no paga» (z > 0) y «paga» (z < 0)."""
        corners = [(X_RANGE[0], Y_RANGE[0]), (X_RANGE[1], Y_RANGE[0]),
                   (X_RANGE[1], Y_RANGE[1]), (X_RANGE[0], Y_RANGE[1])]
        ends = self.clip_line(b, w1, w2)
        out = VGroup()
        for sign, col in [(1, ORANGE), (-1, BLUE)]:
            poly = []
            ring = corners + [corners[0]]
            for (ax, ay), (bx, by) in zip(ring, ring[1:]):
                za = b + w1 * ax + w2 * ay
                zb = b + w1 * bx + w2 * by
                if sign * za >= 0:
                    poly.append((ax, ay))
                if za * zb < 0:
                    t = za / (za - zb)
                    poly.append((ax + t * (bx - ax), ay + t * (by - ay)))
            if len(poly) >= 3:
                out.add(Polygon(*[self.p(*q) for q in poly], stroke_width=0,
                                fill_color=col, fill_opacity=opacity))
        return out

    def stump_regions(self, t: float, opacity: float = 0.16) -> VGroup:
        return self.half_planes(-t, 0.0, 1.0, opacity)

    def grid_regions(self, grid: np.ndarray, opacity: float = 0.22) -> VGroup:
        xs = np.linspace(X_RANGE[0], X_RANGE[1], GRID_NX + 1)
        ys = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_NY + 1)
        w = (xs[1] - xs[0]) * self.unit
        h = (ys[1] - ys[0]) * self.unit
        cells = VGroup()
        for j in range(GRID_NY):
            i = 0
            while i < GRID_NX:          # une celdas consecutivas de la misma clase
                k = i
                while k + 1 < GRID_NX and grid[j, k + 1] == grid[j, i]:
                    k += 1
                col = interpolate_color(ManimColor(BG), ManimColor(CLASS_COLOR[int(grid[j, i])]),
                                        opacity)
                c = Rectangle(width=w * (k - i + 1) + 0.006, height=h + 0.006, stroke_width=0,
                              fill_color=col, fill_opacity=1.0)
                c.move_to(self.p((xs[i] + xs[k + 1]) / 2, (ys[j] + ys[j + 1]) / 2))
                cells.add(c)
                i = k + 1
        cells.grid = grid
        return cells
