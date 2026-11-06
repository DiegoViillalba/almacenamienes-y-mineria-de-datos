"""Planos con escala isotrópica, fronteras lineales y una proyección pseudo-3D.

La misma unidad en ambos ejes hace que un ángulo recto se vea recto: el vector
w se ve perpendicular a la frontera y una distancia se ve como distancia.
"""
from __future__ import annotations

from math import cos, sin

import numpy as np
from manim import *

from sections.theme import BLUE, GRID, INK, MUTED, ORANGE, PURPLE, YELLOW, marker, safe_text


class Plane(VGroup):
    def __init__(self, x_range=(0.0, 10.0), y_range=(0.0, 8.0), *, height: float = 4.8,
                 center=(-3.0, -0.45, 0), xlabel: str | None = "x₁",
                 ylabel: str | None = "x₂", tick_step: float | None = 2.0,
                 axis_color: str = GRID):
        super().__init__()
        self.xr, self.yr = tuple(x_range), tuple(y_range)
        unit = height / (self.yr[1] - self.yr[0])
        self.unit = unit
        width = unit * (self.xr[1] - self.xr[0])
        self.axes = Axes(x_range=[self.xr[0], self.xr[1], 1], y_range=[self.yr[0], self.yr[1], 1],
                         x_length=width, y_length=height, tips=False,
                         axis_config={"color": axis_color, "stroke_width": 2,
                                      "include_ticks": False})
        self.axes.move_to(center)
        self.add(self.axes)
        self.ticks = VGroup()
        if tick_step:
            v = self.xr[0] + tick_step
            while v < self.xr[1] - 1e-9:
                self.ticks.add(safe_text(f"{v:g}", size=14, color=MUTED).next_to(
                    self.p(v, self.yr[0]), DOWN, buff=0.08))
                v += tick_step
            v = self.yr[0] + tick_step
            while v < self.yr[1] - 1e-9:
                self.ticks.add(safe_text(f"{v:g}", size=14, color=MUTED).next_to(
                    self.p(self.xr[0], v), LEFT, buff=0.08))
                v += tick_step
        self.add(self.ticks)
        if xlabel:
            self.xlab = safe_text(xlabel, size=20, color=MUTED).next_to(self.axes.x_axis, RIGHT,
                                                                       buff=0.12)
            self.add(self.xlab)
        if ylabel:
            self.ylab = safe_text(ylabel, size=20, color=MUTED).next_to(self.axes.y_axis, UP,
                                                                       buff=0.10)
            self.add(self.ylab)

    # ------------------------------------------------------------------
    def p(self, x: float, y: float):
        return self.axes.c2p(x, y)

    def dot(self, x, y, cls, r: float = 0.12) -> VMobject:
        m = marker(cls, r=r).move_to(self.p(x, y))
        m.xy = (x, y)
        m.cls = cls
        return m

    def dots(self, pts, r: float = 0.12) -> VGroup:
        return VGroup(*[self.dot(x, y, c, r) for x, y, c in pts])

    # ------------------------------------------------------------------
    def clip(self, b: float, w1: float, w2: float):
        """Extremos (coordenadas de datos) de b + w1·x + w2·y = 0 dentro del rectángulo."""
        (x0, x1), (y0, y1) = self.xr, self.yr
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

    def boundary(self, b, w1, w2, color=YELLOW, width: float = 5, dashed=False) -> VMobject:
        ends = self.clip(b, w1, w2)
        if ends is None:
            return VectorizedPoint(self.p(*self.xr[:1], self.yr[0]))
        cls = DashedLine if dashed else Line
        kw = {"dash_length": 0.12} if dashed else {}
        return cls(self.p(*ends[0]), self.p(*ends[1]), color=color, stroke_width=width, **kw)

    def half_planes(self, b, w1, w2, opacity: float = 0.14, colors=(ORANGE, BLUE)) -> VGroup:
        """Dos polígonos: región g > 0 (naranja, +1) y g < 0 (azul, −1)."""
        (x0, x1), (y0, y1) = self.xr, self.yr
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        out = VGroup()
        for sign, col in [(1, colors[0]), (-1, colors[1])]:
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

    def foot(self, b, w1, w2, x, y):
        """Pie de la perpendicular desde (x, y) a la recta."""
        n2 = w1 * w1 + w2 * w2
        t = (b + w1 * x + w2 * y) / n2
        return x - t * w1, y - t * w2

    def w_arrow(self, b, w1, w2, length: float = 1.3, anchor=None, color=PURPLE) -> Arrow:
        """Flecha de w (en pantalla, longitud fija) desde un punto de la frontera."""
        if anchor is None:
            cx = 0.5 * (self.xr[0] + self.xr[1])
            cy = 0.5 * (self.yr[0] + self.yr[1])
            anchor = self.foot(b, w1, w2, cx, cy)
        start = self.p(*anchor)
        d = np.array([w1, w2, 0.0])
        d /= np.linalg.norm(d)
        return Arrow(start, start + length * d, buff=0, color=color, stroke_width=6,
                     max_tip_length_to_length_ratio=0.22)


def legend(pos_text="clase +1", neg_text="clase −1", size: float = 19) -> VGroup:
    a = VGroup(marker(-1, r=0.10), safe_text(neg_text, size=size, color=MUTED)).arrange(RIGHT,
                                                                                    buff=0.12)
    b = VGroup(marker(1, r=0.10), safe_text(pos_text, size=size, color=MUTED)).arrange(RIGHT,
                                                                                   buff=0.12)
    return VGroup(a, b).arrange(RIGHT, buff=0.4)


# ---------------------------------------------------------------------------
# Pseudo-3D: proyección ortográfica con azimut y elevación.
# ---------------------------------------------------------------------------
class View3D:
    def __init__(self, origin=(0, 0, 0), scale: float = 1.0, az: float = -0.6,
                 el: float = 0.45):
        self.origin = np.array(origin, float)
        self.scale = scale
        self.az = az
        self.el = el

    def __call__(self, x, y, z, az=None, el=None):
        az = self.az if az is None else az
        el = self.el if el is None else el
        u = x * cos(az) - y * sin(az)
        v = x * sin(az) + y * cos(az)
        sx = u
        sy = z * cos(el) + v * sin(el)
        return self.origin + self.scale * np.array([sx, sy, 0.0])

    def depth(self, x, y, z, az=None):
        az = self.az if az is None else az
        return -(x * sin(az) + y * cos(az))


def box_edges(view: View3D, lo=(0, 0, 0), hi=(4, 4, 4), az=None, el=None,
              color=GRID, width: float = 2) -> VGroup:
    xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    V = {(i, j, k): view(xs[i], ys[j], zs[k], az, el) for i in (0, 1) for j in (0, 1)
         for k in (0, 1)}
    out = VGroup()
    for a in V:
        for d in range(3):
            if a[d] == 0:
                bb = list(a)
                bb[d] = 1
                out.add(Line(V[a], V[tuple(bb)], color=color, stroke_width=width))
    return out


def clip_poly(poly, a: float, b: float, c: float):
    """Sutherland–Hodgman: conserva la parte de ``poly`` con a + b·x + c·y ≥ 0."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp = a + b * p[0] + c * p[1]
        fq = a + b * q[0] + c * q[1]
        if fp >= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def plane_cube_polygon(w, b, lo=(0, 0, 0), hi=(4, 4, 4)):
    """Polígono (en 3D) de {x : wᵀx + b = 0} dentro del cubo, ordenado."""
    w = np.asarray(w, float)
    xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    V = {(i, j, k): np.array([xs[i], ys[j], zs[k]], float) for i in (0, 1) for j in (0, 1)
         for k in (0, 1)}
    pts = []
    for a in V:
        for d in range(3):
            if a[d] == 0:
                bb = list(a)
                bb[d] = 1
                p, q = V[a], V[tuple(bb)]
                fp, fq = w @ p + b, w @ q + b
                if fp * fq < 0:
                    t = fp / (fp - fq)
                    pts.append(p + t * (q - p))
    if len(pts) < 3:
        return []
    c = np.mean(pts, axis=0)
    n = w / np.linalg.norm(w)
    e1 = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    pts.sort(key=lambda p: np.arctan2((p - c) @ e2, (p - c) @ e1))
    return pts
