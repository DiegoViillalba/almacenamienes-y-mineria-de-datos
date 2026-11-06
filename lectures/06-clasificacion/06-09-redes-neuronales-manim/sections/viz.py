"""Piezas visuales del mapa: lienzo, polígonos, rásteres, rectas y contornos."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (BE_SOUTH, COUNTER, ENCLAVES, H_MAP, W_MAP, country, forward, grid,
                           logit_grid, to_unit)
from sections.theme import BG, BLUE, GRID, INK, MUTED, ORANGE, YELLOW, marker, safe_text


def _rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


BLUE_RGB, ORANGE_RGB, BG_RGB = _rgb(BLUE), _rgb(ORANGE), _rgb(BG)


class MapView:
    """Convierte coordenadas del mapa (10 × 7) en puntos de la escena."""

    def __init__(self, width: float = 7.0, center=(0.0, -0.4, 0.0)):
        self.w = width
        self.s = width / W_MAP
        self.h = H_MAP * self.s
        self.c = np.array(center, float)

    def p(self, x: float, y: float) -> np.ndarray:
        return self.c + np.array([(x - W_MAP / 2) * self.s, (y - H_MAP / 2) * self.s, 0.0])

    # -- marco y rótulos ------------------------------------------------
    def frame(self, color=GRID) -> Rectangle:
        return Rectangle(width=self.w, height=self.h, stroke_color=color,
                         stroke_width=2).move_to(self.c)

    # -- rásteres --------------------------------------------------------
    def image(self, rgba: np.ndarray) -> ImageMobject:
        img = ImageMobject(rgba.astype(np.uint8))
        img.stretch_to_fit_width(self.w)
        img.stretch_to_fit_height(self.h)
        img.move_to(self.c)
        return img

    def truth_image(self, nx=600, ny=420, alpha=0.32) -> ImageMobject:
        X, Y = grid(nx, ny)
        c = country(X, Y)[..., None]
        rgb = np.where(c == 0, BLUE_RGB, ORANGE_RGB)
        rgb = BG_RGB * (1 - alpha) + rgb * alpha
        return self.image(np.dstack([rgb, np.full(c.shape, 255.0)]))

    def prob_image(self, params, nx=600, ny=420, alpha=0.42, *, linear=False,
                   sharp: float = 1.0) -> ImageMobject:
        z = logit_grid(params, nx, ny, linear=linear)
        p = 1 / (1 + np.exp(-np.clip(z * sharp, -40, 40)))
        rgb = BLUE_RGB * (1 - p[..., None]) + ORANGE_RGB * p[..., None]
        rgb = BG_RGB * (1 - alpha) + rgb * alpha
        return self.image(np.dstack([rgb, np.full(p.shape, 255.0)]))

    def field_image(self, F: np.ndarray, *, alpha=0.5, scale=None) -> ImageMobject:
        """Campo escalar: azul si F > 0 (Bélgica), naranja si F < 0, intensidad |F|."""
        s = scale if scale is not None else max(np.abs(F).max(), 1e-9)
        t = np.clip(F / s, -1, 1)[..., None]
        rgb = np.where(t > 0, BLUE_RGB, ORANGE_RGB)
        a = alpha * np.abs(t) ** 0.6
        rgb = BG_RGB * (1 - a) + rgb * a
        return self.image(np.dstack([rgb, np.full(t.shape[:2], 255.0)]))

    def region_image(self, ids: np.ndarray, alpha=0.38, seed=4) -> ImageMobject:
        rng = np.random.default_rng(seed)
        n = ids.max() + 1
        hues = rng.uniform(0, 1, n)
        pal = np.array([_hsv(h, 0.55, 0.95) for h in hues])
        rgb = pal[ids]
        rgb = BG_RGB * (1 - alpha) + rgb * alpha
        return self.image(np.dstack([rgb, np.full(ids.shape, 255.0)]))

    # -- vectores --------------------------------------------------------
    def polygon(self, pts, color, *, width=3, fill=0.0) -> Polygon:
        return Polygon(*[self.p(*q) for q in pts], stroke_color=color, stroke_width=width,
                       fill_color=color, fill_opacity=fill)

    def borders(self, color=INK, width=2.5) -> VGroup:
        """Contornos de los pedazos belgas y del contraenclave."""
        south = [q for q in BE_SOUTH if not (q[1] == 0.0)]
        g = VGroup(VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
            [self.p(*q) for q in south]))
        for pts in ENCLAVES.values():
            g.add(self.polygon(pts, color, width=width))
        for pts in COUNTER.values():
            g.add(self.polygon(pts, color, width=width).set_stroke(opacity=0.9))
        return g

    def dots(self, P, Y, r=0.055) -> VGroup:
        g = VGroup()
        for (x, y), c in zip(P, Y):
            m = marker(int(c), r=r)
            m.set_stroke(width=1.2)
            g.add(m.move_to(self.p(x, y)))
        return g

    def line(self, a: float, c: float, d: float, color=YELLOW, width=3) -> Line | None:
        """Recta a·x + c·y = d recortada al rectángulo del mapa."""
        pts = []
        if abs(c) > 1e-9:
            for x in (0.0, W_MAP):
                y = (d - a * x) / c
                if 0 <= y <= H_MAP:
                    pts.append((x, y))
        if abs(a) > 1e-9:
            for y in (0.0, H_MAP):
                x = (d - c * y) / a
                if 0 <= x <= W_MAP:
                    pts.append((x, y))
        uniq = []
        for q in pts:
            if all(np.hypot(q[0] - u[0], q[1] - u[1]) > 1e-6 for u in uniq):
                uniq.append(q)
        if len(uniq) < 2:
            return None
        return Line(self.p(*uniq[0]), self.p(*uniq[1]), color=color, stroke_width=width)

    def contour(self, F: np.ndarray, color=YELLOW, width=5, level=0.0) -> VMobject:
        """Curva de nivel de un campo evaluado en ``grid`` (fila 0 = arriba)."""
        ny, nx = F.shape
        xs = (np.arange(nx) + 0.5) / nx * W_MAP
        ys = ((np.arange(ny) + 0.5) / ny * H_MAP)[::-1]
        mob = VMobject(stroke_color=color, stroke_width=width)
        for a, b in marching_squares(F - level, xs, ys):
            mob.start_new_path(self.p(*a))
            mob.add_line_to(self.p(*b))
        return mob


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return np.array([r, g, b]) * 255


def marching_squares(F, xs, ys):
    """Segmentos del nivel 0 de F (F[i, j] en (xs[j], ys[i]))."""
    segs = []
    ny, nx = F.shape
    sign = F > 0
    for i in range(ny - 1):
        row_change = (sign[i, :-1] != sign[i, 1:]) | (sign[i + 1, :-1] != sign[i + 1, 1:]) | \
                     (sign[i, :-1] != sign[i + 1, :-1])
        for j in np.nonzero(row_change)[0]:
            corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            pts = []
            for k in range(4):
                (i1, j1), (i2, j2) = corners[k], corners[(k + 1) % 4]
                f1, f2 = F[i1, j1], F[i2, j2]
                if (f1 > 0) != (f2 > 0):
                    t = f1 / (f1 - f2)
                    pts.append((xs[j1] + t * (xs[j2] - xs[j1]), ys[i1] + t * (ys[i2] - ys[i1])))
            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                segs.append((pts[0], pts[1]))
                segs.append((pts[2], pts[3]))
    return segs


def hidden_fields(params, layer: int, nx=300, ny=210):
    """Preactivaciones z_k de una capa oculta en la malla."""
    X, Y = grid(nx, ny)
    P = np.column_stack([X.ravel(), Y.ravel()])
    _, acts = forward(params, to_unit(P))
    z = acts[layer][0]
    return [z[:, k].reshape(X.shape) for k in range(z.shape[1])]


def map_legend(size=20) -> VGroup:
    return VGroup(
        VGroup(marker(0, r=0.1), safe_text("Bélgica (Baarle-Hertog)", size=size, color=MUTED)
               ).arrange(RIGHT, buff=0.15),
        VGroup(marker(1, r=0.1), safe_text("Países Bajos (Baarle-Nassau)", size=size,
                                           color=MUTED)).arrange(RIGHT, buff=0.15),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
