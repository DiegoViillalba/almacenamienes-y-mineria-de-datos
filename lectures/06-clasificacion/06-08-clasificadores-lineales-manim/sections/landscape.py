"""Paisajes de pérdida en pseudo-3D: superficie sombreada + curvas de nivel.

La superficie z = pérdida(β₀, β₁) es un cuenco limitado por la curva de nivel más
externa: se construye con una malla polar alrededor del mínimo (cada rayo llega
exactamente a esa curva), de modo que el borde es liso y las curvas de nivel
encajan en la malla. Los parches se ordenan por profundidad (algoritmo del
pintor) sobre la proyección ``View3D`` que ya usa la clase.

Las curvas de nivel pueden «subir» desde el piso (``lift`` de 0 a 1) hasta su
altura sobre la superficie: así se ve que el mapa de contornos es el mismo
paisaje visto desde arriba.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import level_curve
from sections.plane import View3D
from sections.theme import GRID, INK, MUTED, PANEL, YELLOW, safe_text, tex

DEEP = "#162022"          # tono del fondo del cuenco
LIGHT = np.array([-0.35, -0.55, 0.76])
LIGHT = LIGHT / np.linalg.norm(LIGHT)   # luz desde arriba y de frente
N_ANG, N_RAD = 240, 36   # malla fina (cuadro final); la animación usa 1/3 de ella
STRIDE = 3
AZ, EL = 0.55, 0.70      # vista final: el valle largo se ve de lado


class Landscape3D:
    def __init__(self, f, center, levels, color, *, origin, scale, wx=4.2, wy=2.4, zh=1.5):
        self.f = f
        self.center = np.asarray(center, float)
        self.levels = list(levels)
        self.color = color
        self.view = View3D(origin=origin, scale=scale)
        self.wx, self.wy, self.zh = wx, wy, zh
        self.lmin = float(f(self.center))
        self.ltop = self.levels[-1]
        # Curvas de nivel en los mismos ángulos que la malla
        self.rings = [level_curve(f, self.center, lv, n=N_ANG) for lv in self.levels]
        self.rim = self.rings[-1]
        lo, hi = self.rim.min(0), self.rim.max(0)
        margin = 0.06 * (hi - lo)
        lo, hi = lo - margin, hi + margin
        self.mid = (lo + hi) / 2
        self.span = hi - lo
        # Rejilla polar de parámetros β
        s = np.linspace(0.0, 1.0, N_RAD + 1)
        self.grid = np.array([[self.center + sj * (self.rim[k % N_ANG] - self.center)
                               for sj in s] for k in range(N_ANG + 1)])   # (ang, rad, 2)
        self.gval = np.array([[f(b) for b in row] for row in self.grid])

    # --- geometría -----------------------------------------------------
    def wxy(self, b0, b1):
        return ((b0 - self.mid[0]) / self.span[0] * self.wx,
                (b1 - self.mid[1]) / self.span[1] * self.wy)

    def zfrac(self, value):
        return (value - self.lmin) / (self.ltop - self.lmin)

    def point(self, b0, b1, az, el, z=None):
        x, y = self.wxy(b0, b1)
        if z is None:
            z = self.zh * self.zfrac(self.f((b0, b1)))
        return self.view(x, y, z, az, el)

    # --- dibujo --------------------------------------------------------
    def draw(self, az: float, el: float, lift: float = 1.0, opacity: float = 1.0,
             labels: bool = True, fine: bool = True) -> VGroup:
        """``fine=False`` usa 1/3 de la malla: para animar la cámara con fluidez."""
        v, wx, wy, zh = self.view, self.wx, self.wy, self.zh
        st = 1 if fine else STRIDE
        g = VGroup()
        corners = [(-wx / 2, -wy / 2), (wx / 2, -wy / 2), (wx / 2, wy / 2), (-wx / 2, wy / 2)]
        g.add(Polygon(*[v(x, y, 0, az, el) for x, y in corners], stroke_color=GRID,
                      stroke_width=1.6, fill_color=PANEL, fill_opacity=0.55))
        items = []
        key = {}
        dark, light = ManimColor(DEEP), ManimColor(self.color)
        for k in range(0, N_ANG, st):
            for j in range(0, N_RAD, st):
                idx = [(k, j), (k + st, j), (k + st, j + st), (k, j + st)]
                world, zs, depth, w3 = [], [], 0.0, []
                for a, b in idx:
                    x, y = self.wxy(*self.grid[a, b])
                    z = zh * self.zfrac(self.gval[a, b])
                    world.append(v(x, y, z, az, el))
                    w3.append((x, y, z))
                    zs.append(z / zh)
                    depth += v.depth(x, y, z, az) / 4
                # Sombreado por orientación: la luz resalta la curvatura del cuenco.
                n = np.cross(np.subtract(w3[1], w3[0]), np.subtract(w3[3], w3[0])) + \
                    np.cross(np.subtract(w3[2], w3[1]), np.subtract(w3[2], w3[3]))
                norm = np.linalg.norm(n)
                shade = 0.5 if norm < 1e-9 else float(np.clip(abs(n @ LIGHT) / norm, 0, 1))
                frac = float(np.mean(zs))
                t = (0.10 + 0.78 * min(frac, 1.0) ** 0.8) * (0.55 + 0.75 * shade)
                col = interpolate_color(dark, light, float(np.clip(t, 0.0, 1.0)))
                items.append((depth, 0, Polygon(*world, stroke_color=col, stroke_width=0.6,
                                                stroke_opacity=opacity, fill_color=col,
                                                fill_opacity=opacity)))
                key[(k, j)] = depth
        # Curvas de nivel: el tramo k–k+1 del anillo i cae en la celda (k, j)
        radial = [np.linalg.norm(self.rim[k] - self.center) for k in range(N_ANG)]
        light_line = interpolate_color(light, ManimColor(INK), 0.7)
        for lv, ring in zip(self.levels, self.rings):
            zl = lift * zh * self.zfrac(lv)
            for k in range(0, N_ANG, st):
                p, q = ring[k], ring[(k + st) % N_ANG]
                s = np.linalg.norm(p - self.center) / radial[k]
                j = (min(int(s * N_RAD), N_RAD - 1) // st) * st
                seg = Line(v(*self.wxy(*p), zl, az, el), v(*self.wxy(*q), zl, az, el),
                           stroke_color=light_line, stroke_width=2.6,
                           stroke_opacity=opacity)
                items.append((key[(k, j)] + 0.06, 1, seg))
        for _, _, m in sorted(items, key=lambda t: (t[0], t[1])):
            g.add(m)
        if labels:
            g.add(*self.labels(az, el))
        return g

    def labels(self, az: float, el: float) -> list:
        v, wx, wy = self.view, self.wx, self.wy
        c = v(0, 0, 0, az, el)

        def out(p, d=0.30):
            direction = p - c
            n = np.linalg.norm(direction)
            return p + (direction / n * d if n > 1e-6 else 0.0)

        if not hasattr(self, "_labs"):
            self._labs = (tex(r"\beta_0", size=26, color=MUTED),
                          tex(r"\beta_1", size=26, color=MUTED),
                          safe_text("pérdida", size=17, color=MUTED))
        b0, b1, z_lab = self._labs
        b0.move_to(out(v(0, -wy / 2, 0, az, el)))
        b1.move_to(out(v(wx / 2, -0.32 * wy, 0, az, el), 0.34))
        pts = [(-wx / 2, -wy / 2), (wx / 2, -wy / 2), (wx / 2, wy / 2), (-wx / 2, wy / 2)]
        x0, y0 = min(pts, key=lambda p: v(p[0], p[1], 0, az, el)[0])
        bottom, top = v(x0, y0, 0, az, el), v(x0, y0, self.zh * 1.15, az, el)
        axis = Arrow(bottom, top, buff=0, color=MUTED, stroke_width=2.2,
                     max_tip_length_to_length_ratio=0.12)
        z_lab.next_to(top, UP, buff=0.06)
        return [b0, b1, axis, z_lab]

    def star(self, az: float, el: float) -> Star:
        return Star(n=5, outer_radius=0.13, color=YELLOW, fill_opacity=1).move_to(
            self.point(*self.center, az, el)).set_z_index(30)

    def path(self, pts, az: float, el: float, color=YELLOW, width=3.0, dense: int = 24) -> VMobject:
        """Trayectoria sobre la superficie; cada tramo se densifica para seguir la curvatura."""
        pts = [np.asarray(p, float) for p in pts]
        dense_pts = []
        for p, q in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, dense, endpoint=False):
                dense_pts.append(p + t * (q - p))
        dense_pts.append(pts[-1])
        return VMobject(stroke_color=color, stroke_width=width).set_points_smoothly(
            [self.point(a, b, az, el) for a, b in dense_pts]).set_z_index(30)
