"""Piezas visuales de la clase: diagrama de árbol, plano 2D, fichas y barras de impureza.

- ``TreeView``: dibuja un árbol a partir de una especificación anidada y permite
  hacerlo crecer nodo por nodo sin que nada cambie de lugar.
- ``Plane``: plano con escala isotrópica, puntos, cortes y regiones rectangulares.
- ``chip`` / ``Bucket``: fichas de ejemplos (D1…D15) y cubetas que las agrupan.
- ``area_bars``: la entropía ponderada como suma de áreas (ancho = proporción,
  alto = impureza).
- ``impurity_axes``: ejes p ∈ [0, 1] para las curvas de impureza.
"""
from __future__ import annotations

from manim import *

from sections.theme import (BG, BLUE, CLASS_COLOR, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, TEAL,
                            YELLOW, marker, safe_text)


# ============================================================================
# Árbol
# ============================================================================
def leaf_spec(counts, names=("0", "1")) -> dict:
    return {"counts": tuple(counts), "names": names}


def q_spec(question: str, counts, kids, names=("0", "1")) -> dict:
    """``kids`` = [(etiqueta de rama, spec), …]."""
    return {"q": question, "counts": tuple(counts), "kids": kids, "names": names}


def spec_from_node(node, fmt_q, names=("0", "1"), edge=("sí", "no")) -> dict:
    """Convierte un ``data.Node`` en especificación; ``fmt_q(node)`` da el texto."""
    if node.is_leaf:
        return leaf_spec(node.counts, names)
    return q_spec(fmt_q(node), node.counts,
                  [(edge[0], spec_from_node(node.left, fmt_q, names, edge)),
                   (edge[1], spec_from_node(node.right, fmt_q, names, edge))], names)


def counts_row(counts, size: float = 18, r: float = 0.08) -> VGroup:
    parts = VGroup()
    for cls, c in enumerate(counts):
        parts.add(VGroup(marker(cls, r=r), safe_text(str(c), size=size, color=INK)
                         ).arrange(RIGHT, buff=0.06))
    return parts.arrange(RIGHT, buff=0.18)


class TreeView(VGroup):
    """Árbol con nodos fijos en su posición final.

    Cada nodo tiene dos apariencias: ``leaf[path]`` (hoja con conteos) y
    ``qbox[path]`` (pregunta). ``grow(path)`` devuelve las animaciones que
    convierten la hoja en pregunta y hacen aparecer las ramas e hijos.
    Las rutas son cadenas: "" es la raíz, "0" su primer hijo, "01"…
    """

    def __init__(self, spec: dict, *, width: float = 6.0, level_h: float = 1.35,
                 top=(3.5, 1.9, 0), node_w: float = 1.9, leaf_w: float = 1.25,
                 q_size: float = 19, label_size: float = 16, names_size: float = 15,
                 show_names: bool = True):
        super().__init__()
        self.spec = spec
        self.specs: dict[str, dict] = {}
        self.pos: dict[str, np.ndarray] = {}
        self.leaf: dict[str, VGroup] = {}
        self.qbox: dict[str, VGroup] = {}
        self.edges: dict[str, VGroup] = {}
        self._collect(spec, "")
        n_leaves = self._count(spec)
        unit = width / max(n_leaves, 1)
        top = np.array(top, float)
        self._layout(spec, "", top[0] - width / 2, unit, top[1], level_h)
        for path, sp in self.specs.items():
            self.leaf[path] = self._leaf_box(sp, leaf_w, names_size, show_names).move_to(
                self.pos[path])
            if "q" in sp:
                self.qbox[path] = self._q_box(sp, node_w, q_size).move_to(self.pos[path])
        for path, sp in self.specs.items():
            for i, (lab, _) in enumerate(sp.get("kids", [])):
                child = path + str(i)
                a = self.qbox[path].get_bottom()
                b = self._top_of(child)
                ln = Line(a, b, color=TEAL, stroke_width=3)
                t = safe_text(lab, size=label_size, color=TEAL, weight=BOLD)
                n_kids = len(sp["kids"])
                if 0 < i < n_kids - 1:          # rama central (árboles multivía)
                    t.next_to(ln.point_from_proportion(0.72), LEFT, buff=0.08)
                else:
                    t.next_to(ln.point_from_proportion(0.45), LEFT if i == 0 else RIGHT,
                              buff=0.08)
                self.edges[child] = VGroup(ln, t)

    # -- construcción ---------------------------------------------------------
    def _collect(self, sp, path):
        self.specs[path] = sp
        for i, (_, k) in enumerate(sp.get("kids", [])):
            self._collect(k, path + str(i))

    def _count(self, sp) -> int:
        return 1 if "kids" not in sp else sum(self._count(k) for _, k in sp["kids"])

    def _layout(self, sp, path, x0, unit, y, dy):
        n = self._count(sp)
        self.pos[path] = np.array([x0 + n * unit / 2, y, 0])
        x = x0
        for i, (_, k) in enumerate(sp.get("kids", [])):
            self._layout(k, path + str(i), x, unit, y - dy, dy)
            x += self._count(k) * unit

    def _top_of(self, path):
        box = self.qbox.get(path, self.leaf[path])
        return box.get_top()

    def _leaf_box(self, sp, w, names_size, show_names) -> VGroup:
        a, b = sp["counts"]
        cls = int(b > a)
        col = MUTED if a == b else CLASS_COLOR[cls]
        name = "empate" if a == b else sp["names"][cls]
        rows = VGroup(counts_row(sp["counts"]))
        if show_names:
            rows.add(safe_text(name, size=names_size, color=col, weight=BOLD))
        rows.arrange(DOWN, buff=0.07)
        box = RoundedRectangle(corner_radius=0.12, width=max(w, rows.width + 0.25),
                               height=rows.height + 0.24, stroke_color=col, stroke_width=2.5,
                               fill_color=col, fill_opacity=0.14)
        return VGroup(box, rows.move_to(box))

    def _q_box(self, sp, w, size) -> VGroup:
        t = safe_text(sp["q"], size=size, color=INK, weight=BOLD, max_width=w - 0.2)
        box = RoundedRectangle(corner_radius=0.14, width=max(w, t.width + 0.3),
                               height=t.height + 0.32, stroke_color=TEAL, stroke_width=2.5,
                               fill_color=PANEL, fill_opacity=0.98)
        return VGroup(box, t.move_to(box))

    # -- consultas ------------------------------------------------------------
    def kids(self, path) -> list[str]:
        return [path + str(i) for i in range(len(self.specs[path].get("kids", [])))]

    def final(self) -> VGroup:
        """Árbol completo (preguntas en nodos internos, hojas en las puntas)."""
        g = VGroup()
        for path, sp in self.specs.items():
            g.add(self.qbox[path] if "q" in sp else self.leaf[path])
            if path:
                g.add(self.edges[path])
        return g

    def leaves_paths(self) -> list[str]:
        return [p for p, sp in self.specs.items() if "q" not in sp]

    def node(self, path) -> VGroup:
        return self.qbox[path] if path in self.qbox else self.leaf[path]

    # -- animaciones ----------------------------------------------------------
    def grow(self, path, run_time: float = 0.9) -> list[Animation]:
        anims: list[Animation] = [ReplacementTransform(self.leaf[path], self.qbox[path])]
        for k in self.kids(path):
            anims.append(Create(self.edges[k][0]))
            anims.append(FadeIn(self.edges[k][1]))
            anims.append(FadeIn(self.leaf[k], shift=DOWN * 0.15))
        return anims

    def path_to(self, leaf_path: str) -> list[str]:
        return [leaf_path[:i] for i in range(len(leaf_path) + 1)]


# ============================================================================
# Plano
# ============================================================================
class Plane(VGroup):
    """Plano con la misma unidad en ambos ejes (un rectángulo se ve como tal)."""

    def __init__(self, xr=(0, 10), yr=(0, 10), *, height: float = 5.0, center=(-3.4, -0.3, 0),
                 xlabel: str = "x₁", ylabel: str = "x₂", xticks=None, yticks=None,
                 x_scale: float = 1.0, tick_size: float = 15, label_size: float = 18):
        super().__init__()
        self.xr, self.yr = xr, yr
        unit = height / (yr[1] - yr[0])
        self.unit = unit
        self.x_scale = x_scale
        width = unit * (xr[1] - xr[0]) * x_scale
        self.axes = Axes(x_range=[xr[0], xr[1], 1], y_range=[yr[0], yr[1], 1],
                         x_length=width, y_length=height, tips=False,
                         axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        self.axes.move_to(center)
        self.frame = Rectangle(width=width, height=height, stroke_color=GRID, stroke_width=1.5
                               ).move_to(self.axes)
        self.add(self.frame, self.axes)
        self.ticks = VGroup()
        for v in (xticks or []):
            self.ticks.add(safe_text(f"{v:g}", size=tick_size, color=MUTED).next_to(
                self.p(v, yr[0]), DOWN, buff=0.1))
        for v in (yticks or []):
            self.ticks.add(safe_text(f"{v:g}", size=tick_size, color=MUTED).next_to(
                self.p(xr[0], v), LEFT, buff=0.1))
        self.add(self.ticks)
        self.xlab = safe_text(xlabel, size=label_size, color=MUTED).next_to(
            self.frame, DOWN, buff=0.42 if xticks else 0.15)
        self.ylab = safe_text(ylabel, size=label_size, color=MUTED).rotate(PI / 2).next_to(
            self.frame, LEFT, buff=0.45 if yticks else 0.15)
        self.add(self.xlab, self.ylab)

    def p(self, x, y):
        return self.axes.c2p(x, y)

    def dot(self, x, y, cls, *, r: float = 0.11, label: str | None = None,
              label_size: float = 14) -> VGroup:
        m = marker(cls, r=r).move_to(self.p(x, y))
        g = VGroup(m)
        if label is not None:
            g.add(safe_text(label, size=label_size, color=INK).next_to(m, UR, buff=0.02))
        g.cls = cls
        g.xy = (x, y)
        return g

    def dots(self, X, y, *, r: float = 0.11, labels=None, **kw) -> VGroup:
        return VGroup(*[self.dot(X[i][0], X[i][1], int(y[i]), r=r,
                                   label=None if labels is None else str(labels[i]), **kw)
                        for i in range(len(y))])

    def rect(self, box, cls: int | None, opacity: float = 0.20, stroke: bool = False) -> Rectangle:
        """Rectángulo de una región; ``cls=None`` = empate (gris)."""
        x0, x1, y0, y1 = box
        a, b = self.p(x0, y0), self.p(x1, y1)
        base = MUTED if cls is None else CLASS_COLOR[cls]
        col = interpolate_color(ManimColor(BG), ManimColor(base), opacity)
        r = Rectangle(width=abs(b[0] - a[0]), height=abs(b[1] - a[1]), fill_color=col,
                      fill_opacity=1.0, stroke_width=1.5 if stroke else 0, stroke_color=GRID)
        return r.move_to((a + b) / 2)

    def regions(self, leaves, opacity: float = 0.20) -> VGroup:
        return VGroup(*[self.rect(l.box, None if l.counts[0] == l.counts[1] else l.cls,
                                  opacity, stroke=True) for l in leaves])

    def vcut(self, x, y0=None, y1=None, color=YELLOW, width: float = 4) -> Line:
        y0 = self.yr[0] if y0 is None else y0
        y1 = self.yr[1] if y1 is None else y1
        return Line(self.p(x, y0), self.p(x, y1), color=color, stroke_width=width)

    def hcut(self, y, x0=None, x1=None, color=YELLOW, width: float = 4) -> Line:
        x0 = self.xr[0] if x0 is None else x0
        x1 = self.xr[1] if x1 is None else x1
        return Line(self.p(x0, y), self.p(x1, y), color=color, stroke_width=width)

    def cut_of(self, node, color=YELLOW, width: float = 4) -> Line:
        x0, x1, y0, y1 = node.box
        if node.feature == 0:
            return self.vcut(node.threshold, y0, y1, color, width)
        return self.hcut(node.threshold, x0, x1, color, width)


# ============================================================================
# Fichas y cubetas
# ============================================================================
def chip(cls: int, label: str | None = None, *, r: float = 0.17, size: float = 13) -> VGroup:
    m = marker(cls, r=r)
    g = VGroup(m)
    if label:
        g.add(safe_text(label, size=size, color=BG, weight=BOLD).move_to(m))
    g.cls = cls
    return g


def arrange_chips(chips: VGroup, cols: int = 5, buff: float = 0.08) -> VGroup:
    chips.arrange_in_grid(cols=cols, buff=buff)
    return chips


class Bucket(VGroup):
    """Recuadro con título, fichas y (opcional) su impureza."""

    def __init__(self, title: str, width: float = 2.4, height: float = 1.9,
                 color=GRID, title_size: float = 20):
        super().__init__()
        self.box = RoundedRectangle(corner_radius=0.14, width=width, height=height,
                                    stroke_color=color, stroke_width=2, fill_color=PANEL,
                                    fill_opacity=0.9)
        self.title = safe_text(title, size=title_size, color=INK, weight=BOLD,
                               max_width=width - 0.1).next_to(self.box, UP, buff=0.1)
        self.add(self.box, self.title)

    def slots(self, n: int, cols: int = 4, r: float = 0.17, buff: float = 0.08):
        """Posiciones (centros) para n fichas dentro de la caja."""
        dummies = VGroup(*[Circle(radius=r) for _ in range(max(n, 1))])
        dummies.arrange_in_grid(cols=min(cols, max(n, 1)), buff=buff).move_to(self.box)
        return [d.get_center() for d in dummies][:n]


# ============================================================================
# Impureza
# ============================================================================
def area_bars(groups, *, total_w: float = 6.0, unit_h: float = 2.2, origin=(0, 0, 0),
              colors=None, values=None, value_fmt="{:.3f}", label_size: float = 18,
              show_values: bool = True, labels=None) -> VGroup:
    """Una barra por grupo: ancho = |S_v|/|S|, alto = impureza. Suma de áreas = promedio
    ponderado. ``groups`` = [(n_v, impureza_v), …]."""
    n = sum(g[0] for g in groups)
    x = origin[0]
    out = VGroup()
    for i, (nv, hv) in enumerate(groups):
        w = total_w * nv / n
        h = max(unit_h * hv, 0.02)
        col = colors[i] if colors else PURPLE
        r = Rectangle(width=w - 0.04, height=h, stroke_color=col, stroke_width=2,
                      fill_color=col, fill_opacity=0.35)
        r.move_to([x + w / 2, origin[1] + h / 2, 0])
        g = VGroup(r)
        if show_values:
            v = values[i] if values else value_fmt.format(abs(hv))
            t = safe_text(v, size=label_size, color=INK)
            if h > t.height + 0.3:          # dentro de la barra, junto a su borde superior
                t.move_to(r.get_top() + DOWN * (t.height / 2 + 0.12))
            else:
                t.next_to(r, UP, buff=0.08)
            g.add(t)
        if labels:
            g.add(safe_text(labels[i], size=label_size - 2, color=MUTED,
                            max_width=max(w - 0.05, 0.3)).next_to(r, DOWN, buff=0.1)
                  .set_y(origin[1] - 0.22))
        out.add(g)
        x += w
    return out


def impurity_axes(width: float = 5.6, height: float = 3.4, center=(0, 0, 0),
                  ymax: float = 1.0, ystep: float = 0.25) -> Axes:
    ax = Axes(x_range=[0, 1, 0.25], y_range=[0, ymax, ystep], x_length=width,
              y_length=height, tips=False,
              axis_config={"color": GRID, "stroke_width": 2, "include_ticks": True,
                           "tick_size": 0.05})
    return ax.move_to(center)


def axis_numbers(ax: Axes, xs, ys, size: float = 15, nd: int = 2) -> VGroup:
    g = VGroup()
    for v in xs:
        g.add(safe_text(f"{v:g}", size=size, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1))
    for v in ys:
        g.add(safe_text(f"{v:g}", size=size, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1))
    return g
