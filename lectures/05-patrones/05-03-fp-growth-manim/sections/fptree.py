"""Dibujo animable de un FP-tree (o de un árbol de prefijos de palabras).

La distribución se calcula una sola vez con el árbol *final*: hojas en orden de
creación, cada padre centrado sobre sus hijos. Así, al insertar transacciones,
los nodos aparecen directamente en su lugar definitivo y nada se reacomoda.
"""
from __future__ import annotations

from manim import *

from sections.theme import BG, GRID, INK, MUTED, YELLOW, color_of, safe_text

Path = tuple[str, ...]


class TreeView(VGroup):
    def __init__(self, counts: dict[Path, int], *, root_pos=ORIGIN, dx: float = 2.3,
                 dy: float = 1.05, node_w: float = 2.0, node_h: float = 0.52,
                 size: float = 22, colorize: bool = True, root_label: str = "∅") -> None:
        super().__init__()
        self.final = dict(counts)
        self.node_w, self.node_h, self.size, self.colorize = node_w, node_h, size, colorize
        self.children: dict[Path, list[Path]] = {(): []}
        for p in counts:
            self.children.setdefault(p, [])
            self.children.setdefault(p[:-1], []).append(p)
        xs: dict[Path, float] = {}
        leaf = [0]

        def place(p: Path) -> None:
            kids = self.children[p]
            if not kids:
                xs[p] = leaf[0]
                leaf[0] += 1
                return
            for k in kids:
                place(k)
            xs[p] = sum(xs[k] for k in kids) / len(kids)

        place(())
        mid = (leaf[0] - 1) / 2
        root_pos = np.array(root_pos, dtype=float)
        self.pos = {p: root_pos + RIGHT * (x - mid) * dx + DOWN * len(p) * dy
                    for p, x in xs.items()}
        circle = Circle(radius=0.26, stroke_color=MUTED, stroke_width=2, fill_color=BG,
                        fill_opacity=1).move_to(self.pos[()])
        self.root = VGroup(circle, safe_text(root_label, size=22, color=MUTED).move_to(circle))
        self.add(self.root)
        self.nodes: dict[Path, VGroup] = {}
        self.edges: dict[Path, Line] = {}
        self.now: dict[Path, int] = {}

    # ── piezas ───────────────────────────────────────────────────────────
    def item_color(self, p: Path) -> str:
        return color_of(p[-1]) if self.colorize else INK

    def label(self, p: Path, count: int) -> Text:
        return safe_text(f"{p[-1]}: {count}", size=self.size, color=INK, weight=BOLD,
                         max_width=self.node_w - 0.16).move_to(self.pos[p])

    def make_node(self, p: Path, count: int) -> VGroup:
        col = self.item_color(p)
        box = RoundedRectangle(corner_radius=0.12, width=self.node_w, height=self.node_h,
                               stroke_color=col, stroke_width=2.5, fill_color=col,
                               fill_opacity=0.18).move_to(self.pos[p])
        backing = box.copy().set_fill(BG, 1).set_stroke(width=0)
        return VGroup(backing, box, self.label(p, count))

    def make_edge(self, p: Path) -> Line:
        parent = p[:-1]
        top = self.pos[parent] + DOWN * (0.26 if not parent else self.node_h / 2)
        bottom = self.pos[p] + UP * self.node_h / 2
        return Line(top, bottom, color=GRID, stroke_width=3)

    # ── construcción ─────────────────────────────────────────────────────
    def build_all(self, counts: dict[Path, int] | None = None) -> VGroup:
        """Crea todos los nodos (con ``counts`` o con los finales) sin animar."""
        counts = counts or self.final
        for p, c in counts.items():
            self.edges[p] = self.make_edge(p)
            self.nodes[p] = self.make_node(p, c)
            self.now[p] = c
            self.add(self.edges[p])
        for p in counts:
            self.add(self.nodes[p])
        return self

    def insert(self, path: list[str], weight: int = 1) -> list[Animation]:
        """Animaciones (en orden) para insertar una transacción ordenada."""
        anims = []
        for d in range(1, len(path) + 1):
            p = tuple(path[:d])
            if p in self.nodes:
                self.now[p] += weight
                anims.append(AnimationGroup(
                    Transform(self.nodes[p][2], self.label(p, self.now[p])),
                    Indicate(self.nodes[p][1], color=YELLOW, scale_factor=1.08),
                    run_time=0.55))
            else:
                self.now[p] = weight
                self.edges[p] = self.make_edge(p)
                self.nodes[p] = self.make_node(p, weight)
                self.add(self.edges[p], self.nodes[p])
                anims.append(AnimationGroup(Create(self.edges[p]),
                                            FadeIn(self.nodes[p], scale=0.6),
                                            run_time=0.55))
        return anims

    def set_count(self, p: Path, count: int) -> Animation:
        self.now[p] = count
        return Transform(self.nodes[p][2], self.label(p, count))

    # ── lectura ──────────────────────────────────────────────────────────
    def path_group(self, p: Path) -> VGroup:
        """Nodos y aristas desde la raíz hasta ``p`` (incluido)."""
        g = VGroup()
        for d in range(1, len(p) + 1):
            q = p[:d]
            g.add(self.edges[q], self.nodes[q])
        return g

    def links(self, item: str, color: str = YELLOW) -> VGroup:
        """Enlaces de nodo: arcos discontinuos entre apariciones del mismo ítem."""
        ps = sorted((p for p in self.nodes if p[-1] == item), key=lambda p: self.pos[p][0])
        arcs = VGroup()
        for a, b in zip(ps, ps[1:]):
            if abs(self.pos[a][1] - self.pos[b][1]) < 1e-6:
                # Misma profundidad: de base a base, con el arco por debajo.
                start, end, angle = (self.nodes[a][1].get_bottom(),
                                     self.nodes[b][1].get_bottom(), PI / 2)
            else:
                # Profundidades distintas: de costado a costado, arco suave.
                start, end = self.nodes[a][1].get_right(), self.nodes[b][1].get_left()
                angle = PI / 4 if end[1] > start[1] else -PI / 4
            arc = ArcBetweenPoints(start, end, angle=angle, color=color, stroke_width=3)
            arcs.add(DashedVMobject(arc, num_dashes=18))
        return arcs

    def nodes_of(self, item: str) -> list[Path]:
        return [p for p in self.nodes if p[-1] == item]
