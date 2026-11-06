"""Acto 1 · Preguntas que parten el plano (0–10 min).

Retoma los diez clientes de la clase anterior y su regla de un corte. Cada
pregunta nueva vive sólo dentro de su rectángulo: el diagrama del árbol crece
a la derecha, sincronizado con la partición del plano a la izquierda. Un
cliente nuevo «cae» por el árbol. Cierra con la forma general f(x) = Σ c·1[x∈R].
"""
from __future__ import annotations

from manim import *

from sections.data import C_RANGE, CLIENTS, CX, CY, NEW_CLIENT, client_tree, internal, leaves
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL,
                            YELLOW, card, legend_item, marker, pill, safe_text, tex)
from sections.widgets import Plane, TreeView, spec_from_node

CLIENT_NAMES = ("paga", "no paga")
FEATURE = ("ingreso", "deuda")


def client_question(node) -> str:
    return f"¿{FEATURE[node.feature]} ≤ {node.threshold:g}?"


def client_plane(center=(-3.75, -0.35, 0), height: float = 4.75) -> Plane:
    x0, x1, y0, y1 = C_RANGE
    return Plane((x0, x1), (y0, y1), height=height, center=center,
                 xlabel="ingreso (miles $/mes)", ylabel="deuda (% del ingreso)",
                 xticks=[10, 20, 30, 40], yticks=[20, 30, 40, 50, 60])


def client_points(plane: Plane) -> VGroup:
    return plane.dots(CX, CY, r=0.13, labels=[c[0] for c in CLIENTS])


def client_treeview(**kw) -> TreeView:
    spec = spec_from_node(client_tree(), client_question, CLIENT_NAMES)
    opts = dict(width=6.2, level_h=1.42, top=(3.35, 1.85, 0), node_w=2.35, leaf_w=1.35)
    opts.update(kw)
    return TreeView(spec, **opts)


class PartitionSection:
    # ------------------------------------------------------------------
    def opening(self) -> None:
        self.beat("0–1 min. Portada. Pregunte: ¿alguna vez han seguido un diagrama de flujo para "
                  "decidir algo (un trámite, un diagnóstico, «¿qué serie veo?»)? Hoy aprendemos "
                  "a que la computadora construya ese diagrama a partir de ejemplos.")
        eyebrow = safe_text("CLASIFICACIÓN · ÁRBOLES DE DECISIÓN", size=22, color=TEAL,
                            weight=BOLD)
        title = safe_text("¿Qué preguntar primero?", size=62, weight=BOLD, max_width=12.6)
        subtitle = safe_text("De preguntas a rectángulos: entropía, ID3, C4.5, Gini, CART y poda",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Ruta. Tres ejemplos del libro: los diez clientes (ya los conocen), "
                  "quince días de béisbol y catorce compradores de laptop. Cerramos con 60 puntos "
                  "ruidosos para ver el sobreajuste y la poda. Habrá un ejercicio a mano al final.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "Una pregunta: ¿cuál es la mejor pregunta?", YELLOW)
        route = VGroup(*[pill(t, c, size=23, width=5.9) for t, c in [
            ("1 · Preguntas que parten el plano", TEAL),
            ("2 · Medir la mezcla: entropía (ID3)", PURPLE),
            ("3 · Trampas y números: C4.5", ORANGE),
            ("4 · Gini y cortes binarios: CART", BLUE),
            ("5 · Crecer, sobreajustar y podar", RED),
            ("6 · Ejercicio y cierre", YELLOW)]]).arrange(DOWN, buff=0.2)
        route.move_to([-3.3, -0.4, 0])
        promise = VGroup(
            safe_text("Al terminar sabrás", size=26, color=MUTED),
            safe_text("· leer un árbol como reglas y rectángulos", size=24),
            safe_text("· calcular entropía, ganancia y Gini a mano", size=24),
            safe_text("· explicar por qué C4.5 corrige a ID3", size=24),
            safe_text("· podar un árbol con costo-complejidad", size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([3.4, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in route], lag_ratio=0.12))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in promise], lag_ratio=0.12))
        self.partition()
        self.new_client()
        self.formal_partition()

    # ------------------------------------------------------------------
    def partition(self) -> None:
        self.beat("2–3 min. Recuerdo de la clase anterior: diez clientes y la regla de un solo "
                  "corte «deuda > 44 ⇒ no paga», que se equivocaba sólo con el cliente 6. "
                  "Nuevo: esa regla ya es un árbol de una pregunta. Señale la correspondencia "
                  "entre la recta del plano y la caja de la derecha.")
        head = self.start("Del corte al árbol",
                          "La regla de la clase pasada ya era un árbol", TEAL)
        plane = client_plane()
        pts = client_points(plane)
        legend = VGroup(legend_item(0, "pagó"), legend_item(1, "no pagó")
                        ).arrange(RIGHT, buff=0.4).next_to(plane.frame, UP, buff=0.12)
        self.play(FadeIn(plane), FadeIn(legend), LaggedStart(*[FadeIn(p) for p in pts],
                                                              lag_ratio=0.05))
        tv = client_treeview()
        root = tv.leaf[""]
        self.play(FadeIn(root, shift=DOWN * 0.1))
        tree = client_tree()
        nodes = internal(tree)           # [raíz, ingreso ≤ 29, ingreso ≤ 32.5]
        regions = plane.regions([tree])
        self.play(FadeIn(regions))
        self.bring_to_front(pts)
        note = safe_text("antes de preguntar: 6 pagan, 4 no", size=20, color=MUTED
                         ).next_to(root, DOWN, buff=0.25)
        self.play(FadeIn(note))

        self.beat("3–4 min. Primera pregunta. La recta horizontal parte el plano en DOS "
                  "rectángulos y, a la derecha, la caja se vuelve una pregunta con dos ramas. "
                  "Abajo: 5 pagan y 0 no (puro). Arriba: 1 paga (el 6) y 4 no. Pregunte: ¿qué "
                  "harían con el rectángulo de arriba?")
        cut1 = plane.cut_of(nodes[0])
        t1 = client_tree(max_depth=1)
        reg1 = plane.regions(leaves(t1))
        self.play(FadeOut(note), Create(cut1))
        self.play(*tv.grow(""), ReplacementTransform(regions, reg1))
        self.bring_to_front(pts, cut1)
        self.play(Indicate(pts[5], color=YELLOW, scale_factor=1.8))

        self.beat("4–6 min. La idea clave: la segunda pregunta vive SÓLO dentro del rectángulo de "
                  "arriba. Por eso el corte vertical no atraviesa todo el plano, se detiene en "
                  "deuda = 44. Pregunte por qué. Ahora el rectángulo de la izquierda queda puro "
                  "(3 no pagan) y el de la derecha empata 1 a 1.")
        cut2 = plane.cut_of(nodes[1])
        t2 = client_tree(max_depth=2)
        reg2 = plane.regions(leaves(t2))
        self.play(Create(cut2))
        self.play(*tv.grow("1"), ReplacementTransform(reg1, reg2))
        self.bring_to_front(pts, cut1, cut2)

        self.beat("6–7 min. Tercera pregunta: separa al 6 del 9. Ahora todas las hojas son "
                  "puras y el árbol no se equivoca con NINGÚN cliente de entrenamiento. "
                  "Pregunte: ¿se creen esa franja delgada para el cliente 6? Guarde la duda: "
                  "volverá en el acto 5 (sobreajuste).")
        cut3 = plane.cut_of(nodes[2])
        reg3 = plane.regions(leaves(tree))
        self.play(Create(cut3))
        self.play(*tv.grow("11"), ReplacementTransform(reg2, reg3))
        self.bring_to_front(pts, cut1, cut2, cut3)
        strip = SurroundingRectangle(reg3[2], color=YELLOW, buff=0.02, stroke_width=3)
        doubt = safe_text("¿una franja sólo para el 6?", size=20, color=YELLOW
                          ).next_to(plane.frame, UP, buff=0.12)
        self.play(Create(strip), FadeOut(legend), FadeIn(doubt))
        self._act1 = dict(plane=plane, pts=pts, tv=tv, regions=reg3,
                          cuts=VGroup(cut1, cut2, cut3), head=head, extra=VGroup(strip, doubt))

    # ------------------------------------------------------------------
    def new_client(self) -> None:
        self.beat("7–8 min. Llega un cliente nuevo: ingreso 33, deuda 52. Pidan que lo bajen por "
                  "el árbol en voz alta antes de animarlo: ¿deuda ≤ 44? no. ¿ingreso ≤ 29? no. "
                  "¿ingreso ≤ 32.5? no ⇒ «no paga». En el plano cae en el rectángulo del 9.")
        a = self._act1
        plane, tv = a["plane"], a["tv"]
        self.play(FadeOut(a["extra"]))
        q = marker(None, r=0.15).move_to(plane.p(*NEW_CLIENT))
        ql = safe_text("nuevo", size=16, color=YELLOW, weight=BOLD).next_to(q, UR, buff=0.03)
        self.play(GrowFromCenter(q), FadeIn(ql))
        path = ["", "1", "11", "111"]
        trail = VGroup()
        for i, p in enumerate(path):
            box = tv.node(p)
            hl = SurroundingRectangle(box, color=YELLOW, buff=0.06, stroke_width=4)
            anims = [Create(hl)]
            if p:
                anims.append(tv.edges[p][0].animate.set_color(YELLOW).set_stroke(width=5))
            self.play(*anims, run_time=0.55)
            trail.add(hl)
        ans = pill("ŷ = no paga", ORANGE, size=24).next_to(tv.node("111"), DOWN, buff=0.22)
        self.play(FadeIn(ans, shift=UP * 0.1),
                  Indicate(a["regions"][3], color=YELLOW, scale_factor=1.0))
        self._act1.update(q=VGroup(q, ql), trail=trail, ans=ans)

    # ------------------------------------------------------------------
    def formal_partition(self) -> None:
        self.beat("8–10 min. Formalice lo que ya vieron. Un árbol con J hojas parte el espacio en "
                  "J rectángulos (regiones) que no se traslapan; en cada uno predice una "
                  "constante, la clase mayoritaria. Cerrar con la pregunta del día: ¿quién eligió "
                  "«deuda ≤ 44» y no otra? Necesitamos medir qué tan buena es una pregunta.")
        self.clear()
        self.start("Formalización", "Un árbol = una partición en rectángulos", TEAL)
        plane = client_plane(center=(-4.2, -0.45, 0), height=4.3)
        tree = client_tree()
        regs = plane.regions(leaves(tree))
        pts = client_points(plane)
        cuts = VGroup(*[plane.cut_of(n) for n in internal(tree)])
        spots = [(12, 25), (18, 50), (30.75, 64.5), (42, 52)]   # dónde escribir R1…R4
        labels = VGroup(*[safe_text(f"R{i + 1}", size=20, color=YELLOW, weight=BOLD)
                          .move_to(plane.p(*xy)) for i, xy in enumerate(spots)])
        self.play(FadeIn(plane), FadeIn(regs), FadeIn(cuts), FadeIn(pts))
        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.15))
        f = tex(r"\hat{y} = f(\mathbf{x}) = \sum_{h=1}^{J} c_h \, \mathbf{1}[\mathbf{x} \in R_h]",
                size=40, color=INK).move_to([2.9, 1.35, 0])
        expl = VGroup(
            VGroup(tex(r"R_h", size=34, color=YELLOW),
                   safe_text("rectángulo de la hoja h", size=22)).arrange(RIGHT, buff=0.25),
            VGroup(tex(r"c_h", size=34, color=ORANGE),
                   safe_text("clase mayoritaria en R_h", size=22)).arrange(RIGHT, buff=0.25),
            VGroup(tex(r"J", size=34, color=TEAL),
                   safe_text("número de hojas (aquí J = 4)", size=22)).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(f, DOWN, buff=0.4)
        self.play(Write(f))
        self.play(LaggedStart(*[FadeIn(e, shift=RIGHT * 0.1) for e in expl], lag_ratio=0.2))
        q = VGroup(safe_text("La pregunta de hoy", size=22, color=MUTED),
                   safe_text("¿Por qué «deuda ≤ 44» y no otra?", size=28, color=YELLOW,
                             weight=BOLD),
                   safe_text("Hay que medir qué tan buena es una pregunta.", size=22)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = card(q.width + 0.5, q.height + 0.4, YELLOW)
        self.play(FadeIn(VGroup(box, q.move_to(box)).next_to(expl, DOWN, buff=0.45)
                         .align_to(f, LEFT)))
