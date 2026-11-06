"""Acto 4 · Gini y cortes binarios: CART (31–43 min).

Las tres curvas de impureza en un mismo plano; la concavidad se ve con una
cuerda: si la curva es una recta entre los hijos (error de clasificación), el
promedio de los hijos cae SOBRE la curva y la división «no gana» nada; si es
estrictamente cóncava (Gini, entropía), cae debajo. Después: Gini con los
compradores de laptop y preguntas binarias, y el barrido de cortes en el plano
de los clientes, que reconstruye el árbol del acto 1. Cierra con la
invariancia a transformaciones monótonas (no hace falta estandarizar).
"""
from __future__ import annotations

from manim import *

from sections.data import (AGE_QUESTIONS, BEST_QUESTIONS, C_RANGE, CONCAVE_KIDS, CONCAVE_PARENT,
                           CX, CY, E2, G2, G_LAPTOP_ROOT, H2, LAPTOP, client_tree, cut_scan, gini,
                           internal, lap_question, leaves)
from sections.part01_partition import client_plane, client_points
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, TEAL, YELLOW, card, marker, pill, safe_text, table, tex)
from sections.widgets import Bucket, axis_numbers, chip, counts_row, impurity_axes

BUY = {"sí": 0, "no": 1}


class CartSection:
    def cart(self) -> None:
        self.three_curves()
        self.concavity()
        self.gini_laptop()
        self.id3_vs_cart()
        self.scan_cuts()
        self.monotone()

    # ------------------------------------------------------------------
    def three_curves(self) -> None:
        self.beat("31–32 min. Tres maneras de medir la mezcla con dos clases. Las tres valen 0 "
                  "en un grupo puro y son máximas con p = 0.5. Gini (la de CART) es la "
                  "probabilidad de equivocarse si etiquetamos al azar según las proporciones "
                  "del grupo. Pregunte: ¿qué curva se ve distinta? (el error es un «techo» de "
                  "dos rectas).")
        self.start("CART: otra medida de mezcla", "Entropía, Gini y error de clasificación", BLUE)
        ax = impurity_axes(width=6.4, height=3.8, center=(-2.2, -0.35, 0))
        nums = axis_numbers(ax, [0, 0.5, 1], [0.5, 1])
        xl = safe_text("p = proporción de la clase 1", size=19, color=MUTED
                       ).next_to(ax, DOWN, buff=0.42)
        curves = [
            (H2, PURPLE, r"H = -p\log_2 p - (1-p)\log_2(1-p)", "entropía (ID3, C4.5)"),
            (G2, TEAL, r"G = 1 - p^2 - (1-p)^2 = 2p(1-p)", "Gini (CART)"),
            (E2, RED, r"E = 1 - \max(p, 1-p)", "error de clasificación"),
        ]
        self.play(Create(ax), FadeIn(nums), FadeIn(xl))
        legend = VGroup()
        for f, col, formula, name in curves:
            c = ax.plot(f, x_range=[0, 1, 0.005], color=col, stroke_width=4)
            entry = VGroup(safe_text(name, size=21, color=col, weight=BOLD),
                           tex(formula, size=26, color=col, max_width=5.2)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            legend.add(entry)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([4.15, 0.0, 0])
            self.play(Create(c), FadeIn(entry), run_time=1.0)
        note = VGroup(safe_text("Las tres: 0 si el grupo es puro,", size=21, color=YELLOW),
                      safe_text("máximas con p = 0.5", size=21, color=YELLOW)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        note.next_to(legend, DOWN, buff=0.45).align_to(legend, LEFT)
        self.play(FadeIn(note))

    # ------------------------------------------------------------------
    def concavity(self) -> None:
        self.beat("32–34 min. ¿Por qué no usar el error, que es lo que queremos minimizar? Un "
                  "grupo con 25 % de la clase 1 se divide en dos mitades con 5 % y 45 %. Ambas "
                  "mitades siguen siendo mayoría de clase 0: el error no cambia (gana 0). Pero "
                  "la división SÍ separó las proporciones. Con una curva estrictamente cóncava "
                  "(Gini), el promedio de los hijos cae por debajo: gana 0.08 (con entropía, "
                  "0.17). Pida que lo vean como la distancia vertical entre el punto amarillo "
                  "(padre) y el morado (promedio de los hijos).")
        self.start("Por qué Gini y no el error", "La cuerda debajo de la curva es la ganancia",
                   BLUE)
        pp = CONCAVE_PARENT
        a, b = CONCAVE_KIDS
        panels = VGroup()
        for i, (f, col, name) in enumerate([(E2, RED, "error"), (G2, TEAL, "Gini")]):
            cx = -3.4 if i == 0 else 3.4
            ax = impurity_axes(width=5.2, height=3.1, center=(cx, -0.35, 0), ymax=0.5, ystep=0.1)
            nums = axis_numbers(ax, [0, 0.5, 1], [0.25, 0.5])
            curve = ax.plot(f, x_range=[0, 1, 0.005], color=col, stroke_width=4)
            title = safe_text(name, size=24, color=col, weight=BOLD).next_to(ax, UP, buff=0.15)
            ka, kb = Dot(ax.c2p(a, f(a)), color=col), Dot(ax.c2p(b, f(b)), color=col)
            chord = Line(ka.get_center(), kb.get_center(), color=INK, stroke_width=3)
            avg = (f(a) + f(b)) / 2
            mid = Dot(ax.c2p(pp, avg), color=PURPLE, radius=0.09)
            par = Dot(ax.c2p(pp, f(pp)), color=YELLOW, radius=0.09)
            gap = Line(mid.get_center(), par.get_center(), color=YELLOW, stroke_width=5)
            gain = f(pp) - avg
            lbl = safe_text(f"ganancia = {gain:.2f}", size=22, color=YELLOW if gain > 1e-9 else RED,
                            weight=BOLD).next_to(ax, DOWN, buff=0.45)
            panels.add(VGroup(ax, nums, curve, title, ka, kb, chord, mid, par, gap, lbl))
        setup = safe_text(f"padre con p = {pp:g}  →  dos mitades con p = {a:g} y p = {b:g}",
                          size=21,
                          color=YELLOW).move_to([0, 2.05, 0])
        self.play(FadeIn(setup))
        for pnl in panels:
            ax, nums, curve, title = pnl[:4]
            self.play(Create(ax), FadeIn(nums), Create(curve), FadeIn(title), run_time=0.8)
        for pnl in panels:
            ka, kb, chord, mid, par, gap, lbl = pnl[4:]
            self.play(FadeIn(ka), FadeIn(kb))
            self.play(Create(chord), FadeIn(mid))
            self.play(FadeIn(par), Create(gap), FadeIn(lbl))
        msg = safe_text("Curva estrictamente cóncava ⇒ toda división que separa proporciones gana",
                        size=21, color=INK).move_to([0, -3.05, 0])
        self.play(FadeIn(msg))

    # ------------------------------------------------------------------
    def gini_laptop(self) -> None:
        self.beat("34–36 min. Gini a mano. Catorce clientes de una tienda: 9 compran, 5 no. "
                  "G = 1 − (9/14)² − (5/14)² = 0.459. CART sólo hace preguntas de sí/no: con "
                  "edad (joven, adulto, mayor) hay 3 preguntas binarias. «¿Es adulto?» deja 4 "
                  "compradores puros y 5/5 al otro lado: Gini ponderado 0.357, reduce 0.102. "
                  "Pida que calculen «¿es joven?» en parejas (0.394).")
        self.start("Gini a mano", "¿Quién compra una laptop? (CART categórico)", BLUE)
        f = tex(r"G = 1 - \sum_k p_k^2 \qquad G_{\text{ra\'iz}} = 1 - \left(\tfrac{9}{14}\right)^2"
                r" - \left(\tfrac{5}{14}\right)^2 = 0.459", size=32).move_to([0, 2.0, 0])
        self.play(Write(f))
        yes_c, no_c = AGE_QUESTIONS["adulto"][0], AGE_QUESTIONS["adulto"][1]
        q = safe_text("¿edad = adulto?", size=24, color=YELLOW, weight=BOLD).move_to([-3.6, 1.05, 0])
        bk = VGroup(Bucket("sí", width=2.2, height=1.55, title_size=20),
                    Bucket("no", width=2.2, height=1.55, title_size=20)).arrange(RIGHT, buff=0.35)
        bk.move_to([-3.6, -0.6, 0])
        chips = VGroup()
        for side, rows in [(0, [r for r in LAPTOP if r[0] == "adulto"]),
                           (1, [r for r in LAPTOP if r[0] != "adulto"])]:
            slots = bk[side].slots(len(rows), cols=5, r=0.15)
            for r, pos in zip(rows, slots):
                chips.add(marker(BUY[r[-1]], r=0.15).move_to(pos))
        vals = VGroup(
            safe_text(f"G = {gini(yes_c):.3f}", size=20, color=GREEN).next_to(bk[0], DOWN, buff=0.12),
            safe_text(f"G = {gini(no_c):.3f}", size=20, color=MUTED).next_to(bk[1], DOWN, buff=0.12))
        w = tex(r"\tfrac{4}{14}(0) + \tfrac{10}{14}(0.5) = 0.357", size=28).move_to([-3.6, -2.45, 0])
        self.play(FadeIn(q), FadeIn(bk))
        self.play(LaggedStart(*[GrowFromCenter(c) for c in chips], lag_ratio=0.04))
        self.play(FadeIn(vals), Write(w))
        rows = [["Pregunta", "Sí (●/■)", "No (●/■)", "Gini pond.", "ΔG"]]
        cols = []
        for v in ("adulto", "joven", "mayor"):
            yc, nc, wg, dg = AGE_QUESTIONS[v]
            rows.append([f"¿{v}?", f"{yc[0]} / {yc[1]}", f"{nc[0]} / {nc[1]}", f"{wg:.3f}",
                         f"{dg:.3f}"])
            cols.append([INK, MUTED, MUTED, INK, YELLOW if v == "adulto" else INK])
        tb = table(rows, col_widths=[1.3, 1.25, 1.25, 1.35, 0.9], size=20, row_h=0.5,
                   colors=cols).move_to([3.25, 0.35, 0])
        self.play(FadeIn(tb))
        best = VGroup(safe_text("Mejor pregunta de cada atributo:", size=19, color=MUTED),
                      *[safe_text(f"¿{a} = {v}?  ΔG = {lap_question(a, v)[3]:.3f}", size=19,
                                  color=YELLOW if i == 0 else INK)
                        for i, (a, v) in enumerate(BEST_QUESTIONS)]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(tb, DOWN, buff=0.35)
        self.play(FadeIn(best))

    # ------------------------------------------------------------------
    def id3_vs_cart(self) -> None:
        self.beat("36–37 min. Las diferencias de diseño. Lo esencial: CART SIEMPRE hace "
                  "preguntas binarias, así que evita el sesgo hacia atributos con muchos valores "
                  "y un mismo atributo puede reaparecer más abajo. Con variables numéricas, "
                  "cada pregunta es una recta vertical u horizontal: el árbol parte el plano en "
                  "rectángulos. Es el algoritmo de scikit-learn y la pieza de los bosques y del "
                  "boosting.")
        self.start("ID3 / C4.5 frente a CART", "Cuatro decisiones de diseño", BLUE)
        rows = [["", "ID3 / C4.5", "CART"],
                ["divisiones", "multivía (una rama por valor)", "siempre binarias"],
                ["impureza", "entropía, razón de ganancia", "Gini (o entropía)"],
                ["numéricas", "C4.5: cortes A ≤ t", "cortes A ≤ t, reutilizables"],
                ["complejidad", "poda pesimista", "poda de costo-complejidad"],
                ["regresión", "no", "sí (error cuadrático)"]]
        cols = [[MUTED, INK, BLUE]] * 5
        tb = table(rows, col_widths=[2.3, 4.6, 4.3], size=23, row_h=0.62, colors=cols)
        tb.move_to([0, -0.2, 0])
        self.play(FadeIn(tb[0]), FadeIn(tb[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in tb[1:-1]], lag_ratio=0.18))

    # ------------------------------------------------------------------
    def scan_cuts(self) -> None:
        self.beat("37–40 min. De vuelta a los diez clientes. CART prueba TODOS los cortes de "
                  "ambas variables (puntos medios) y dibuja su Gini ponderado. Pregunte dónde "
                  "esperan el mínimo antes de barrer. Deuda ≤ 44 deja 0.160; el mejor de "
                  "ingreso apenas llega a 0.400. Al repetir en el rectángulo de arriba salen "
                  "ingreso ≤ 29 y ingreso ≤ 32.5: ¡es exactamente el árbol del acto 1!")
        self.start("Cortes numéricos en el plano", "Barrer todos los cortes de las dos variables",
                   BLUE)
        plane = client_plane(center=(-4.15, -0.4, 0), height=4.6)
        pts = client_points(plane)
        self.play(FadeIn(plane), FadeIn(pts))
        idx = np.arange(len(CY))
        graphs = VGroup()
        specs = [(1, "deuda", C_RANGE[2:], 1.0), (0, "ingreso", C_RANGE[:2], -1.75)]
        results = {}
        for f, name, rng, cy in specs:
            ax = Axes(x_range=[rng[0], rng[1], 10], y_range=[0, 0.5, 0.25], x_length=5.6,
                      y_length=1.75, tips=False,
                      axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
            ax.move_to([3.45, cy, 0])
            lab = safe_text(f"Gini ponderado · corte en {name}", size=18, color=MUTED
                            ).next_to(ax, UP, buff=0.08).align_to(ax, LEFT)
            yl = VGroup(safe_text("0.5", size=14, color=MUTED).next_to(ax.c2p(rng[0], 0.5), LEFT,
                                                                      buff=0.08),
                        safe_text("0", size=14, color=MUTED).next_to(ax.c2p(rng[0], 0), LEFT,
                                                                    buff=0.08))
            graphs.add(VGroup(ax, lab, yl))
            results[f] = (ax, cut_scan(CX, CY, idx, f))
        self.play(FadeIn(graphs))
        for f, name, rng, _ in specs:
            ax, scan = results[f]
            line = None
            dots = VGroup()
            for t, wg in scan:
                cut = plane.hcut(t) if f == 1 else plane.vcut(t)
                cut.set_stroke(YELLOW, 3, opacity=0.9)
                d = Dot(ax.c2p(t, wg), radius=0.06, color=PURPLE)
                if line is None:
                    self.play(Create(cut), FadeIn(d), run_time=0.3)
                else:
                    self.play(Transform(line, cut), FadeIn(d), run_time=0.3)
                line = line or cut
                dots.add(d)
            poly = VMobject(color=PURPLE, stroke_width=3).set_points_as_corners(
                [d.get_center() for d in dots])
            self.play(Create(poly), FadeOut(line), run_time=0.5)
            tb, wb = min(scan, key=lambda z: z[1])
            star = Dot(ax.c2p(tb, wb), radius=0.1, color=YELLOW)
            lab = safe_text(f"{name} ≤ {tb:g}: {wb:.3f}", size=18,
                            color=YELLOW if f == 1 else MUTED, weight=BOLD
                            ).next_to(star, DOWN, buff=0.15)
            if lab.get_left()[0] < ax.get_left()[0]:
                lab.align_to(ax, LEFT).shift(RIGHT * 0.15)
            self.play(FadeIn(star), FadeIn(lab), run_time=0.4)

        self.beat("40–41 min. Ganó la deuda. Recursión en cada rectángulo (sólo con sus "
                  "puntos). Señale que el árbol de CART es el mismo que dibujamos a ojo: la "
                  "intuición «buscar grupos puros» y Gini dicen lo mismo aquí.")
        tree = client_tree()
        for k, nd in enumerate(internal(tree)):
            c = plane.cut_of(nd)
            self.play(Create(c), run_time=0.6)
        regs = plane.regions(leaves(tree))
        self.play(FadeIn(regs))
        self.bring_to_front(pts)
        same = pill("= el árbol del acto 1", TEAL, size=22).move_to([3.45, -3.0, 0])
        self.play(FadeIn(same))
        self._scan = dict(plane=plane, pts=pts, regs=regs)

    # ------------------------------------------------------------------
    def monotone(self) -> None:
        self.beat("41–43 min. Una ventaja práctica. Deformamos el eje del ingreso con una "
                  "transformación monótona (raíz cuadrada; podría ser log o cambiar de pesos a "
                  "dólares). Los puntos se mueven, pero NINGUNO cambia de rectángulo: los "
                  "cortes sólo dependen del ORDEN. Contraste con k-NN de la clase pasada, donde "
                  "las unidades cambiaban a los vecinos. Los árboles no necesitan estandarizar.")
        self.clear()
        self.start("Sin estandarizar", "Sólo importa el orden de los valores", BLUE)
        plane = client_plane(center=(-1.0, -0.4, 0), height=4.6)
        tree = client_tree()
        regs = plane.regions(leaves(tree))
        cuts = VGroup(*[plane.cut_of(n) for n in internal(tree)])
        pts = client_points(plane)
        self.play(FadeIn(plane), FadeIn(regs), FadeIn(cuts), FadeIn(pts))
        x0 = plane.frame.get_left()[0]
        w = plane.frame.width
        stretch = 2.6

        def warp(p):
            u = (p[0] - x0) / w
            return np.array([x0 + w * stretch * np.sqrt(max(u, 0.0)), p[1], p[2]])

        new_xlab = safe_text("√ingreso (u otra transformación monótona)", size=17, color=YELLOW)
        # Cada pieza por separado: un VGroup nuevo se agregaría encima y taparía los puntos.
        anims = [ApplyPointwiseFunction(warp, m) for m in (regs, cuts, plane.frame, plane.axes)]
        for p in pts:
            anims.append(p.animate.shift(warp(p[0].get_center()) - p[0].get_center()))
        self.play(FadeOut(plane.ticks), FadeOut(plane.xlab), run_time=0.4)
        self.play(*anims, run_time=2.4)
        new_xlab.next_to(plane.frame, DOWN, buff=0.2)
        self.play(FadeIn(new_xlab))
        same = VGroup(safe_text("cada punto sigue en su rectángulo:", size=21),
                      safe_text("el árbol (y sus predicciones) no cambia", size=21, color=GREEN,
                                weight=BOLD)).arrange(DOWN, buff=0.08)
        same.next_to(plane.frame, UP, buff=0.15)
        self.play(FadeIn(same))
