"""Acto 5 · Crecer, sobreajustar y podar (43–54 min).

60 puntos ruidosos del laboratorio de CART. El árbol crece nivel por nivel
hasta tener hojas puras (18 hojas, 100 % en entrenamiento) y 300 puntos de
prueba revelan el sobreajuste (71.7 %). La poda de costo-complejidad se dibuja
como una familia de rectas R(T) + α|T|: para cada α gana la más baja (la
envolvente inferior) y el plano muestra el árbol elegido. Después: el eslabón
más débil con números y la curva entrenamiento/prueba a lo largo de la ruta.
"""
from __future__ import annotations

from manim import *

from sections.data import (LAB_ORDER, LX, LY, R, R_tree, TX, TY, accuracy, grow, internal,
                           lab_four_leaves, lab_path, lab_tree, leaves, weakest_link)
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, TEAL, YELLOW, card, legend_item, marker, pill, safe_text, table,
                            tex)
from sections.widgets import Plane, TreeView, spec_from_node

LAB_NAMES = ("clase 0", "clase 1")


def lab_plane(center=(-3.95, -0.35, 0), height: float = 4.7) -> Plane:
    return Plane((0, 10), (0, 10), height=height, center=center, xlabel="x₁", ylabel="x₂",
                 xticks=[0, 5, 10], yticks=[0, 5, 10])


def lab_question(node) -> str:
    return f"¿x{'₁₂'[node.feature]} ≤ {node.threshold:.2f}?"


def depth_of(node) -> int:
    return 0 if node.is_leaf else 1 + max(depth_of(node.left), depth_of(node.right))


class PruningSection:
    def pruning(self) -> None:
        self.grow_full()
        self.test_points()
        self.cost_complexity()
        self.weakest()
        self.path_curve()

    # ------------------------------------------------------------------
    def grow_full(self) -> None:
        self.beat("43–45 min. 60 puntos con ruido: dentro de dos zonas la clase 1 sale con "
                  "probabilidad 0.88 y fuera con 0.12. Dejamos crecer el árbol nivel por nivel "
                  "hasta que todas las hojas sean puras. Pregunte qué pasará con la exactitud de "
                  "entrenamiento (llega a 100 %) y cuántas hojas creen que hacen falta.")
        self.start("Dejarlo crecer", "Hojas puras… ¿a qué precio?", RED)
        plane = lab_plane()
        pts = plane.dots(LX, LY, r=0.09)
        self.play(FadeIn(plane), LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.01))
        full = lab_tree()
        dmax = depth_of(full)
        info = VGroup()

        def readout(d, t):
            return VGroup(
                safe_text(f"profundidad {d}", size=26, color=INK, weight=BOLD),
                safe_text(f"{len(leaves(t))} hojas", size=26, color=TEAL),
                safe_text(f"entrenamiento {accuracy(t, LX, LY) * 100:.1f} %", size=26,
                          color=BLUE),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.6, 0.6, 0])

        regs = None
        for d in range(0, dmax + 1):
            t = grow(LX, LY, order=LAB_ORDER, max_depth=d)
            new = plane.regions(leaves(t))
            new_info = readout(d, t)
            if regs is None:
                self.play(FadeIn(new), FadeIn(new_info), run_time=0.6)
            else:
                self.play(ReplacementTransform(regs, new), ReplacementTransform(info, new_info),
                          run_time=0.55)
            regs, info = new, new_info
            self.bring_to_front(pts)
        self._lab = dict(plane=plane, pts=pts, regs=regs, info=info, full=full)

    # ------------------------------------------------------------------
    def test_points(self) -> None:
        self.beat("45–46 min. 300 puntos NUEVOS de la misma fuente. El árbol completo acierta "
                  "sólo 71.7 %. Las líneas punteadas son las zonas verdaderas: el árbol "
                  "dibujó además islas pequeñas alrededor de puntos ruidosos (como la franja "
                  "del cliente 6 en el acto 1). Eso es sobreajuste: memoriza el ruido.")
        a = self._lab
        plane, full = a["plane"], a["full"]
        tp = VGroup(*[Dot(plane.p(*TX[i]), radius=0.035, color=CLASS_COLOR[int(TY[i])],
                          fill_opacity=0.85) for i in range(len(TY))])
        self.play(FadeOut(a["pts"]), LaggedStart(*[FadeIn(d) for d in tp], lag_ratio=0.002),
                  run_time=1.2)
        zones = VGroup(
            DashedVMobject(Rectangle(width=6 * plane.unit, height=6 * plane.unit, color=INK,
                                     stroke_width=3).move_to(plane.p(7, 3)), num_dashes=40),
            DashedVMobject(Rectangle(width=2.5 * plane.unit, height=3 * plane.unit, color=INK,
                                     stroke_width=3).move_to(plane.p(1.25, 8.5)), num_dashes=24))
        test = safe_text(f"prueba {accuracy(full, TX, TY) * 100:.1f} %", size=28, color=GREEN,
                         weight=BOLD).next_to(a["info"], DOWN, buff=0.3).align_to(a["info"], LEFT)
        gap = safe_text("100 % → 71.7 %: memoriza el ruido", size=22, color=RED
                        ).next_to(test, DOWN, buff=0.35).align_to(test, LEFT)
        zl = safe_text("- - - zonas verdaderas", size=18, color=MUTED).next_to(plane.frame, UP,
                                                                               buff=0.12)
        self.play(Create(zones), FadeIn(zl))
        self.play(FadeIn(test))
        self.play(FadeIn(gap))

    # ------------------------------------------------------------------
    def cost_complexity(self) -> None:
        self.beat("46–49 min. Poda de costo-complejidad (Breiman et al., 1984). Cada subárbol de "
                  "la ruta es una RECTA: su costo es su impureza R(T) más α por cada hoja. Con "
                  "α = 0 gana el árbol completo (impureza 0). Al subir α, las rectas con mucha "
                  "pendiente (muchas hojas) se encarecen y gana una más baja: la envolvente "
                  "amarilla. Mueva α y pida que observen cómo desaparecen primero las islas.")
        self.start("Poda de costo-complejidad", "Pagar α por cada hoja", RED)
        plane = lab_plane(center=(-4.4, -0.45, 0), height=4.4)
        path = lab_path()
        alphas = [a for a, _ in path]
        f = tex(r"R_\alpha(T) = R(T) + \alpha\,|T|", size=36).move_to([3.0, 2.05, 0])
        ax = Axes(x_range=[0, 0.1, 0.02], y_range=[0, 0.6, 0.1], x_length=6.0, y_length=3.5,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": True, "tick_size": 0.04})
        ax.move_to([3.15, -0.35, 0])
        xl = safe_text("α (precio por hoja)", size=18, color=MUTED).next_to(ax, DOWN, buff=0.38)
        yl = safe_text("costo R_α(T)", size=18, color=MUTED).next_to(ax, UP, buff=0.08
                                                                      ).align_to(ax, LEFT)
        nums = VGroup(*[safe_text(f"{v:g}", size=14, color=MUTED).next_to(ax.c2p(v, 0), DOWN,
                                                                          buff=0.08)
                        for v in (0, 0.05, 0.1)])
        lines = VGroup()
        labels = VGroup()
        for a_k, t in path:
            L, r = len(leaves(t)), R_tree(t, 60)
            a_end = min(0.1, (0.6 - r) / L)
            lines.add(Line(ax.c2p(0, r), ax.c2p(a_end, r + a_end * L), color=MUTED,
                           stroke_width=2))
            if L in (18, 7, 4, 1):          # las demás quedan demasiado juntas
                labels.add(safe_text(f"{L}", size=15, color=MUTED).next_to(ax.c2p(0, r), LEFT,
                                                                          buff=0.08))
        env = VGroup()
        bounds = alphas[1:] + [0.1]
        for (a_k, t), a_next in zip(path, bounds):
            L, r = len(leaves(t)), R_tree(t, 60)
            env.add(Line(ax.c2p(a_k, r + a_k * L), ax.c2p(a_next, r + a_next * L), color=YELLOW,
                         stroke_width=5))
        hl = safe_text("hojas", size=14, color=MUTED).next_to(labels, UP, buff=0.08)
        self.play(FadeIn(plane), Write(f))
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(nums))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.12), FadeIn(labels),
                  FadeIn(hl))
        self.play(Create(env), run_time=1.5)

        alpha = ValueTracker(0.0)

        def current():
            a = alpha.get_value()
            k = max(i for i, ak in enumerate(alphas) if ak <= a + 1e-12)
            return path[k][1]

        regs = always_redraw(lambda: plane.regions(leaves(current())))
        pts = plane.dots(LX, LY, r=0.08)
        vline = always_redraw(lambda: DashedLine(ax.c2p(alpha.get_value(), 0),
                                                 ax.c2p(alpha.get_value(), 0.6), color=INK,
                                                 stroke_width=2))
        read = always_redraw(lambda: VGroup(
            safe_text(f"α = {alpha.get_value():.3f}", size=20, color=YELLOW, weight=BOLD),
            safe_text(f"{len(leaves(current()))} hojas", size=20, color=TEAL),
            safe_text(f"entren. {accuracy(current(), LX, LY) * 100:.0f} %", size=20, color=BLUE),
            safe_text(f"prueba {accuracy(current(), TX, TY) * 100:.1f} %", size=20, color=GREEN),
        ).arrange(RIGHT, buff=0.3).move_to([0, -3.05, 0]))
        self.add(regs)
        self.play(FadeIn(pts))
        self.add(vline, read)

        self.beat("49–50 min. α sube: 18 → 14 → 12 → 10 → 7 → 4 hojas en un intervalo muy "
                  "corto (0.014–0.019) y luego, hasta α = 0.090, se queda en 4. Pregunte: ¿por "
                  "qué el árbol de 4 hojas aguanta tanto? Porque sus cortes capturan las zonas "
                  "verdaderas. Más allá de 0.090 sólo queda la raíz.")
        for target, rt in [(0.0145, 1.4), (0.0175, 1.2), (0.03, 1.0), (0.085, 2.0),
                           (0.095, 0.8), (0.03, 1.4)]:
            self.play(alpha.animate.set_value(target), run_time=rt, rate_func=smooth)
        self.wait(0.3)   # ≥ 3 fotogramas: una pausa de 1 fotograma rompe la concatenación

    # ------------------------------------------------------------------
    def weakest(self) -> None:
        self.beat("50–52 min. ¿De dónde salen esos α? Del ESLABÓN MÁS DÉBIL. Para cada nodo "
                  "interno, g(t) = cuánta impureza quita su subárbol por cada hoja extra. El "
                  "nodo con menor g se poda primero, justo cuando α = g. En el árbol de 4 hojas "
                  "es la raíz (0.090 < 0.115 < 0.125): con α > 0.090 el árbol pasa de 4 hojas a "
                  "1. Pida que verifiquen R(T) de la raíz: 13/60·0.260 + 21/60·0.308 + 0 + "
                  "20/60·0.180 = 0.224.")
        self.clear()
        self.start("El eslabón más débil", "Cuánto aporta cada hoja extra", RED)
        _, t4 = lab_four_leaves()
        spec = spec_from_node(t4, lab_question, LAB_NAMES)
        tv = TreeView(spec, width=5.6, level_h=1.5, top=(-3.6, 1.75, 0), node_w=2.15,
                      leaf_w=1.25, show_names=False)
        self.play(FadeIn(tv.final()))
        g = tex(r"g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}", size=36).move_to([3.3, 1.75, 0])
        self.play(Write(g))
        rows = [["nodo", "R(t)", "R(T_t)", "|T_t|", "g(t)"]]
        cols = []
        names = {"": "raíz", "0": "izquierdo", "1": "derecho"}
        gs = []
        for path, nd in zip(["", "0", "1"], internal(t4)):
            gv = weakest_link(nd, 60)
            gs.append(gv)
            rows.append([names[path], f"{R(nd, 60):.3f}", f"{R_tree(nd, 60):.3f}",
                         str(len(leaves(nd))), f"{gv:.3f}"])
        for gv in gs:
            cols.append([INK, MUTED, MUTED, MUTED, YELLOW if gv == min(gs) else INK])
        tb = table(rows, col_widths=[1.6, 1.15, 1.25, 0.9, 1.1], size=22, row_h=0.55,
                   colors=cols).move_to([3.3, -0.2, 0])
        self.play(FadeIn(tb))
        hl = SurroundingRectangle(tv.node(""), color=YELLOW, buff=0.08)
        concl = VGroup(safe_text("menor g(t): la raíz", size=22, color=YELLOW, weight=BOLD),
                       safe_text("con α > 0.090 el árbol queda en 1 hoja", size=21)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(tb, DOWN, buff=0.4)
        self.play(Create(hl), Indicate(tb[1], color=YELLOW, scale_factor=1.04))
        self.play(FadeIn(concl))

    # ------------------------------------------------------------------
    def path_curve(self) -> None:
        self.beat("52–54 min. La ruta completa: al podar, el entrenamiento baja siempre, pero la "
                  "prueba SUBE hasta 4 hojas (85.3 %) y luego se desploma. Advertencia: aquí "
                  "miramos la prueba sólo para ilustrar; en la práctica α se elige con "
                  "validación cruzada (o la regla de un error estándar) y la prueba se usa una "
                  "sola vez al final.")
        self.clear()
        self.start("La ruta de poda", "Menos hojas: peor en entrenamiento, mejor en prueba", RED)
        path = lab_path()
        ax = Axes(x_range=[0, 19, 2], y_range=[0.5, 1.0, 0.1], x_length=8.6, y_length=4.2,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": True, "tick_size": 0.04})
        ax.move_to([-1.2, -0.45, 0])
        xn = VGroup(*[safe_text(str(v), size=15, color=MUTED).next_to(ax.c2p(v, 0.5), DOWN,
                                                                      buff=0.1)
                      for v in (1, 4, 7, 10, 12, 14, 18)])
        yn = VGroup(*[safe_text(f"{int(v * 100)} %", size=15, color=MUTED).next_to(
            ax.c2p(0, v), LEFT, buff=0.1) for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
        xl = safe_text("número de hojas", size=18, color=MUTED).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(xn), FadeIn(yn), FadeIn(xl))
        series = []
        for X, Y, col, name in [(LX, LY, BLUE, "entrenamiento"), (TX, TY, GREEN, "prueba")]:
            pts = [ax.c2p(len(leaves(t)), accuracy(t, X, Y)) for _, t in path]
            line = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, radius=0.07, color=col) for p in pts])
            lab = safe_text(name, size=20, color=col, weight=BOLD).next_to(pts[0], RIGHT,
                                                                           buff=0.15)
            series.append((line, dots, lab))
            self.play(Create(line), FadeIn(dots), FadeIn(lab), run_time=1.2)
        best = max(path, key=lambda z: accuracy(z[1], TX, TY))[1]
        star = ax.c2p(len(leaves(best)), accuracy(best, TX, TY))
        ring = Circle(radius=0.2, color=YELLOW, stroke_width=4).move_to(star)
        note = VGroup(safe_text("4 hojas: 85.3 % en prueba", size=22, color=YELLOW, weight=BOLD),
                      safe_text("elegir α con validación cruzada,", size=20),
                      safe_text("no mirando la prueba", size=20)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([5.0, 0.3, 0])
        self.play(Create(ring), FadeIn(note))
