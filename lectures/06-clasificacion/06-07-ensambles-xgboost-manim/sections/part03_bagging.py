"""Acto 3 · Bagging y bosques aleatorios (17–27 min).

Una urna con las seis laptops: sorteo con reemplazo {A, A, C, D, F, F}; B y E
quedan fuera de la bolsa y sirven para evaluar sin trampa. La fracción fuera
de la bolsa tiende a e⁻¹. En los 40 puntos: árboles bootstrap tenues y su
promedio grueso, más suave y con menos error. Bosques aleatorios: en cada nodo
sólo se miran m variables al azar, lo que baja la correlación entre árboles.
"""
from __future__ import annotations

from manim import *

from sections.data import (BAG_TR, BOOT, LAPTOPS, OOB, SIGMA2, bagged_trees, boot_tree,
                           f_true, mse_vs_truth, oob_errors, oob_fraction, reg_predict, reg_steps,
                           reg_tree)
from sections.part01_regression import laptop_axes, money, step_graph
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, table, tex)


def ball(name: str, color=BLUE, r: float = 0.3) -> VGroup:
    c = Circle(radius=r, color=color, fill_color=color, fill_opacity=0.25, stroke_width=3)
    return VGroup(c, safe_text(name, size=22, color=INK, weight=BOLD).move_to(c))


def mini_tree(root: str, kids: tuple[str, str], color=TEAL, highlight=None) -> VGroup:
    def node(t, col):
        txt = safe_text(t, size=18, color=INK, weight=BOLD)
        box = RoundedRectangle(corner_radius=0.1, width=txt.width + 0.3, height=0.45,
                               stroke_color=col, stroke_width=2.5, fill_color=PANEL,
                               fill_opacity=0.95)
        return VGroup(box, txt.move_to(box))
    r = node(root, highlight or color)
    a, b = node(kids[0], color), node(kids[1], color)
    VGroup(a, b).arrange(RIGHT, buff=0.35).next_to(r, DOWN, buff=0.55)
    edges = VGroup(Line(r.get_bottom(), a.get_top(), color=color, stroke_width=2.5),
                   Line(r.get_bottom(), b.get_top(), color=color, stroke_width=2.5))
    return VGroup(edges, r, a, b)


class BaggingSection:
    def bagging(self) -> None:
        self.bootstrap()
        self.out_of_bag()
        self.average_curves()
        self.forest()

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        self.beat("17–19 min. Si tuviéramos muchos conjuntos de entrenamiento, entrenaríamos un "
                  "árbol en cada uno y los promediaríamos. Sólo tenemos uno: lo SIMULAMOS "
                  "sorteando con reemplazo. Seis extracciones de una urna con A–F: salen A dos "
                  "veces y F dos veces; B y E no salen. Pregunte: ¿cuántas veces esperan que "
                  "quede alguien fuera? (Lo calculamos en un momento.)")
        self.start("Bagging", "Bootstrap: sortear con reemplazo", GREEN)
        urn = RoundedRectangle(corner_radius=0.2, width=3.4, height=2.3, stroke_color=GRID,
                               stroke_width=2, fill_color=PANEL, fill_opacity=0.9
                               ).move_to([-4.3, 0.2, 0])
        ulab = safe_text("datos originales", size=20, color=MUTED).next_to(urn, UP, buff=0.12)
        names = [l[0] for l in LAPTOPS]
        balls = VGroup(*[ball(n) for n in names]).arrange_in_grid(rows=2, cols=3, buff=0.3
                                                                   ).move_to(urn)
        self.play(FadeIn(urn), FadeIn(ulab), LaggedStart(*[FadeIn(b) for b in balls],
                                                          lag_ratio=0.08))
        slots = VGroup(*[RoundedRectangle(corner_radius=0.12, width=0.8, height=0.8,
                                          stroke_color=YELLOW, stroke_width=2)
                         for _ in range(6)]).arrange(RIGHT, buff=0.18).move_to([2.6, 0.9, 0])
        slab = safe_text("muestra bootstrap (n = 6, con reemplazo)", size=20, color=YELLOW
                         ).next_to(slots, UP, buff=0.15)
        self.play(FadeIn(slots), FadeIn(slab))
        drawn = VGroup()
        for i, name in enumerate(BOOT):
            src = balls[names.index(name)]
            cp = src.copy()
            self.play(Indicate(src, color=YELLOW, scale_factor=1.25), run_time=0.35)
            self.play(cp.animate.move_to(slots[i]), run_time=0.45)
            drawn.add(cp)
        out = VGroup(*[balls[names.index(n)] for n in OOB])
        self.play(*[b[0].animate.set_color(GREEN).set_fill(GREEN, 0.35) for b in out])
        oob_lab = VGroup(safe_text("fuera de la bolsa (OOB):", size=21, color=GREEN, weight=BOLD),
                         safe_text("B y E no participan en este árbol", size=20)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([2.6, -0.45, 0])
        self.play(FadeIn(oob_lab))
        f = tex(r"\hat f_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B} \hat f^{*b}(x)", size=36,
                color=YELLOW).move_to([2.6, -1.9, 0])
        fl = safe_text("(en clasificación: voto mayoritario)", size=18, color=MUTED
                       ).next_to(f, DOWN, buff=0.15)
        self.play(Write(f), FadeIn(fl))

    # ------------------------------------------------------------------
    def out_of_bag(self) -> None:
        self.beat("19–21 min. Un árbol con esa muestra: el corte vuelve a ser RAM ≤ 12, pero las "
                  "hojas cambian (8 875 y 24 000). B y E no se usaron: sus errores (−1 675 y "
                  "−3 000) son una evaluación honesta, gratis. Probabilidad de quedar fuera: "
                  "(1 − 1/n)ⁿ, que tiende a e⁻¹ ≈ 0.368. Cada árbol deja fuera ~1/3 de los "
                  "datos: el error OOB es casi una validación cruzada sin costo extra.")
        self.start("Fuera de la bolsa", "Evaluar con los que no salieron", GREEN)
        ax, labels = laptop_axes(center=(-3.3, -0.7, 0))
        self.play(Create(ax), FadeIn(labels))
        counts = {n: BOOT.count(n) for n in set(BOOT)}
        dots = VGroup()
        for name, r, p in LAPTOPS:
            in_bag = name in counts
            col = BLUE if in_bag else GREEN
            d = Dot(ax.c2p(r, p), radius=0.1, color=col)
            txt = name + (f" ×{counts[name]}" if in_bag and counts[name] > 1 else "")
            lab = safe_text(txt, size=16, color=col).next_to(d, LEFT if name in "ACE" else RIGHT,
                                                           buff=0.08)
            dots.add(VGroup(d, lab))
        self.play(FadeIn(dots))
        lm, rm, _ = boot_tree()
        st = step_graph(ax, [(0, 12, lm), (12, 18, rm)])
        hl = VGroup(safe_text(money(lm), size=18, color=YELLOW).next_to(ax.c2p(2, lm), UP, buff=0.08),
                    safe_text(money(rm), size=18, color=YELLOW).next_to(ax.c2p(13.5, rm), DOWN,
                                                                         buff=0.1))
        self.play(Create(st), FadeIn(hl))
        errs = VGroup()
        for name, price, pred, e in oob_errors():
            ram = next(l[1] for l in LAPTOPS if l[0] == name)
            errs.add(Line(ax.c2p(ram, price), ax.c2p(ram, pred), color=RED, stroke_width=5))
        rows = [["OOB", "real", "predicción", "error"]] + [
            [n, money(p), money(q), f"{e:+,.0f}".replace(",", " ")] for n, p, q, e in oob_errors()]
        tb = table(rows, col_widths=[0.9, 1.2, 1.5, 1.2], size=20, row_h=0.48,
                   colors=[[GREEN, INK, YELLOW, RED]] * 2).move_to([3.4, 1.35, 0])
        self.play(LaggedStart(*[Create(e) for e in errs], lag_ratio=0.3), FadeIn(tb))
        ax2 = Axes(x_range=[1, 40, 10], y_range=[0.24, 0.4, 0.05], x_length=4.6, y_length=1.9,
                   tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                            "include_ticks": False})
        ax2.move_to([3.5, -1.35, 0])
        curve = VMobject(color=GREEN, stroke_width=4).set_points_smoothly(
            [ax2.c2p(n, oob_fraction(n)) for n in range(2, 41)])
        asym = DashedLine(ax2.c2p(1, np.exp(-1)), ax2.c2p(40, np.exp(-1)), color=MUTED,
                          stroke_width=2)
        alab = tex(r"e^{-1} \approx 0.368", size=26, color=MUTED).next_to(asym, UP, buff=0.06
                                                                         ).align_to(asym, RIGHT)
        flab = tex(r"P(\text{quedar fuera}) = \left(1 - \tfrac{1}{n}\right)^{n}", size=28,
                   color=GREEN).next_to(ax2, UP, buff=0.12).align_to(ax2, LEFT)
        d6 = Dot(ax2.c2p(6, oob_fraction(6)), radius=0.07, color=YELLOW)
        t6 = safe_text(f"n = 6: {oob_fraction(6):.3f}", size=16, color=YELLOW
                       ).next_to(d6, DR, buff=0.08)
        nlab = safe_text("n", size=16, color=MUTED).next_to(ax2, DOWN, buff=0.08)
        self.play(Create(ax2), FadeIn(flab), FadeIn(nlab))
        self.play(Create(curve), Create(asym), FadeIn(alab))
        self.play(FadeIn(d6), FadeIn(t6))

    # ------------------------------------------------------------------
    def average_curves(self) -> None:
        self.beat("21–24 min. De vuelta a los 40 puntos. Cada línea tenue es un árbol profundo "
                  "entrenado con una muestra bootstrap distinta: todos son nerviosos, pero se "
                  "equivocan en lugares DISTINTOS. Su promedio (amarillo) es mucho más suave. "
                  "Error esperado en datos nuevos: un árbol 0.665, el promedio de 60 árboles "
                  "0.540; el piso es σ² = 0.36 (ruido irreducible). Más árboles nunca "
                  "sobreajustan: sólo cuestan tiempo.")
        self.start("Promediar árboles", "Cada árbol tiembla; el promedio no", GREEN)
        ax = Axes(x_range=[0, 10, 1], y_range=[-2.5, 6, 1], x_length=8.4, y_length=4.4,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False})
        ax.move_to([-2.2, -0.45, 0])
        pts = VGroup(*[Dot(ax.c2p(x, y), radius=0.06, color=BLUE) for x, y in BAG_TR])
        truth = DashedVMobject(ax.plot(lambda x: float(f_true(x)), x_range=[0, 10, 0.05],
                                       color=INK, stroke_width=3), num_dashes=60)
        self.play(Create(ax), FadeIn(pts), Create(truth))
        trees = bagged_trees()
        thin = VGroup()
        for t in trees[:6]:
            g = step_graph(ax, reg_steps(t), color=ORANGE, width=2.5)
            g.set_stroke(opacity=0.45)
            thin.add(g)
        self.play(LaggedStart(*[Create(g) for g in thin], lag_ratio=0.35), run_time=3.0)
        xs = np.linspace(0, 10, 401)
        avg = VMobject(color=YELLOW, stroke_width=6).set_points_as_corners(
            [ax.c2p(x, float(np.mean([reg_predict(t, x) for t in trees]))) for x in xs])
        self.play(Create(avg), run_time=1.8)
        one = reg_tree(BAG_TR[:, 0], BAG_TR[:, 1])
        e1 = mse_vs_truth(lambda x: reg_predict(one, x))
        eb = mse_vs_truth(lambda x: float(np.mean([reg_predict(t, x) for t in trees])))
        legend = VGroup(
            safe_text("árboles bootstrap (6 de 60)", size=19, color=ORANGE),
            safe_text("promedio de 60 árboles", size=19, color=YELLOW, weight=BOLD),
            safe_text("- - - función real", size=19, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.6, 1.4, 0])
        bars = VGroup()
        for i, (name, v, col) in enumerate([("un árbol", e1, ORANGE), ("bagging", eb, YELLOW),
                                            ("σ² ruido", SIGMA2, MUTED)]):
            b = Rectangle(width=v * 3.2, height=0.36, stroke_width=0, fill_color=col,
                          fill_opacity=0.8)
            lab = safe_text(name, size=18, color=col).next_to(b, LEFT, buff=0.12)
            val = safe_text(f"{v:.3f}", size=18, color=INK).next_to(b, RIGHT, buff=0.1)
            row = VGroup(lab, b, val)
            bars.add(row)
        for i, r in enumerate(bars):
            r[1].move_to([3.6 + r[1].width / 2, -0.6 - 0.55 * i, 0])
            r[0].next_to(r[1], LEFT, buff=0.12)
            r[2].next_to(r[1], RIGHT, buff=0.1)
        bt = safe_text("error esperado en datos nuevos", size=18, color=MUTED
                       ).next_to(bars, UP, buff=0.2).align_to(bars, LEFT)
        self.play(FadeIn(legend))
        self.play(FadeIn(bt), LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in bars],
                                          lag_ratio=0.2),
                  LaggedStart(*[FadeIn(VGroup(r[0], r[2])) for r in bars], lag_ratio=0.2))

    # ------------------------------------------------------------------
    def forest(self) -> None:
        self.beat("24–27 min. Problema de bagging: si una variable es muy predictiva, TODOS los "
                  "árboles la ponen en la raíz y se parecen (ρ alto, piso alto). Bosque "
                  "aleatorio (Breiman, 2001): en CADA nodo el mejor corte se busca sólo entre m "
                  "variables sorteadas (√p en clasificación, p/3 en regresión). Otras variables "
                  "llegan a la raíz, los árboles se decorrelacionan y el promedio mejora. Con "
                  "m = p, el bosque es exactamente bagging.")
        self.start("Bosques aleatorios", "Sortear también las variables", GREEN)
        top = safe_text("bagging: todos eligen la variable más fuerte", size=21, color=MUTED
                        ).move_to([-3.4, 1.85, 0])
        bag = VGroup(*[mini_tree("x₁", ("x₂", "x₄"), highlight=RED) for _ in range(3)]
                     ).arrange(RIGHT, buff=0.35).next_to(top, DOWN, buff=0.3)
        rho1 = safe_text("árboles parecidos → ρ alto", size=20, color=RED
                         ).next_to(bag, DOWN, buff=0.25)
        self.play(FadeIn(top), LaggedStart(*[FadeIn(t) for t in bag], lag_ratio=0.2))
        self.play(FadeIn(rho1))
        feats = VGroup(*[safe_text(f"x{s}", size=24, color=MUTED, weight=BOLD)
                         for s in "₁₂₃₄₅₆₇₈₉"]).arrange(RIGHT, buff=0.3).move_to([3.4, 1.85, 0])
        pick = [1, 4, 7]
        boxes = VGroup(*[SurroundingRectangle(feats[i], color=YELLOW, buff=0.06) for i in pick])
        flab = safe_text("en cada nodo: m = 3 de p = 9 al azar", size=19, color=YELLOW
                         ).next_to(feats, DOWN, buff=0.2)
        self.play(FadeIn(feats))
        self.play(Create(boxes), FadeIn(flab))
        rf = VGroup(mini_tree("x₂", ("x₅", "x₁")), mini_tree("x₁", ("x₈", "x₃")),
                    mini_tree("x₅", ("x₂", "x₇"))).arrange(RIGHT, buff=0.35)
        rf.next_to(flab, DOWN, buff=0.35)
        rho2 = safe_text("árboles distintos → ρ bajo", size=20, color=GREEN
                         ).next_to(rf, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(t) for t in rf], lag_ratio=0.2))
        self.play(FadeIn(rho2))
        rows = [["hiperparámetro", "scikit-learn", "habitual"],
                ["árboles B", "n_estimators", "200–1000"],
                ["variables m", "max_features", "√p (clasif.)"],
                ["hoja mínima", "min_samples_leaf", "1–5"]]
        tb = table(rows, col_widths=[2.3, 2.5, 1.9], size=18, row_h=0.4,
                   colors=[[INK, TEAL, MUTED]] * 3).move_to([0, -2.55, 0])
        self.play(FadeIn(tb))
