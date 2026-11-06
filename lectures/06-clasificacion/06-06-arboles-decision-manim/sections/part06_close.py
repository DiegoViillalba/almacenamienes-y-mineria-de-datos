"""Acto 6 · Límites, ejercicio y cierre (54–60 min).

Inestabilidad: quitar UN punto de 60 cambia la raíz y el número de hojas (puente
a ensambles). Fronteras en escalera: una diagonal necesita muchos cortes.
Ejercicio a mano con ocho puntos (Gini y tres cortes candidatos), su solución
y el cierre con ticket de salida.
"""
from __future__ import annotations

from manim import *

from sections.data import (EX_CANDIDATES, EX_POINTS, EX_X, EX_Y, LAB_ORDER, LX, LY,
                           UNSTABLE_POINT, exercise_tree, exercise_weighted, gini, internal,
                           lab_tree, lab_tree_without, leaves)
from sections.part05_pruning import lab_plane
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, TEAL, YELLOW, card, marker, pill, safe_text, table, tex)
from sections.widgets import Plane


class CloseSection:
    def close(self) -> None:
        self.instability()
        self.staircase()
        self.exercise()
        self.closing()

    # ------------------------------------------------------------------
    def instability(self) -> None:
        self.beat("54–55 min. Una debilidad seria: los árboles son INESTABLES. Quitamos un solo "
                  "punto (marcado en rojo) de los 60 y volvemos a crecer el árbol: la raíz pasa "
                  "de x₂ ≤ 5.70 a x₁ ≤ 8.75 y salen 20 hojas en vez de 18. Pregunte: si un dato "
                  "cambia tanto el modelo, ¿qué podríamos hacer? (Promediar muchos árboles: "
                  "bagging, bosques aleatorios y boosting, la próxima clase.)")
        self.start("Una debilidad: la varianza", "Quitar 1 punto de 60 cambia todo el árbol", RED)
        full = lab_tree()
        alt = lab_tree_without(UNSTABLE_POINT)
        keep = [i for i in range(len(LY)) if i != UNSTABLE_POINT]
        panels = VGroup()
        for k, (tree, idx, title) in enumerate([(full, list(range(len(LY))), "60 puntos"),
                                                (alt, keep, "sin el punto rojo")]):
            pl = lab_plane(center=(-3.4 + 6.8 * k, -0.5, 0), height=4.1)
            regs = pl.regions(leaves(tree))
            pts = pl.dots(LX[idx], LY[idx], r=0.07)
            root = pl.cut_of(tree, color=YELLOW, width=6)
            cap = VGroup(safe_text(title, size=22, color=INK, weight=BOLD),
                         safe_text(f"raíz x{'₁₂'[tree.feature]} ≤ {tree.threshold:.2f} · "
                                   f"{len(leaves(tree))} hojas", size=19, color=YELLOW)
                         ).arrange(DOWN, buff=0.08).next_to(pl.frame, UP, buff=0.12)
            panels.add(VGroup(pl, regs, pts, root, cap))
        lost = Circle(radius=0.16, color=RED, stroke_width=5).move_to(
            panels[0][0].p(*LX[UNSTABLE_POINT]))
        p0 = panels[0]
        self.play(FadeIn(p0[0]), FadeIn(p0[1]), FadeIn(p0[2]), Create(p0[3]), FadeIn(p0[4]))
        self.play(Create(lost))
        arrow = Arrow([-0.4, -0.5, 0], [0.4, -0.5, 0], color=RED, buff=0)
        p1 = panels[1]
        self.play(GrowArrow(arrow))
        self.play(FadeIn(p1[0]), FadeIn(p1[1]), FadeIn(p1[2]), Create(p1[3]), FadeIn(p1[4]))

    # ------------------------------------------------------------------
    def staircase(self) -> None:
        self.beat("55–56 min. Otra limitación geométrica: cada corte es vertical u horizontal. "
                  "Una frontera diagonal (tan fácil para la regresión logística) exige una "
                  "escalera: 4 escalones, 8, 16… cada vez más hojas para imitarla. Después, "
                  "ventajas y límites en una lámina.")
        self.start("Ventajas y límites", "Cortes paralelos a los ejes", RED)
        sq = Square(side_length=3.6, color=GRID, stroke_width=2).move_to([-4.2, -0.35, 0])
        x0, y0 = sq.get_corner(DL)[:2]
        L = 3.6
        diag = Line(sq.get_corner(DL), sq.get_corner(UR), color=YELLOW, stroke_width=4)

        def stairs(n):
            pts = [[x0, y0, 0]]
            for i in range(n):
                xa, xb = x0 + L * i / n, x0 + L * (i + 1) / n
                yb = y0 + L * (i + 1) / n
                pts += [[xb, pts[-1][1], 0], [xb, yb, 0]]
            return VMobject(color=TEAL, stroke_width=4).set_points_as_corners(pts)

        st = stairs(4)
        cap = safe_text("4 escalones", size=20, color=TEAL).next_to(sq, DOWN, buff=0.15)
        self.play(Create(sq), Create(diag))
        self.play(Create(st), FadeIn(cap))
        for n in (8, 16):
            new_cap = safe_text(f"{n} escalones", size=20, color=TEAL).next_to(sq, DOWN, buff=0.15)
            self.play(Transform(st, stairs(n)), Transform(cap, new_cap), run_time=0.8)
        pros = VGroup(safe_text("Ventajas", size=26, color=GREEN, weight=BOLD),
                      *[safe_text("· " + t, size=21) for t in [
                          "se leen como reglas SI… ENTONCES",
                          "no hay que estandarizar ni escalar",
                          "mezclan variables numéricas y categóricas",
                          "capturan interacciones sin pedirlas"]]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        cons = VGroup(safe_text("Límites", size=26, color=RED, weight=BOLD),
                      *[safe_text("· " + t, size=21) for t in [
                          "inestables: alta varianza",
                          "voraces: no garantizan el mejor árbol",
                          "fronteras en escalera",
                          "sobreajustan si no se podan"]]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        cols = VGroup(pros, cons).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([2.6, -0.3, 0])
        self.play(FadeIn(pros, shift=LEFT * 0.1))
        self.play(FadeIn(cons, shift=LEFT * 0.1))

    # ------------------------------------------------------------------
    def exercise(self) -> None:
        self.beat("56–59 min. Ejercicio individual (3 min, papel). Ocho puntos: 3 cuadrados y 5 "
                  "círculos. (1) Gini de la raíz. (2) Gini ponderado de los tres cortes. (3) ¿Cuál "
                  "elige CART? (4) Termina el árbol. Camine por el salón; error típico: olvidar "
                  "ponderar por el tamaño de cada lado.")
        self.start("Tu turno", "Construye el árbol a mano", YELLOW)
        pl = Plane((0, 9), (0, 9), height=4.3, center=(-4.0, -0.45, 0), xlabel="x₁",
                   ylabel="x₂", xticks=[0, 3, 6, 9], yticks=[0, 3, 6, 9])
        pts = pl.dots(EX_X, EX_Y, r=0.14,
                      labels=[f"({int(p[0])},{int(p[1])})" for p in EX_POINTS], label_size=13)
        self.play(FadeIn(pl), LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.08))
        names = {0: "x₁", 1: "x₂"}
        tasks = VGroup(
            safe_text("1. Gini de la raíz (3 ■ · 5 ●)", size=23),
            safe_text("2. Gini ponderado de cada corte:", size=23),
            safe_text("     " + "    ".join(f"{names[f]} ≤ {t:g}" for f, t in EX_CANDIDATES),
                      size=23, color=YELLOW, weight=BOLD),
            safe_text("3. ¿Cuál elige CART?", size=23),
            safe_text("4. Termina el árbol hasta hojas puras", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([2.5, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.1) for t in tasks], lag_ratio=0.15))
        timer = pill("3 minutos", YELLOW, size=24).move_to([2.5, -2.5, 0])
        self.play(FadeIn(timer))
        self._ex = dict(pl=pl, pts=pts, tasks=tasks, timer=timer)

        self.beat("59 min. Solución. Raíz 1 − (3/8)² − (5/8)² = 0.469. Cortes: x₁ ≤ 2.5 → 0.208; "
                  "x₁ ≤ 4.5 → 0.188 (gana); x₂ ≤ 4.5 → 0.300. A la derecha de 4.5 quedan 4 "
                  "círculos (puro). A la izquierda (3 ■, 1 ●) basta x₂ ≤ 4: árbol de 3 hojas.")
        a = self._ex
        self.play(FadeOut(a["timer"]), FadeOut(a["tasks"]))
        g0 = gini((5, 3))
        sol = VGroup(tex(rf"G_{{\text{{ra\'iz}}}} = 1 - \left(\tfrac38\right)^2 - "
                         rf"\left(\tfrac58\right)^2 = {g0:.3f}", size=30))
        for f, t in EX_CANDIDATES:
            w = exercise_weighted(f, t)
            best = (f, t) == (0, 4.5)
            sol.add(safe_text(f"{names[f]} ≤ {t:g}:  {w:.3f}" + ("   ← gana" if best else ""),
                              size=24, color=YELLOW if best else INK,
                              weight=BOLD if best else NORMAL))
        sol.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([2.6, 0.7, 0])
        self.play(Write(sol[0]))
        self.play(LaggedStart(*[FadeIn(s) for s in sol[1:]], lag_ratio=0.25))
        tree = exercise_tree()
        cuts = VGroup(*[pl.cut_of(n) for n in internal(tree)])
        regs = pl.regions(leaves(tree))
        self.play(FadeIn(regs), Create(cuts))
        self.bring_to_front(a["pts"])
        fin = safe_text("x₁ ≤ 4.5 → (sí) x₂ ≤ 4 → hojas puras: 3 hojas", size=22, color=GREEN,
                        weight=BOLD).next_to(sol, DOWN, buff=0.45).align_to(sol, LEFT)
        self.play(FadeIn(fin))

    # ------------------------------------------------------------------
    def closing(self) -> None:
        self.beat("60 min. Cierre con cuatro ideas y el ticket de salida. Próxima clase: si un "
                  "árbol es inestable, ¿qué pasa si promediamos cien? Bagging, bosques "
                  "aleatorios, boosting y XGBoost.")
        self.start("Cierre", "Lo que nos llevamos", YELLOW)
        ideas = VGroup(*[VGroup(safe_text(n, size=32, color=c, weight=BOLD),
                                safe_text(t, size=24)).arrange(RIGHT, buff=0.3)
                         for n, t, c in [
                             ("árbol", "preguntas en cadena = rectángulos en el plano", TEAL),
                             ("H, G", "miden la mezcla; la ganancia es cuánto baja", PURPLE),
                             ("C4.5", "razón de ganancia y cortes numéricos", ORANGE),
                             ("CART", "binario, Gini y poda con R(T) + α|T|", BLUE),
                         ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to([0, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = VGroup(
            safe_text("Ticket de salida", size=24, color=YELLOW, weight=BOLD),
            safe_text("¿Por qué un árbol de 18 hojas puras puede predecir peor que uno de 4?",
                      size=21, max_width=11),
            safe_text("Explica qué papel juega α en la poda.", size=21, max_width=11),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = card(ticket.width + 0.6, ticket.height + 0.4, YELLOW)
        tg = VGroup(box, ticket.move_to(box)).move_to([0, -1.95, 0])
        nxt = safe_text("Próxima clase: del árbol al bosque — bagging, bosques aleatorios y XGBoost",
                        size=20, color=MUTED).move_to([0, -3.05, 0])
        self.play(FadeIn(tg, shift=UP * 0.1))
        self.play(FadeIn(nxt))
