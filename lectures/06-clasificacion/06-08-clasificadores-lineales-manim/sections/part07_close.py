"""Acto 7 · Comparar, ejercicio y cierre (56–60 min)."""
from __future__ import annotations

from manim import *

from sections.data import EX_B, EX_ROWS, EX_W
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, table, tex)


class CloseSection:
    def close(self) -> None:
        self.beat("60")
        head = self.start("Comparar", "Elegir el criterio; después evaluar", YELLOW)
        rows = [
            ["método", "criterio", "¿fórmula cerrada?", "frontera"],
            ["perceptrón", "corregir errores", "no (iterativo)", "hiperplano"],
            ["mínimos cuadrados", "errores al cuadrado", "sí: (XᵀX)⁻¹Xᵀy", "hiperplano"],
            ["Fisher", "separar sombras", "sí: S_W⁻¹·(mA − mB)", "hiperplano"],
            ["logística", "verosimilitud", "no: Newton / gradiente", "hiperplano (p = ½)"],
            ["kernel", "cualquiera, sobre φ(x)", "según la pérdida", "hiperplano en φ"],
        ]
        colors = [[ORANGE, INK, RED, YELLOW], [BLUE, INK, GREEN, YELLOW],
                  [TEAL, INK, GREEN, YELLOW], [GREEN, INK, RED, YELLOW],
                  [PINK, INK, MUTED, YELLOW]]
        tbl = table(rows, col_widths=[2.6, 3.3, 3.6, 3.0], size=22, row_h=0.62, colors=colors)
        tbl.move_to([0, -0.2, 0])
        self.play(FadeIn(tbl[0]), FadeIn(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in tbl[1:-1]], lag_ratio=0.25),
                  run_time=2.2)
        last = SurroundingRectangle(VGroup(*[r[3] for r in tbl[1:-1]]), color=YELLOW, buff=0.12)
        self.play(Create(last))
        key = safe_text("Lineal en x o en φ(x): especifica el espacio.", size=26, color=YELLOW,
                        weight=BOLD).next_to(tbl, DOWN, buff=0.45)
        self.play(FadeIn(key))

        self.beat("61")
        head = self.start("Tu turno", "Leer un clasificador lineal a mano", ORANGE)
        prompt = VGroup(
            tex(rf"\mathbf{{w}} = ({EX_W[0]:g},\ {EX_W[1]:g}),\quad b = {EX_B:g}", size=40,
                color=PURPLE),
            tex(r"z = \mathbf{w}^\top\mathbf{x} + b,\qquad p = \sigma(z)", size=36),
        ).arrange(DOWN, buff=0.3).move_to([0, 1.2, 0])
        rows = [["punto", "x", "z", "p", "clase"]]
        rows += [[n, f"({x[0]:g}, {x[1]:g})", "?", "?", "?"] for n, x, _, _ in EX_ROWS]
        q = table(rows, col_widths=[1.6, 2.2, 1.6, 1.6, 2.4], size=26, row_h=0.6).move_to(
            [0, -1.3, 0])
        self.play(Write(prompt))
        self.play(FadeIn(q))

        rows = [["punto", "x", "z", "p", "clase"]]
        cols = []
        for n, x, z, p in EX_ROWS:
            cl = "+1" if z > 0 else ("−1" if z < 0 else "frontera")
            rows.append([n, f"({x[0]:g}, {x[1]:g})", (f"{z:+g}" if z else "0"), f"{p:.2f}", cl])
            c = ORANGE if z > 0 else (BLUE if z < 0 else YELLOW)
            cols.append([INK, MUTED, PURPLE, GREEN, c])
        a = table(rows, col_widths=[1.6, 2.2, 1.6, 1.6, 2.4], size=26, row_h=0.6,
                  colors=cols).move_to(q)
        for i, (n,x,z,p) in enumerate(EX_ROWS,start=1):
            self.beat(f"{61+i:02}")
            calc = tex(rf"{n}:\quad z=2({x[0]:g})-{x[1]:g}-3={z:g}",size=30,color=PURPLE).move_to([0,-3.0,0])
            self.play(Transform(q[i],a[i]),Write(calc))
            self.wait(0.4)
            self.play(FadeOut(calc))

        self.beat("65")
        self.start("Para llevarse", "Cuatro ideas", YELLOW)
        ideas = VGroup(*[pill(t, c, size=24, width=11.6) for t, c in [
            ("1 · Un clasificador lineal es signo(wᵀx + b): w gira, b desliza", PURPLE),
            ("2 · MCO y Fisher se resuelven con una fórmula: su óptimo es un sistema lineal", BLUE),
            ("3 · La logística no: β queda dentro de σ; es convexa y se itera (Newton)", GREEN),
            ("4 · Un kernel puede curvar la frontera en x; es lineal en φ(x)", PINK)]]
                      ).arrange(DOWN, buff=0.24).move_to([0, 0.15, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = safe_text("Pausa: ¿en qué espacio es lineal una SVM con kernel RBF?",
                           size=24, color=YELLOW, weight=BOLD).next_to(ideas, DOWN, buff=0.5)
        self.play(FadeIn(ticket))
