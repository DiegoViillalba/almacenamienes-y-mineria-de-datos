"""Acto 7 · Comparar, ejercicio y cierre (56–60 min)."""
from __future__ import annotations

from manim import *

from sections.data import EX_B, EX_ROWS, EX_W
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, table, tex)


class CloseSection:
    def close(self) -> None:
        self.beat("56–57 min. Resumen en una tabla. Lea por columnas: el criterio cambia, la "
                  "existencia de fórmula cerrada cambia, pero la última columna no: la frontera "
                  "es siempre un hiperplano (en x o en φ(x)). Pregunte por qué la logística es "
                  "la única sin fórmula (β dentro de σ).")
        head = self.start("Comparar", "Cuatro criterios, una misma forma", YELLOW)
        rows = [
            ["método", "criterio", "¿fórmula cerrada?", "frontera"],
            ["perceptrón", "corregir errores", "no (iterativo)", "hiperplano"],
            ["mínimos cuadrados", "errores al cuadrado", "sí: (XᵀX)⁻¹Xᵀy", "hiperplano"],
            ["Fisher", "separar sombras", "sí: S⁻¹·(mA − mB)", "hiperplano"],
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
        key = safe_text("«Lineal» significa lineal en los parámetros.", size=26, color=YELLOW,
                        weight=BOLD).next_to(tbl, DOWN, buff=0.45)
        self.play(FadeIn(key))

        self.beat("57–59 min. Ejercicio (3 minutos, en parejas). Con w = (2, −1) y b = −3, "
                  "calcular z y p = σ(z) para P, Q y R, decidir la clase y decir cuál está "
                  "exactamente sobre la frontera. No avance hasta que la mayoría tenga los tres "
                  "valores de z.")
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

        self.beat("Solución. P: z = 6 − 1 − 3 = 2, p = 0.88, clase +1. Q: z = 2 − 2 − 3 = −3, "
                  "p = 0.05, clase −1. R: z = 4 − 1 − 3 = 0, p = 0.5: está SOBRE la frontera. "
                  "Pregunte: ¿qué cambiaría si duplicamos w y b? (Misma frontera, probabilidades "
                  "más extremas.)")
        rows = [["punto", "x", "z", "p", "clase"]]
        cols = []
        for n, x, z, p in EX_ROWS:
            cl = "+1" if z > 0 else ("−1" if z < 0 else "frontera")
            rows.append([n, f"({x[0]:g}, {x[1]:g})", (f"{z:+g}" if z else "0"), f"{p:.2f}", cl])
            c = ORANGE if z > 0 else (BLUE if z < 0 else YELLOW)
            cols.append([INK, MUTED, PURPLE, GREEN, c])
        a = table(rows, col_widths=[1.6, 2.2, 1.6, 1.6, 2.4], size=26, row_h=0.6,
                  colors=cols).move_to(q)
        self.play(Transform(q, a))

        self.beat("59–60 min. Cierre y ticket de salida: en una frase, ¿por qué un SVM con "
                  "kernel RBF, cuya frontera es curva, se considera un método lineal? Respuesta "
                  "esperada: es un hiperplano en el espacio φ(x); el modelo es lineal en sus "
                  "pesos α.")
        self.start("Para llevarse", "Cuatro ideas", YELLOW)
        ideas = VGroup(*[pill(t, c, size=24, width=11.6) for t, c in [
            ("1 · Un clasificador lineal es signo(wᵀx + b): w gira, b desliza", PURPLE),
            ("2 · MCO y Fisher se resuelven con una fórmula: su óptimo es un sistema lineal", BLUE),
            ("3 · La logística no: β queda dentro de σ; es convexa y se itera (Newton)", GREEN),
            ("4 · Los kernels curvan la frontera en x, pero es un hiperplano en φ(x)", PINK)]]
                      ).arrange(DOWN, buff=0.24).move_to([0, 0.15, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = safe_text("Ticket: ¿por qué un SVM con kernel RBF es un método lineal?",
                           size=24, color=YELLOW, weight=BOLD).next_to(ideas, DOWN, buff=0.5)
        self.play(FadeIn(ticket))
