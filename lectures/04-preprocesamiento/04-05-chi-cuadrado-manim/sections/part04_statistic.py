"""Acto 4 · Construir χ² (21–27 min).

Elevar al cuadrado elimina el signo. Un error de 20 no pesa igual sobre 40
que sobre 1,000: se divide entre E. Las cuatro contribuciones se suman y la
fórmula aparece como resumen de lo que ya se hizo. Asiento da 0.08.
"""
from __future__ import annotations

from manim import *

from sections.data import SEAT_CHI2, TOY_CHI2, TOY_COLS, TOY_CONTRIB, TOY_E, TOY_RESID, TOY_ROWS
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, PINK, PURPLE, RED, YELLOW, Table2,
                            card, fmt, safe_text, tex)


class StatisticSection:
    def statistic(self) -> None:
        self.beat("21–22 min. Para quitar el signo hay dos opciones: valor absoluto o cuadrado. "
                  "Usamos el cuadrado (castiga más las desviaciones grandes y tiene buena "
                  "teoría detrás). Las cuatro celdas valen 400. ¿Ya terminamos? Pregunte si "
                  "un error de 20 pesa lo mismo en todas las celdas.")
        head = self.start("Construir el estadístico", "Paso 1: quitar el signo", PINK)
        res = Table2(TOY_RESID.astype(int), TOY_ROWS, TOY_COLS, cw=1.6, ch=0.85, color=PINK,
                     row_w=2.3)
        for (i, j), n in list(res.nums.items()):
            new = safe_text(f"{TOY_RESID[i, j]:+.0f}", size=30, color=PINK, weight=BOLD).move_to(n)
            res.remove(n)
            res.nums[i, j] = new
            res.add(new)
        res.move_to([-3.3, 0.2, 0])
        self.play(FadeIn(res))
        sq = Table2((TOY_RESID ** 2).astype(int), TOY_ROWS, TOY_COLS, cw=1.6, ch=0.85,
                    color=PINK, row_w=2.3)
        sq.move_to([3.3, 0.2, 0])
        arrow = Arrow(res.get_right(), sq.get_left(), buff=0.25, color=MUTED)
        lab = tex(r"(O-E)^2", size=34, color=PINK).next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), Write(lab))
        self.play(TransformFromCopy(res, sq))

        self.beat("22–24 min. Dos tiendas: una esperaba 40 clientes y llegaron 60; otra "
                  "esperaba 1,000 y llegaron 1,020. La diferencia es 20 en ambas. ¿Cuál "
                  "sorpresa es mayor? La primera (50 % más); la segunda es ruido normal. "
                  "Por eso cada cuadrado se mide RELATIVO a lo esperado: se divide entre E.")
        head2 = self.swap_heading(head, "Construir el estadístico", "Paso 2: ¿20 es mucho o poco?", PINK)
        self.play(FadeOut(VGroup(res, sq, arrow, lab)))
        scale = 0.0042
        panels = VGroup()
        for k, (e, o) in enumerate([(40, 60), (1000, 1020)]):
            x = -3.3 if k == 0 else 3.3
            sc = 0.0029 if k else 0.05
            base = Line([x - 1.4, -2.4, 0], [x + 1.4, -2.4, 0], color=GRID)
            be = Rectangle(width=0.9, height=e * sc, stroke_color=YELLOW, stroke_width=3,
                           fill_opacity=0).move_to([x - 0.6, -2.4 + e * sc / 2, 0])
            be = DashedVMobject(be, num_dashes=30)
            bo = Rectangle(width=0.9, height=o * sc, stroke_width=0, fill_color=BLUE,
                           fill_opacity=0.85).move_to([x + 0.6, -2.4 + o * sc / 2, 0])
            te = safe_text(f"E = {e:,}".replace(",", " "), size=20, color=YELLOW).next_to(be, LEFT, buff=0.12)
            to = safe_text(f"O = {o:,}".replace(",", " "), size=20, color=BLUE).next_to(bo, RIGHT, buff=0.12)
            pct = safe_text(f"+20 → {100 * 20 / e:.0f} % de E", size=26, color=PINK, weight=BOLD)
            pct.move_to([x, 1.55, 0])
            panels.add(VGroup(base, be, bo, te, to, pct))
        self.play(*[Create(p[0]) for p in panels], *[FadeIn(p[1:5]) for p in panels])
        self.play(*[Write(p[5]) for p in panels])
        div = tex(r"\frac{(O-E)^2}{E}", size=48, color=PURPLE).move_to([0, -0.2, 0])
        self.play(Write(div))

        self.beat("24–25 min. Dividimos cada 400 entre su esperado: 400/60 = 6.67 en la fila "
                  "de 120 pasajeros y 400/40 = 10 en la de 80. La fila pequeña contribuye "
                  "más porque el mismo error es más sorprendente allí. Cada número es la "
                  "CONTRIBUCIÓN de su celda.")
        head3 = self.swap_heading(head2, "Construir el estadístico", "Paso 3: contribuciones y suma", PURPLE)
        self.play(FadeOut(panels), FadeOut(div))
        cells = VGroup()
        for i in range(2):
            for j in range(2):
                box = Rectangle(width=2.9, height=1.05, stroke_color=PURPLE, stroke_width=2,
                                fill_color=PURPLE, fill_opacity=0.10)
                box.move_to([-3.2 + j * 2.9, 1.25 - i * 1.05, 0])
                expr = tex(rf"\frac{{400}}{{{TOY_E[i, j]:.0f}}} = {TOY_CONTRIB[i, j]:.2f}",
                           size=34, color=PURPLE).move_to(box)
                cells.add(VGroup(box, expr))
        rl = VGroup(*[safe_text(r, size=20, color=MUTED).next_to(cells[2 * i], LEFT, buff=0.15)
                      for i, r in enumerate(TOY_ROWS)])
        cl = VGroup(*[safe_text(c, size=20, color=MUTED).next_to(cells[j], UP, buff=0.12)
                      for j, c in enumerate(TOY_COLS)])
        self.play(FadeIn(rl), FadeIn(cl), LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in cells],
                                                      lag_ratio=0.25))
        total = tex([r"6.67 + 6.67 + 10 + 10", r"=", rf"{TOY_CHI2:.2f}"], size=36)
        total[2].set_color(PURPLE)
        total.move_to([4.4, 0.6, 0])
        name = tex(r"\chi^2", size=60, color=PURPLE).next_to(total, UP, buff=0.4)
        self.play(Write(total))
        self.play(Write(name), Indicate(total[2], color=PURPLE))

        self.beat("25–26 min. AHORA sí la fórmula: no es nueva, es el resumen de los tres "
                  "pasos. Lea cada pieza señalando su color: O azul, E amarilla, el cuadrado "
                  "quita el signo, la división pone la escala, la suma recorre todas las "
                  "celdas (i filas, j columnas).")
        formula = tex([r"\chi^2", r"=", r"\sum_{i}\sum_{j}", r"\frac{(", r"O_{ij}", r"-",
                       r"E_{ij}", r")^2}{", r"E_{ij}", r"}"], size=50)
        formula[0].set_color(PURPLE)
        formula[4].set_color(BLUE)
        formula[6].set_color(YELLOW)
        formula[8].set_color(YELLOW)
        formula.move_to([0, -1.9, 0])
        box = SurroundingRectangle(formula, color=PURPLE, buff=0.2, corner_radius=0.12)
        self.play(Write(formula), Create(box))
        notes = VGroup(
            safe_text("cuadrado: quita el signo", size=20, color=PINK),
            safe_text("÷ E: escala relativa", size=20, color=YELLOW),
            safe_text("Σ: todas las celdas", size=20, color=PURPLE),
        ).arrange(RIGHT, buff=0.8).next_to(box, DOWN, buff=0.18)
        self.play(FadeIn(notes))

        self.beat("26–27 min. Comparación: con la misma receta, el asiento da χ² = 0.08 y el "
                  "retraso 33.3. Las barras pegadas producen un χ² casi nulo. Si O = E exacto, "
                  "χ² = 0. Pero ¿33.3 es 'mucho'? ¿0.08 es 'poco'? ¿Dónde está la frontera? "
                  "Deje la pregunta abierta: es el siguiente acto.")
        self.play(FadeOut(VGroup(cells, rl, cl, total, name, notes)),
                  VGroup(formula, box).animate.scale(0.7).move_to([0, 1.3, 0]))
        axis = NumberLine(x_range=[0, 36, 4], length=11, color=GRID, include_numbers=True,
                          font_size=22).move_to([0, -1.2, 0])
        seat = Dot(axis.n2p(SEAT_CHI2), color=RED, radius=0.13)
        ret = Dot(axis.n2p(TOY_CHI2), color=GREEN, radius=0.13)
        seat_l = safe_text(f"Asiento  {fmt(SEAT_CHI2, 2)}", size=24, color=RED, weight=BOLD)
        seat_l.next_to(seat, UP, buff=0.3).shift(RIGHT * 0.8)
        ret_l = safe_text(f"Retraso  {TOY_CHI2:.1f}", size=24, color=GREEN, weight=BOLD)
        ret_l.next_to(ret, UP, buff=0.3).shift(LEFT * 0.5)
        self.play(Create(axis))
        self.play(GrowFromCenter(seat), FadeIn(seat_l), GrowFromCenter(ret), FadeIn(ret_l))
        q = safe_text("¿Dónde termina el «poco» y empieza el «mucho»?", size=30, color=YELLOW,
                      weight=BOLD).move_to([0, -2.6, 0])
        self.play(Write(q))
