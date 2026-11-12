"""Acto 3 · El mundo independiente (14–21 min).

Las barras se vuelven tabla observada O. ¿Cuántos satisfechos habría entre
los 120 sin retraso si el retraso no importara? 50 % de 120 = 60. Así se
construye la tabla esperada E con los mismos totales. Residuos O − E = ±20,
que siempre suman cero.
"""
from __future__ import annotations

from manim import *

from sections.data import TOY, TOY_COLS, TOY_E, TOY_RESID, TOY_ROWS
from sections.theme import (BLUE, GREEN, INK, MUTED, ORANGE, PINK, PURPLE, RED, YELLOW, Table2,
                            card, safe_text, tex)


class ExpectedSection:
    def expected(self) -> None:
        self.beat("14–15 min. Las dos barras del retraso, escritas como conteos, son una TABLA "
                  "DE CONTINGENCIA. Los totales de fila (120, 80) y de columna (100, 100) se "
                  "llaman marginales. Llámela O: lo OBSERVADO.")
        head = self.start("El mundo independiente", "De las barras a la tabla observada O", BLUE)
        obs = Table2(TOY, TOY_ROWS, TOY_COLS, cw=1.45, ch=0.82, color=BLUE, totals=True,
                     row_w=2.3)
        obs.move_to([-3.4, 0.1, 0])
        o_tag = safe_text("O · observado", size=26, color=BLUE, weight=BOLD)
        o_tag.next_to(obs, UP, buff=0.3)
        n_tag = safe_text("n = 200", size=24, color=INK, weight=BOLD)
        n_tag.next_to(obs.col_tot, RIGHT, buff=0.45).align_to(obs.col_tot, DOWN)
        self.play(FadeIn(o_tag), FadeIn(obs.row_labels), FadeIn(obs.col_labels),
                  LaggedStart(*[FadeIn(VGroup(obs.cell(i, j), obs.num(i, j)), scale=0.8)
                                for i in range(2) for j in range(2)], lag_ratio=0.15))
        self.play(FadeIn(obs.row_tot), FadeIn(obs.col_tot), FadeIn(n_tag))

        self.beat("15–17 min. PREDICCIÓN. En un mundo donde el retraso NO importara, ¿cuántos "
                  "de los 120 pasajeros sin retraso estarían satisfechos? Espere la respuesta: "
                  "el 50 % general → 60. Escriba el cálculo: 120 × 100 / 200. Es exactamente "
                  "la regla del producto con frecuencias en lugar de probabilidades.")
        exp_t = Table2([[""] * 2] * 2, TOY_ROWS, TOY_COLS, cw=1.45, ch=0.82, color=YELLOW,
                       dashed=True, row_w=2.3)
        exp_t.move_to([3.4, 0.1, 0])
        exp_t.shift(UP * (obs.cell(0, 0).get_y() - exp_t.cell(0, 0).get_y()))
        e_tag = safe_text("E · esperado si fueran independientes", size=24, color=YELLOW,
                          weight=BOLD, max_width=6.0)
        e_tag.next_to(exp_t, UP, buff=0.3).align_to(o_tag, DOWN)
        self.play(FadeIn(e_tag), Create(exp_t.grid_part()), FadeIn(exp_t.row_labels),
                  FadeIn(exp_t.col_labels))
        hi_row = SurroundingRectangle(obs.row_tot[0], color=YELLOW, buff=0.08)
        hi_col = SurroundingRectangle(obs.col_tot[0], color=YELLOW, buff=0.08)
        hi_n = SurroundingRectangle(n_tag, color=YELLOW, buff=0.08)
        calc = tex(r"\frac{120 \times 100}{200} = 60", size=42, color=YELLOW)
        calc.move_to([3.4, -2.2, 0])
        self.play(Create(hi_row), Create(hi_col), Create(hi_n))
        self.play(Write(calc))
        v00 = safe_text("60", size=30, color=YELLOW, weight=BOLD).move_to(exp_t.cell(0, 0))
        self.play(TransformFromCopy(calc[0][-2:], v00))

        self.beat("17–18 min. Las otras tres celdas siguen la misma receta: total de su fila "
                  "por total de su columna, entre n. Pida al grupo que dicte los valores "
                  "antes de que aparezcan (60, 40, 40). Verifique que E conserva los MISMOS "
                  "totales que O: sólo cambia cómo se reparten por dentro.")
        rule = tex(r"E_{ij} = \frac{(\text{total fila } i)\,(\text{total columna } j)}{n}",
                   size=38, color=YELLOW)
        rule.move_to(calc)
        vals = VGroup(*[safe_text(f"{TOY_E[i, j]:.0f}", size=30, color=YELLOW, weight=BOLD)
                        .move_to(exp_t.cell(i, j)) for i, j in [(0, 1), (1, 0), (1, 1)]])
        self.play(FadeOut(hi_row), FadeOut(hi_col), FadeOut(hi_n), ReplacementTransform(calc, rule))
        self.play(LaggedStart(*[FadeIn(v, scale=1.3) for v in vals], lag_ratio=0.3))
        e_tot = VGroup(
            *[safe_text(t, size=26, color=INK).next_to(exp_t.cell(i, 1), RIGHT, buff=0.35)
              for i, t in enumerate(["120", "80"])],
            *[safe_text("100", size=26, color=INK).next_to(exp_t.cell(1, j), DOWN, buff=0.25)
              for j in range(2)])
        self.play(FadeIn(e_tot))
        same = safe_text("mismos totales que O", size=22, color=MUTED).move_to([3.4, -3.05, 0])
        self.play(FadeIn(same))

        self.beat("18–20 min. Ahora restamos celda a celda: O − E. Señale el patrón: +20 donde "
                  "hay MÁS de lo esperado (sin retraso y satisfecho; con retraso e insatisfecho) "
                  "y −20 donde hay menos. Esos residuos cuentan la historia de la dependencia.")
        self.play(FadeOut(rule), FadeOut(same))
        res = Table2(TOY_RESID, TOY_ROWS, TOY_COLS, cw=1.45, ch=0.82, color=PINK, row_w=2.3)
        res.move_to([0, -2.1, 0])
        for (i, j), n in res.nums.items():
            v = TOY_RESID[i, j]
            new = safe_text(f"{v:+.0f}", size=30, color=PINK, weight=BOLD).move_to(n)
            res.remove(n)
            res.nums[i, j] = new
            res.add(new)
        top = VGroup(obs, o_tag, n_tag, exp_t, e_tag, vals, v00, e_tot)
        self.play(top.animate.scale(0.78).shift(UP * 0.55))
        minus = tex(r"O - E", size=36, color=PINK).next_to(res, LEFT, buff=0.4)
        self.play(FadeIn(res.row_labels), FadeIn(res.col_labels), Write(minus),
                  *[FadeIn(res.cell(i, j)) for i in range(2) for j in range(2)])
        for i in range(2):
            for j in range(2):
                self.play(TransformFromCopy(VGroup(obs.num(i, j)), res.num(i, j)), run_time=0.45)

        self.beat("20–21 min. PREGUNTA CLAVE: ¿por qué no sumar simplemente los residuos para "
                  "medir la distancia? Pida que lo hagan: +20 − 20 − 20 + 20 = 0. Siempre da "
                  "cero, porque O y E tienen los mismos totales. Los positivos y negativos se "
                  "cancelan: hay que deshacerse del signo.")
        total = tex(r"+20 - 20 - 20 + 20 = 0", size=40, color=PINK)
        total.next_to(res, RIGHT, buff=0.6)
        self.play(Write(total))
        warn = safe_text("¡siempre cero!", size=26, color=RED, weight=BOLD)
        warn.next_to(total, DOWN, buff=0.2)
        self.play(FadeIn(warn, shift=UP * 0.1))
