"""Acto 5 · ¿Cuánto χ² produce el azar? (27–38 min).

Barajar las etiquetas de satisfacción destruye cualquier relación con el
retraso y conserva los totales. Cada barajada produce un χ²; el histograma de
400 barajadas tiene la forma de la distribución χ² con 1 grado de libertad.
Valor crítico 3.84, valor p. Grados de libertad como celdas libres, curvas
para gl = 1, 2, 4.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (AIR_RETRASO, CRIT_1, LEVELS, SEAT_CHI2, SEAT_P, TOY, TOY_CHI2, TOY_P,
                           chi2_crit, chi2_pdf, chi2_stat, permutation_null)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL, YELLOW,
                            Table2, card, fmt, person, safe_text, tex)

STEP = 0.19
SIZE = 0.12
BIN = 0.5
XMAX = 13


def positions(n: int, cols: int, center) -> list[np.ndarray]:
    rows = (n + cols - 1) // cols
    w, h = (cols - 1) * STEP, (rows - 1) * STEP
    origin = np.array(center, dtype=float) + np.array([-w / 2, h / 2, 0])
    return [origin + RIGHT * (i % cols) * STEP + DOWN * (i // cols) * STEP for i in range(n)]


def hist_bars(ax: Axes, chis: np.ndarray) -> VGroup:
    edges = np.arange(0, XMAX + BIN, BIN)
    counts, _ = np.histogram(np.clip(chis, 0, XMAX - 1e-6), bins=edges)
    dens = counts / (len(chis) * BIN)
    bars = VGroup()
    for k, d in enumerate(dens):
        x0, x1 = edges[k], edges[k + 1]
        h = max(float(d), 0.0)
        r = Rectangle(width=ax.c2p(x1, 0)[0] - ax.c2p(x0, 0)[0],
                      height=max(ax.c2p(0, h)[1] - ax.c2p(0, 0)[1], 0.001),
                      stroke_color=BLUE, stroke_width=1, fill_color=BLUE, fill_opacity=0.55)
        r.move_to(ax.c2p((x0 + x1) / 2, 0), aligned_edge=DOWN)
        bars.add(r)
    return bars


class NullSection:
    def null_world(self) -> None:
        labels, perms, chis = permutation_null()
        n0 = int(TOY[0].sum())

        self.beat("27–29 min. Idea central: fabriquemos mundos donde el retraso NO importa. "
                  "Tomamos las 200 tarjetas de satisfacción y las BARAJAMOS entre los pasajeros. "
                  "Los totales se conservan (120/80 y 100/100), pero cualquier relación con el "
                  "retraso queda destruida. Pregunte: ¿qué χ² esperan obtener?, ¿exactamente 0?")
        head = self.start("¿Cuánto χ² produce el azar?", "Barajar destruye la relación", YELLOW)
        cA, cB = [-5.0, 0.15, 0], [-2.2, 0.15, 0]
        posA, posB = positions(n0, 10, cA), positions(200 - n0, 10, cB)
        allpos = posA + posB
        dots = VGroup(*[person(int(s), SIZE).move_to(p) for s, p in zip(labels, allpos)])
        ga = safe_text("Sin retraso", size=20, color=INK, weight=BOLD).move_to([cA[0], 1.75, 0])
        gb = safe_text("Con retraso", size=20, color=INK, weight=BOLD).move_to([cB[0], 1.75, 0])
        self.play(FadeIn(dots, lag_ratio=0.003, run_time=1.0), FadeIn(ga), FadeIn(gb))
        ax = Axes(x_range=[0, XMAX, 1], y_range=[0, 1.2, 0.4], x_length=6.2, y_length=3.6,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2})
        ax.move_to([3.4, -0.4, 0])
        xl = VGroup(*[safe_text(str(v), size=16, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
                      for v in range(0, XMAX + 1, 2)])
        xt = safe_text("χ² de cada barajada", size=20, color=MUTED).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(xl), FadeIn(xt))
        obs_txt = safe_text(f"real: χ² = {TOY_CHI2:.1f}", size=22, color=GREEN, weight=BOLD)
        obs_txt.move_to([-3.6, -2.15, 0])
        self.play(FadeIn(obs_txt))

        self.beat("29–31 min. Seis barajadas, una por una. Cada vez se rehace la tabla y se "
                  "calcula χ². Los valores quedan entre 0 y unos pocos: el azar produce χ² "
                  "PEQUEÑOS, casi nunca cero. Pida que comparen con el 33.3 real.")
        tally = VGroup()
        cur = safe_text("", size=22)
        for k, perm in enumerate(perms):
            new_dots = VGroup(*[person(int(s), SIZE).move_to(p) for s, p in zip(perm, allpos)])
            a = int(perm[:n0].sum())
            x2 = chi2_stat([[a, n0 - a], [100 - a, 80 - 100 + a]])
            txt = safe_text(f"barajada {k + 1}: χ² = {x2:.2f}", size=22, color=YELLOW, weight=BOLD)
            txt.move_to([-3.6, -2.65, 0])
            mark = Dot(ax.c2p(min(x2, XMAX - 0.1), 0.06 + 0.08 * sum(
                1 for t in tally if abs(t.x2 - x2) < BIN / 2)), color=YELLOW, radius=0.07)
            mark.x2 = x2
            self.play(Transform(dots, new_dots), Transform(cur, txt) if k else FadeIn(txt),
                      run_time=0.7)
            if k == 0:
                cur = txt
            self.play(FadeIn(mark, scale=2), run_time=0.3)
            tally.add(mark)

        self.beat("31–33 min. La computadora hace 400 barajadas. El histograma es la "
                  "distribución del χ² CUANDO NO HAY RELACIÓN (hipótesis nula H₀). Casi todo "
                  "está por debajo de 4; el máximo de las 400 fue 12. Nuestro 33.3 ni siquiera "
                  "cabe en la gráfica.")
        bars = hist_bars(ax, chis)
        self.play(FadeOut(tally), FadeOut(cur),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03,
                              run_time=1.6))
        n_txt = safe_text(f"400 barajadas · máximo {chis.max():.0f}", size=22, color=BLUE,
                          weight=BOLD).move_to([-3.6, -2.65, 0])
        self.play(FadeIn(n_txt))
        off = Arrow(ax.c2p(XMAX - 4, 0.75), ax.c2p(XMAX, 0.75), buff=0, color=GREEN,
                    stroke_width=5)
        off_t = safe_text("real: 33.3,\nfuera de la gráfica", size=20, color=GREEN, weight=BOLD)
        off_t.next_to(off, UP, buff=0.1)
        self.play(GrowArrow(off), FadeIn(off_t))

        self.beat("33–35 min. Formalización. No hace falta simular siempre: la estadística "
                  "ya conoce esta forma. Cada contribución es aproximadamente un normal estándar "
                  "al cuadrado, y una suma de normales al cuadrado sigue una χ². Aquí con 1 "
                  "grado de libertad (en un momento veremos por qué 1). La curva morada cae "
                  "sobre el histograma.")
        head2 = self.swap_heading(head, "De la simulación a la teoría", "La distribución χ² con 1 g. l.",
                                  PURPLE)
        self.play(FadeOut(VGroup(dots, ga, gb, obs_txt, n_txt)))
        curve = ax.plot(lambda x: min(chi2_pdf(x, 1), 1.18), x_range=[0.03, XMAX - 0.05, 0.02],
                        color=PURPLE, stroke_width=5)
        self.play(Create(curve), run_time=1.4)
        f = tex(r"\frac{O_{ij}-E_{ij}}{\sqrt{E_{ij}}} \approx Z \sim \mathcal N(0,1)", size=34)
        g = tex(r"\chi^2 = \sum Z^2 \;\sim\; \chi^2_{\text{g.l.}}", size=38, color=PURPLE)
        VGroup(f, g).arrange(DOWN, buff=0.4).move_to([-3.5, 0.3, 0])
        self.play(Write(f))
        self.play(Write(g))

        self.beat("35–36 min. La regla de decisión: buscamos el punto que el azar supera sólo "
                  "el 5 % de las veces (α = 0.05). Con 1 g. l. es 3.84 (en las 400 barajadas, "
                  "el 4.5 % lo superó). El valor p es el área a la derecha de lo observado: "
                  "asiento p = 0.77 (el azar lo produce todo el tiempo); retraso "
                  "p ≈ 8 × 10⁻⁹ (el azar casi nunca).")
        tail = ax.get_area(curve, x_range=[CRIT_1, XMAX - 0.05], color=RED, opacity=0.55)
        crit = DashedLine(ax.c2p(CRIT_1, 0), ax.c2p(CRIT_1, 1.1), color=RED, stroke_width=3)
        crit_t = safe_text(f"3.84 · 5 %", size=20, color=RED, weight=BOLD).next_to(crit, UP, buff=0.05)
        self.play(FadeIn(tail), Create(crit), FadeIn(crit_t))
        seat_m = Triangle(color=ORANGE, fill_opacity=1).scale(0.1).rotate(PI)
        seat_m.move_to(ax.c2p(SEAT_CHI2, 0) + DOWN * 0.0 + UP * 0.12)
        self.play(FadeOut(f), FadeOut(g))
        pv = tex(r"p = P\big(\chi^2 \ge \chi^2_{\text{obs}} \mid H_0\big)", size=38)
        pv.move_to([-3.5, 1.2, 0])
        rows = VGroup(
            safe_text(f"Asiento: χ² = {SEAT_CHI2:.2f} → p = {SEAT_P:.2f}", size=24, color=ORANGE,
                      weight=BOLD),
            safe_text("no rechazo H₀: parece ruido", size=20, color=MUTED),
            safe_text(f"Retraso: χ² = {TOY_CHI2:.1f} → p ≈ 8 × 10⁻⁹", size=24, color=GREEN,
                      weight=BOLD),
            safe_text("rechazo H₀: hay relación", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([-3.5, -0.9, 0])
        self.play(Write(pv), FadeIn(seat_m))
        self.play(FadeIn(rows, shift=UP * 0.1))

        self.beat("36–37 min. ¿Por qué UN grado de libertad? Con los totales fijos, escriba sólo "
                  "una celda (80): las otras tres quedan obligadas (120 − 80, 100 − 80, ...). "
                  "Sólo una celda es libre de variar. En general, se fija la última fila y la "
                  "última columna: (r − 1)(c − 1).")
        self.clear()
        head = self.heading("Grados de libertad", "¿Cuántas celdas son libres?", TEAL)
        self.play(FadeIn(head), run_time=0.35)
        t2 = Table2([["", ""], ["", ""]], ["Sin retraso", "Con retraso"], ["Alta", "Baja"],
                    cw=1.4, ch=0.8, color=TEAL, row_w=2.2)
        t2.move_to([-3.6, 0.5, 0])
        rt = VGroup(*[safe_text(v, size=24, color=INK).next_to(t2.cell(i, 1), RIGHT, buff=0.3)
                      for i, v in enumerate(["120", "80"])])
        ct = VGroup(*[safe_text("100", size=24, color=INK).next_to(t2.cell(1, j), DOWN, buff=0.22)
                      for j in range(2)])
        self.play(FadeIn(t2), FadeIn(rt), FadeIn(ct))
        free = safe_text("80", size=30, color=YELLOW, weight=BOLD).move_to(t2.cell(0, 0))
        self.play(FadeIn(free, scale=1.4), t2.cell(0, 0).animate.set_fill(YELLOW, 0.25))
        forced = VGroup(*[safe_text(v, size=30, color=MUTED, weight=BOLD).move_to(t2.cell(i, j))
                          for (i, j), v in [((0, 1), "40"), ((1, 0), "20"), ((1, 1), "60")]])
        self.play(LaggedStart(*[FadeIn(f) for f in forced], lag_ratio=0.35))
        g1 = tex(r"(2-1)(2-1) = 1", size=36, color=TEAL).next_to(ct, DOWN, buff=0.35)
        self.play(Write(g1))

        air = AIR_RETRASO
        cols3 = ["Baja", "Media", "Alta"]
        t3 = Table2([[""] * 3] * 3, LEVELS["Retraso"], cols3, cw=1.1, ch=0.7, color=TEAL,
                    row_w=2.0, label_size=20, size=24)
        t3.move_to([2.9, 0.5, 0])
        rt3 = VGroup(*[safe_text(str(air[i].sum()), size=20, color=INK)
                       .next_to(t3.cell(i, 2), RIGHT, buff=0.22) for i in range(3)])
        ct3 = VGroup(*[safe_text(str(air[:, j].sum()), size=20, color=INK)
                       .next_to(t3.cell(2, j), DOWN, buff=0.18) for j in range(3)])
        self.play(FadeIn(t3), FadeIn(rt3), FadeIn(ct3))
        free3 = VGroup(*[safe_text(str(air[i, j]), size=24, color=YELLOW, weight=BOLD)
                         .move_to(t3.cell(i, j)) for i in range(2) for j in range(2)])
        self.play(LaggedStart(*[FadeIn(f, scale=1.3) for f in free3], lag_ratio=0.2),
                  *[t3.cell(i, j).animate.set_fill(YELLOW, 0.25) for i in range(2) for j in range(2)])
        forced3 = VGroup(*[safe_text(str(air[i, j]), size=24, color=MUTED, weight=BOLD)
                           .move_to(t3.cell(i, j)) for i in range(3) for j in range(3)
                           if i == 2 or j == 2])
        self.play(LaggedStart(*[FadeIn(f) for f in forced3], lag_ratio=0.15))
        g3 = tex(r"(3-1)(3-1) = 4", size=36, color=TEAL).next_to(ct3, DOWN, buff=0.35)
        self.play(Write(g3))
        src = safe_text("encuesta de 1,000 pasajeros", size=18, color=MUTED).next_to(t3, UP, buff=0.12)
        self.play(FadeIn(src))

        self.beat("37–38 min. Más celdas libres = más lugares donde el azar mete ruido, así que "
                  "la curva se desplaza a la derecha y el valor crítico crece: 3.84, 5.99, "
                  "9.49. Por eso un χ² sólo se interpreta junto con sus grados de libertad. "
                  "La tabla 3×3 real da χ² = 108.8 ≫ 9.49.")
        self.play(FadeOut(VGroup(t2, rt, ct, free, forced, g1, t3, rt3, ct3, free3, forced3, src,
                                 g3)))
        ax2 = Axes(x_range=[0, 16, 2], y_range=[0, 0.5, 0.1], x_length=9.5, y_length=3.8,
                   tips=False, axis_config={"color": GRID, "stroke_width": 2})
        ax2.move_to([-0.6, -0.5, 0])
        xl2 = VGroup(*[safe_text(str(v), size=16, color=MUTED).next_to(ax2.c2p(v, 0), DOWN, buff=0.1)
                       for v in range(0, 17, 2)])
        self.play(Create(ax2), FadeIn(xl2))
        legend = VGroup()
        for k, col in [(1, PURPLE), (2, ORANGE), (4, TEAL)]:
            c = ax2.plot(lambda x, k=k: min(chi2_pdf(x, k), 0.5), x_range=[0.05, 16, 0.03],
                         color=col, stroke_width=4)
            cv = chi2_crit(0.05, k)
            ln = DashedLine(ax2.c2p(cv, 0), ax2.c2p(cv, chi2_pdf(cv, k) + 0.06), color=col)
            lab = safe_text(f"g.l. = {k} · crítico {cv:.2f}", size=22, color=col, weight=BOLD)
            legend.add(lab)
            self.play(Create(c), Create(ln), run_time=0.8)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.7, 0.6, 0])
        self.play(FadeIn(legend))
