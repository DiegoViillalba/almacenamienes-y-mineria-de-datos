"""Acto 5 · Gradient boosting: ajustar residuos (37–46 min).

Cinco casas. Arriba, los datos y la predicción acumulada F; abajo, los
residuos. Cada ronda: los segmentos rojos bajan al panel de residuos, un tocón
se ajusta a ellos y sólo la mitad (η = 0.5) se suma a F. La tabla de rondas se
llena a la derecha (SSE 106 → 43.5 → 16.83 → 8.43). Cierra mostrando que el
residuo es el gradiente negativo de la pérdida cuadrática.
"""
from __future__ import annotations

from manim import *

from sections.data import ETA, GB, GB_X, GB_Y
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, safe_text, table, tex)


def steps_from(F) -> list[tuple[float, float, float]]:
    """Predicción por casa → escalones en [0.5, 5.5] (cortes en puntos medios)."""
    out, start = [], 0.5
    for i in range(len(F)):
        end = 5.5 if i == len(F) - 1 else (GB_X[i] + GB_X[i + 1]) / 2
        out.append((start, end, float(F[i])))
        start = end
    return out


def num(v: float) -> str:
    """+2.67, −6, 0: dos decimales sin ceros sobrantes."""
    if abs(v) < 5e-3:
        return "0"
    s = f"{v:+.2f}"
    return s[:-3] if s.endswith(".00") else s.rstrip("0")


def step_line(ax: Axes, steps, color=YELLOW, width: float = 5) -> VGroup:
    g = VGroup()
    for i, (a, b, v) in enumerate(steps):
        g.add(Line(ax.c2p(a, v), ax.c2p(b, v), color=color, stroke_width=width))
        if i and abs(steps[i - 1][2] - v) > 1e-9:
            g.add(Line(ax.c2p(a, steps[i - 1][2]), ax.c2p(a, v), color=color,
                       stroke_width=width * 0.4, stroke_opacity=0.6))
    return g


class GradientSection:
    def gradient(self) -> None:
        self.gb_setup()
        for m in (1, 2, 3):
            self.gb_round(m)
        self.gb_formula()
        self.gb_gradient()

    # ------------------------------------------------------------------
    def gb_setup(self) -> None:
        self.beat("37–39 min. Gradient boosting. Imaginen tasadores en fila: el primero da una "
                  "cifra; el segundo NO tasa desde cero, estudia cuánto se equivocó el primero y "
                  "propone una corrección; y así sucesivamente. Cinco casas (tamaño, precio en "
                  "millones). Ronda 0: la mejor constante es el promedio, 8. Residuos en rojo; "
                  "SSE = 106.")
        self.start("Gradient boosting", "Cada árbol corrige al anterior", BLUE)
        top = Axes(x_range=[0, 6, 1], y_range=[0, 16, 4], x_length=6.3, y_length=2.5, tips=False,
                   axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        top.move_to([-3.35, 0.85, 0])
        bot = Axes(x_range=[0, 6, 1], y_range=[-7, 7, 7], x_length=6.3, y_length=1.55, tips=False,
                   axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        bot.move_to([-3.35, -2.1, 0])
        tl = safe_text("datos y predicción F", size=17, color=MUTED).next_to(top, UP, buff=0.06
                                                                            ).align_to(top, LEFT)
        bl = safe_text("residuos r = y − F", size=17, color=RED).next_to(bot, UP, buff=0.06
                                                                       ).align_to(bot, LEFT)
        xt = VGroup(*[safe_text(str(int(x)), size=15, color=MUTED).next_to(top.c2p(x, 0), DOWN,
                                                                          buff=0.08)
                      for x in GB_X])
        dots = VGroup(*[Dot(top.c2p(x, y), radius=0.09, color=BLUE) for x, y in zip(GB_X, GB_Y)])
        ylab = VGroup(*[safe_text(f"{y:g}", size=14, color=BLUE).next_to(d, LEFT, buff=0.06)
                        for d, y in zip(dots, GB_Y)])
        self.play(Create(top), FadeIn(tl), FadeIn(xt), FadeIn(dots), FadeIn(ylab))
        r0 = GB[0]
        F = step_line(top, steps_from(r0["F"]))
        res = VGroup(*[Line(top.c2p(x, y), top.c2p(x, f), color=RED, stroke_width=4)
                       for x, y, f in zip(GB_X, GB_Y, r0["F"])])
        flab = safe_text("F₀ = 8", size=19, color=YELLOW, weight=BOLD).next_to(tl, RIGHT, buff=0.5)
        self.play(Create(F), FadeIn(flab))
        self.play(LaggedStart(*[Create(r) for r in res], lag_ratio=0.1))
        self.bring_to_front(dots)
        rows = [["ronda", "predicciones F", "corte", "SSE"]]
        tb = table(rows + [["0", "8, 8, 8, 8, 8", "—", f"{r0['sse']:.2f}"]],
                   col_widths=[0.9, 3.0, 1.0, 1.0], size=18, row_h=0.44,
                   colors=[[MUTED, YELLOW, MUTED, RED]]).move_to([3.55, 1.45, 0])
        tb.align_to([0.75, 0, 0], LEFT).align_to([0, 2.15, 0], UP)
        self.play(FadeIn(tb))
        self._gb = dict(top=top, bot=bot, bl=bl, dots=dots, F=F, res=res, flab=flab, tb=tb,
                        rows=VGroup(), resbars=VGroup(), stump=VGroup())

    # ------------------------------------------------------------------
    def gb_round(self, m: int) -> None:
        g = self._gb
        top, bot = g["top"], g["bot"]
        r_prev, r = GB[m - 1], GB[m]
        t, a, b = r["stump"]
        if m == 1:
            self.beat("39–41 min. Ronda 1. Los residuos bajan al panel inferior: (−6, −4, 1, 2, 7). "
                      "Un tocón se ajusta a los RESIDUOS, no a los precios: corta entre 2 y 3, "
                      "promedios −5 y 3.33. Pregunte: ¿sumamos la corrección completa? No: con "
                      "η = 0.5 sumamos la mitad. F pasa a 5.5 | 9.67 y la SSE baja a 43.5.")
        elif m == 2:
            self.beat("41–42 min. Ronda 2. Nuevos residuos (−3.5, −1.5, −0.67, 0.33, 5.33): el "
                      "gran error que queda es la última casa. El tocón corta entre 4 y 5 "
                      "(−1.33 | 5.33). SSE = 16.83.")
        else:
            self.beat("42–43 min. Ronda 3. El corte 2.5 REAPARECE: cada ronda sólo ve los "
                      "residuos actuales. SSE = 8.43. La tabla resume: cada ronda corrige un poco.")
        bars = VGroup(*[Line(bot.c2p(x, 0), bot.c2p(x, rv), color=RED, stroke_width=10)
                        for x, rv in zip(GB_X, r["fitted_r"])])
        vals = VGroup(*[safe_text(num(rv), size=14, color=RED)
                        .next_to(bot.c2p(x, rv), UP if rv > 0 else DOWN, buff=0.05)
                        for x, rv in zip(GB_X, r["fitted_r"])])
        anims = [FadeOut(g["resbars"]), FadeOut(g["stump"])]
        if m == 1:
            anims += [Create(bot), FadeIn(g["bl"])]
        self.play(*anims, run_time=0.4)
        self.play(*[ReplacementTransform(src.copy(), dst) for src, dst in zip(g["res"], bars)],
                  FadeIn(vals), run_time=1.0)
        stump = step_line(bot, [(0.5, t, a), (t, 5.5, b)], color=TEAL, width=5)
        slab = safe_text(f"tocón h{m}: {a:.2f} | {b:.2f}", size=17, color=TEAL, weight=BOLD
                         ).next_to(g["bl"], RIGHT, buff=0.5)
        self.play(Create(stump), FadeIn(slab))
        newF = step_line(top, steps_from(r["F"]))
        new_res = VGroup(*[Line(top.c2p(x, y), top.c2p(x, f), color=RED, stroke_width=4)
                           for x, y, f in zip(GB_X, GB_Y, r["F"])])
        eta = safe_text(f"F{m} = F{m - 1} + {ETA:g}·h{m}", size=19, color=YELLOW, weight=BOLD
                        ).next_to(g["top"], UP, buff=0.06).align_to(g["top"], RIGHT)
        self.play(FadeOut(g["flab"]), FadeIn(eta))
        self.play(Transform(g["F"], newF), Transform(g["res"], new_res), run_time=1.3)
        self.bring_to_front(g["dots"])
        F_txt = ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") for v in r["F"])
        row = table([["ronda", "predicciones F", "corte", "SSE"],
                     [str(m), F_txt, f"x ≤ {t:g}", f"{r['sse']:.2f}"]],
                    col_widths=[0.9, 3.0, 1.0, 1.0], size=18, row_h=0.44,
                    colors=[[MUTED, YELLOW, TEAL, RED]])[1]
        row.move_to(g["tb"][1]).shift(DOWN * 0.44 * (len(g["rows"]) + 1))
        self.play(FadeIn(row, shift=DOWN * 0.1))
        g["rows"].add(row)
        g.update(resbars=VGroup(bars, vals), stump=VGroup(stump, slab), flab=eta)

    # ------------------------------------------------------------------
    def gb_formula(self) -> None:
        self.beat("43–44 min. La receta en una línea. η es la tasa de aprendizaje (contracción): "
                  "con η = 1 el tocón corrige toda su parte del error… y también aprende todo su "
                  "ruido. Con η pequeño hacen falta más rondas, pero el modelo es más estable. "
                  "En la práctica: η entre 0.01 y 0.3 y el número de rondas por validación.")
        g = self._gb
        f = tex(r"F_m(x) = F_{m-1}(x) + \eta\, h_m(x)", size=34, color=YELLOW)
        f2 = tex(r"h_m \text{ se ajusta a los residuos de } F_{m-1}", size=26, color=MUTED)
        box = VGroup(f, f2).arrange(DOWN, buff=0.15)
        bg = card(box.width + 0.5, box.height + 0.4, YELLOW)
        grp = VGroup(bg, box.move_to(bg)).move_to([3.75, -1.55, 0])
        self.play(FadeIn(grp))

    # ------------------------------------------------------------------
    def gb_gradient(self) -> None:
        self.beat("44–46 min. ¿Por qué «gradient»? Con la pérdida L = ½(y − F)², la derivada "
                  "respecto a la predicción es g = F − y. El residuo es EXACTAMENTE −g: ajustar "
                  "residuos es dar un paso de descenso de gradiente en el espacio de funciones. "
                  "Para otras pérdidas (log-loss, absoluta) se ajustan los «pseudo-residuos» −g. "
                  "Señale la casa 5: en F = 8 la pendiente es −7; bajar la pérdida es subir F.")
        self.start("¿Por qué «gradient»?", "El residuo es el gradiente negativo", BLUE)
        ax = Axes(x_range=[4, 20, 2], y_range=[0, 40, 10], x_length=6.4, y_length=4.0, tips=False,
                  axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        ax.move_to([-3.1, -0.45, 0])
        y5 = 15.0
        curve = ax.plot(lambda F: 0.5 * (y5 - F) ** 2, x_range=[6.1, 20, 0.05], color=PURPLE,
                        stroke_width=4)
        xl = safe_text("predicción F de la casa 5 (y = 15)", size=18, color=MUTED
                       ).next_to(ax, DOWN, buff=0.95)
        yl = tex(r"L = \tfrac12 (y - F)^2", size=28, color=PURPLE).next_to(ax, UP, buff=0.05
                                                                         ).align_to(ax, LEFT)
        self.play(Create(ax), Create(curve), FadeIn(xl), FadeIn(yl))
        F0 = 8.0
        p = Dot(ax.c2p(F0, 0.5 * (y5 - F0) ** 2), radius=0.1, color=YELLOW)
        slope = F0 - y5
        tan = Line(ax.c2p(F0 - 2, 0.5 * (y5 - F0) ** 2 - 2 * slope),
                   ax.c2p(F0 + 2, 0.5 * (y5 - F0) ** 2 + 2 * slope), color=YELLOW, stroke_width=3)
        lab = safe_text("pendiente g = F − y = −7", size=19, color=YELLOW
                        ).next_to(p, UR, buff=0.1)
        arr = Arrow(ax.c2p(F0, 0) + DOWN * 0.3, ax.c2p(F0 + 7, 0) + DOWN * 0.3, buff=0,
                    color=GREEN, stroke_width=6)
        alab = safe_text("mover F en −g = +7 = residuo", size=19, color=GREEN, weight=BOLD
                         ).next_to(arr, DOWN, buff=0.08)
        guide = DashedLine(p.get_center(), ax.c2p(F0, 0), color=YELLOW, stroke_width=2)
        self.play(FadeIn(p), Create(tan), FadeIn(lab))
        self.play(Create(guide), GrowArrow(arr), FadeIn(alab))
        eqs = VGroup(
            tex(r"g_i = \frac{\partial L(y_i, F)}{\partial F}\Big|_{F_{m-1}(x_i)}", size=32),
            tex(r"\text{cuadr\'atica: } -g_i = y_i - F = r_i", size=30, color=GREEN),
            tex(r"\text{log-loss: } -g_i = y_i - p_i", size=30, color=ORANGE),
            tex(r"h_m \approx -g: \text{ descenso de gradiente}", size=28),
            safe_text("en el espacio de funciones", size=21, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.6, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.1) for e in eqs], lag_ratio=0.25))
