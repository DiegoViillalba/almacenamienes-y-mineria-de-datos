"""Acto 6 · XGBoost: segundo orden y regularización (46–55 min).

La función objetivo regularizada; la aproximación de Taylor de segundo orden
dibujada sobre la log-loss (recta tangente contra parábola); el peso óptimo de
una hoja como el vértice de una parábola que λ empuja hacia 0 (−5 → −3.33); la
ganancia de cada corte como barras que γ «poda» al subir; y el mismo
mecanismo en clasificación, donde el hessiano p(1 − p) frena las hojas seguras.
"""
from __future__ import annotations

from manim import *

from sections.data import (G_I, H_I, LAMBDA, XGB_GAINS, clf_leaf, leaf_weight, logloss,
                           logloss_taylor)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE, RED,
                            TEAL, YELLOW, card, safe_text, table, tex)


def f2(v: float) -> str:
    """Dos decimales redondeando la mitad hacia arriba (23.625 → 23.63, como el libro)."""
    return f"{v + 1e-9:.2f}"


class XGBoostSection:
    def xgboost(self) -> None:
        self.objective()
        self.leaf_weight()
        self.gain_gamma()
        self.classification()
        self.extras()

    # ------------------------------------------------------------------
    def objective(self) -> None:
        self.beat("46–48 min. XGBoost (Chen y Guestrin, 2016) es gradient boosting con dos "
                  "ideas. (1) La función objetivo incluye la complejidad del árbol: γ por hoja y "
                  "λ sobre los pesos. (2) Aproxima la pérdida con Taylor de SEGUNDO orden: no "
                  "sólo la pendiente (gradiente g) sino la curvatura (hessiano h). En la "
                  "gráfica: la recta tangente se aleja pronto; la parábola sigue a la curva "
                  "mucho mejor cerca de F₀.")
        self.start("XGBoost", "Una función objetivo con regularización", PINK)
        obj = tex(r"\text{Obj}^{(t)} = \sum_i l\big(y_i,\, F_{t-1}(x_i) + f_t(x_i)\big)"
                  r" + \gamma\, T + \tfrac12 \lambda \sum_j w_j^2", size=32).move_to([0, 2.0, 0])
        brace_t = safe_text("T = hojas · w = valores de las hojas", size=18, color=MUTED
                            ).next_to(obj, DOWN, buff=0.12).align_to(obj, RIGHT)
        self.play(Write(obj), FadeIn(brace_t))
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 3.2, 1], x_length=6.4, y_length=3.5, tips=False,
                  axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        ax.move_to([-3.1, -1.05, 0])
        F0 = -0.3
        real = ax.plot(lambda F: logloss(F), x_range=[-3, 3, 0.02], color=INK, stroke_width=4)
        lin = ax.plot(lambda F: logloss_taylor(F, F0)[0], x_range=[-2.75, 1.1, 0.02], color=ORANGE,
                      stroke_width=3)
        quad = ax.plot(lambda F: logloss_taylor(F, F0)[1], x_range=[-3, 2.4, 0.02], color=TEAL,
                       stroke_width=4)
        p0 = Dot(ax.c2p(F0, logloss(F0)), radius=0.08, color=YELLOW)
        xl = safe_text("F (log-odds) · ejemplo con y = 1", size=17, color=MUTED
                       ).next_to(ax, DOWN, buff=0.1)
        legend = VGroup(safe_text("log-loss real", size=19, color=INK),
                        safe_text("1er orden: l + g·f", size=19, color=ORANGE),
                        safe_text("2º orden: l + g·f + ½h·f²", size=19, color=TEAL, weight=BOLD)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.next_to(ax, UP, buff=0.0).align_to(ax, RIGHT).shift(DOWN * 0.7 + RIGHT * 0.2)
        self.play(Create(ax), Create(real), FadeIn(xl), FadeIn(legend[0]))
        self.play(FadeIn(p0), Create(lin), FadeIn(legend[1]))
        self.play(Create(quad), FadeIn(legend[2]))
        eqs = VGroup(
            tex(r"l(y, F_{t-1} + f) \approx l + g\,f + \tfrac12 h\,f^2", size=30),
            tex(r"g_i = \partial_F\, l, \qquad h_i = \partial^2_F\, l", size=30, color=TEAL),
            safe_text("Agrupando por hoja j  (G = Σg, H = Σh):", size=20, color=MUTED),
            tex(r"\text{Obj} \approx \sum_j \Big[G_j w_j + \tfrac12 (H_j + \lambda) w_j^2\Big]"
                r" + \gamma T", size=30, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.55, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.1) for e in eqs], lag_ratio=0.3))

    # ------------------------------------------------------------------
    def leaf_weight(self) -> None:
        self.beat("48–50 min. Lo que queda por hoja es una PARÁBOLA en w: G·w + ½(H + λ)·w². Su "
                  "vértice es w* = −G/(H + λ). Con las cinco casas (ronda 1): g = F − y = (6, 4, "
                  "−1, −2, −7), h = 1. Hoja izquierda (casas 1–2): G = 10, H = 2. Sin "
                  "regularización (λ = 0) el vértice es −5, el promedio de residuos del tocón "
                  "clásico. Mueva λ: el vértice se acerca a 0. λ actúa como una muestra ficticia "
                  "que tira la hoja hacia la predicción actual.")
        self.start("El peso óptimo de una hoja", "El vértice de una parábola", PINK)
        G, H = float(G_I[:2].sum()), float(H_I[:2].sum())
        lam = ValueTracker(0.0)
        ax = Axes(x_range=[-7, 1, 1], y_range=[-27, 5, 10], x_length=7.0, y_length=4.4, tips=False,
                  axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        ax.move_to([-2.6, -0.5, 0])
        xt = VGroup(*[safe_text(str(v), size=15, color=MUTED).next_to(ax.c2p(v, -27), DOWN,
                                                                     buff=0.08)
                      for v in (-6, -5, -4, -3, -2, -1, 0)])
        xl = safe_text("w (valor de la hoja izquierda)", size=17, color=MUTED
                       ).next_to(xt, DOWN, buff=0.12)
        zero = DashedLine(ax.c2p(0, -27), ax.c2p(0, 5), color=GRID, stroke_width=2)
        para = always_redraw(lambda: ax.plot(
            lambda w: G * w + 0.5 * (H + lam.get_value()) * w * w, x_range=[-7, 1, 0.02],
            color=PINK, stroke_width=4))
        vtx = always_redraw(lambda: Dot(ax.c2p(-G / (H + lam.get_value()),
                                               -0.5 * G * G / (H + lam.get_value())),
                                        radius=0.1, color=YELLOW))
        read = always_redraw(lambda: VGroup(
            safe_text(f"λ = {lam.get_value():.1f}", size=24, color=INK, weight=BOLD),
            safe_text(f"w* = {-G / (H + lam.get_value()):.2f}", size=24, color=YELLOW,
                      weight=BOLD),
        ).arrange(RIGHT, buff=0.5).move_to([4.2, 0.85, 0]))
        f = tex(r"w_j^{*} = -\frac{G_j}{H_j + \lambda}", size=40, color=YELLOW
                ).move_to([4.2, 1.95, 0])
        self.play(Create(ax), FadeIn(xt), FadeIn(xl), Create(zero))
        self.add(para, vtx, read)
        self.play(Write(f))
        mean_lab = safe_text("λ = 0: −5 = promedio de residuos", size=18, color=MUTED
                             ).next_to(ax.c2p(-5, -25), RIGHT, buff=0.1)
        self.play(FadeIn(mean_lab))
        self.play(lam.animate.set_value(LAMBDA), run_time=1.5)
        self.play(lam.animate.set_value(6.0), run_time=2.0)
        self.play(lam.animate.set_value(LAMBDA), run_time=1.2)
        wl, wr = leaf_weight(G, H), leaf_weight(float(G_I[2:].sum()), float(H_I[2:].sum()))
        rows = [["hoja", "G", "H", "λ = 0", "λ = 1"],
                ["casas 1–2", f"{G:g}", f"{H:g}", "−5.00", f"{wl:.2f}"],
                ["casas 3–5", f"{G_I[2:].sum():g}", f"{H_I[2:].sum():g}", "3.33", f"{wr:.2f}"]]
        tb = table(rows, col_widths=[1.5, 0.7, 0.7, 1.0, 1.0], size=20, row_h=0.48,
                   colors=[[INK, MUTED, MUTED, MUTED, YELLOW]] * 2).move_to([4.2, -0.35, 0])
        shrink = safe_text("λ encoge más a la hoja pequeña", size=19, color=PINK
                           ).next_to(tb, DOWN, buff=0.2)
        objv = tex(r"\text{Obj}^{*} = -\tfrac12 \sum_j \frac{G_j^2}{H_j + \lambda} + \gamma T",
                   size=28, color=INK).next_to(shrink, DOWN, buff=0.25)
        self.play(FadeIn(tb))
        self.play(FadeIn(shrink), Write(objv))

    # ------------------------------------------------------------------
    def gain_gamma(self) -> None:
        self.beat("50–52 min. ¿Qué corte elegir? El que más baja Obj*: la GANANCIA. Pida que "
                  "calculen el corte x ≤ 2.5: G_L = 10, G_R = −10, ½(100/3 + 100/4 − 0) = 29.17. "
                  "Gana ese corte. γ es el precio de cada hoja nueva: si sube por encima de "
                  "29.17 ningún corte compensa y el árbol se queda en una hoja. Así poda XGBoost.")
        self.start("La ganancia de un corte", "γ: el precio de cada hoja nueva", PINK)
        f = tex(r"\text{Ganancia} = \tfrac12\Big[\frac{G_L^2}{H_L+\lambda} + "
                r"\frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma",
                size=32).move_to([0, 1.95, 0])
        self.play(Write(f))
        ax = Axes(x_range=[0.5, 4.5, 1], y_range=[0, 32, 10], x_length=6.2, y_length=3.6,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False})
        ax.move_to([-2.9, -1.0, 0])
        bars = VGroup()
        for i, (t, GL, GR, HL, HR, g) in enumerate(XGB_GAINS):
            col = YELLOW if g == max(z[-1] for z in XGB_GAINS) else PINK
            b = Rectangle(width=0.95, height=g / 32 * 3.6, stroke_width=0, fill_color=col,
                          fill_opacity=0.8)
            b.move_to(ax.c2p(i + 1, 0), aligned_edge=DOWN)
            v = safe_text(f2(g), size=19, color=INK).next_to(b, UP, buff=0.06)
            n = safe_text(f"x ≤ {t:g}", size=18, color=MUTED).next_to(ax.c2p(i + 1, 0), DOWN,
                                                                     buff=0.1)
            gl = safe_text(f"G: {GL:g} | {GR:g}", size=15, color=MUTED).next_to(n, DOWN, buff=0.05)
            bars.add(VGroup(b, v, n, gl))
        self.play(Create(ax), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars],
                                          lag_ratio=0.15),
                  LaggedStart(*[FadeIn(VGroup(*b[1:])) for b in bars], lag_ratio=0.15))
        calc = tex(r"x \le 2.5:\;\tfrac12\Big[\tfrac{10^2}{2+1} + \tfrac{(-10)^2}{3+1} - "
                   r"\tfrac{0^2}{5+1}\Big] = 29.17", size=28, color=YELLOW)
        calc.move_to([0, 0.75, 0]).align_to([1.95, 0, 0], LEFT)
        self.play(Write(calc))
        gamma = ValueTracker(0.0)
        gline = always_redraw(lambda: DashedLine(ax.c2p(0.5, gamma.get_value()),
                                                 ax.c2p(4.5, gamma.get_value()), color=RED,
                                                 stroke_width=3))
        glab = always_redraw(lambda: safe_text(f"γ = {gamma.get_value():.1f}", size=22,
                                               color=RED, weight=BOLD).next_to(
            ax.c2p(4.5, gamma.get_value()), RIGHT, buff=0.12))
        status = always_redraw(lambda: safe_text(
            ("se corta en x ≤ 2.5" if gamma.get_value() < XGB_GAINS[1][-1]
             else "ningún corte compensa: 1 hoja (poda)"),
            size=22, color=GREEN if gamma.get_value() < XGB_GAINS[1][-1] else RED,
            weight=BOLD).move_to([4.2, -0.5, 0]))
        self.add(gline, glab, status)
        self.play(gamma.animate.set_value(20.0), run_time=1.6)
        self.play(gamma.animate.set_value(31.0), run_time=1.4)
        self.wait(0.3)   # ≥ 3 fotogramas: una pausa de 1 fotograma rompe la concatenación
        note = safe_text("xgboost reporta 2 × ganancia (58.33)", size=17, color=MUTED
                         ).move_to([4.2, -1.3, 0])
        self.play(FadeIn(note))

    # ------------------------------------------------------------------
    def classification(self) -> None:
        self.beat("52–54 min. En clasificación se suma en la escala de log-odds y p = σ(F). Con "
                  "la log-loss: g = p − y y h = p(1 − p). Una hoja con y = (1, 1, 0) y p = 0.5: "
                  "G = −0.5, H = 0.75, w = 0.5/1.75 = 0.286. El hessiano es grande cuando el "
                  "modelo duda (p ≈ 0.5) y pequeño cuando ya está seguro: la curvatura decide "
                  "cuánto corregir.")
        self.start("XGBoost para clasificar", "El mismo mecanismo con la log-loss", PINK)
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 0.3, 0.1], x_length=5.6, y_length=3.2,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False})
        ax.move_to([-3.3, -0.55, 0])
        hcurve = ax.plot(lambda p: p * (1 - p), x_range=[0, 1, 0.01], color=TEAL, stroke_width=4)
        xt = VGroup(*[safe_text(f"{v:g}", size=15, color=MUTED).next_to(ax.c2p(v, 0), DOWN,
                                                                        buff=0.08)
                      for v in (0, 0.5, 1)])
        xl = safe_text("p = probabilidad predicha", size=17, color=MUTED
                       ).next_to(xt, DOWN, buff=0.1)
        hl = tex(r"h = p(1-p)", size=30, color=TEAL).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
        doubt = safe_text("duda: corrige más", size=17, color=TEAL
                          ).next_to(ax.c2p(0.5, 0.25), UP, buff=0.08)
        sure = safe_text("seguro: corrige poco", size=17, color=MUTED
                         ).next_to(ax.c2p(0.86, 0.06), LEFT, buff=0.05)
        self.play(Create(ax), Create(hcurve), FadeIn(xt), FadeIn(xl), FadeIn(hl))
        self.play(FadeIn(doubt), FadeIn(sure))
        g, h, G, H, w = clf_leaf()
        eqs = VGroup(
            tex(r"p = \sigma(F), \qquad g = p - y, \qquad h = p(1-p)", size=30),
            safe_text("Hoja con y = (1, 1, 0) y p = 0.5:", size=21, color=MUTED),
            tex(rf"G = {g[0]:g} {g[1]:+g} {g[2]:+g} = {G:g}, \quad H = 3(0.25) = {H:g}", size=28),
            tex(rf"w^{{*}} = -\frac{{G}}{{H + \lambda}} = \frac{{0.5}}{{1.75}} = {w:.3f}",
                size=30, color=YELLOW),
            safe_text("los log-odds de la hoja suben 0.286·η", size=20, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.3, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.1) for e in eqs], lag_ratio=0.3))

    # ------------------------------------------------------------------
    def extras(self) -> None:
        self.beat("54–55 min. Además de la matemática, XGBoost es ingeniería: submuestreo de "
                  "filas y columnas (como en los bosques), búsqueda aproximada de cortes con "
                  "histogramas, una dirección por defecto para los faltantes, early stopping con "
                  "un conjunto de validación y paralelismo. Mencione que SHAP explica cada "
                  "predicción como suma de contribuciones (capítulo del libro).")
        self.start("Lo que agrega XGBoost", "Regularización + ingeniería", PINK)
        items = [
            ("γ y λ en el objetivo", "poda y encoge hojas desde la función objetivo", PINK),
            ("Segundo orden", "g y h: pasos de Newton por hoja", TEAL),
            ("Submuestreo", "subsample y colsample: menos correlación", GREEN),
            ("Cortes aproximados", "histogramas por cuantiles: millones de filas", BLUE),
            ("Faltantes", "aprende hacia qué lado enviarlos", ORANGE),
            ("Early stopping", "detener cuando la validación deja de mejorar", YELLOW),
        ]
        cards = VGroup()
        for t, d, c in items:
            g = VGroup(safe_text(t, size=24, color=c, weight=BOLD),
                       safe_text(d, size=19, max_width=5.3)).arrange(DOWN, aligned_edge=LEFT,
                                                                    buff=0.12)
            bx = card(5.9, g.height + 0.42, c)
            cards.add(VGroup(bx, g.move_to(bx).align_to(bx, LEFT).shift(RIGHT * 0.28)))
        cards.arrange_in_grid(rows=3, cols=2, buff=(0.35, 0.28)).move_to([0, -0.35, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cards], lag_ratio=0.15))
