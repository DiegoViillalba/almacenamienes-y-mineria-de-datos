"""Acto 4 · AdaBoost: repesar los errores (27–37 min).

Diez puntos en una recta que ningún corte separa. Los pesos se ven como barras
debajo de cada punto. Cada ronda: el mejor tocón ponderado (una banda de
color arriba), sus errores en rojo, ε y α, y el repesado (las barras de los
errores crecen hasta sumar exactamente 0.5). Al final, la suma ponderada de
los tres tocones se apila punto por punto y su signo clasifica bien los diez.
"""
from __future__ import annotations

from manim import *

from sections.data import ADA, ADA_X, ADA_Y, ada_score
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, marker, safe_text, tex)

SIGN_CLS = {1: 0, -1: 1}               # +1 → círculo azul, −1 → cuadrado naranja
ROUND_COLORS = [TEAL, PURPLE, YELLOW]
X0, DX = -5.45, 1.1                     # posición en pantalla del punto x = 1 y separación
LINE_Y, BAR_Y0, BAR_SCALE = 0.55, -2.35, 9.0


def sx(x: float) -> float:
    return X0 + (x - 1) * DX


def weight_bars(w) -> VGroup:
    out = VGroup()
    for x, wi in zip(ADA_X, w):
        h = wi * BAR_SCALE
        r = Rectangle(width=0.62, height=h, stroke_width=0, fill_color=MUTED, fill_opacity=0.7)
        r.move_to([sx(x), BAR_Y0 + h / 2, 0])
        t = safe_text(f"{wi:.3f}", size=14, color=MUTED).next_to(r, UP, buff=0.05)
        out.add(VGroup(r, t))
    return out


def stump_band(t: float, s: int, y: float = 1.55) -> VGroup:
    left = Line([sx(0.5), y, 0], [sx(t), y, 0], stroke_width=14,
                color=BLUE if s == 1 else ORANGE, stroke_opacity=0.7)
    right = Line([sx(t), y, 0], [sx(10.5), y, 0], stroke_width=14,
                 color=ORANGE if s == 1 else BLUE, stroke_opacity=0.7)
    cut = DashedLine([sx(t), y + 0.3, 0], [sx(t), BAR_Y0, 0], color=INK, stroke_width=2)
    sl = safe_text("+" if s == 1 else "−", size=26, color=INK, weight=BOLD).move_to(left)
    sr = safe_text("−" if s == 1 else "+", size=26, color=INK, weight=BOLD).move_to(right)
    return VGroup(left, right, cut, sl, sr)


class AdaBoostSection:
    def adaboost(self) -> None:
        self.ada_setup()
        for i in range(3):
            self.ada_round(i)
        self.ada_sum()
        self.ada_formal()

    # ------------------------------------------------------------------
    def ada_setup(self) -> None:
        self.beat("27–28 min. Otra familia: boosting. En vez de árboles independientes, una "
                  "SECUENCIA donde cada modelo se concentra en lo que los anteriores fallaron. "
                  "Diez puntos: + + + − − − + + + −. Pregunte: ¿cuántos errores comete el mejor "
                  "corte único? (3). El modelo débil será un tocón: un solo corte. Las barras "
                  "grises son los PESOS de cada punto: al inicio, todos 0.1.")
        self.start("AdaBoost", "Cada ronda se concentra en los errores", ORANGE)
        nl = Line([sx(0.5), LINE_Y, 0], [sx(10.5), LINE_Y, 0], color=GRID, stroke_width=2)
        ticks = VGroup(*[safe_text(str(x), size=17, color=MUTED).move_to([sx(x), LINE_Y - 0.28, 0])
                         for x in ADA_X])
        pts = VGroup(*[marker(SIGN_CLS[int(y)], r=0.17).move_to([sx(x), LINE_Y + 0.4, 0])
                       for x, y in zip(ADA_X, ADA_Y)])
        base = Line([sx(0.5), BAR_Y0, 0], [sx(10.5), BAR_Y0, 0], color=GRID, stroke_width=1.5)
        wl = safe_text("pesos", size=17, color=MUTED).next_to(base, LEFT, buff=0.1).shift(UP * 0.35)
        bars = weight_bars(np.full(10, 0.1))
        self.play(Create(nl), FadeIn(ticks), LaggedStart(*[GrowFromCenter(p) for p in pts],
                                                         lag_ratio=0.06))
        self.play(Create(base), FadeIn(wl), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars],
                                                        lag_ratio=0.05),
                  FadeIn(VGroup(*[b[1] for b in bars])))
        self._ada = dict(pts=pts, bars=bars, info=VGroup(), band=VGroup(), marks=VGroup())

    # ------------------------------------------------------------------
    def ada_round(self, i: int) -> None:
        r = ADA[i]
        a = self._ada
        notes = [
            ("28–30 min. Ronda 1, pesos iguales. Mejor tocón: x ≤ 3.5 ⇒ +, si no −. Falla en 7, "
             "8 y 9: ε = 0.3. Su voto pesa α = ½ ln(0.7/0.3) = 0.424: un modelo con poco error "
             "vota fuerte; con ε = 0.5 (moneda) votaría 0.",
             "30–31 min. Repesar: los errores se multiplican por e^α = 1.528 y los aciertos por "
             "e^−α = 0.655; luego se normaliza. Los tres errores pasan a pesar 0.167 cada uno y "
             "suman EXACTAMENTE 0.5: para la ronda siguiente, el tocón anterior es una moneda."),
            ("31–32 min. Ronda 2. Pregunte qué tocón elegirían ahora que 7, 8 y 9 pesan mucho. "
             "Gana x ≤ 9.5 ⇒ +: falla 4, 5 y 6, que pesan poco (ε = 0.214, α = 0.650). Solo "
             "tendría sentido POR los pesos.",
             "32–33 min. Repesar otra vez: ahora 4, 5 y 6 son los pesados."),
            ("33–34 min. Ronda 3. Gana x ≤ 6.5 ⇒ −, si no +: falla 1, 2, 3 y 10, ahora ligeros "
             "(ε = 0.182, α = 0.752).",
             "34 min. Últimos pesos. No los necesitamos: tres rondas bastan aquí."),
        ][i]
        self.beat(notes[0])
        band = stump_band(r["t"], r["s"])
        info = VGroup(
            safe_text(f"Ronda {i + 1}", size=24, color=ROUND_COLORS[i], weight=BOLD),
            safe_text(f"x ≤ {r['t']:g} ⇒ {'+' if r['s'] == 1 else '−'}", size=22, color=INK),
            safe_text(f"ε = {r['err']:.3f}", size=22, color=RED),
            tex(rf"\alpha = \tfrac12\ln\tfrac{{1-\varepsilon}}{{\varepsilon}} = {r['alpha']:.3f}",
                size=28, color=ROUND_COLORS[i]),
        ).arrange(RIGHT, buff=0.45).move_to([1.4, 2.2, 0])
        marks = VGroup(*[Circle(radius=0.27, color=RED, stroke_width=4).move_to(a["pts"][x - 1])
                         for x in r["wrong"]])
        self.play(FadeOut(a["band"]), FadeOut(a["marks"]), FadeOut(a["info"]))
        self.play(FadeIn(band), FadeIn(info[:2]))
        self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.15),
                  *[a["bars"][x - 1][0].animate.set_fill(RED, 0.8) for x in r["wrong"]])
        self.play(FadeIn(info[2]))
        self.play(Write(info[3]))
        a.update(band=band, info=info, marks=marks)

        self.beat(notes[1])
        new = weight_bars(r["w_after"])
        for x in r["wrong"]:
            new[x - 1][0].set_fill(RED, 0.8)
        rule = tex(r"w_i \leftarrow w_i\, e^{-\alpha\, y_i h(x_i)} \;/\; Z", size=28, color=INK
                   ).move_to([0, -3.0, 0])
        self.play(Transform(a["bars"], new), FadeIn(rule), run_time=1.3)
        tot = sum(r["w_after"][x - 1] for x in r["wrong"])
        s = safe_text(f"peso de los errores = {tot:.2f}", size=20, color=RED
                      ).next_to(rule, RIGHT, buff=0.6)
        self.play(FadeIn(s))
        self.play(FadeOut(rule), FadeOut(s))
        for b in a["bars"]:
            b[0].set_fill(MUTED, 0.7)

    # ------------------------------------------------------------------
    def ada_sum(self) -> None:
        self.beat("34–36 min. El ensamble: F(x) = 0.424 h₁ + 0.650 h₂ + 0.752 h₃ y se decide por "
                  "el signo. Hay sólo cuatro grupos de puntos (1–3, 4–6, 7–9 y 10). En cada uno "
                  "apilamos los votos de las tres rondas (arriba +, abajo −). Pregunte el signo de "
                  "F en x = 10 antes de mostrarlo. Tres modelos que fallan 3 o 4 de 10 producen "
                  "uno perfecto en entrenamiento: ninguno podía dibujar tres cambios de clase; "
                  "su suma sí.")
        self.start("La suma ponderada", "Sumar los votos y decidir por el signo", ORANGE)
        groups = [("x = 1–3", 0), ("x = 4–6", 3), ("x = 7–9", 6), ("x = 10", 9)]
        cols = [-4.6, -1.55, 1.5, 4.55]
        y0, k = -0.75, 1.3
        axis = Line([-6.4, y0, 0], [6.4, y0, 0], color=INK, stroke_width=2)
        lab0 = safe_text("F = 0", size=17, color=MUTED).next_to(axis, UP, buff=0.06
                                                                ).align_to(axis, LEFT)
        heads = VGroup()
        for (name, j), cx in zip(groups, cols):
            m = marker(SIGN_CLS[int(ADA_Y[j])], r=0.16)
            heads.add(VGroup(safe_text(name, size=20, color=INK, weight=BOLD), m
                             ).arrange(RIGHT, buff=0.2).move_to([cx, 1.75, 0]))
        rl = safe_text("clase real:", size=17, color=MUTED).next_to(heads, UP, buff=0.12
                                                                    ).align_to(heads, LEFT)
        self.play(Create(axis), FadeIn(lab0), FadeIn(heads), FadeIn(rl))
        level = np.zeros(4)
        legend = VGroup()
        for i, r in enumerate(ADA):
            arrows = VGroup()
            for g, ((_, j), cx) in enumerate(zip(groups, cols)):
                d = r["alpha"] * r["pred"][j]
                x = cx - 0.45 + 0.42 * i
                arrows.add(Arrow([x, y0 + level[g] * k, 0], [x, y0 + (level[g] + d) * k, 0],
                                 buff=0, color=ROUND_COLORS[i], stroke_width=7,
                                 max_tip_length_to_length_ratio=0.3, tip_length=0.18))
                if i < 2:
                    nxt = cx - 0.45 + 0.42 * (i + 1)
                    arrows.add(DashedLine([x, y0 + (level[g] + d) * k, 0],
                                          [nxt, y0 + (level[g] + d) * k, 0],
                                          color=MUTED, stroke_width=1.5, dash_length=0.05))
                level[g] += d
            sign = "+" if r["s"] == 1 else "−"
            legend.add(safe_text(f"h{i + 1}: x ≤ {r['t']:g} ⇒ {sign}   α = {r['alpha']:.3f}",
                                 size=19, color=ROUND_COLORS[i], weight=BOLD))
            self.play(LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else Create(a)
                                    for a in arrows], lag_ratio=0.05), run_time=1.0)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UR, buff=0.45)
        self.play(FadeIn(legend))
        F = ada_score()
        finals = VGroup()
        for g, ((_, j), cx) in enumerate(zip(groups, cols)):
            f = F[j]
            m = marker(SIGN_CLS[int(np.sign(f))], r=0.17).move_to([cx + 0.6, y0 + f * k, 0])
            t = safe_text(f"F = {f:+.3f}", size=19, color=INK, weight=BOLD
                          ).next_to(m, RIGHT, buff=0.12)
            finals.add(VGroup(m, t))
        self.play(LaggedStart(*[GrowFromCenter(fm) for fm in finals], lag_ratio=0.15))
        ok = safe_text("signo(F) = clase real en los 10 puntos", size=22, color=GREEN,
                       weight=BOLD).move_to([0, -3.05, 0])
        self.play(FadeIn(ok))

    # ------------------------------------------------------------------
    def ada_formal(self) -> None:
        self.beat("36–37 min. Formalización. AdaBoost (Freund y Schapire, 1997) en tres pasos. "
                  "Friedman, Hastie y Tibshirani mostraron que minimiza, paso a paso, la pérdida "
                  "exponencial: cada ronda agrega el término que más la reduce. Esa lectura como "
                  "optimización por etapas es la puerta al gradient boosting. Advertencia: la "
                  "pérdida exponencial castiga sin límite a los mal clasificados, así que es "
                  "sensible a etiquetas erróneas.")
        self.start("Formalización", "AdaBoost minimiza la pérdida exponencial", ORANGE)
        steps = VGroup(
            safe_text("Con wᵢ = 1/n, para m = 1, …, M:", size=23, color=MUTED),
            tex(r"1.\;\; h_m = \arg\min_h \sum_i w_i\,\mathbf{1}[h(x_i) \ne y_i]"
                r"\quad (\varepsilon_m)", size=32),
            tex(r"2.\;\; \alpha_m = \tfrac12 \ln\frac{1-\varepsilon_m}{\varepsilon_m}", size=32),
            tex(r"3.\;\; w_i \leftarrow w_i\, e^{-\alpha_m y_i h_m(x_i)}, \text{ normalizar}",
                size=32),
            tex(r"\hat y = \operatorname{signo}\Big(\sum_m \alpha_m h_m(x)\Big)", size=34,
                color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([-2.2, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in steps], lag_ratio=0.3))
        side = VGroup(
            tex(r"L = \sum_i e^{-y_i F(x_i)}", size=36, color=ORANGE),
            safe_text("pérdida exponencial", size=20, color=MUTED),
            safe_text("· reduce sobre todo el SESGO", size=21),
            safe_text("· modelo base débil (tocón)", size=21),
            safe_text("· sensible a etiquetas erróneas", size=21, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.4, -0.3, 0])
        self.play(FadeIn(side, shift=LEFT * 0.1))
