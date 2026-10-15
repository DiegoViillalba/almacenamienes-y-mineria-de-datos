"""Acto 4 · Más allá del lift: leverage, conviction y Zhang como regiones y alturas.

Todas las métricas se leen en el mismo cuadrado de café ⇒ pan dulce:
- leverage: franja entre el nivel P(B) y el agua dentro de A (área extra);
- conviction: errores esperados (sobre el nivel) / errores observados (sobre el agua);
- Zhang: diferencia de alturas dentro y fuera de A, normalizada por la mayor.
Después se comparan las cuatro reglas de la tienda.
"""
from __future__ import annotations

from manim import *

from sections.base import UnitSquare
from sections.data import CAFE, STORE_RULES, fmt
from sections.theme import (BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, pill, safe_text, table, tex)

SIDE = 4.6
CENTER = np.array([-3.1, -0.3, 0])
RX = 3.55


def cafe_square() -> tuple[UnitSquare, VGroup]:
    r = CAFE
    sq = UnitSquare(SIDE, r.pa, r.conf, r.h_not_a).move_to(CENTER)
    level = sq.level_line(r.pb)
    a_lab = safe_text("café", size=22, color=BLUE, weight=BOLD).next_to(sq.p(r.pa / 2, 1), UP, buff=0.1)
    b_lab = safe_text("pan dulce", size=20, color=BG, weight=BOLD)
    b_lab.move_to(sq.p((1 + r.pa) / 2, r.h_not_a / 2))
    lvl = sq.level_label(rf"P(B)={fmt(r.pb)}", r.pb)
    return sq, VGroup(level, a_lab, b_lab, lvl)


def mini_square(r, side: float = 1.75) -> VGroup:
    sq = UnitSquare(side, r.pa, r.conf, r.h_not_a)
    sq.frame.set_stroke(width=1.8)
    level = sq.level_line(r.pb)
    level.set_stroke(width=2.2)
    return VGroup(sq, level)


class MetricsSection:
    def metrics(self) -> None:
        r = CAFE
        self.beat("30–32 min. Leverage: la franja verde entre el nivel (0.30) y el agua (0.50) "
                  "dentro de café. Su área 0.40 × 0.20 = 0.08 son tickets que aparecen juntos "
                  "de más: 8 de cada 100. El lift es relativo; el leverage es volumen.")
        head = self.start("3 · Medir", "Leverage: tickets de más", PURPLE)
        sq, extras = cafe_square()
        self.play(FadeIn(sq), FadeIn(extras))
        lev_band = sq.region(0, r.pa, r.pb, r.conf, GREEN, opacity=0.75)
        self.play(FadeIn(lev_band, scale=1.05))
        lev = VGroup(
            tex(r"\mathrm{lev}(A\Rightarrow B)=P(A\cup B)-P(A)\,P(B)", size=32),
            tex(rf"={fmt(r.pab)}-{fmt(r.pa * r.pb)}={fmt(r.lev)}", size=36, color=GREEN),
            safe_text(f"× {r.n} tickets = {r.lev * r.n:.0f} tickets de más", size=26,
                      color=GREEN, weight=BOLD),
            safe_text("Mide impacto absoluto; cero ⇔ independencia.", size=22, color=MUTED),
        ).arrange(DOWN, buff=0.3).move_to([RX, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in lev], lag_ratio=0.3))

        self.beat("32–34 min. Conviction mira los ERRORES de la regla: tickets con café y SIN "
                  "pan dulce (zona roja sobre el agua, 0.20). Si fueran independientes, los "
                  "errores llegarían hasta el nivel: 0.40 × 0.70 = 0.28. Cociente 1.4: la regla "
                  "falla 1.4 veces menos que el azar.")
        head = self.swap_heading(head, "3 · Medir", "Conviction: ¿cuánto menos se equivoca?", PURPLE)
        self.play(FadeOut(lev), lev_band.animate.set_fill(opacity=0.0).set_stroke(opacity=0))
        err_obs = sq.region(0, r.pa, r.conf, 1, RED, opacity=0.55)
        err_exp = DashedVMobject(sq.region(0, r.pa, r.pb, 1, YELLOW, opacity=0).set_stroke(width=4),
                                 num_dashes=40)
        self.play(FadeIn(err_obs))
        self.play(Create(err_exp))
        conv = VGroup(
            tex(r"\mathrm{conv}(A\Rightarrow B)=\frac{1-P(B)}{1-\mathrm{conf}(A\Rightarrow B)}",
                size=32),
            tex(rf"=\frac{{P(A)\,(1-P(B))}}{{P(A)\,(1-\mathrm{{conf}})}}"
                rf"=\frac{{{fmt(r.pa * (1 - r.pb))}}}{{{fmt(r.pa * (1 - r.conf))}}}={fmt(r.conv)}",
                size=32, color=RED),
            safe_text("errores esperados (amarillo) / observados (rojo)", size=21, color=MUTED),
            safe_text(f"Dirigida: conv(pan dulce ⇒ café) = {fmt(r.reverse.conv)}", size=22),
            safe_text("Si conf = 1 no hay errores: conviction → ∞", size=21, color=MUTED),
        ).arrange(DOWN, buff=0.28).move_to([RX, 0.25, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in conv], lag_ratio=0.3))

        self.beat("34–36 min. Zhang compara dos alturas: agua dentro de café (0.50) y fuera "
                  "(0.17). (0.50 − 0.17) / 0.50 = 0.67. Vive entre −1 y 1: vale 1 si fuera de A "
                  "no hay nada de B y −1 si dentro de A no hay nada. Ojo: tampoco es simétrica.")
        head = self.swap_heading(head, "3 · Medir", "Zhang: dentro contra fuera, de −1 a 1", PURPLE)
        self.play(FadeOut(conv), FadeOut(err_obs), FadeOut(err_exp))
        top_in = Line(sq.p(0, r.conf), sq.p(r.pa, r.conf), color=ORANGE, stroke_width=7)
        top_out = Line(sq.p(r.pa, r.h_not_a), sq.p(1, r.h_not_a), color=ORANGE, stroke_width=7)
        gap = DoubleArrow(sq.p(r.pa + 0.06, r.h_not_a), sq.p(r.pa + 0.06, r.conf), buff=0,
                          color=PURPLE, stroke_width=5, tip_length=0.18)
        guide = DashedLine(sq.p(r.pa, r.conf), sq.p(r.pa + 0.12, r.conf), color=ORANGE)
        self.play(Create(top_in), Create(top_out))
        self.play(Create(guide), GrowFromCenter(gap))
        z = VGroup(
            tex(r"Z=\frac{P(B\mid A)-P(B\mid\neg A)}{\max\{P(B\mid A),\,P(B\mid\neg A)\}}", size=32),
            tex(rf"=\frac{{{fmt(r.conf)}-{fmt(r.h_not_a)}}}{{{fmt(r.conf)}}}={fmt(r.zhang)}",
                size=34, color=PURPLE),
            safe_text("Misma fórmula que la del libro, escrita con alturas", size=21, color=MUTED),
            safe_text(f"Z(pan dulce ⇒ café) = {fmt(r.reverse.zhang)}", size=22),
            pill(f"Lift relativo = lift / lift máx = {fmt(r.lift)} / {fmt(r.lift_max, 1)} = "
                 f"{fmt(r.lift / r.lift_max)}", YELLOW, size=18),
        ).arrange(DOWN, buff=0.28).move_to([RX, 0.25, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in z], lag_ratio=0.3))

        self.compare_rules()

    # ------------------------------------------------------------------
    def compare_rules(self) -> None:
        self.beat("36–38 min. Cuatro reglas de la misma tienda, cada una con su cuadrado. Antes "
                  "de la tabla: ¿en cuál el agua sube más dentro de la columna? ¿Cuál tiene el "
                  "agua POR DEBAJO del nivel? ¿Cuál está nivelada?")
        head = self.start("3 · Medir", "Cuatro reglas, cuatro cuadrados", PURPLE)
        minis = VGroup()
        for k, r in enumerate(STORE_RULES):
            m = mini_square(r)
            lab = VGroup(safe_text(r.a, size=20, color=BLUE, weight=BOLD),
                         safe_text("⇒", size=20),
                         safe_text(r.b, size=20, color=ORANGE, weight=BOLD)).arrange(RIGHT, buff=0.1)
            lab.next_to(m, DOWN, buff=0.14)
            minis.add(VGroup(m, lab))
        minis.arrange(RIGHT, buff=1.15).move_to([0, 0.95, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in minis], lag_ratio=0.2))

        self.beat("38–40 min. La tabla confirma la lectura visual. Pregunte: ¿cuál regla genera "
                  "más tickets extra? Muchos dirán vino ⇒ queso (lift 10), pero café ⇒ pan dulce "
                  "suma 8 contra 3.6. Té ⇒ café es negativa en todas las métricas: sustitutos.")
        data = [["Regla", "supp", "conf", "lift", "lev × 100", "conv", "Zhang"]]
        colors = []
        for r in STORE_RULES:
            tone = GREEN if r.lift > 1.05 else (RED if r.lift < 0.95 else MUTED)
            data.append([f"{r.a} ⇒ {r.b}", fmt(r.pab), fmt(r.conf), fmt(r.lift),
                         f"{r.lev * r.n:+.1f}".replace("+0.0", "0.0"), fmt(r.conv), f"{r.zhang:+.2f}".replace("+0.00", "0.00")])
            colors.append([INK, MUTED, MUTED, tone, tone, tone, tone])
        tbl = table(data, col_widths=[3.4, 1.15, 1.15, 1.25, 1.65, 1.2, 1.25], size=22,
                    row_h=0.46, colors=colors)
        tbl.move_to([0, -1.85, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.1) for row in tbl[1:-1]],
                              lag_ratio=0.2), run_time=1.2)
        hl = VGroup(*[SurroundingRectangle(tbl[k][4], color=YELLOW, buff=0.08) for k in (1, 3)])
        self.play(Create(hl))
