"""Acto 3 · Probabilidad como área: soporte, confianza y lift en el cuadrado unitario.

Regla de la tienda: café ⇒ pan dulce (40, 30 y 20 tickets de 100). El
consecuente se dibuja como agua: bajo independencia se nivela a P(B); con
asociación sube dentro de la columna del antecedente y baja fuera, sin cambiar
su área total.
"""
from __future__ import annotations

from manim import *

from sections.base import UnitSquare
from sections.data import CAFE, fmt
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, pill, safe_text, table, tex)

SIDE = 4.6
CENTER = np.array([-3.1, -0.3, 0])
RX = 3.55          # centro del panel derecho


def heights(nab: float, r=CAFE) -> tuple[float, float]:
    return nab / r.na, (r.nb - nab) / (r.n - r.na)


def square_for(nab: float, r=CAFE) -> UnitSquare:
    h_a, h_na = heights(nab, r)
    return UnitSquare(SIDE, r.pa, h_a, h_na).move_to(CENTER)


class SquareSection:
    def square_view(self) -> None:
        r = CAFE
        self.beat("20–21 min. Volvemos a la tienda con otra regla: café ⇒ pan dulce. El cuadrado "
                  "son los 100 tickets (área 1). La columna azul son los tickets con café: su "
                  "ANCHO es P(A) = 0.40. El soporte de un itemset es un área.")
        head = self.start("2 · Ver", "Probabilidad como área", ORANGE)
        empty = UnitSquare(SIDE, r.pa, 0, 0).move_to(CENTER)
        omega = safe_text("100 tickets · área = 1", size=22, color=MUTED)
        omega.next_to(empty, DOWN, buff=0.15)
        self.play(Create(empty.frame), FadeIn(omega))
        a_lab = safe_text("café", size=22, color=BLUE, weight=BOLD)
        a_lab.next_to(empty.p(r.pa / 2, 1), UP, buff=0.1)
        a_brace = BraceBetweenPoints(empty.p(0, 0), empty.p(r.pa, 0), DOWN, color=BLUE)
        a_val = tex(rf"P(A)={fmt(r.pa)}", size=30, color=BLUE).next_to(a_brace, DOWN, buff=0.08)
        self.play(FadeIn(empty.a_col), Create(empty.divider), FadeIn(a_lab))
        self.play(FadeOut(omega), GrowFromCenter(a_brace), FadeIn(a_val))
        rule = VGroup(safe_text("café", size=30, color=BLUE, weight=BOLD), tex(r"\Rightarrow", size=40),
                      safe_text("pan dulce", size=30, color=ORANGE, weight=BOLD)).arrange(RIGHT, buff=0.2)
        rule.move_to([RX, 1.75, 0])
        self.play(FadeIn(rule))

        self.beat("21–22 min. Hipótesis de independencia: el pan dulce (30 % del área) se reparte "
                  "como agua que se nivela: misma altura 0.30 dentro y fuera de la columna. "
                  "El rectángulo amarillo es lo que esperaríamos juntos por azar: 0.40 × 0.30.")
        nab = ValueTracker(r.pa * r.pb * r.n)          # 12 tickets: independencia
        water = always_redraw(lambda: VGroup(*square_for(nab.get_value())[1:3]))
        self.add(water)
        self.bring_to_front(empty.frame, empty.divider)
        self.play(FadeIn(water))
        level = empty.level_line(r.pb)
        level_lab = empty.level_label(rf"P(B)={fmt(r.pb)}", r.pb)
        indep = empty.region(0, r.pa, 0, r.pb, YELLOW, opacity=0)
        indep.set_stroke(YELLOW, width=4)
        indep_txt = VGroup(
            safe_text("Esperado si fueran independientes", size=22, color=YELLOW),
            tex(rf"P(A)\,P(B)={fmt(r.pa)}\times{fmt(r.pb)}={fmt(r.pa * r.pb)}", size=32, color=YELLOW),
        ).arrange(DOWN, buff=0.15).move_to([RX, 0.75, 0])
        self.play(Create(level), FadeIn(level_lab), Create(indep), FadeIn(indep_txt))

        self.beat("22–23 min. Predicción: si quien compra café tiende a llevar pan dulce, ¿qué "
                  "le pasa al agua dentro y fuera de la columna? Pida que lo dibujen con la mano. "
                  "Luego anime: sube dentro, baja fuera; el área naranja total NO cambia (30).")
        self.play(nab.animate.set_value(r.nab), run_time=2.2)
        obs = VGroup(
            safe_text("Observado: 20 tickets con ambos", size=22, color=GREEN),
            tex(rf"P(A\cup B)={fmt(r.pab)}", size=32, color=GREEN),
        ).arrange(DOWN, buff=0.15).next_to(indep_txt, DOWN, buff=0.35)
        self.play(FadeIn(obs))

        self.beat("23–24 min. Lea alturas, no áreas: dentro de la columna café el agua llega a "
                  "0.50 = 20/40: ESO es la confianza (probabilidad de B dentro del mundo A). "
                  "Fuera llega a 10/60 ≈ 0.17.")
        h_a, h_na = heights(r.nab)
        b_in = BraceBetweenPoints(empty.p(0, 0), empty.p(0, h_a), LEFT, color=ORANGE)
        b_in_lab = tex(rf"\mathrm{{conf}}={fmt(h_a)}", size=28, color=ORANGE)
        b_in_lab.next_to(empty.p(r.pa / 2, h_a), UP, buff=0.08)
        na_lab = tex(rf"P(B\mid\neg A)={fmt(h_na)}", size=26, color=MUTED)
        na_lab.move_to(empty.p((1 + r.pa) / 2, (h_na + r.pb) / 2))
        self.play(GrowFromCenter(b_in), FadeIn(b_in_lab), FadeIn(na_lab))

        self.beat("24–25 min. El lift compara la altura del agua en la columna con el nivel "
                  "de independencia: 0.50 / 0.30 = 1.67. «Comprar café hace 1.67 veces más "
                  "probable el pan dulce.» Lift = 1 ⇔ el agua está nivelada.")
        arrow = Arrow(empty.p(r.pa * 0.55, r.pb), empty.p(r.pa * 0.55, h_a), buff=0,
                      color=PURPLE, stroke_width=6, max_tip_length_to_length_ratio=0.35)
        lift_eq = tex(rf"\mathrm{{lift}}=\frac{{\text{{altura en }}A}}{{\text{{nivel}}}}"
                      rf"=\frac{{{fmt(h_a)}}}{{{fmt(r.pb)}}}={fmt(r.lift)}", size=34, color=PURPLE)
        self.play(FadeOut(indep_txt), FadeOut(obs))
        lift_eq.move_to([RX, 0.7, 0])
        self.play(GrowArrow(arrow), Write(lift_eq))

        self.beat("25–26 min. Misma cifra vista como áreas: verde observado (0.20) contra "
                  "amarillo esperado (0.12). Como el cociente de áreas no distingue cuál es A y "
                  "cuál B, el lift es SIMÉTRICO. La confianza no: conf(pan dulce ⇒ café) = 0.67.")
        obs_rect = empty.region(0, r.pa, 0, h_a, GREEN, opacity=0.0).set_stroke(GREEN, width=5)
        area_eq = tex(r"\mathrm{lift}=\frac{P(A\cup B)}{P(A)\,P(B)}=\frac{"
                      + fmt(r.pab) + "}{" + fmt(r.pa * r.pb) + "}", size=34)
        area_eq.next_to(lift_eq, DOWN, buff=0.45)
        sym = VGroup(
            safe_text("Simétrico: lift(B ⇒ A) = lift(A ⇒ B)", size=21, color=INK),
            safe_text(f"Asimétrica: conf(B ⇒ A) = 20/30 = {fmt(r.reverse.conf)}", size=21, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(area_eq, DOWN, buff=0.4)
        self.play(Create(obs_rect), Write(area_eq))
        self.play(FadeIn(sym, shift=UP * 0.1))

        self.beat("26–27 min. Movamos los datos. Primero hacia la independencia (12 tickets, "
                  "lift 1) y luego hacia la exclusión (2 tickets): el agua de la columna queda "
                  "por debajo del nivel, lift < 1. Pida que anticipen el valor antes de cada paso.")
        self.play(FadeOut(VGroup(arrow, lift_eq, area_eq, sym, b_in, b_in_lab, na_lab, obs_rect)),
                  run_time=0.4)
        readout = self.live_readout(nab, r).move_to([RX, 0.35, 0])
        self.play(FadeIn(readout))
        self.play(nab.animate.set_value(12), run_time=1.8)
        self.wait(0.4)
        self.play(nab.animate.set_value(2), run_time=1.8)

        self.beat("27–28 min. Ahora al extremo opuesto: todo el pan dulce cae dentro de café "
                  "(30 tickets). El agua llega a 0.75 y fuera queda seca: lift máximo = "
                  "1 / max(P(A), P(B)) = 2.5. Un lift de 10 es imposible con estos soportes.")
        self.play(nab.animate.set_value(30), run_time=2.4)
        ceiling = tex(r"\mathrm{lift}_{\max}=\frac{1}{\max\{P(A),P(B)\}}=\frac{1}{0.40}=2.5",
                      size=32, color=RED).move_to([RX, -2.25, 0])
        self.play(Write(ceiling))

        self.beat("28–30 min. Formalización. Las cuatro regiones del cuadrado son la tabla de "
                  "contingencia 2 × 2. Deduzca en voz alta: lift > k ⇔ conf > k·P(B) (por "
                  "ejemplo, con leche al 60 %, lift > 1.5 exige conf > 0.90).")
        self.play(nab.animate.set_value(r.nab), FadeOut(ceiling), FadeOut(readout), run_time=1.2)
        regions = VGroup(
            safe_text("20", size=26, weight=BOLD).move_to(empty.p(r.pa / 2, h_a / 2)),
            safe_text("20", size=26, weight=BOLD, color=MUTED).move_to(empty.p(r.pa / 2, (1 + h_a) / 2)),
            safe_text("10", size=24, weight=BOLD).move_to(empty.p((1 + r.pa) / 2, h_na / 2)),
            safe_text("50", size=26, weight=BOLD, color=MUTED).move_to(empty.p((1 + r.pa) / 2, (1 + h_na) / 2)),
        )
        self.play(FadeIn(regions))
        ct = table([["", "B", "¬B", "total"],
                    ["A", "20", "20", "40"],
                    ["¬A", "10", "50", "60"],
                    ["total", "30", "70", "100"]],
                   col_widths=[1.2, 1.15, 1.15, 1.25], size=24,
                   colors=[[BLUE, ORANGE, MUTED, INK], [MUTED, ORANGE, MUTED, INK],
                           [INK, ORANGE, MUTED, INK]])
        ct.move_to([RX, 0.45, 0])
        defs = VGroup(
            tex(r"\mathrm{supp}=P(A\cup B)=\text{área}", size=28),
            tex(r"\mathrm{conf}=P(B\mid A)=\text{altura dentro de }A", size=28, color=ORANGE),
            tex(r"\mathrm{lift}>k\iff\mathrm{conf}>k\,P(B)", size=28, color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(ct, DOWN, buff=0.4)
        self.play(FadeIn(ct))
        self.play(LaggedStart(*[FadeIn(d, shift=UP * 0.1) for d in defs], lag_ratio=0.25))
        self.cafe_square = (empty, water, level, level_lab, a_lab, a_brace, a_val, rule, regions)

    def live_readout(self, nab: ValueTracker, r) -> VGroup:
        """Ficha con conteo, confianza y lift que se actualizan con ``nab``."""
        specs = [
            ("tickets con ambos", lambda: nab.get_value(), 0, GREEN),
            ("conf = altura en A", lambda: nab.get_value() / r.na, 2, ORANGE),
            ("lift", lambda: nab.get_value() / r.n / (r.pa * r.pb), 2, PURPLE),
        ]
        rows = VGroup()
        for i, (label, fn, d, color) in enumerate(specs):
            lab = safe_text(label, size=24, color=MUTED)
            lab.move_to([0, -0.62 * i, 0], aligned_edge=LEFT)
            num = DecimalNumber(fn(), num_decimal_places=d, font_size=36, color=color)
            num.move_to([3.6, -0.62 * i, 0])
            num.add_updater(lambda m, f=fn: m.set_value(f()))
            rows.add(VGroup(lab, num))
        box = card(rows.width + 0.8, rows.height + 0.55).move_to(rows)
        return VGroup(box, rows)
