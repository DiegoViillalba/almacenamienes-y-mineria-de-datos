"""Acto 2 · De tickets a reglas: matriz binaria, soporte, Apriori, confianza y lift.

Usa los cinco tickets de los capítulos 5.0 y 5.1 del libro, con conteo mínimo 2
y confianza mínima 0.60, para que el grupo reconozca los mismos números.
"""
from __future__ import annotations

from manim import *

from sections.data import (ITEMS, MIN_CONF, MIN_COUNT, PAIR_COUNTS, SINGLE_COUNTS,
                           TICKETS, TOY_RULES, TRIPLE_COUNT, fmt)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, YELLOW, card, pill, safe_text, table, tex)

COL_W = 1.42
ROW_H = 0.60
X0 = -4.85             # centro de la primera columna de ítems
Y0 = 1.15              # centro de la primera fila (T1)
SHORT = {"leche": "leche", "pan": "pan", "mantequilla": "mantequilla", "queso": "queso"}


def mx(j: int) -> float:
    return X0 + j * COL_W


def my(i: int) -> float:
    return Y0 - i * ROW_H


class ConstructionSection:
    def construction(self) -> None:
        self.beat("10–11 min. Cambiamos a una tienda diminuta: los cinco tickets del libro. "
                  "Pida que, sin calcular, digan qué par de productos «va junto».")
        head = self.start("1 · Construir", "Cinco tickets", BLUE)
        rows = VGroup()
        for i, t in enumerate(TICKETS):
            name = safe_text(f"T{i + 1}", size=26, color=MUTED, weight=BOLD)
            items = safe_text("  ·  ".join(x for x in ITEMS if x in t), size=28)
            rows.add(VGroup(name, items).arrange(RIGHT, buff=0.45))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-1.5, -0.2, 0])
        box = card(rows.width + 0.8, rows.height + 0.6).move_to(rows)
        self.play(FadeIn(box), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows],
                                           lag_ratio=0.15))
        ask = pill("¿Qué par va junto?", YELLOW, size=22).move_to([4.6, 1.2, 0])
        self.play(FadeIn(ask))

        self.beat("11–12 min. Pasamos a matriz binaria ticket × producto: sólo presencia o "
                  "ausencia; ni cantidades ni precios. Pregunte qué información se perdió.")
        headers = VGroup(*[safe_text(SHORT[x], size=21, color=MUTED, weight=BOLD,
                                     max_width=COL_W - 0.12).move_to([mx(j), Y0 + 0.62, 0])
                           for j, x in enumerate(ITEMS)])
        names = VGroup(*[safe_text(f"T{i + 1}", size=22, color=MUTED, weight=BOLD)
                         .move_to([X0 - 1.05, my(i), 0]) for i in range(len(TICKETS))])
        cells, digits = VGroup(), VGroup()
        self.cell = {}
        for i, t in enumerate(TICKETS):
            for j, x in enumerate(ITEMS):
                on = x in t
                sq = Rectangle(width=COL_W - 0.1, height=ROW_H - 0.1, stroke_color=GRID,
                               stroke_width=1.2, fill_color=BLUE if on else PANEL,
                               fill_opacity=0.30 if on else 0.9).move_to([mx(j), my(i), 0])
                d = safe_text("1" if on else "0", size=24, color=INK if on else GRID,
                              weight=BOLD if on else NORMAL).move_to(sq)
                self.cell[(i, j)] = sq
                cells.add(sq)
                digits.add(d)
        self.play(FadeOut(box), FadeOut(ask), FadeOut(rows), run_time=0.5)
        self.play(FadeIn(cells), FadeIn(digits), FadeIn(names), FadeIn(headers), run_time=0.9)

        self.beat("12–13 min. Soporte de un ítem = proporción de tickets que lo contienen. "
                  "Cuenten por columna antes de revelar. Umbral: al menos 2 tickets (0.40). "
                  "Queso queda fuera.")
        counts = VGroup()
        for j, x in enumerate(ITEMS):
            c = SINGLE_COUNTS[x]
            col = GREEN if c >= MIN_COUNT else RED
            counts.add(VGroup(
                safe_text(f"{c}", size=26, color=col, weight=BOLD),
                safe_text(f"{c / len(TICKETS):.2f}", size=20, color=MUTED),
            ).arrange(DOWN, buff=0.08).move_to([mx(j), my(len(TICKETS)) - 0.1, 0]))
        sep = Line([X0 - 0.7, my(4) - 0.42, 0], [mx(3) + 0.7, my(4) - 0.42, 0], color=GRID,
                   stroke_width=2)
        supp = tex(r"\mathrm{supp}(X)=\frac{\#\{\text{tickets}\supseteq X\}}{n}", size=38)
        supp.move_to([3.75, 1.3, 0])
        rule_txt = safe_text(f"Frecuente: al menos {MIN_COUNT} de 5 tickets", size=24,
                             color=MUTED).next_to(supp, DOWN, buff=0.35)
        self.play(Create(sep), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in counts],
                                           lag_ratio=0.2), Write(supp), FadeIn(rule_txt))
        cross = Cross(VGroup(*[self.cell[(i, 3)] for i in range(5)]), stroke_color=RED,
                      stroke_width=4)
        self.play(Create(cross), headers[3].animate.set_color(RED))

        self.beat("13–15 min. Pares: resalte las filas donde leche y pan valen 1 (T1, T2, T5). "
                  "Recalque la antimonotonía: ningún par con queso puede ser frecuente, así que "
                  "ni se cuenta. El trío aparece sólo en T1.")
        both = [i for i, t in enumerate(TICKETS) if {"leche", "pan"} <= t]
        hl = VGroup(*[SurroundingRectangle(VGroup(self.cell[(i, 0)], self.cell[(i, 1)]),
                                           color=GREEN, buff=0.03, stroke_width=4)
                      for i in both])
        self.play(Create(hl))
        pair_rows = VGroup(*[
            VGroup(safe_text("{" + ", ".join(p) + "}", size=24, max_width=4.0),
                   safe_text(f"{c}", size=26, color=GREEN, weight=BOLD))
            .arrange(RIGHT, buff=0.4)
            for p, c in PAIR_COUNTS.items()])
        trio = VGroup(safe_text("{leche, pan, mantequilla}", size=24, color=MUTED, max_width=4.4),
                      safe_text(f"{TRIPLE_COUNT}", size=26, color=RED, weight=BOLD)
                      ).arrange(RIGHT, buff=0.4)
        listing = VGroup(*pair_rows, trio).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        listing.next_to(rule_txt, DOWN, buff=0.45).align_to(rule_txt, LEFT)
        anti = pill("{queso} infrecuente ⇒ sus superconjuntos también", RED, size=19)
        anti.next_to(listing, DOWN, buff=0.35).set_x(3.75)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in listing], lag_ratio=0.2))
        self.play(FadeIn(anti))

        self.beat("15–17 min. De un itemset frecuente salen reglas. pan ⇒ leche: nos quedamos "
                  "SÓLO con los tickets que tienen pan (4) y contamos cuántos tienen leche (3). "
                  "La confianza es una probabilidad condicional: el universo se encoge.")
        self.play(FadeOut(VGroup(supp, rule_txt, listing, anti, hl)), run_time=0.4)
        with_pan = [i for i, t in enumerate(TICKETS) if "pan" in t]
        others = [i for i in range(5) if i not in with_pan]
        dim = VGroup(*[self.cell[(i, j)] for i in others for j in range(4)])
        dim_d = VGroup(*[digits[i * 4 + j] for i in others for j in range(4)], names[others[0]])
        self.play(dim.animate.set_opacity(0.15), dim_d.animate.set_opacity(0.2))
        hits = VGroup(*[SurroundingRectangle(self.cell[(i, 0)], color=ORANGE, buff=0.03,
                                             stroke_width=4)
                        for i in with_pan if "leche" in TICKETS[i]])
        pan_box = SurroundingRectangle(VGroup(*[self.cell[(i, 1)] for i in range(5)]),
                                       color=BLUE, buff=0.04, stroke_width=3)
        self.play(Create(pan_box), Create(hits))
        rule = VGroup(safe_text("pan", size=34, color=BLUE, weight=BOLD),
                      tex(r"\Rightarrow", size=44),
                      safe_text("leche", size=34, color=ORANGE, weight=BOLD)).arrange(RIGHT, buff=0.25)
        rule.move_to([3.75, 1.5, 0])
        conf = tex(r"\mathrm{conf}(A\Rightarrow B)=\frac{\mathrm{supp}(A\cup B)}{\mathrm{supp}(A)}"
                   r"=P(B\mid A)", size=34).next_to(rule, DOWN, buff=0.45)
        conf_num = tex(r"=\frac{3/5}{4/5}=\frac{3}{4}=0.75", size=40, color=ORANGE)
        conf_num.next_to(conf, DOWN, buff=0.35)
        self.play(FadeIn(rule, shift=DOWN * 0.1))
        self.play(Write(conf))
        self.play(Write(conf_num))
        note = safe_text("A ∪ B: tickets que contienen ambos itemsets", size=19,
                         color=MUTED).next_to(conf_num, DOWN, buff=0.3)
        self.play(FadeIn(note))

        self.beat("17–18 min. Las seis reglas posibles con los tres pares frecuentes. Pregunte "
                  "por qué pan ⇒ leche y leche ⇒ pan tienen la misma confianza aquí "
                  "(ambos ítems tienen soporte 4). Umbral 0.60: pasan 4 de 6.")
        self.clear()
        self.play(FadeIn(self.heading("1 · Construir", "Seis reglas candidatas", BLUE)),
                  run_time=0.3)
        data = [["Regla", "Conteo", "Confianza", "≥ 0.60"]]
        cols = []
        for r in TOY_RULES:
            data.append([f"{r['a']} ⇒ {r['b']}", f"{r['nab']}/{r['na']}", fmt(r["conf"]),
                         "sí" if r["ok"] else "no"])
            cols.append([INK, MUTED, INK, GREEN if r["ok"] else RED])
        tbl = table(data, col_widths=[4.6, 1.6, 1.9, 1.4], size=25, colors=cols)
        tbl.move_to([-0.9, -0.35, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in tbl[1:-1]],
                              lag_ratio=0.15), run_time=1.2)
        self.rule_table = tbl

        self.beat("18–19 min. Predicción: ¿alguna de estas reglas «aceptadas» describe una "
                  "asociación real? Revele la columna lift = confianza / P(B). Todas < 1: los "
                  "pares aparecen juntos MENOS de lo esperado. Es la leche del pan dulce otra vez.")
        lift_head = safe_text("Lift", size=25, color=PURPLE, weight=BOLD)
        lift_head.move_to(tbl[0][-1]).shift(RIGHT * 1.7)
        lifts = VGroup()
        for k, r in enumerate(TOY_RULES):
            lifts.add(safe_text(f"{r['lift']:.4f}", size=25, color=RED, weight=BOLD)
                      .move_to(tbl[k + 1][-1]).shift(RIGHT * 1.7))
        self.play(FadeIn(lift_head), LaggedStart(*[FadeIn(l) for l in lifts], lag_ratio=0.1))
        lift_def = tex(r"\mathrm{lift}(A\Rightarrow B)=\frac{\mathrm{conf}(A\Rightarrow B)}{P(B)}"
                       r"=\frac{0.75}{0.80}=0.9375", size=34, color=PURPLE)
        lift_def.move_to([0, -2.85, 0])
        self.play(Write(lift_def))

        self.beat("19–20 min. Moraleja: Apriori encuentra lo frecuente y lo confiable, no lo "
                  "interesante. Formalice la búsqueda en dos líneas (antimonotonía + reglas "
                  "desde itemsets frecuentes) y anuncie: ahora veremos POR QUÉ se divide entre P(B).")
        moral = VGroup(
            pill("Frecuente + confiable ≠ interesante", YELLOW, size=22),
            safe_text("Apriori, en dos pasos:", size=23, weight=BOLD),
            safe_text("1) itemsets frecuentes nivel por nivel,", size=22, color=MUTED),
            safe_text("    podando con la antimonotonía", size=22, color=MUTED),
            safe_text("2) reglas A ⇒ Z∖A con confianza ≥ mínimo,", size=22, color=MUTED),
            safe_text("    sin volver a leer los datos", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        self.play(VGroup(tbl, lift_head, lifts).animate.scale(0.72).move_to([-3.1, 0.35, 0]),
                  lift_def.animate.scale(0.8).move_to([-3.1, -2.35, 0]))
        moral.move_to([3.95, 0.0, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=LEFT * 0.1) for m in moral], lag_ratio=0.25))
