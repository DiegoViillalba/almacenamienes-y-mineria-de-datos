"""Acto 3 · Trampas y números: C4.5 (23–31 min).

La columna «Día» parte los quince días en quince cubetas puras y gana con la
ganancia máxima, aunque no sirve para predecir. La información de la partición
(la entropía de los TAMAÑOS de los grupos) se ve como una barra segmentada; la
razón de ganancia atenúa el sesgo, pero no lo elimina. Después: un corte que
barre la humedad numérica y dibuja la ganancia de cada umbral.
"""
from __future__ import annotations

from manim import *

from sections.data import DAYS, H_ROOT, HUM_BEST, HUM_CUTS, HUM_NUM, RATIOS, partition
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, TEAL, YELLOW, card, marker, pill, safe_text, table, tex)
from sections.widgets import chip, counts_row
from sections.part02_entropy import PLAY, day_chip


def size_bar(sizes, width: float, colors, height: float = 0.55, labels=True) -> VGroup:
    """Barra horizontal partida en segmentos proporcionales a ``sizes``."""
    n = sum(sizes)
    out = VGroup()
    x = 0.0
    for i, s in enumerate(sizes):
        w = width * s / n
        r = Rectangle(width=w, height=height, stroke_color=INK, stroke_width=1.2,
                      fill_color=colors[i % len(colors)], fill_opacity=0.55)
        r.move_to([x + w / 2, 0, 0])
        seg = VGroup(r)
        if labels:
            seg.add(safe_text(str(s), size=18, color=INK).move_to(r))
        out.add(seg)
        x += w
    return out


class C45Section:
    def c45(self) -> None:
        self.day_trap()
        self.split_info()
        self.ratio_table()
        self.numeric_cut()
        self.c45_summary()

    # ------------------------------------------------------------------
    def day_trap(self) -> None:
        self.beat("23–25 min. Trampa. Agregamos la columna Día (D1…D15) como si fuera un "
                  "atributo. Pregunte: ¿qué ganancia tendrá? Cada día queda solo en su cubeta, "
                  "cada cubeta es pura, la entropía después es 0 y la ganancia es la máxima "
                  "posible: 0.971. ID3 la pondría en la raíz. Pero un día nuevo (D16) no tiene "
                  "rama: el atributo no sirve para predecir.")
        self.start("C4.5: el sesgo de la ganancia", "¿Y si preguntamos por el día?", ORANGE)
        boxes = VGroup()
        for r in DAYS:
            b = RoundedRectangle(corner_radius=0.08, width=0.72, height=0.72,
                                 stroke_color=GRID, stroke_width=1.5, fill_color=PANEL,
                                 fill_opacity=0.9)
            c = day_chip(r, r=0.2).move_to(b)
            lab = safe_text(r[0], size=15, color=MUTED).next_to(b, UP, buff=0.08)
            boxes.add(VGroup(b, c, lab))
        boxes.arrange(RIGHT, buff=0.1).move_to([0, 1.1, 0])
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in boxes], lag_ratio=0.05))
        pure = safe_text("15 cubetas · todas puras", size=24, color=GREEN, weight=BOLD
                         ).next_to(boxes, DOWN, buff=0.3)
        self.play(FadeIn(pure))
        e1 = tex(r"H_{\text{desp}} = \sum_{v=1}^{15} \tfrac{1}{15}\cdot 0 = 0", size=34)
        e2 = tex(r"\text{Gan}(\text{D\'ia}) = 0.971 - 0 = 0.971", size=36, color=YELLOW)
        e3 = tex(r"> \text{Gan}(\text{Pron\'ostico}) = 0.280", size=34, color=MUTED)
        eqs = VGroup(e1, VGroup(e2, e3).arrange(RIGHT, buff=0.3)).arrange(DOWN, buff=0.3)
        eqs.move_to([0, -1.2, 0])
        self.play(Write(e1))
        self.play(Write(e2), FadeIn(e3))
        d16 = VGroup(marker(None, r=0.2), safe_text("D16: ¿a qué rama va?", size=22,
                                                  color=YELLOW, weight=BOLD)
                     ).arrange(RIGHT, buff=0.2).move_to([0, -2.75, 0])
        self.play(FadeIn(d16, shift=UP * 0.1))

    # ------------------------------------------------------------------
    def split_info(self) -> None:
        self.beat("25–27 min. La corrección de C4.5 (Quinlan, 1993): dividir la ganancia entre "
                  "la entropía de los TAMAÑOS de los grupos (información de la partición). Una "
                  "pregunta que reparte en muchos grupos pequeños tiene una barra muy picada y "
                  "un SplitInfo grande: queda penalizada. Pida que calculen el de Pronóstico "
                  "(6, 4, 5) antes de mostrarlo.")
        self.start("Información de la partición", "Castigar las preguntas que pican demasiado",
                   ORANGE)
        f = tex(r"\text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|}\log_2 \frac{|S_v|}{|S|}",
                size=34).move_to([0, 1.65, 0])
        self.play(Write(f))
        rows = VGroup()
        for name, sizes, si, cols in [
            ("Pronóstico", [6, 4, 5], RATIOS["Pronóstico"][1], [TEAL, PURPLE, BLUE]),
            ("Humedad", [7, 8], RATIOS["Humedad"][1], [TEAL, PURPLE]),
            ("Día", [1] * 15, RATIOS["Día"][1], [TEAL, PURPLE]),
        ]:
            lab = safe_text(name, size=24, weight=BOLD).set_x(-5.3)
            bar = size_bar(sizes, 7.2, cols, labels=len(sizes) < 10).next_to(
                [-4.2, 0, 0], RIGHT, buff=0)
            val = safe_text(f"{si:.3f} bits", size=24, color=ORANGE, weight=BOLD
                            ).next_to(bar, RIGHT, buff=0.3)
            rows.add(VGroup(lab, bar, val))
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, -0.35, 0])
        for r in rows:
            self.play(FadeIn(r[0]), LaggedStart(*[GrowFromEdge(s, LEFT) for s in r[1]],
                                                lag_ratio=0.04), run_time=0.9)
            self.play(FadeIn(r[2]), run_time=0.4)
        g = tex(r"\text{Raz\'onGan}(S, A) = \frac{\text{Gan}(S, A)}{\text{SplitInfo}(S, A)}",
                size=36, color=YELLOW).move_to([0, -2.55, 0])
        self.play(Write(g))

    # ------------------------------------------------------------------
    def ratio_table(self) -> None:
        self.beat("27–28 min. Dos lecciones honestas. (1) La razón atenúa el sesgo pero no lo "
                  "elimina: con 15 ejemplos, Día sigue ganando (0.249). Los identificadores se "
                  "EXCLUYEN antes de entrenar. (2) Criterios distintos eligen raíces distintas: "
                  "sin Día, la razón prefiere Humedad (dos grupos) a Pronóstico. ¿Cuál árbol es "
                  "mejor? Lo decide la validación: dejando uno fuera, ganancia 14/15, razón "
                  "12/15, Gini 13/15; con Día, 9/15 = siempre la mayoría.")
        self.start("Ganancia contra razón de ganancia", "¿Cambia la raíz?", ORANGE)
        hdr = ["Atributo", "Ganancia", "SplitInfo", "Razón"]
        rows = [hdr]
        colors = []
        for a in ["Día", "Humedad", "Pronóstico", "Viento", "Temperatura"]:
            g, si, gr = RATIOS[a]
            rows.append([a, f"{g:.3f}", f"{si:.3f}", f"{gr:.3f}"])
            colors.append([RED if a == "Día" else INK, INK, MUTED,
                           YELLOW if a in ("Día", "Humedad") else INK])
        tb = table(rows, col_widths=[2.3, 1.7, 1.7, 1.5], size=24, row_h=0.55, colors=colors)
        tb.move_to([-2.6, -0.25, 0])
        self.play(FadeIn(tb))
        l1 = VGroup(safe_text("1 · Día sigue ganando:", size=22, color=RED, weight=BOLD),
                    safe_text("los identificadores se excluyen", size=21),
                    safe_text("antes de entrenar.", size=21)).arrange(DOWN, aligned_edge=LEFT,
                                                                      buff=0.08)
        l2 = VGroup(safe_text("2 · Sin Día: Humedad > Pronóstico", size=22, color=YELLOW,
                              weight=BOLD),
                    safe_text("otro criterio, otra raíz.", size=21),
                    safe_text("Decide la validación:", size=21),
                    safe_text("LOOCV 14/15 (ganancia) · 12/15 (razón)", size=19, color=MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ls = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([4.2, -0.2, 0])
        self.play(Indicate(tb[1], color=RED, scale_factor=1.03), FadeIn(l1))
        self.play(Indicate(tb[2], color=YELLOW, scale_factor=1.03), FadeIn(l2))

    # ------------------------------------------------------------------
    def numeric_cut(self) -> None:
        self.beat("28–30 min. Atributos numéricos. Si la humedad se mide en %, C4.5 pregunta "
                  "«¿humedad ≤ t?». Se ordenan los valores y se prueban los puntos medios entre "
                  "valores consecutivos. Pida que griten «¡alto!» cuando crean que el corte es "
                  "el mejor. Gana 82.5 %: a la izquierda 6 sí y 1 no. Dato: el óptimo siempre "
                  "está entre dos valores de clases distintas (Fayyad e Irani), y un atributo "
                  "numérico puede reusarse más abajo con otro corte.")
        self.start("Atributos numéricos", "¿Humedad ≤ t? Barrer todos los cortes", ORANGE)
        nl = NumberLine(x_range=[62, 98, 2], length=12, color=GRID, include_numbers=False,
                        include_tip=False).move_to([0.5, -2.05, 0])
        ticks = VGroup(*[safe_text(str(v), size=16, color=MUTED).next_to(nl.n2p(v), DOWN, buff=0.12)
                         for v in range(65, 100, 5)])
        xlab = safe_text("humedad (%)", size=18, color=MUTED).next_to(nl, DOWN, buff=0.5)
        chips = VGroup()
        seen: dict[int, int] = {}
        for v, c in HUM_NUM:
            k = seen.get(v, 0)
            seen[v] = k + 1
            chips.add(marker(1 - c, r=0.15).move_to(nl.n2p(v) + UP * (0.25 + 0.34 * k)))
        self.play(Create(nl), FadeIn(ticks), FadeIn(xlab),
                  LaggedStart(*[GrowFromCenter(c) for c in chips], lag_ratio=0.05))
        scale = 9.0
        axis = Line(nl.n2p(62) + UP * 1.75, nl.n2p(98) + UP * 1.75, color=GRID, stroke_width=1.5)
        glab = safe_text("ganancia de cada corte", size=18, color=PURPLE
                         ).next_to(axis, DOWN, buff=0.1).align_to(axis, RIGHT)
        self.play(Create(axis), FadeIn(glab))
        t = ValueTracker(HUM_CUTS[0][0])
        cut = always_redraw(lambda: DashedLine(nl.n2p(t.get_value()) + DOWN * 0.1,
                                               nl.n2p(t.get_value()) + UP * 3.6, color=YELLOW,
                                               stroke_width=3))
        self.add(cut)
        stems = VGroup()
        for i, (th, L, R, g) in enumerate(HUM_CUTS):
            if i:
                self.play(t.animate.set_value(th), run_time=0.35)
            x = nl.n2p(th)
            y0 = axis.get_y()
            stem = Line([x[0], y0, 0], [x[0], y0 + g * scale, 0], color=PURPLE, stroke_width=6)
            dot = Dot([x[0], y0 + g * scale, 0], radius=0.06, color=PURPLE)
            stems.add(VGroup(stem, dot))
            self.play(Create(stem), FadeIn(dot), run_time=0.25)
        th, L, R, g = HUM_BEST
        self.play(t.animate.set_value(th), run_time=0.8)
        best = stems[[c[0] for c in HUM_CUTS].index(th)]
        self.play(best.animate.set_color(YELLOW))
        info = VGroup(
            safe_text(f"t = {th:g} %", size=26, color=YELLOW, weight=BOLD),
            VGroup(safe_text("≤ t:", size=22, color=MUTED), counts_row(L, size=22, r=0.1)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(safe_text("> t:", size=22, color=MUTED), counts_row(R, size=22, r=0.1)
                   ).arrange(RIGHT, buff=0.2),
            safe_text(f"Gan = {g:.3f}", size=24, color=PURPLE, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        bx = card(info.width + 0.4, info.height + 0.35, YELLOW)
        grp = VGroup(bx, info.move_to(bx)).move_to([-4.9, 1.25, 0])
        self.play(FadeIn(grp))

    # ------------------------------------------------------------------
    def c45_summary(self) -> None:
        self.beat("30–31 min. C4.5 en una lámina. Además de la razón de ganancia y los cortes "
                  "numéricos: con valores faltantes reparte el ejemplo entre las ramas con pesos "
                  "proporcionales; y poda con una estimación pesimista del error de cada hoja. "
                  "La poda la veremos con CART en el acto 5.")
        self.start("C4.5 en una lámina", "Lo que agrega a ID3", ORANGE)
        items = [
            ("Razón de ganancia", "Gan / SplitInfo: castiga preguntas con muchos valores", ORANGE),
            ("Cortes numéricos", "«A ≤ t» en puntos frontera; el atributo puede reusarse", YELLOW),
            ("Valores faltantes", "el ejemplo baja por todas las ramas con pesos", BLUE),
            ("Poda pesimista", "cambia subárboles por hojas si no empeora el error", RED),
        ]
        cards = VGroup()
        for t, d, c in items:
            g = VGroup(safe_text(t, size=26, color=c, weight=BOLD),
                       safe_text(d, size=21, max_width=5.2)).arrange(DOWN, aligned_edge=LEFT,
                                                                    buff=0.15)
            bx = card(5.9, g.height + 0.5, c)
            cards.add(VGroup(bx, g.move_to(bx).align_to(bx, LEFT).shift(RIGHT * 0.3)))
        cards.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.35)).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cards], lag_ratio=0.2))
