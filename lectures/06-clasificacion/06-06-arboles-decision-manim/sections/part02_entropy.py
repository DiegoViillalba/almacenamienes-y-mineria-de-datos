"""Acto 2 · Medir la mezcla: entropía y ganancia de información, ID3 (10–23 min).

Quince días de béisbol. Las filas de la tabla se vuelven fichas; dos preguntas
candidatas las reparten en cubetas y se ve cuál deja grupos «puros». La
entropía nace como un medidor de mezcla (curva H(p) con una cubeta que cambia
de composición). El promedio ponderado se dibuja como suma de áreas: la
ganancia es el área que «desaparece». Después: recursión, árbol, reglas y la
forma general (algoritmo de Hunt).
"""
from __future__ import annotations

from manim import *

from sections.data import (ATTRS, BRANCH_GAINS, DAYS, H2, H_ROOT, ROOT_GAINS, VALUES, entropy,
                           partition, yes_no)
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, TEAL, YELLOW, bullet_list, card, marker, pill, safe_text, table,
                            tex)
from sections.widgets import (Bucket, TreeView, area_bars, axis_numbers, chip, counts_row,
                              impurity_axes, leaf_spec, q_spec)

PLAY = {"Sí": 0, "No": 1}           # clase 0 = juega (azul), 1 = no juega (naranja)
NAMES = ("Sí", "No")


def day_chip(row, r: float = 0.17) -> VGroup:
    return chip(PLAY[row[-1]], row[0][1:], r=r, size=12)


def days_table(size: float = 17) -> VGroup:
    rows = [["Día", "Pronóstico", "Temperatura", "Humedad", "Viento", "¿Juega?"]]
    cols = []
    for r in DAYS:
        rows.append(list(r))
        cols.append([MUTED, INK, INK, INK, INK, CLASS_COLOR[PLAY[r[-1]]]])
    return table(rows, col_widths=[0.8, 1.55, 1.6, 1.35, 1.2, 1.05], size=size, row_h=0.3,
                 colors=cols)


def id3_spec() -> dict:
    sunny = q_spec("Humedad", (2, 4), [("Alta", leaf_spec((0, 4), NAMES)),
                                       ("Normal", leaf_spec((2, 0), NAMES))], NAMES)
    rain = q_spec("Viento", (3, 2), [("Débil", leaf_spec((3, 0), NAMES)),
                                     ("Fuerte", leaf_spec((0, 2), NAMES))], NAMES)
    return q_spec("Pronóstico", (9, 6), [("Soleado", sunny), ("Nublado", leaf_spec((4, 0), NAMES)),
                                        ("Lluvia", rain)], NAMES)


class EntropySection:
    def entropy(self) -> None:
        self.weather_table()
        self.two_questions()
        self.entropy_meter()
        self.weighted_area()
        self.gain_race()
        self.recurse()
        self.rules()
        self.hunt()

    # ------------------------------------------------------------------
    def weather_table(self) -> None:
        self.beat("10–11 min. Nuevo ejemplo: quince días con el clima y si se jugó béisbol. "
                  "9 sí, 6 no. Pregunte: si sólo pudieran hacer UNA pregunta sobre el clima "
                  "para adivinar si se juega, ¿cuál harían? Pida una votación rápida a mano "
                  "alzada y anote el resultado en la pizarra.")
        self.start("Medir la mezcla", "¿Se juega el partido?", PURPLE)
        tb = days_table().move_to([-2.4, -0.5, 0])
        self.play(FadeIn(tb[-1]), LaggedStart(*[FadeIn(r) for r in tb[:-1]], lag_ratio=0.04))
        side = VGroup(
            counts_row((9, 6), size=30, r=0.15),
            safe_text("9 días sí · 6 días no", size=22, color=MUTED),
            safe_text("Si sólo pudieras hacer", size=26),
            safe_text("UNA pregunta…", size=30, color=YELLOW, weight=BOLD),
            safe_text("¿cuál harías?", size=26),
        ).arrange(DOWN, buff=0.22).move_to([4.9, -0.2, 0])
        self.play(FadeIn(side, shift=LEFT * 0.1))
        self._tb, self._side = tb, side

    # ------------------------------------------------------------------
    def two_questions(self) -> None:
        self.beat("11–13 min. Cada fila se vuelve una ficha (círculo azul = se jugó, cuadrado "
                  "naranja = no). Repartimos las fichas por Pronóstico y por Temperatura. Pida "
                  "que comparen: ¿qué reparto les deja más seguros de la respuesta? Nublado es "
                  "PURO (4 de 4). La temperatura deja grupos tan mezclados como al principio.")
        tb, side = self._tb, self._side
        chips = VGroup(*[day_chip(r) for r in DAYS]).arrange(RIGHT, buff=0.12).move_to([0, 1.55, 0])
        self.play(FadeOut(side),
                  *[ReplacementTransform(tb[i + 1], chips[i]) for i in range(len(DAYS))],
                  FadeOut(tb[0]), FadeOut(tb[-1]), run_time=1.3)
        groups = VGroup()
        for attr, cx in [("Pronóstico", -3.45), ("Temperatura", 3.45)]:
            vals = VALUES[attr]
            bk = VGroup(*[Bucket(v, width=1.85, height=1.35, title_size=19) for v in vals])
            bk.arrange(RIGHT, buff=0.18).move_to([cx, -0.75, 0])
            lab = safe_text(f"por {attr}", size=24, color=YELLOW, weight=BOLD
                            ).next_to(bk, UP, buff=0.55)
            groups.add(VGroup(lab, bk))
        self.play(FadeIn(groups))
        moves, verdicts = [], VGroup()
        for gi, attr in enumerate(["Pronóstico", "Temperatura"]):
            bk = groups[gi][1]
            parts = partition(DAYS, attr)
            for vi, v in enumerate(VALUES[attr]):
                rows = parts[v]
                slots = bk[vi].slots(len(rows), cols=3)
                for r, pos in zip(rows, slots):
                    src = chips[DAYS.index(r)]
                    c = src.copy()
                    moves.append(c.animate.move_to(pos))
                    self.add(c)
                sy, sn = yes_no(rows)
                h = entropy((sy, sn))
                txt = "puro" if h == 0 else f"{sy} sí · {sn} no"
                verdicts.add(safe_text(txt, size=18, color=GREEN if h == 0 else MUTED,
                                       weight=BOLD if h == 0 else NORMAL)
                             .next_to(bk[vi], DOWN, buff=0.12))
        self.play(*moves, run_time=1.6)
        self.play(FadeOut(chips), FadeIn(verdicts))
        nub = groups[0][1][1]
        self.play(nub.box.animate.set_stroke(GREEN, 4))
        ask = safe_text("Necesitamos un número que mida qué tan mezclado está un grupo",
                        size=24, color=YELLOW).move_to([0, -2.85, 0])
        self.play(FadeIn(ask, shift=UP * 0.1))

    # ------------------------------------------------------------------
    def entropy_meter(self) -> None:
        self.beat("13–15 min. Un medidor de mezcla. Movemos la composición de una cubeta de 10 "
                  "fichas: con todas iguales la incertidumbre es 0; con mitad y mitad es "
                  "máxima (1 bit). Pregunte antes de mover: ¿qué forma tendrá la curva? "
                  "Interpretación: bits = preguntas de sí/no que faltan, en promedio, para "
                  "conocer la clase.")
        self.start("Entropía", "Un medidor de qué tan mezclado está un grupo", PURPLE)
        ax = impurity_axes(width=5.6, height=3.5, center=(2.9, -0.15, 0))
        nums = axis_numbers(ax, [0, 0.5, 1], [0.5, 1])
        xl = safe_text("p = proporción de «no»", size=19, color=MUTED).next_to(ax, DOWN,
                                                                             buff=0.42)
        yl = safe_text("H (bits)", size=19, color=PURPLE).next_to(ax, UP, buff=0.1
                                                                  ).align_to(ax, LEFT)
        curve = ax.plot(H2, x_range=[0, 1, 0.005], color=PURPLE, stroke_width=4)
        p = ValueTracker(0.0)
        box = Bucket("una cubeta de 10 días", width=3.4, height=1.75, title_size=20
                     ).move_to([-3.6, 0.55, 0])

        def fill():
            k = int(round(p.get_value() * 10))
            ms = VGroup(*[marker(1 if i < k else 0, r=0.2) for i in range(10)])
            ms.arrange_in_grid(rows=2, cols=5, buff=0.16).move_to(box.box)
            return ms

        content = always_redraw(fill)
        dot = always_redraw(lambda: Dot(ax.c2p(p.get_value(), H2(p.get_value())), radius=0.09,
                                        color=YELLOW))
        read = always_redraw(lambda: VGroup(
            safe_text(f"p = {p.get_value():.1f}", size=26, color=INK),
            safe_text(f"H = {H2(p.get_value()):.3f} bits", size=26, color=PURPLE, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-3.6, -1.25, 0]))
        self.play(FadeIn(box), FadeIn(content), Create(ax), FadeIn(nums), FadeIn(xl),
                  FadeIn(yl))
        self.add(dot, read)
        trace = TracedPath(dot.get_center, stroke_color=PURPLE, stroke_width=4)
        self.add(trace)
        self.play(p.animate.set_value(0.5), run_time=2.2, rate_func=linear)
        self.play(p.animate.set_value(1.0), run_time=2.2, rate_func=linear)
        self.wait(0.3)   # ≥ 3 fotogramas: una pausa de 1 fotograma rompe la concatenación
        self.remove(trace)
        self.add(curve)
        self._meter = dict(p=p, ax=ax, dot=dot, read=read, content=content)

        self.beat("15–16 min. La fórmula. Con dos clases y p = 6/15 = 0.4 de «no», los quince "
                  "días tienen 0.971 bits: casi la mezcla máxima. Pida que lo verifiquen con "
                  "calculadora (log base 2). Recalque la convención 0·log 0 = 0.")
        self.play(p.animate.set_value(0.4), run_time=1.2)
        f = tex(r"H(S) = -\sum_{k} p_k \log_2 p_k", size=40, color=INK).move_to([-3.6, -2.45, 0])
        calc = tex(r"H = -\tfrac{9}{15}\log_2\tfrac{9}{15} - \tfrac{6}{15}\log_2\tfrac{6}{15}"
                   r" = 0.971", size=32, color=PURPLE)
        calc.move_to([2.9, -2.95, 0])
        guide = DashedLine(ax.c2p(0.4, 0), ax.c2p(0.4, H2(0.4)), color=YELLOW, stroke_width=2)
        self.play(Write(f))
        self.play(Create(guide), Write(calc))

    # ------------------------------------------------------------------
    def weighted_area(self) -> None:
        self.beat("16–18 min. Después de preguntar por el pronóstico hay tres grupos, cada uno "
                  "con su entropía. El promedio debe pesar por tamaño: dibujamos cada grupo como "
                  "una barra de ANCHO = proporción de días y ALTO = entropía. El área total es la "
                  "entropía después de preguntar (0.691). Lo que bajó respecto al marco punteado "
                  "(0.971) es la GANANCIA: el área amarilla, 0.280 bits.")
        self.start("Ganancia de información", "¿Cuánta mezcla quita una pregunta?", PURPLE)
        parts = partition(DAYS, "Pronóstico")
        vals = VALUES["Pronóstico"]
        groups = [(len(parts[v]), entropy(yes_no(parts[v]))) for v in vals]
        total_w, unit_h, ox, oy = 6.4, 2.55, -6.6, -2.5
        frame = DashedVMobject(Rectangle(width=total_w, height=unit_h * H_ROOT, color=MUTED,
                                         stroke_width=2), num_dashes=60)
        frame.move_to([ox + total_w / 2, oy + unit_h * H_ROOT / 2, 0])
        flab = safe_text(f"antes: H = {H_ROOT:.3f}", size=19, color=MUTED
                         ).next_to(frame, UP, buff=0.08).align_to(frame, LEFT)
        bars = area_bars(groups, total_w=total_w, unit_h=unit_h, origin=(ox, oy, 0),
                         colors=[PURPLE] * 3, labels=vals, label_size=19)
        chips = VGroup()
        for v, b in zip(vals, bars):
            row = VGroup(*[day_chip(r, r=0.13) for r in parts[v]]).arrange_in_grid(
                cols=3, buff=0.06)
            row.next_to(frame, UP, buff=0.55).set_x(b[0].get_x())
            chips.add(row)
        self.play(Create(frame), FadeIn(flab))
        self.play(LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.2))
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(VGroup(*b[1:])) for b in bars], lag_ratio=0.25))
        gaps = VGroup()
        for b, (nv, hv) in zip(bars, groups):
            top = oy + unit_h * H_ROOT
            bot = oy + unit_h * hv
            if top - bot > 0.01:
                gaps.add(Rectangle(width=b[0].width, height=top - bot, stroke_width=0,
                                   fill_color=YELLOW, fill_opacity=0.35)
                         .move_to([b[0].get_x(), (top + bot) / 2, 0]))
        eq1 = tex(r"H_{\text{desp}} = \tfrac{6}{15}(0.918) + \tfrac{4}{15}(0) + \tfrac{5}{15}(0.971)",
                  size=30)
        eq2 = tex(r"= 0.367 + 0 + 0.324 = 0.691", size=30)
        eq3 = tex(r"\text{Gan} = 0.971 - 0.691 = 0.280 \text{ bits}", size=34, color=YELLOW)
        eqs = VGroup(eq1, eq2, eq3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        eqs.move_to([3.55, -0.6, 0])
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.play(FadeIn(gaps), Write(eq3))

    # ------------------------------------------------------------------
    def gain_race(self) -> None:
        self.beat("18–19 min. Repetimos el cálculo para los cuatro atributos. Antes de mostrar "
                  "las barras, pida que ordenen los atributos de mayor a menor ganancia. "
                  "ID3 elige el máximo: Pronóstico (0.280). La temperatura casi no informa "
                  "(0.013), como sugería la intuición.")
        self.start("ID3 elige la mayor ganancia", "Las cuatro preguntas candidatas", PURPLE)
        order = sorted(ATTRS, key=lambda a: -ROOT_GAINS[a])
        scale = 16.0
        rows = VGroup()
        for i, a in enumerate(order):
            g = ROOT_GAINS[a]
            lab = safe_text(a, size=26, color=INK, weight=BOLD).move_to([-4.4, 1.3 - i * 1.0, 0])
            lab.align_to([-3.4, 0, 0], RIGHT)
            bar = Rectangle(width=max(g * scale, 0.04), height=0.55, stroke_width=0,
                            fill_color=YELLOW if i == 0 else PURPLE, fill_opacity=0.8)
            bar.next_to([-3.15, lab.get_y(), 0], RIGHT, buff=0)
            val = safe_text(f"{g:.3f}", size=24, color=INK).next_to(bar, RIGHT, buff=0.15)
            after = safe_text(f"H después {H_ROOT - g:.3f}", size=18, color=MUTED
                              ).next_to(val, RIGHT, buff=0.4)
            rows.add(VGroup(lab, bar, val, after))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(VGroup(r[2], r[3])) for r in rows], lag_ratio=0.15))
        win = SurroundingRectangle(rows[0], color=YELLOW, buff=0.12)
        rule = safe_text("A* = argmax Gan(S, A)  →  raíz: Pronóstico", size=26, color=YELLOW,
                         weight=BOLD).move_to([0, -2.6, 0])
        self.play(Create(win), FadeIn(rule))

    # ------------------------------------------------------------------
    def recurse(self) -> None:
        self.beat("19–21 min. Recursión. Nublado es puro: hoja «Sí». En Soleado y en Lluvia "
                  "repetimos TODO el procedimiento sólo con sus días (y sin volver a usar "
                  "Pronóstico). Pregunte qué atributo esperan en cada rama antes de mostrar las "
                  "tablas. Cuando la ganancia iguala la entropía de la rama, los hijos son puros.")
        self.start("Repetir en cada rama", "El mismo cálculo, sólo con los días de la rama", PURPLE)
        tv = TreeView(id3_spec(), width=7.4, level_h=1.55, top=(-2.4, 1.75, 0), node_w=2.0,
                      leaf_w=1.15, label_size=17)
        self.play(FadeIn(tv.leaf[""]))
        self.play(*tv.grow(""))
        self.play(Indicate(tv.leaf["1"], color=GREEN))
        tables = VGroup()
        for branch, color in [("Soleado", YELLOW), ("Lluvia", YELLOW)]:
            gains = BRANCH_GAINS[branch]
            n = len(partition(DAYS, "Pronóstico")[branch])
            sy, sn = yes_no(partition(DAYS, "Pronóstico")[branch])
            head = safe_text(f"Rama {branch}: {n} días ({sy} sí, {sn} no), "
                             f"H = {entropy((sy, sn)):.3f}", size=19, color=color, weight=BOLD)
            items = VGroup(*[
                safe_text(f"Gan({a}) = {g:.3f}", size=19,
                          color=INK if g == max(gains.values()) else MUTED,
                          weight=BOLD if g == max(gains.values()) else NORMAL)
                for a, g in sorted(gains.items(), key=lambda kv: -kv[1])
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            tables.add(VGroup(head, items).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        tables.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([4.15, 0.2, 0])
        self.play(FadeIn(tables[0]))
        self.play(*tv.grow("0"))
        self.play(FadeIn(tables[1]))
        self.play(*tv.grow("2"))
        done = safe_text("5 hojas puras · 15 de 15 bien clasificados", size=22, color=GREEN,
                         weight=BOLD).move_to([0, -3.1, 0])
        self.play(FadeIn(done))
        self._id3 = dict(tv=tv, tables=tables, done=done)

    # ------------------------------------------------------------------
    def rules(self) -> None:
        self.beat("21–22 min. Cada camino de la raíz a una hoja es una regla SI… ENTONCES. "
                  "El árbol usa 3 de 4 atributos: la temperatura nunca fue útil. Pregunte: "
                  "¿qué regla le darían a un entrenador? Conexión: son casi las reglas que "
                  "encontró el clasificador asociativo (CBA) en la unidad de patrones.")
        a = self._id3
        tv = a["tv"]
        texts = [
            ("SI Nublado ⇒ Sí", "1"),
            ("SI Soleado Y Humedad Alta ⇒ No", "00"),
            ("SI Soleado Y Humedad Normal ⇒ Sí", "01"),
            ("SI Lluvia Y Viento Débil ⇒ Sí", "20"),
            ("SI Lluvia Y Viento Fuerte ⇒ No", "21"),
        ]
        rules = VGroup(*[safe_text(f"{i + 1}. {t}", size=19, color=INK, max_width=5.1)
                         for i, (t, _) in enumerate(texts)]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        rules.move_to([0, 0.3, 0]).align_to([1.75, 0, 0], LEFT)
        self.play(FadeOut(a["tables"]))
        for r, (_, path) in zip(rules, texts):
            hl = VGroup(*[SurroundingRectangle(tv.node(p), color=YELLOW, buff=0.05)
                          for p in tv.path_to(path)])
            self.play(FadeIn(r, shift=LEFT * 0.1), Create(hl), run_time=0.5)
            self.play(FadeOut(hl), run_time=0.25)
        temp = VGroup(safe_text("La temperatura no aparece:", size=19, color=MUTED),
                      safe_text("nunca fue la mejor pregunta.", size=19, color=MUTED)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        temp.next_to(rules, DOWN, buff=0.35).align_to(rules, LEFT)
        self.play(FadeIn(temp))

    # ------------------------------------------------------------------
    def hunt(self) -> None:
        self.beat("22–23 min. Formalización. El esquema recursivo de Hunt: si el nodo es puro, "
                  "hoja; si no quedan atributos o se cumple un criterio de parada, hoja con la "
                  "mayoría; si no, elegir la mejor prueba, dividir y repetir. Es VORAZ: elige la "
                  "mejor pregunta local sin volver atrás (el árbol óptimo global es NP-difícil). "
                  "La ganancia es la información mutua entre el atributo y la clase.")
        self.start("Formalización", "Algoritmo de Hunt e ID3", PURPLE)
        steps = VGroup(
            safe_text("Hunt(D_t, A):", size=24, color=TEAL, weight=BOLD, font="Menlo"),
            safe_text("1. si D_t es puro → hoja con su clase", size=21),
            safe_text("2. si A = ∅ o hay parada → hoja con la mayoría", size=21),
            safe_text("3. si no: mejor prueba, dividir D_t", size=21),
            safe_text("    y repetir Hunt en cada hijo", size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        steps[-1].shift(RIGHT * 0.3)
        box = card(steps.width + 0.6, steps.height + 0.5, TEAL)
        g1 = VGroup(box, steps.move_to(box))
        g1.move_to([0, 0.85, 0]).align_to([-6.6, 0, 0], LEFT)
        greedy = safe_text("Voraz: la mejor pregunta LOCAL, sin volver atrás.", size=21,
                           color=YELLOW).next_to(g1, DOWN, buff=0.35).align_to(g1, LEFT)
        f1 = tex(r"\text{Gan}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|}\, H(S_v)", size=33)
        f2 = tex(r"= H(Y) - H(Y \mid A) = I(Y; A)", size=33, color=PURPLE)
        f3 = tex(r"A^{*} = \arg\max_{A} \text{Gan}(S, A)", size=33, color=YELLOW)
        fs = VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        fs.move_to([0, -1.6, 0]).align_to([6.7, 0, 0], RIGHT)
        self.play(FadeIn(g1))
        self.play(FadeIn(greedy))
        self.play(Write(f1))
        self.play(Write(f2))
        self.play(Write(f3))
