"""Acto 2 · Matriz de confusión (10–22 min).

Los clientes viajan del plano a las cuatro celdas; exactitud, precisión y
sensibilidad se leen como diagonal, columna y fila antes de escribir fórmulas.
Mover el corte muestra el compromiso precisión–sensibilidad; cien clientes con
tres morosos muestran por qué la exactitud sola engaña.
"""
from __future__ import annotations

from manim import *

from sections.confmat import CELL_NAME, ConfMatrix, metric_bars
from sections.data import (LAZY_N, LAZY_POS, STUMP_T, TRAIN, XTR, YTR, cell_of, confusion,
                           metrics, stump_predict, threshold_table)
from sections.plane import ClientPlane
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, YELLOW, card,
                            legend_item, marker, pill, safe_text, tex)

POS_NOTE = "positivo = «no paga»: la alarma que queremos encender"


class ConfusionSection:
    def _plane_with_rule(self, t: float):
        plane = ClientPlane()
        pts = plane.clients()
        regions = plane.stump_regions(t)
        line = plane.hline(t)
        legend = VGroup(legend_item(0, "pagó"), legend_item(1, "no pagó")
                        ).arrange(RIGHT, buff=0.4).next_to(plane.axes, UP, buff=0.12)
        return plane, pts, regions, line, legend

    def _fly(self, cm: ConfMatrix, pts: VGroup, t: float, keep_source: bool = True):
        """Animaciones que llevan una copia de cada cliente a su celda."""
        pred = stump_predict(XTR[:, 1], t)
        used = {k: 0 for k in cm.slots_used}
        tokens = VGroup()
        anims = []
        for i, p in enumerate(pts):
            code = cell_of(int(YTR[i]), int(pred[i]))
            target = cm.slot(code, used[code])
            used[code] += 1
            tok = p[0].copy()
            tok.code = code
            tokens.add(tok)
            anims.append(tok.animate.scale(0.75).move_to(target))
        return tokens, anims, used

    def confusion(self) -> None:
        self.beat("10–11 min. Volvemos a la regla «deuda > 44 ⇒ no paga». Fije la convención: "
                  "la clase POSITIVA es la que queremos detectar (no paga). «Positivo» no es "
                  "bueno ni malo: es encender la alarma. Pregunte: ¿de cuántas maneras puede "
                  "equivocarse la alarma?")
        head = self.start("¿Cómo medimos errores?", "Dos maneras de equivocarse", RED)
        plane, pts, regions, line, legend = self._plane_with_rule(STUMP_T)
        self.play(FadeIn(plane), FadeIn(regions), Create(line), FadeIn(legend))
        self.play(LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.05))
        pos = pill(POS_NOTE, ORANGE, size=18, width=6.4).move_to([3.4, 2.0, 0])
        self.play(FadeIn(pos))
        ways = VGroup(
            VGroup(safe_text("Falsa alarma", size=28, color=RED, weight=BOLD),
                   safe_text("dijimos «no paga» y sí pagó", size=22, color=MUTED),
                   safe_text("→ perdimos un buen cliente", size=22)),
            VGroup(safe_text("Alarma que no sonó", size=28, color=RED, weight=BOLD),
                   safe_text("dijimos «paga» y no pagó", size=22, color=MUTED),
                   safe_text("→ perdimos el dinero prestado", size=22)),
        )
        for w in ways:
            w.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        ways.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([3.3, 0.0, 0])
        self.play(FadeIn(ways[0], shift=LEFT * 0.1))
        self.play(Indicate(pts[5], color=RED, scale_factor=1.6))
        self.play(FadeIn(ways[1], shift=LEFT * 0.1))
        q = safe_text("¿Cuál le duele más al banco?", size=26, color=YELLOW, weight=BOLD)
        q.move_to([3.3, -2.3, 0])
        self.play(FadeIn(q))

        self.beat("11–13 min. Construya la matriz moviendo a cada cliente. Dos preguntas por "
                  "cliente: ¿qué era en realidad? (fila) ¿qué dijo la regla? (columna). Pida "
                  "a la clase que diga la celda del cliente 6 ANTES de que llegue (falso "
                  "positivo).")
        self.play(FadeOut(VGroup(ways, q, pos)))
        cm = ConfMatrix(center=(3.6, 0.0, 0), cw=1.85, ch=1.3)
        self.play(FadeIn(cm))
        tokens, anims, used = self._fly(cm, pts, STUMP_T)
        self.play(LaggedStart(*anims, lag_ratio=0.18), run_time=3.2)
        counts = VGroup(*[cm.count_mob(k, used[k]) for k in ("TP", "FN", "FP", "TN")])
        cm.add(cm.code_labels)
        self.play(FadeIn(counts), FadeIn(cm.code_labels))

        self.beat("13–14 min. Lectura de los nombres: la SEGUNDA palabra es lo que dijo el "
                  "modelo (positivo/negativo); la PRIMERA, si acertó (verdadero/falso). Así "
                  "«falso positivo» = encendió la alarma y se equivocó. La diagonal verde son "
                  "los aciertos.")
        names = VGroup()
        for code in ("TP", "FN", "FP", "TN"):
            box = cm.cells[code]
            n = safe_text(CELL_NAME[code], size=15, color=GREEN if code in ("TP", "TN") else RED)
            n.next_to(box, DOWN if code in ("FP", "TN") else UP, buff=0.06)
            names.add(n)
        self.play(FadeOut(cm.headers[0]), FadeOut(cm.headers[1]), FadeIn(names[0]),
                  FadeIn(names[1]))
        self.play(FadeIn(names[2]), FadeIn(names[3]))
        diag = VGroup(cm.highlight(["TP"], GREEN), cm.highlight(["TN"], GREEN))
        self.play(Create(diag))
        acc_t = tex(r"\text{exactitud} = \frac{TP + TN}{n} = \frac{4 + 5}{10} = 0.90",
                    size=30, color=GREEN, max_width=6.0).move_to([3.6, -2.4, 0])
        self.play(Write(acc_t))
        self.play(FadeOut(VGroup(names[0], names[1])), FadeIn(cm.headers[0]),
                  FadeIn(cm.headers[1]))
        self.play(FadeOut(names[2:]), run_time=0.3)

        matrix = VGroup(cm, tokens, counts)
        self.remove(cm, cm.boxes, cm.headers, cm.code_labels, tokens, counts)
        self.add(matrix)
        self.play(FadeOut(VGroup(plane, pts, regions, line, legend, acc_t, diag)),
                  matrix.animate.move_to([-3.65, 0.2, 0]))
        head = self.swap_heading(head, "¿Cómo medimos errores?", "Una pregunta por métrica", RED)
        views = [
            ("Exactitud", "¿Cuántos aciertos hubo en total?", ["TP", "TN"], GREEN,
             r"\frac{TP + TN}{n} = \frac{9}{10} = 0.90",
             "9 aciertos entre 10 clientes",
             "La exactitud cuenta los aciertos sobre todos los clientes. Resaltar sólo la diagonal."),
            ("Precisión", "Cuando suena la alarma, ¿acierta?", ["TP", "FP"], PURPLE,
             r"\frac{TP}{TP + FP} = \frac{4}{5} = 0.80",
             "4 alarmas correctas de 5",
             "La precisión mira la columna de las alarmas: cuatro correctas entre cinco."),
            ("Sensibilidad", "De los impagos, ¿cuántos detecta?", ["TP", "FN"], YELLOW,
             r"\frac{TP}{TP + FN} = \frac{4}{4} = 1.00",
             "Detectamos los 4 impagos",
             "La sensibilidad mira la fila de los impagos: detectamos cuatro de cuatro. "
             "Conectar con la alarma del inicio, que no detectaba ninguno."),
        ]
        current = highlight = None
        for name, question, codes, color, formula, meaning, notes in views:
            self.beat(notes)
            new = VGroup(
                safe_text(name, size=38, color=color, weight=BOLD),
                safe_text(question, size=25, max_width=6.3),
                tex(formula, size=44, max_width=6.2),
                safe_text(meaning, size=27, color=color, max_width=6.2),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([3.2, 0.15, 0])
            hi = VGroup(*[cm.highlight([code], color) for code in codes])
            if current is not None:
                self.play(FadeOut(current), FadeOut(highlight), run_time=0.3)
            self.play(FadeIn(new), Create(hi), run_time=0.7)
            current, highlight = new, hi
        if not self.video_mode:
            self.beat("F1 resume precisión y sensibilidad con la media armónica. "
                      "No incluye verdaderos negativos ni sustituye el costo de los errores.")
            f1 = VGroup(safe_text("F1: media armónica de P y R", size=28),
                        tex(r"F_1 = \frac{2PR}{P+R} \approx 0.89", size=44))
            f1.arrange(DOWN, buff=0.4).move_to(current)
            self.play(FadeOut(current), FadeOut(highlight), FadeIn(f1))
        self.threshold_tradeoff(head)

    # ------------------------------------------------------------------
    def threshold_tradeoff(self, head) -> None:
        self.beat("17–19 min. Si movemos el corte, los clientes cambian de celda. Prediga "
                  "con la clase: si EXIGIMOS más deuda para encender la alarma (subir el "
                  "corte), sonarán menos alarmas. La sensibilidad no aumenta; la precisión "
                  "depende de cuáles se retiren y no tiene por qué ser monótona. Avance corte por "
                  "corte y deje que lean las barras.")
        self.clear()
        head = self.heading("¿Cómo medimos errores?", "Mover el corte: nada es gratis", RED)
        self.add(head)
        rows = threshold_table()
        t0 = rows[0][0]
        plane, pts, regions, line, legend = self._plane_with_rule(t0)
        cm = ConfMatrix(center=(3.7, 1.05, 0), cw=1.75, ch=0.95, header_size=17, codes=False)
        self.play(FadeIn(plane), FadeIn(regions), FadeIn(line), FadeIn(legend), FadeIn(pts),
                  FadeIn(cm))
        tokens, anims, used = self._fly(cm, pts, t0)
        self.play(LaggedStart(*anims, lag_ratio=0.05), run_time=1.2)
        counts = VGroup(*[cm.count_mob(k, used[k], size=28) for k in ("TP", "FN", "FP", "TN")])
        self.play(FadeIn(counts))
        m0 = rows[0][2]
        bars, tr = metric_bars({"exactitud": (m0["acc"], GREEN), "precisión": (m0["prec"], PURPLE),
                                "sensibilidad": (m0["rec"], YELLOW)},
                               origin=(0.6, -1.15, 0), length=3.3)
        t_lab = safe_text(f"corte: {t0:.1f} %", size=24, color=YELLOW, weight=BOLD)
        t_lab.move_to([3.6, -2.75, 0])
        self.play(FadeIn(bars), FadeIn(t_lab))

        notes = {
            44.0: "18–19 min. Subimos a 44: el cliente 5 (deuda 40) sale de la alarma; una "
                  "falsa alarma menos. Sube la precisión, la sensibilidad sigue en 1.",
            51.5: "19–20 min. En 51.5 el cliente 10 (deuda 48) se escapa: aparece un falso "
                  "negativo y la sensibilidad baja a 0.75.",
            59.0: "20 min. En 59 ya no hay falsas alarmas (precisión 1) pero sólo detectamos "
                  "a la mitad de los morosos. Moraleja: el negocio elige el corte según "
                  "cuánto cuesta cada error.",
        }
        for t, cmv, m in rows[1:]:
            self.beat(notes[t])
            targets = self._targets(cm, t)
            new_counts = VGroup(*[cm.count_mob(k, cmv[k], size=28)
                                  for k in ("TP", "FN", "FP", "TN")])
            self.play(Transform(regions, plane.stump_regions(t)), Transform(line, plane.hline(t)),
                      Transform(t_lab, safe_text(f"corte: {t:.1f} %", size=24, color=YELLOW,
                                                 weight=BOLD).move_to(t_lab)), run_time=0.8)
            self.play(*[tok.animate.move_to(pos) for tok, pos in zip(tokens, targets)],
                      Transform(counts, new_counts),
                      tr["exactitud"].animate.set_value(m["acc"]),
                      tr["precisión"].animate.set_value(m["prec"]),
                      tr["sensibilidad"].animate.set_value(m["rec"]), run_time=1.2)
        trade = pill("subir el corte: menos alarmas; pueden escapar impagos", YELLOW,
                     size=18, width=6.4).move_to([3.4, -0.5, 0])
        self.play(FadeIn(trade))
        if not self.video_mode:
            self.accuracy_paradox()

    def _targets(self, cm: ConfMatrix, t: float):
        pred = stump_predict(XTR[:, 1], t)
        used = {}
        out = []
        for i in range(len(TRAIN)):
            code = cell_of(int(YTR[i]), int(pred[i]))
            k = used.get(code, 0)
            used[code] = k + 1
            out.append(cm.slot(code, k))
        return out

    def accuracy_paradox(self) -> None:
        self.beat("20–21 min. Cien clientes; tres no pagan (naranja). Un modelo perezoso dice "
                  "«todos pagan». Pida que calculen su exactitud ANTES de revelarla: 97 %. "
                  "Luego su sensibilidad: 0. Con clases desbalanceadas la exactitud engaña: "
                  "mire siempre la matriz completa.")
        self.start("¿Cómo medimos errores?", "La exactitud puede mentir", RED)
        dots = VGroup()
        pos_idx = {17, 54, 86}
        for i in range(LAZY_N):
            dots.add(marker(1 if i in pos_idx else 0, r=0.12))
        dots.arrange_in_grid(rows=10, cols=10, buff=0.16).move_to([-3.4, -0.35, 0])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.005),
                  run_time=1.2)
        lazy = VGroup(
            safe_text("Modelo perezoso:", size=28, weight=BOLD),
            safe_text("«todos pagan»", size=28, color=BLUE, weight=BOLD),
        ).arrange(RIGHT, buff=0.2).move_to([3.3, 1.7, 0])
        self.play(FadeIn(lazy))
        cmv = {"TP": 0, "FN": LAZY_POS, "FP": 0, "TN": LAZY_N - LAZY_POS}
        m = metrics(cmv)
        res = VGroup(
            tex(rf"\text{{exactitud}} = \frac{{{LAZY_N - LAZY_POS}}}{{{LAZY_N}}} = "
                rf"{m['acc']:.2f}", size=38, color=GREEN),
            tex(rf"\text{{sensibilidad}} = \frac{{0}}{{{LAZY_POS}}} = 0", size=38, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.3, 0.1, 0])
        self.play(Write(res[0]))
        rings = VGroup(*[Circle(radius=0.22, color=RED, stroke_width=4).move_to(dots[i])
                         for i in sorted(pos_idx)])
        self.play(Write(res[1]), Create(rings))
        moral = pill("Con clases desbalanceadas, mira la matriz completa", YELLOW, size=20,
                     width=6.2)
        moral.move_to([3.3, -1.7, 0])
        self.play(FadeIn(moral, shift=UP * 0.1))

        self.beat("21–22 min. Tu turno (1 min, individual). Un detector de fraude: TP = 8, "
                  "FN = 2, FP = 4, TN = 86. Calculen exactitud, precisión y sensibilidad. "
                  "Respuesta: 0.94, 0.67, 0.80. Pregunte qué número le importaría al banco.")
        self.start("Tu turno", "Lee la matriz", YELLOW)
        cm = ConfMatrix(center=(-2.6, -0.2, 0), cw=2.1, ch=1.45, header_size=20)
        vals = {"TP": 8, "FN": 2, "FP": 4, "TN": 86}
        nums = VGroup(*[safe_text(str(v), size=44, color=INK, weight=BOLD).move_to(cm.cells[k])
                        for k, v in vals.items()])
        hdr = VGroup(safe_text("fraude", size=20, color=ORANGE, weight=BOLD),
                     safe_text("legítima", size=20, color=BLUE, weight=BOLD))
        hdr2 = hdr.copy()
        hdr[0].move_to(cm.headers[0][0])
        hdr[1].move_to(cm.headers[0][1])
        hdr2[0].move_to(cm.headers[2][0], aligned_edge=RIGHT)
        hdr2[1].move_to(cm.headers[2][1], aligned_edge=RIGHT)
        frame = VGroup(cm.boxes, cm.headers[1], cm.headers[3], hdr, hdr2, cm.code_labels)
        self.play(FadeIn(frame), FadeIn(nums))
        qs = VGroup(*[safe_text(t, size=28) for t in
                      ("exactitud = ?", "precisión = ?", "sensibilidad = ?")]
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.4, 0.3, 0])
        self.play(LaggedStart(*[FadeIn(q) for q in qs], lag_ratio=0.2))
        timer = safe_text("1 min", size=22, color=MUTED).next_to(qs, DOWN, buff=0.5)
        self.play(FadeIn(timer))

        self.beat("Solución. La exactitud alta (0.94) esconde que 1 de cada 3 alarmas es "
                  "falsa y que se escapan 2 de 10 fraudes.")
        ans = VGroup(
            tex(r"\text{exactitud} = \frac{8 + 86}{100} = 0.94", size=32, color=GREEN),
            tex(r"\text{precisión} = \frac{8}{8 + 4} \approx 0.67", size=32, color=PURPLE),
            tex(r"\text{sensibilidad} = \frac{8}{8 + 2} = 0.80", size=32, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(qs, aligned_edge=LEFT)
        self.play(FadeOut(timer), *[ReplacementTransform(q, a) for q, a in zip(qs, ans)])
