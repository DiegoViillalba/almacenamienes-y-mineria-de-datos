"""Acto 5 · Comparación y ejercicio (52–60 min).

Los cinco clientes de prueba pasan por 3-NN y por la regresión logística; sus
matrices de confusión se comparan lado a lado y se discute qué error cuesta
más. Cierra con un ejercicio a mano y un ticket de salida.
"""
from __future__ import annotations

from math import exp

from manim import *

from sections.confmat import ConfMatrix
from sections.data import (EX_B, EX_CLIENT, EX_W1, EX_W2, K_FINAL, TEST, TRAIN, YTE, YTR,
                           confusion, ex_score, knn_order, metrics, test_predictions)
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, marker, pill, safe_text, table, tex)


def small_matrix(cmv: dict, center, title: str, color) -> VGroup:
    cm = ConfMatrix(center=center, cw=1.45, ch=0.95, header_size=16, codes=True)
    nums = VGroup(*[cm.count_mob(k, cmv[k], size=34).move_to(cm.cells[k].get_center()
                                                                + DOWN * 0.08 + RIGHT * 0.15)
                    for k in ("TP", "FN", "FP", "TN")])
    t = safe_text(title, size=24, color=color, weight=BOLD).next_to(cm.headers[1], UP, buff=0.15)
    return VGroup(cm, cm.code_labels, nums, t)


class CompareSection:
    def compare(self) -> None:
        knn, p, lr = test_predictions()
        self.beat("52–54 min. El examen real: cinco clientes que ningún modelo vio. Pida que "
                  "predigan en voz alta qué modelo ganará. Recorra fila por fila: real, 3-NN "
                  "y la probabilidad de la logística. Verde = acierto, rojo = error.")
        head = self.start("Comparar y practicar", "Cinco clientes nuevos", YELLOW)
        rows = [["cliente", "ingreso", "deuda", "real", "3-NN", "logística (p)"]]
        cols = []
        for (cid, x, y, c), kn, pp, ll in zip(TEST, knn, p, lr):
            rows.append([cid, str(x), str(y), "no paga" if c else "paga",
                         ("no paga" if kn else "paga") + (" ✓" if kn == c else " ✗"),
                         f"{pp:.2f}  " + ("no paga" if ll else "paga") + (" ✓" if ll == c else " ✗")])
            cols.append([YELLOW, INK, INK, CLASS_COLOR[c], GREEN if kn == c else RED,
                         GREEN if ll == c else RED])
        tbl = table(rows, col_widths=[1.4, 1.3, 1.2, 1.6, 2.1, 3.0], size=24, row_h=0.6,
                    colors=cols).move_to([0, 0.15, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        for r in tbl[1:-1]:
            self.play(FadeIn(r[:4], shift=RIGHT * 0.1), run_time=0.4)
            self.play(FadeIn(r[4:]), run_time=0.5)
        score = VGroup(pill(f"3-NN: {int((knn == YTE).sum())}/5", GREEN, size=24),
                       pill(f"logística: {int((lr == YTE).sum())}/5", PURPLE, size=24)
                       ).arrange(RIGHT, buff=0.6).move_to([0, -2.4, 0])
        self.play(FadeIn(score, shift=UP * 0.1))

        self.beat("54–55 min. Las matrices, con positivo = no paga. Ambos dejan escapar al "
                  "cliente C (falso negativo). 3-NN además da una falsa alarma con E. La "
                  "logística tiene precisión perfecta pero misma sensibilidad. Pregunte: "
                  "¿qué error le cuesta más al banco y qué harían?")
        self.clear()
        head = self.heading("Comparar y practicar", "¿En qué se equivoca cada uno?", YELLOW)
        self.add(head)
        cm_k = confusion(YTE, knn)
        cm_l = confusion(YTE, lr)
        mk, ml = metrics(cm_k), metrics(cm_l)
        A = small_matrix(cm_k, (-3.5, 0.6, 0), "3-NN", GREEN)
        B = small_matrix(cm_l, (3.6, 0.6, 0), "Regresión logística", PURPLE)
        self.play(FadeIn(A), FadeIn(B))

        def mrow(m, x):
            return VGroup(*[safe_text(f"{n} {m[k]:.2f}", size=22, color=c) for n, k, c in [
                ("exactitud", "acc", GREEN), ("precisión", "prec", PURPLE),
                ("sensibilidad", "rec", YELLOW)]]).arrange(DOWN, aligned_edge=LEFT, buff=0.12
                                                           ).move_to([x, -1.55, 0])

        row_k, row_l = mrow(mk, -3.0), mrow(ml, 4.1)
        self.play(FadeIn(row_k), FadeIn(row_l))

        self.beat("55–56 min. Conecte con el acto 2: la logística da probabilidades, así que "
                  "podemos MOVER EL UMBRAL. Si el banco teme más al impago, baja el umbral a "
                  "0.2: C (p = 0.27) se detecta, pero E (p = 0.25) se vuelve falsa alarma. "
                  "Sensibilidad 1, precisión 0.67. Otra vez: nada es gratis.")
        lr2 = (p >= 0.2).astype(int)
        cm_l2 = confusion(YTE, lr2)
        m2 = metrics(cm_l2)
        B2 = small_matrix(cm_l2, (3.6, 0.6, 0), "Logística, umbral 0.2", ORANGE)
        self.play(Transform(B, B2))
        self.play(Transform(row_l, mrow(m2, 4.1)))
        mid = VGroup(safe_text("umbral 0.5 → 0.2", size=24, color=ORANGE, weight=BOLD),
                     safe_text("detecta a C… y acusa a E", size=20, color=MUTED)
                     ).arrange(DOWN, buff=0.12).move_to([0.5, -1.55, 0])
        self.play(FadeIn(mid))
        warn = pill("Con 5 clientes, un solo caso mueve la exactitud 0.20", RED, size=20)
        warn.move_to([0, -2.85, 0])
        self.play(FadeIn(warn))
        self.exercise()

    def exercise(self) -> None:
        z = ex_score(*EX_CLIENT)
        pz = 1 / (1 + exp(-z))
        order, dist = knn_order(EX_CLIENT)
        self.beat("56–58 min. Ejercicio en parejas (3 min). Cliente nuevo: ingreso 20, deuda "
                  "50. (a) 3-NN con las distancias dadas. (b) Logística con pesos redondeados: "
                  "calcular z y leer p con la tabla de σ. (c) Si resulta que SÍ pagó, ¿en qué "
                  "celda de la matriz cae cada modelo? No revele.")
        self.start("Tu turno", "Clasifica a mano", YELLOW)
        dl = table([["vecino", "distancia", "clase"]] + [
            [str(TRAIN[i][0]), f"{dist[i]:.2f}", "no paga" if YTR[i] else "paga"]
            for i in order[:4]], col_widths=[1.3, 1.6, 1.6], size=22, row_h=0.46,
            colors=[[INK, INK, CLASS_COLOR[int(YTR[i])]] for i in order[:4]])
        left = VGroup(
            safe_text("Cliente: ingreso 20 · deuda 50 %", size=26, color=YELLOW, weight=BOLD),
            safe_text("(a) los 4 más cercanos:", size=22, color=MUTED),
            dl,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-3.3, 0.35, 0])
        sig_tbl = table([["z", "−2", "−1", "0", "1", "2", "2.5", "3"],
                         ["σ(z)", "0.12", "0.27", "0.50", "0.73", "0.88", "0.92", "0.95"]],
                        col_widths=[0.9] + [0.78] * 7, size=19, row_h=0.42)
        right = VGroup(
            safe_text("(b) pesos redondeados:", size=22, color=MUTED),
            tex(rf"z = {EX_B:g} {EX_W1:+.2f}\,\text{{ingreso}} {EX_W2:+.2f}\,\text{{deuda}}",
                size=30, color=PURPLE, max_width=6.2),
            sig_tbl,
            safe_text("(c) si en realidad pagó: ¿TP, FN, FP o TN?", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.3, 0.35, 0])
        self.play(FadeIn(left))
        self.play(FadeIn(right))
        timer = pill("3 min · en parejas", YELLOW, size=22).move_to([0, -2.75, 0])
        self.play(FadeIn(timer))

        self.beat("58 min. Solución. (a) 10, 7 y 8: tres «no paga» → no paga. (b) z = −8 − 2 + "
                  "12.5 = 2.5 → p ≈ 0.92 → no paga. (c) Si pagó, ambos cometen un FALSO "
                  "POSITIVO: le negaríamos el crédito a un buen cliente.")
        sol = VGroup(
            safe_text("(a) 10, 7, 8 → 3 votos «no paga»", size=22, color=ORANGE),
            tex(rf"(b)\ z = -8 - 2 + 12.5 = {z:.1f} \;\Rightarrow\; p \approx {pz:.2f}", size=30,
                color=PURPLE),
            safe_text("(c) ambos: falso positivo (FP)", size=22, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        box = card(sol.width + 0.5, sol.height + 0.4, YELLOW)
        g = VGroup(box, sol.move_to(box)).move_to([0, -2.45, 0])
        self.play(FadeOut(timer), VGroup(left, right).animate.shift(UP * 0.35))
        self.play(FadeIn(g, shift=UP * 0.1))
        self.closing()

    def closing(self) -> None:
        self.beat("58–60 min. Cierre con cuatro ideas y el ticket de salida (individual, en "
                  "papel). Próxima clase: árboles de decisión, una tercera manera de partir el "
                  "plano.")
        self.start("Cierre", "Lo que nos llevamos", YELLOW)
        ideas = VGroup(*[VGroup(safe_text(n, size=34, color=c, weight=BOLD),
                                safe_text(t, size=25)).arrange(RIGHT, buff=0.3)
                         for n, t, c in [
                             ("x → ŷ", "clasificar es partir el espacio de atributos", BLUE),
                             ("TP FP", "la matriz de confusión cuenta CÓMO nos equivocamos", RED),
                             ("k-NN", "decidir como los parecidos: flexible, sensible a escala",
                              GREEN),
                             ("σ(z)", "una recta + sigmoide: probabilidades explicables", PURPLE),
                         ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to([0, 0.55, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = VGroup(
            safe_text("Ticket de salida", size=24, color=YELLOW, weight=BOLD),
            safe_text("Da un ejemplo donde un falso negativo cueste más que un falso positivo.",
                      size=21, max_width=11),
            safe_text("¿Qué métrica vigilarías y hacia dónde moverías el umbral?", size=21,
                      max_width=11),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = card(ticket.width + 0.6, ticket.height + 0.4, YELLOW)
        tg = VGroup(box, ticket.move_to(box)).move_to([0, -2.35, 0])
        self.play(FadeIn(tg, shift=UP * 0.1))
