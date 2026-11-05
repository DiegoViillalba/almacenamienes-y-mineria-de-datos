"""Acto 3 · k vecinos más cercanos (22–37 min).

Un círculo crece alrededor del cliente nuevo hasta tocar a sus vecinos; una lupa
convierte la distancia en un triángulo rectángulo (Pitágoras) antes de escribir
la fórmula. Después: votación, regiones de decisión para varios k, la trampa de
la escala, «tu turno» y la diferencia entre entrenamiento y prueba.
"""
from __future__ import annotations

from math import sqrt

from manim import *

from sections.data import (QUERY, TEST, TRAIN, TURN, XTE, YTE, YTR, knn_grid, knn_order,
                           knn_predict, knn_train_acc)
from sections.plane import ClientPlane
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, YELLOW, card, legend_item, marker, pill, safe_text, table, tex)


def vote_bars(n_no: int, n_si: int, center, unit: float = 0.55) -> VGroup:
    """Dos columnas de fichas: votos «no paga» y «paga»."""
    cols = VGroup()
    for cls, n, name in [(1, n_no, "no paga"), (0, n_si, "paga")]:
        stack = VGroup(*[marker(cls, r=0.16) for _ in range(n)])
        if n:
            stack.arrange(UP, buff=0.12)
        lab = safe_text(name, size=20, color=CLASS_COLOR[cls], weight=BOLD)
        col = VGroup(stack, lab)
        cols.add(col)
    for c in cols:
        c[1].move_to(ORIGIN)
        if len(c[0]):
            c[0].next_to(c[1], UP, buff=0.18)
    cols.arrange(RIGHT, buff=0.8, aligned_edge=DOWN).move_to(center)
    return cols


class KnnSection:
    def _base_plane(self):
        plane = ClientPlane()
        pts = plane.clients()
        legend = VGroup(legend_item(0, "pagó"), legend_item(1, "no pagó")
                        ).arrange(RIGHT, buff=0.4).next_to(plane.axes, UP, buff=0.12)
        return plane, pts, legend

    def _ball(self, plane, center, r: ValueTracker, color=YELLOW):
        return always_redraw(lambda: Circle(
            radius=max(r.get_value(), 1e-3) * plane.unit, color=color, stroke_width=3,
            fill_color=color, fill_opacity=0.07).move_to(plane.p(*center)))

    def knn(self) -> None:
        self.beat("22–23 min. Otra estrategia, sin reglas: «dime con quién andas». Llega el "
                  "cliente B (ingreso 31, deuda 60). Pregunte: si sólo pudieran preguntarle a "
                  "UN cliente conocido, ¿a cuál le preguntarían? Que lo señalen.")
        head = self.start("Preguntar a los vecinos", "¿A quién se parece el cliente nuevo?",
                          GREEN)
        plane, pts, legend = self._base_plane()
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(pts))
        q = plane.client(("B", *QUERY, None), unknown=True, r=0.15)
        self.play(GrowFromCenter(q))
        info = VGroup(safe_text("Cliente B", size=28, color=YELLOW, weight=BOLD),
                      safe_text("ingreso 31 · deuda 60 %", size=24),
                      safe_text("¿pagará?", size=28, color=YELLOW, weight=BOLD)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.3, 1.5, 0])
        self.play(FadeIn(info))

        self.beat("23–24 min. Un círculo crece desde B: el primer cliente que toca es su "
                  "vecino más cercano. ¡Es el 6, la excepción que pagó! El clasificador 1-NN "
                  "respondería «paga». Pregunte si les convence.")
        order, dist = knn_order(QUERY)
        r = ValueTracker(0.01)
        ball = self._ball(plane, QUERY, r)
        self.add(ball)
        self.bring_to_front(pts, q)
        self.play(r.animate.set_value(dist[order[0]]), run_time=1.6, rate_func=linear)
        self.play(Indicate(pts[order[0]], color=YELLOW, scale_factor=1.8))
        one = VGroup(safe_text("1 vecino:", size=26, color=MUTED),
                     safe_text("cliente 6 → «paga»", size=26, color=BLUE, weight=BOLD)
                     ).arrange(RIGHT, buff=0.2).move_to([3.3, 0.2, 0])
        self.play(FadeIn(one))

        self.beat("24–26 min. ¿Qué significa «cerca»? Abrimos una lupa. Entre B y el cliente 6 "
                  "hay 1 de ingreso y 2 de deuda: dos catetos. La distancia es la hipotenusa: "
                  "Pitágoras. Pida que calculen la de B al cliente 9 (catetos 4 y 2) antes de "
                  "mostrarla.")
        self.play(FadeOut(VGroup(info, one)))
        box_lo, box_hi = (28.0, 56.0), (37.0, 64.0)
        zone = Rectangle(width=(box_hi[0] - box_lo[0]) * plane.unit,
                         height=(box_hi[1] - box_lo[1]) * plane.unit, color=PURPLE,
                         stroke_width=3).move_to(plane.p((box_lo[0] + box_hi[0]) / 2,
                                                         (box_lo[1] + box_hi[1]) / 2))
        s = 0.42
        ic = np.array([3.4, 0.55, 0])
        mid = ((box_lo[0] + box_hi[0]) / 2, (box_lo[1] + box_hi[1]) / 2)

        def ip(x, y):
            return ic + np.array([(x - mid[0]) * s, (y - mid[1]) * s, 0])

        panel = Rectangle(width=(box_hi[0] - box_lo[0]) * s, height=(box_hi[1] - box_lo[1]) * s,
                          color=PURPLE, stroke_width=3, fill_color=PANEL,
                          fill_opacity=0.95).move_to(ic)
        links = VGroup(DashedLine(zone.get_corner(UR), panel.get_corner(UL), color=PURPLE,
                                  stroke_width=2),
                       DashedLine(zone.get_corner(DR), panel.get_corner(DL), color=PURPLE,
                                  stroke_width=2))
        self.play(Create(zone))
        self.play(Create(links), FadeIn(panel))
        big = VGroup()
        for row in TRAIN:
            if box_lo[0] < row[1] < box_hi[0] and box_lo[1] < row[2] < box_hi[1]:
                m = marker(row[3], r=0.15).move_to(ip(row[1], row[2]))
                lab = safe_text(str(row[0]), size=20).next_to(m, DR, buff=0.04)
                big.add(VGroup(m, lab))
        bq = marker(None, r=0.15).move_to(ip(*QUERY))
        bql = safe_text("B", size=20, color=YELLOW, weight=BOLD).next_to(bq, UL, buff=0.04)
        self.play(FadeIn(big), FadeIn(bq), FadeIn(bql))

        def triangle(x, y, col):
            a, c = ip(*QUERY), ip(x, y)
            corner = np.array([c[0], a[1], 0])
            leg1 = Line(a, corner, color=MUTED, stroke_width=4)
            leg2 = Line(corner, c, color=MUTED, stroke_width=4)
            hyp = Line(a, c, color=col, stroke_width=6)
            dx, dy = abs(x - QUERY[0]), abs(y - QUERY[1])
            l1 = safe_text(f"{dx:g}", size=22, color=MUTED).next_to(leg1, UP if y < QUERY[1]
                                                                      else DOWN, buff=0.08)
            l2 = safe_text(f"{dy:g}", size=22, color=MUTED).next_to(leg2, RIGHT if x > QUERY[0]
                                                                      else LEFT, buff=0.08)
            return VGroup(leg1, leg2, l1, l2), hyp

        legs6, hyp6 = triangle(30, 58, YELLOW)
        self.play(Create(legs6[:2]), FadeIn(legs6[2:]))
        self.play(Create(hyp6))
        d6 = tex(r"d(B, 6) = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.24", size=30, color=YELLOW)
        d6.next_to(panel, DOWN, buff=0.2)
        self.play(Write(d6))

        self.beat("26 min. Revele la distancia a 9: raíz de 20, 4.47. Ahora sí la fórmula: es "
                  "Pitágoras con los dos atributos. Con más atributos se agregan más "
                  "cuadrados dentro de la raíz.")
        legs9, hyp9 = triangle(35, 62, ORANGE)
        self.play(Create(legs9[:2]), FadeIn(legs9[2:]), Create(hyp9))
        d9 = tex(r"d(B, 9) = \sqrt{4^2 + 2^2} = \sqrt{20} \approx 4.47", size=30, color=ORANGE)
        d9.next_to(d6, DOWN, buff=0.18)
        self.play(Write(d9))
        formula = tex(r"d(\mathbf{x}, \mathbf{x}') = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2}",
                      size=30, color=INK, max_width=6.0)
        formula.next_to(d9, DOWN, buff=0.25)
        self.play(Write(formula))

        self.beat("26–28 min. Cerramos la lupa y seguimos creciendo el círculo hasta tres "
                  "vecinos: 6 (paga), 9 y 8 (no pagan). Votan: 2 contra 1. Con k = 3 la "
                  "excepción queda en minoría: 3-NN responde «no paga». Pregunte por qué "
                  "conviene que k sea impar.")
        self.play(FadeOut(VGroup(zone, links, panel, big, bq, bql, legs6, hyp6, legs9, hyp9,
                                 d6, d9, formula)))
        self.play(r.animate.set_value(dist[order[2]] + 0.05), run_time=1.6, rate_func=linear)
        nb = VGroup(*[pts[i] for i in order[:3]])
        self.play(LaggedStart(*[Indicate(p, color=YELLOW, scale_factor=1.6) for p in nb],
                              lag_ratio=0.3))
        dl = table([["vecino", "distancia", "clase"]] + [
            [str(TRAIN[i][0]), f"{dist[i]:.2f}", "paga" if YTR[i] == 0 else "no paga"]
            for i in order[:3]], col_widths=[1.4, 1.7, 1.7], size=22, row_h=0.46,
            colors=[[INK, INK, CLASS_COLOR[int(YTR[i])]] for i in order[:3]])
        dl.move_to([3.3, 1.35, 0])
        self.play(FadeIn(dl))
        votes = vote_bars(2, 1, [3.3, -1.0, 0])
        self.play(FadeIn(votes, shift=UP * 0.1))
        verdict = pill("3-NN: «no paga»", ORANGE, size=26).move_to([3.3, -2.55, 0])
        self.play(FadeIn(verdict))
        self.plane, self.pts = plane, pts

        self.beat("28–29 min. Formalice el algoritmo que acaban de ejecutar: medir, ordenar, "
                  "votar. No hay entrenamiento: el «modelo» son los datos guardados. Por eso "
                  "decimos que k-NN es perezoso y no paramétrico.")
        self.start("Preguntar a los vecinos", "El algoritmo k-NN", GREEN)
        steps = VGroup(*[VGroup(safe_text(n, size=40, color=GREEN, weight=BOLD),
                                safe_text(t, size=28)).arrange(RIGHT, buff=0.35)
                         for n, t in [("1", "Medir la distancia del nuevo caso a todos"),
                                      ("2", "Quedarse con los k más cercanos"),
                                      ("3", "Votar: gana la clase más frecuente")]]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([-0.6, 0.85, 0])
        self.play(LaggedStart(*[FadeIn(st, shift=RIGHT * 0.1) for st in steps], lag_ratio=0.3))
        rule = tex(r"\hat{y}(\mathbf{x}) = \text{moda}\,\{\, y_i \;:\; i \in N_k(\mathbf{x}) \,\}",
                   size=44, color=YELLOW)
        sub = safe_text("N_k(x): los k clientes de entrenamiento más cercanos a x", size=22,
                        color=MUTED)
        VGroup(rule, sub).arrange(DOWN, buff=0.25).move_to([0, -1.6, 0])
        self.play(Write(rule), FadeIn(sub))
        self.regions()

    # ------------------------------------------------------------------
    def regions(self) -> None:
        self.beat("29–31 min. Si preguntamos por TODOS los puntos posibles del plano y "
                  "pintamos la respuesta, vemos la frontera de k-NN. Con k = 1 aparece una "
                  "isla azul alrededor del cliente 6: el modelo memorizó la excepción. "
                  "Pregunte: ¿qué le pasará a la isla con k = 3?")
        head = self.start("Preguntar a los vecinos", "Pintar el plano: regiones de decisión",
                          GREEN)
        plane, pts, legend = self._base_plane()
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(pts))
        info = {}
        for k in (1, 3, 9):
            info[k] = VGroup(
                safe_text(f"k = {k}", size=44, color=YELLOW, weight=BOLD),
                safe_text(f"exactitud en entrenamiento: {knn_train_acc(k):.2f}", size=24),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.3, 1.2, 0])
        reg = plane.grid_regions(knn_grid(1))
        self.play(FadeIn(reg, run_time=1.2), FadeIn(info[1]))
        self.bring_to_front(pts)
        island = Circle(radius=0.55, color=YELLOW, stroke_width=4).move_to(pts[5][0])
        memo = safe_text("isla: memorizó al cliente 6", size=22, color=YELLOW)
        memo.move_to([3.3, -0.3, 0])
        self.play(Create(island), FadeIn(memo))

        self.beat("31–32 min. Con k = 3 la isla desaparece: la frontera se suaviza. Con k = 9 "
                  "casi todo el plano dice «paga»: con tantos votantes gana siempre la "
                  "mayoría global (6 de 10) y el modelo se acerca a la referencia. k pequeño "
                  "memoriza (sobreajuste); k grande ignora los datos (subajuste).")
        self.play(FadeOut(island), FadeOut(memo))
        for k in (3, 9):
            new = plane.grid_regions(knn_grid(k))
            self.play(FadeOut(reg), FadeIn(new), Transform(info[1], info[k]), run_time=1.2)
            reg = new
            self.bring_to_front(pts)
        scale = VGroup(
            safe_text("k pequeño", size=24, color=RED, weight=BOLD),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=4).scale(0.8),
            safe_text("k grande", size=24, color=BLUE, weight=BOLD),
        ).arrange(RIGHT, buff=0.2)
        under = VGroup(safe_text("memoriza", size=20, color=MUTED).next_to(scale[0], DOWN),
                       safe_text("ignora", size=20, color=MUTED).next_to(scale[2], DOWN))
        VGroup(scale, under).move_to([3.3, -1.0, 0])
        self.play(FadeIn(scale), FadeIn(under))
        self.scale_trap()

    def scale_trap(self) -> None:
        self.beat("32–33 min. Una trampa. Alguien guarda la deuda como proporción (0.60 en "
                  "lugar de 60). Mismos clientes, otra unidad. Pida predecir si cambia la "
                  "respuesta para B. Al aplastarse el eje, la deuda deja de pesar en la "
                  "distancia: sólo cuenta el ingreso.")
        head = self.start("Preguntar a los vecinos", "Cuidado con las unidades", GREEN)
        plane, pts, legend = self._base_plane()
        q = plane.client(("B", *QUERY, None), unknown=True, r=0.15)
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(pts), FadeIn(q))
        y0 = 15.0
        squash = [pts[i].animate.move_to(plane.p(TRAIN[i][1], y0 + 3 + (TRAIN[i][2] - 15) * 0.08))
                  for i in range(len(TRAIN))]
        lab = VGroup(safe_text("deuda en %", size=24, color=MUTED),
                     safe_text("60", size=40, weight=BOLD)).arrange(DOWN, buff=0.1)
        lab.move_to([1.5, 1.6, 0])
        lab2 = VGroup(safe_text("deuda como proporción", size=24, color=MUTED),
                      safe_text("0.60", size=40, weight=BOLD, color=RED)).arrange(DOWN, buff=0.1)
        lab2.move_to([5.0, 1.6, 0])
        arr = Arrow(lab.get_right(), lab2.get_left(), color=MUTED, buff=0.15)
        self.play(FadeIn(lab), GrowArrow(arr), FadeIn(lab2))
        self.play(*squash, q.animate.move_to(plane.p(QUERY[0], y0 + 3 + (QUERY[1] - 15) * 0.08)),
                  run_time=2.0)
        order, dist = knn_order(QUERY, sy=0.01)
        nb = [TRAIN[i][0] for i in order[:3]]
        res = VGroup(
            safe_text(f"3 vecinos ahora: {nb[0]}, {nb[1]} y {nb[2]}", size=26),
            safe_text("2 pagan, 1 no  →  «paga»", size=26, color=BLUE, weight=BOLD),
            safe_text("antes: 6, 9 y 8  →  «no paga»", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.4, -0.1, 0])
        self.play(LaggedStart(*[Indicate(pts[i], color=YELLOW, scale_factor=1.8)
                                for i in order[:3]], lag_ratio=0.3))
        self.play(FadeIn(res))

        self.beat("33–34 min. La solución: poner todas las variables en una escala común "
                  "antes de medir distancias (estandarizar). Lo verán con detalle en el "
                  "capítulo de k-NN; aquí basta el mensaje: en k-NN las unidades SON parte "
                  "del modelo.")
        fix = VGroup(
            safe_text("Estandarizar antes de medir", size=26, color=GREEN, weight=BOLD),
            tex(r"z = \frac{x - \bar{x}}{s}", size=44, color=GREEN),
        ).arrange(DOWN, buff=0.2).move_to([3.4, -2.2, 0])
        self.play(FadeIn(fix, shift=UP * 0.1))
        self.your_turn_knn()

    def your_turn_knn(self) -> None:
        self.beat("34–35 min. Tu turno (en parejas, 1 min). Cliente C: ingreso 26, deuda 45. "
                  "Usen los ojos (o una regla) para encontrar sus 3 vecinos y votar. No "
                  "revele el círculo hasta que la mayoría tenga respuesta.")
        head = self.start("Tu turno", "Clasifica al cliente C con 3-NN", YELLOW)
        plane, pts, legend = self._base_plane()
        q = plane.client(("C", *TURN, None), unknown=True, r=0.15)
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(pts), GrowFromCenter(q))
        ask = VGroup(safe_text("Cliente C", size=28, color=YELLOW, weight=BOLD),
                     safe_text("ingreso 26 · deuda 45 %", size=24),
                     safe_text("¿quiénes son sus 3 vecinos?", size=24),
                     safe_text("¿qué predice 3-NN?", size=24)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.3, 1.2, 0])
        self.play(FadeIn(ask))

        self.beat("35–36 min. Solución: 10 (3.6, no paga), 5 (12.1, paga) y 6 (13.6, paga): "
                  "3-NN dice «paga». Ahora la verdad: C NO pagó. El modelo falló. Pregunte: "
                  "¿cómo sabríamos de antemano cuánto falla? Necesitamos clientes que el "
                  "modelo no haya visto.")
        order, dist = knn_order(TURN)
        r = ValueTracker(0.01)
        ball = self._ball(plane, TURN, r)
        self.add(ball)
        self.bring_to_front(pts, q)
        self.play(r.animate.set_value(dist[order[2]] + 0.1), run_time=2.0, rate_func=linear)
        dl = table([["vecino", "distancia", "clase"]] + [
            [str(TRAIN[i][0]), f"{dist[i]:.1f}", "paga" if YTR[i] == 0 else "no paga"]
            for i in order[:3]], col_widths=[1.4, 1.7, 1.7], size=22, row_h=0.46,
            colors=[[INK, INK, CLASS_COLOR[int(YTR[i])]] for i in order[:3]])
        dl.move_to([3.3, 0.6, 0])
        self.play(FadeOut(ask), FadeIn(dl))
        pred = pill("3-NN: «paga»", BLUE, size=24).move_to([1.9, -1.4, 0])
        truth = pill("real: no pagó ✗", RED, size=24).move_to([4.9, -1.4, 0])
        self.play(FadeIn(pred))
        self.play(FadeIn(truth), q[0].animate.set_fill(ORANGE).set_stroke(ORANGE))
        self.train_vs_test()

    def train_vs_test(self) -> None:
        self.beat("36–37 min. Cinco clientes NUEVOS (A–E, contorno blanco) que no se usaron "
                  "para construir nada. 1-NN acertaba 10 de 10 en entrenamiento… y aquí sólo "
                  "3 de 5, igual que la referencia. Memorizar no es aprender: el error que "
                  "importa es el de PRUEBA.")
        head = self.start("Preguntar a los vecinos", "Entrenamiento contra prueba", GREEN)
        plane, pts, legend = self._base_plane()
        reg = plane.grid_regions(knn_grid(1))
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(reg), FadeIn(pts.set_opacity(0.45)))
        pred = knn_predict(XTE, 1)
        tests = VGroup()
        for row, p in zip(TEST, pred):
            g = plane.client(row, r=0.15)
            g[0].set_stroke(WHITE, 3)
            g[1].set_color(YELLOW).scale(1.3)
            ok = p == row[3]
            mark = safe_text("✓" if ok else "✗", size=22, color=GREEN if ok else RED,
                             weight=BOLD).next_to(g[0], DL, buff=0.02)
            g.add(mark)
            tests.add(g)
        self.play(LaggedStart(*[GrowFromCenter(t[:2]) for t in tests], lag_ratio=0.2))
        self.play(LaggedStart(*[FadeIn(t[2]) for t in tests], lag_ratio=0.2))
        acc_te = {k: float(np.mean(knn_predict(XTE, k) == YTE)) for k in (1, 3, 9)}
        tbl = table([["k", "entrenamiento", "prueba"]] + [
            [str(k), f"{knn_train_acc(k):.2f}", f"{acc_te[k]:.2f}"] for k in (1, 3, 9)],
            col_widths=[0.9, 2.5, 1.6], size=24, row_h=0.55,
            colors=[[YELLOW, GREEN if k == 1 else INK, RED if k == 1 else INK] for k in (1, 3, 9)])
        tbl.move_to([3.3, 1.0, 0])
        self.play(FadeIn(tbl))
        moral = VGroup(safe_text("Memorizar ≠ aprender", size=30, color=YELLOW, weight=BOLD),
                       safe_text("el error que importa es el de prueba", size=22, color=MUTED)
                       ).arrange(DOWN, buff=0.12).move_to([3.3, -1.3, 0])
        self.play(FadeIn(moral, shift=UP * 0.1))
