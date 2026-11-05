"""Acto 4 · Clasificadores lineales y regresión logística (37–52 min).

El corte horizontal ya era una frontera lineal; se inclina para usar ambos
atributos. La distancia con signo a la recta se vuelve un puntaje z; los
clientes viajan a una recta numérica y suben a la curva sigmoide, que convierte
z en probabilidad. Después: bandas de probabilidad, el entrenamiento como una
recta que se acomoda para reducir la «sorpresa» total y, al final, la fórmula.
"""
from __future__ import annotations

from math import exp, log

from manim import *

from sections.data import (BETA, STUMP_T, TRAIN, XTR, YTR, gd_path, log_loss, logit_proba,
                           raw_coefs, stump_predict)
from sections.plane import ClientPlane
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, YELLOW, card, legend_item, marker, pill, safe_text, table, tex)

Z_RANGE = (-6.0, 4.0)


def sig(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


class LinearSection:
    def _lplane(self):
        plane = ClientPlane()
        pts = plane.clients()
        legend = VGroup(legend_item(0, "pagó"), legend_item(1, "no pagó")
                        ).arrange(RIGHT, buff=0.4).next_to(plane.axes, UP, buff=0.12)
        return plane, pts, legend

    def linear(self) -> None:
        b, w1, w2 = raw_coefs()
        self.beat("37–38 min. Regresamos al corte de 44: ¡ya era un clasificador lineal! Una "
                  "recta parte el plano en dos. Pero sólo mira la deuda. Pregunte: ¿qué "
                  "ganaríamos si la recta pudiera inclinarse? (Usar también el ingreso: a más "
                  "ingreso, se tolera más deuda.)")
        head = self.start("Trazar una frontera", "Del corte a una recta inclinada", PURPLE)
        plane, pts, legend = self._lplane()
        regions = plane.stump_regions(STUMP_T)
        line = plane.boundary(-STUMP_T, 0.0, 1.0)
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(regions), Create(line), FadeIn(pts))
        txt = VGroup(safe_text("corte:  deuda = 44", size=26, color=YELLOW, weight=BOLD),
                     safe_text("sólo usa un atributo", size=22, color=MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.2, 1.6, 0])
        self.play(FadeIn(txt))

        self.beat("38–39 min. La recta gira hasta la que encuentra la regresión logística "
                  "(veremos cómo la encuentra). Ecuación general: b + w₁·ingreso + w₂·deuda = 0. "
                  "Los pesos dicen hacia dónde apunta la recta. No se detenga en los números.")
        ang = ValueTracker(0.0)
        b_s, w1_s, w2_s = -STUMP_T, 0.0, 1.0
        # interpolación lineal de coeficientes normalizados por w2
        tgt = (b / w2, w1 / w2, 1.0)

        def coefs(a):
            return (b_s + a * (tgt[0] - b_s), w1_s + a * (tgt[1] - w1_s), 1.0)

        line_d = always_redraw(lambda: plane.boundary(*coefs(ang.get_value())))
        reg_d = always_redraw(lambda: plane.half_planes(*coefs(ang.get_value())))
        self.remove(line, regions)
        self.add(reg_d, line_d)
        self.bring_to_front(pts)
        self.play(ang.animate.set_value(1.0), run_time=2.5)
        line_d.clear_updaters()
        reg_d.clear_updaters()
        eq = tex(rf"z = {b:.2f} {w1:+.3f}\,\text{{ingreso}} {w2:+.3f}\,\text{{deuda}}", size=30,
                 color=PURPLE, max_width=6.4)
        eq0 = tex(r"\text{frontera: } z = 0", size=30, color=YELLOW)
        VGroup(eq, eq0).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.3, 0.0, 0])
        self.play(FadeOut(txt), Write(eq), Write(eq0))
        note = safe_text("a más ingreso, se tolera más deuda", size=22, color=MUTED)
        note.next_to(eq0, DOWN, aligned_edge=LEFT, buff=0.35)
        self.play(FadeIn(note))

        self.beat("39–41 min. Idea clave: no todos los clientes del mismo lado están igual de "
                  "seguros. Trace la distancia de cada cliente a la recta: lado naranja = z "
                  "positivo, lado azul = z negativo. Ese número con signo es el PUNTAJE z. "
                  "Pregunte: ¿quién es el «no paga» más seguro? ¿Y el más dudoso?")
        self.play(FadeOut(VGroup(eq0, note)), eq.animate.move_to([3.3, 2.0, 0]))
        head = self.swap_heading(head, "Trazar una frontera", "Distancia a la recta = puntaje",
                                 PURPLE)
        unit_n = np.array([w1, w2]) / np.hypot(w1, w2)
        segs = VGroup()
        zs = []
        for row, p in zip(TRAIN, pts):
            z = b + w1 * row[1] + w2 * row[2]
            zs.append(z)
            dist = z / np.hypot(w1, w2)
            foot = (row[1] - dist * unit_n[0], row[2] - dist * unit_n[1])
            segs.add(Line(plane.p(*foot), plane.p(row[1], row[2]),
                          color=ORANGE if z > 0 else BLUE, stroke_width=4))
        self.play(LaggedStart(*[Create(s) for s in segs], lag_ratio=0.08), run_time=1.6)
        self.bring_to_front(pts)

        # recta numérica de z
        zax = NumberLine(x_range=[Z_RANGE[0], Z_RANGE[1], 1], length=5.8, color=GRID,
                         include_numbers=False, stroke_width=3)
        zax.move_to([3.3, -1.55, 0])
        znums = VGroup(*[safe_text(f"{v}", size=16, color=MUTED).next_to(zax.n2p(v), DOWN, 0.12)
                         for v in range(int(Z_RANGE[0]), int(Z_RANGE[1]) + 1, 2)])
        zlab = safe_text("puntaje z", size=20, color=PURPLE).next_to(zax.get_right(), UP,
                                                                   buff=0.12)
        zero = DashedLine(zax.n2p(0) + DOWN * 0.15, zax.n2p(0) + UP * 1.0, color=YELLOW,
                          stroke_width=3)
        self.play(Create(zax), FadeIn(znums), FadeIn(zlab), Create(zero))
        tokens = VGroup()
        anims = []
        placed: list[float] = []
        for z, p in zip(zs, pts):
            lift = 0.0
            while any(abs(z - q) < 0.35 and abs(lift - l) < 0.2 for q, l in placed):
                lift += 0.26
            placed.append((z, lift))
            tok = p[0].copy()
            tok.z = z
            tok.lift = lift
            tokens.add(tok)
            anims.append(tok.animate.scale(0.85).move_to(zax.n2p(z) + UP * (0.17 + lift)))
        self.play(LaggedStart(*anims, lag_ratio=0.1), run_time=2.4)
        tags = VGroup(
            safe_text("6", size=16).next_to(tokens[5], UP, buff=0.04),
            safe_text("10", size=16).next_to(tokens[9], UP, buff=0.04),
        )
        self.play(FadeIn(tags))

        self.beat("41–43 min. ¿Cómo convertimos un puntaje en una probabilidad entre 0 y 1? "
                  "Con la curva sigmoide: aplasta los extremos y deja el 0 en 0.5. Los "
                  "clientes suben de la recta a la curva. Lea el cliente 6: z = 1.15 → p ≈ 0.76 "
                  "de NO pagar (se equivoca, pero con duda). El cliente 1: casi 0.")
        self.play(FadeOut(VGroup(segs, eq)))
        head = self.swap_heading(head, "Trazar una frontera", "Del puntaje a la probabilidad",
                                 PURPLE)
        sax = Axes(x_range=[Z_RANGE[0], Z_RANGE[1], 1], y_range=[0, 1, 0.5], x_length=5.8,
                   y_length=2.5, tips=False,
                   axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False})
        shift = zax.n2p(0) - sax.c2p(0, 0)
        sax.shift(shift)
        curve = sax.plot(lambda z: sig(z), x_range=[Z_RANGE[0], Z_RANGE[1]], color=PURPLE,
                         stroke_width=5)
        ylabs = VGroup(*[safe_text(f"{v:g}", size=16, color=MUTED).next_to(
            sax.c2p(Z_RANGE[0], v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        half = DashedLine(sax.c2p(Z_RANGE[0], 0.5), sax.c2p(Z_RANGE[1], 0.5), color=YELLOW,
                          stroke_width=2, dash_length=0.08)
        plab = safe_text("p(no paga)", size=18, color=PURPLE).next_to(sax.c2p(Z_RANGE[0], 1),
                                                                         UP, buff=0.08)
        self.play(FadeOut(tags), Create(sax.y_axis), FadeIn(ylabs), FadeIn(plab))
        self.play(Create(curve), run_time=1.4)
        self.play(*[tok.animate.move_to(sax.c2p(tok.z, sig(tok.z))) for tok in tokens],
                  run_time=1.8)
        self.play(Create(half))
        sform = tex(r"p = \sigma(z) = \frac{1}{1 + e^{-z}}", size=34, color=PURPLE)
        sform.move_to([3.3, 2.15, 0])
        r6 = safe_text(f"cliente 6: z = {zs[5]:.2f} → p = {sig(zs[5]):.2f}", size=20,
                       color=INK).move_to([4.3, 1.35, 0])
        self.play(Write(sform))
        self.play(Indicate(tokens[5], color=YELLOW, scale_factor=1.8), FadeIn(r6))
        th = safe_text("p = 0.5  ⇔  z = 0  ⇔  sobre la recta", size=20, color=YELLOW)
        th.next_to(zax, DOWN, buff=0.45)
        self.play(FadeIn(th), FadeOut(znums))
        self.lin = (plane, pts, legend, line_d, reg_d)
        self.bands(head, b, w1, w2)

    def bands(self, head, b, w1, w2) -> None:
        self.beat("43–44 min. De vuelta al plano: las rectas paralelas son curvas de nivel de "
                  "la probabilidad (0.1, 0.5, 0.9). Lejos de la frontera, el modelo está "
                  "seguro; cerca, duda. Ésta es la gran diferencia con el corte duro: el "
                  "modelo dice CUÁNTO cree lo que dice.")
        self.clear()
        head = self.heading("Trazar una frontera", "Bandas de probabilidad", PURPLE)
        plane, pts, legend = self._lplane()
        regions = plane.half_planes(b, w1, w2, opacity=0.12)
        self.add(head)
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(regions), FadeIn(pts))
        lines = VGroup()
        labs = VGroup()
        for pv, col, wd in [(0.1, BLUE, 3), (0.5, YELLOW, 5), (0.9, ORANGE, 3)]:
            zc = log(pv / (1 - pv))
            ln = plane.boundary(b - zc, w1, w2, color=col, width=wd)
            if pv != 0.5:
                ln = DashedLine(ln.get_start(), ln.get_end(), color=col, stroke_width=wd)
            lines.add(ln)
            sample = Line(LEFT * 0.3, RIGHT * 0.3, color=col, stroke_width=wd)
            lab = VGroup(sample, safe_text(f"p = {pv}", size=20, color=col, weight=BOLD)
                         ).arrange(RIGHT, buff=0.15)
            labs.add(lab)
        labs.arrange(RIGHT, buff=0.5).move_to([3.4, 2.1, 0])
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.3), FadeIn(labs))
        self.bring_to_front(pts)
        probs = logit_proba(XTR)
        tbl = table([["cliente", "p(no paga)", "real"]] + [
            [str(TRAIN[i][0]), f"{probs[i]:.2f}", "no paga" if YTR[i] else "paga"]
            for i in (0, 4, 9, 5, 7)], col_widths=[1.3, 1.9, 1.6], size=21, row_h=0.44,
            colors=[[INK, ORANGE if probs[i] >= 0.5 else BLUE, CLASS_COLOR[int(YTR[i])]]
                    for i in (0, 4, 9, 5, 7)])
        tbl.move_to([3.4, 0.3, 0])
        self.play(FadeIn(tbl))
        msg = safe_text("el modelo dice cuánto cree lo que dice", size=22, color=MUTED)
        msg.move_to([3.4, -2.2, 0])
        self.play(FadeIn(msg))
        self.training(b, w1, w2)

    def training(self, b, w1, w2) -> None:
        self.beat("44–47 min. ¿Cómo elige la recta? Cada cliente cobra una MULTA según la "
                  "probabilidad que el modelo le dio a lo que realmente pasó: −log p. Si el "
                  "modelo estaba seguro y acertó, multa casi 0; si estaba seguro y falló, multa "
                  "enorme. Partimos de una recta mala; observe cómo se acomoda y cómo baja la "
                  "multa total. Señale que el 6 y el 10 nunca llegan a cero.")
        self.clear()
        head = self.heading("Trazar una frontera", "Entrenar: reducir la sorpresa", PURPLE)
        self.add(head)
        plane, pts, legend = self._lplane()
        path = gd_path()
        from sections.data import raw_coefs as rc

        def state(beta):
            bb, a1, a2 = rc(beta)
            p = logit_proba(XTR, beta)
            pen = [-log(p[i]) if YTR[i] else -log(1 - p[i]) for i in range(len(YTR))]
            return (bb, a1, a2), pen, log_loss(beta)

        (bb, a1, a2), pen, L = state(path[0])
        reg = plane.half_planes(bb, a1, a2, opacity=0.12)
        ln = plane.boundary(bb, a1, a2)
        self.play(FadeIn(plane), FadeIn(legend), FadeIn(reg), Create(ln), FadeIn(pts))
        # barras de multa
        x0, base_y, bw, scale = 0.7, -1.9, 0.42, 1.1
        axis = Line([x0 - 0.2, base_y, 0], [x0 + 10 * 0.55, base_y, 0], color=GRID)
        names = VGroup(*[safe_text(str(r[0]), size=16, color=MUTED).move_to(
            [x0 + i * 0.55 + bw / 2, base_y - 0.22, 0]) for i, r in enumerate(TRAIN)])

        def bars(pen):
            g = VGroup()
            for i, v in enumerate(pen):
                h = max(min(v, 3.5) * scale, 0.02)
                g.add(Rectangle(width=bw, height=h, stroke_width=0,
                                fill_color=CLASS_COLOR[int(YTR[i])], fill_opacity=0.9)
                      .move_to([x0 + i * 0.55 + bw / 2, base_y + h / 2, 0]))
            return g

        def total(L):
            return VGroup(safe_text("multa promedio", size=22, color=MUTED),
                          safe_text(f"{L:.3f}", size=40, color=RED, weight=BOLD)
                          ).arrange(RIGHT, buff=0.3).move_to([3.4, 2.0, 0])

        cap = safe_text("multa de cada cliente:  −log p(lo que pasó)", size=20, color=MUTED)
        cap.move_to([3.4, 1.35, 0])
        bg = bars(pen)
        tt = total(L)
        self.play(Create(axis), FadeIn(names), FadeIn(cap), FadeIn(bg), FadeIn(tt))
        for beta in path[1:]:
            (bb, a1, a2), pen, L = state(beta)
            self.play(Transform(reg, plane.half_planes(bb, a1, a2, opacity=0.12)),
                      Transform(ln, plane.boundary(bb, a1, a2)), Transform(bg, bars(pen)),
                      Transform(tt, total(L)), run_time=0.9)
            self.bring_to_front(pts)
        self.play(tt[1].animate.set_color(GREEN))
        hard = safe_text("6 y 10: los casos que ninguna recta resuelve", size=20, color=YELLOW)
        hard.move_to([3.4, 0.8, 0])
        self.play(FadeIn(hard), Indicate(pts[5], color=YELLOW, scale_factor=1.6),
                  Indicate(pts[9], color=YELLOW, scale_factor=1.6))
        self.formal(b, w1, w2)

    def formal(self, b, w1, w2) -> None:
        self.beat("47–49 min. Ahora sí, la formalización de TODO lo que vieron: puntaje lineal, "
                  "sigmoide, regla de decisión (frontera = recta) y la función de pérdida "
                  "(la multa promedio, llamada log-loss o entropía cruzada). Se minimiza con "
                  "descenso de gradiente: la recta que se movía.")
        self.start("Trazar una frontera", "Regresión logística, en símbolos", PURPLE)
        rows = VGroup(
            VGroup(safe_text("puntaje", size=22, color=MUTED),
                   tex(r"z = b + w_1 x_1 + w_2 x_2 = b + \mathbf{w}^\top \mathbf{x}", size=36,
                       color=PURPLE)),
            VGroup(safe_text("probabilidad", size=22, color=MUTED),
                   tex(r"\hat{p}(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}", size=36,
                       color=PURPLE)),
            VGroup(safe_text("decisión", size=22, color=MUTED),
                   tex(r"\hat{y} = \text{no paga} \iff \hat{p} \ge 0.5 \iff z \ge 0", size=36,
                       color=YELLOW)),
            VGroup(safe_text("pérdida", size=22, color=MUTED),
                   tex(r"L(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log \hat{p}_i + "
                       r"(1 - y_i)\log(1 - \hat{p}_i)\big]", size=34, color=RED, max_width=9.6)),
        )
        for r in rows:
            r[0].set_width(r[0].width)
        labels = VGroup(*[r[0] for r in rows])
        for r in rows:
            r.arrange(RIGHT, buff=0.5)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to([0, 0.15, 0])
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
            r[0].next_to(r[1], LEFT, buff=0.5)
        rows.move_to([0, 0.15, 0])
        for r in rows:
            self.play(FadeIn(r[0]), Write(r[1]), run_time=0.8)

        self.beat("49–50 min. Leer los pesos: w₂ = +0.19 → cada punto extra de deuda sube el "
                  "puntaje (las «momios» de impago se multiplican por e^0.19 ≈ 1.21). "
                  "w₁ = −0.07 → más ingreso, menos riesgo. Ésta es la gran ventaja de un "
                  "modelo lineal: se puede explicar.")
        weights = VGroup(
            tex(rf"w_2 = {w2:+.2f}", size=34, color=ORANGE),
            safe_text(f"+1 punto de deuda ⇒ momios × {exp(w2):.2f}", size=22),
            tex(rf"w_1 = {w1:+.2f}", size=34, color=BLUE),
            safe_text(f"+1 mil de ingreso ⇒ momios × {exp(w1):.2f}", size=22),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.5, 0.2), col_alignments="ll")
        box = card(weights.width + 0.6, weights.height + 0.5, PURPLE)
        g = VGroup(box, weights.move_to(box)).move_to([0, -2.55, 0])
        self.play(rows.animate.shift(UP * 0.35), FadeIn(g, shift=UP * 0.1))
        self.limits()

    def limits(self) -> None:
        self.beat("50–52 min. Límite: una recta no puede separar todo. Si los morosos están en "
                  "dos esquinas opuestas (patrón tipo «XOR»), ninguna recta sirve; k-NN sí se "
                  "adapta. A cambio, la recta es rápida, estable y explicable. Pregunte: ¿qué "
                  "harían para que un modelo lineal dibuje curvas? (Crear nuevos atributos: "
                  "lo verán más adelante.)")
        self.start("Trazar una frontera", "Lo que una recta no puede hacer", PURPLE)
        rng = np.random.default_rng(7)
        left = VGroup()
        frame = Square(4.2, color=GRID, stroke_width=2).move_to([-3.3, -0.3, 0])
        for cx, cy, cls in [(-1, 1, 1), (1, -1, 1), (-1, -1, 0), (1, 1, 0)]:
            for _ in range(7):
                p = np.array([cx, cy]) * 1.0 + rng.normal(0, 0.32, 2)
                left.add(marker(cls, r=0.1).move_to(frame.get_center() + [p[0], p[1], 0]))
        self.play(Create(frame), LaggedStart(*[FadeIn(m) for m in left], lag_ratio=0.02))
        tries = [Line(frame.get_center() + rotate_vector(LEFT * 2.2, a),
                      frame.get_center() + rotate_vector(RIGHT * 2.2, a), color=RED,
                      stroke_width=4) for a in (0.2, 1.1, 2.3)]
        cur = tries[0]
        self.play(Create(cur))
        for t in tries[1:]:
            self.play(Transform(cur, t), run_time=0.7)
        cross = safe_text("ninguna recta sirve", size=24, color=RED, weight=BOLD)
        cross.next_to(frame, DOWN, buff=0.2)
        self.play(FadeIn(cross))
        cmp = table([["", "k-NN", "logística"],
                     ["frontera", "flexible", "recta"],
                     ["entrenar", "guardar datos", "optimizar w, b"],
                     ["predecir", "comparar con todos", "una fórmula"],
                     ["explicar", "mostrar vecinos", "leer pesos"],
                     ["escala", "muy sensible", "poco sensible"]],
                    col_widths=[1.6, 2.4, 2.4], size=22, row_h=0.56,
                    colors=[[MUTED, GREEN, PURPLE]] * 5)
        cmp.move_to([3.3, -0.2, 0])
        self.play(FadeIn(cmp[0]), Create(cmp[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in cmp[1:-1]], lag_ratio=0.2))
