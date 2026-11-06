"""Acto 5 · ¿Cómo aprende? Pérdida, descenso por gradiente y retropropagación (38–48 min).

La frontera de Baarle durante el entrenamiento junto a la curva de pérdida;
el descenso por gradiente en una dimensión; y una red de una neurona oculta
recorrida hacia adelante y hacia atrás con números, hasta actualizar los pesos.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import BACKPROP, LOSSES, SNAPS, accuracy, logit_grid
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, tex)
from sections.viz import MapView


def box_node(top: str, color, *, w=1.75, h=0.95, size=30) -> VGroup:
    r = RoundedRectangle(corner_radius=0.14, width=w, height=h, stroke_color=color,
                         stroke_width=3, fill_color=PANEL, fill_opacity=1)
    t = MathTex(top, font_size=size, color=INK)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t.move_to(r))


class TrainingSection:
    def training(self) -> None:
        self.train_movie()
        self.gradient_descent()
        self.backprop()
        self.chain_rule()

    # ------------------------------------------------------------------
    def train_movie(self) -> None:
        self.beat("38–40 min. ¿De dónde salieron los 337 números? Se empieza con pesos al azar "
                  "y se mide qué tan mal clasifica la red con la entropía cruzada (la misma "
                  "pérdida de la regresión logística). Avance época por época: primero aprende "
                  "el sur (lo fácil, casi lineal), luego los enclaves grandes, al final el "
                  "contraenclave. La pérdida cae de 0.67 a 0.008.")
        self.start("Entrenamiento", "Pesos al azar → frontera de Baarle", RED)
        mv = MapView(6.6, (-3.15, -0.6, 0))
        loss = tex(r"L = -\frac{1}{n}\sum_i \big[y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)"
                   r"\big]", size=30, max_width=6.2).move_to([3.6, 1.75, 0])
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 0.7, 0.1], x_length=5.6, y_length=3.0,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False}).move_to([3.7, -0.75, 0])
        xl = safe_text("épocas (escala log)", size=18, color=MUTED).next_to(ax, DOWN, buff=0.42)
        yl = safe_text("pérdida", size=18, color=MUTED).next_to(ax, UP, buff=0.08).align_to(
            ax, LEFT)
        ticks = VGroup(*[safe_text(t, size=16, color=MUTED).next_to(ax.c2p(np.log10(v + 1), 0),
                                                                     DOWN, buff=0.08)
                         for v, t in [(10, "10"), (100, "100"), (1000, "1000")]])
        ep = np.array([25 * i for i in range(len(LOSSES) - 1)] + [6000])
        pts = [ax.c2p(np.log10(e + 1), l) for e, l in zip(ep, LOSSES)]
        self.play(Create(mv.frame()), FadeIn(loss), Create(ax), FadeIn(xl), FadeIn(yl),
                  FadeIn(ticks))
        frame = mv.frame()
        img = label = curve = dot = None
        for epoch, params in sorted(SNAPS.items()):
            group = Group(mv.prob_image(params, nx=400, ny=280, sharp=0.5),
                          mv.contour(logit_grid(params, 400, 280), YELLOW, 4))
            new_label = VGroup(safe_text(f"época {epoch}", size=24, color=INK, weight=BOLD),
                               safe_text(f"exactitud {accuracy(params):.1%}", size=22,
                                         color=YELLOW)).arrange(RIGHT, buff=0.4)
            new_label.next_to(frame, UP, buff=0.15)
            n = max(int(np.searchsorted(ep, epoch, side="right")), 2)
            new_curve = VMobject(stroke_color=RED, stroke_width=4).set_points_as_corners(pts[:n])
            new_dot = Dot(pts[n - 1], radius=0.08, color=RED)
            if img is None:
                self.play(FadeIn(group), FadeIn(new_label), Create(new_curve), FadeIn(new_dot))
                curve, dot = new_curve, new_dot
            else:
                self.next_slide(notes=f"Época {epoch}: exactitud {accuracy(params):.1%}. "
                                      "Pida que digan qué pedazo del mapa apareció ahora.")
                self.play(FadeOut(img), FadeIn(group), FadeTransform(label, new_label),
                          Transform(curve, new_curve), Transform(dot, new_dot), run_time=1.2)
            img, label = group, new_label
            self.add(frame)

    # ------------------------------------------------------------------
    def gradient_descent(self) -> None:
        self.beat("40–42 min. La regla para mover los pesos es el descenso por gradiente: mira "
                  "la pendiente de la pérdida respecto a cada peso y da un paso cuesta abajo, "
                  "de tamaño η (tasa de aprendizaje). Con 337 pesos es lo mismo pero en 337 "
                  "dimensiones. Pregunte: ¿qué pasa si η es enorme? (rebota).")
        self.start("Descenso por gradiente", "Bajar la pérdida un paso a la vez", RED)
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 4.5, 1], x_length=7.2, y_length=4.2,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False}).move_to([-2.6, -0.6, 0])

        def f(w):
            return 0.35 * (w - 0.8) ** 2 + 0.25 * np.sin(2.2 * w) + 0.6

        def df(w):
            return 0.7 * (w - 0.8) + 0.55 * np.cos(2.2 * w)

        curve = ax.plot(f, x_range=[-2.4, 3, 0.01], color=RED, stroke_width=5)
        wl = MathTex("w", font_size=34, color=TEAL).next_to(ax.x_axis, RIGHT, buff=0.15)
        Ll = MathTex("L(w)", font_size=34, color=RED).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Create(ax), Create(curve), FadeIn(wl), FadeIn(Ll))
        rule = VGroup(
            tex(r"w \leftarrow w - \eta\,\frac{\partial L}{\partial w}", size=44, color=YELLOW),
            safe_text("pendiente > 0 → mover a la izquierda", size=22, color=MUTED),
            safe_text("pendiente < 0 → mover a la derecha", size=22, color=MUTED),
            safe_text("η = tamaño del paso", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.8, 0.3, 0])
        self.play(FadeIn(rule[0]))
        w = -2.3
        eta = 0.9
        ball = Dot(ax.c2p(w, f(w)), radius=0.13, color=YELLOW)
        self.play(FadeIn(ball))
        self.play(LaggedStart(*[FadeIn(r) for r in rule[1:]], lag_ratio=0.2))
        for _ in range(7):
            s = df(w)
            tangent = Line(ax.c2p(w - 0.7, f(w) - 0.7 * s), ax.c2p(w + 0.7, f(w) + 0.7 * s),
                           color=TEAL, stroke_width=4)
            self.play(Create(tangent), run_time=0.35)
            w_new = w - eta * s
            self.play(ball.animate.move_to(ax.c2p(w_new, f(w_new))), FadeOut(tangent),
                      run_time=0.55)
            w = w_new
        note = safe_text("puede quedarse en un mínimo local: en redes grandes casi nunca "
                         "importa", size=20, color=MUTED, max_width=12.5).move_to([0, -3.1, 0])
        self.play(FadeIn(note))

    # ------------------------------------------------------------------
    def backprop(self) -> None:
        bp = BACKPROP
        self.beat("42–44 min. ¿Cómo se calcula ∂L/∂w para un peso enterrado en la red? "
                  "Retropropagación = regla de la cadena aplicada de atrás hacia adelante. Red "
                  "mínima: x = 2, una neurona ReLU y una salida sigmoide; la etiqueta es y = 1. "
                  "Primero HACIA ADELANTE: z = 0.5, a = 0.5, s = −0.25, ŷ = 0.44, L = 0.83.")
        self.start("Retropropagación", "Hacia adelante: calcular la pérdida", TEAL)
        xs = [-5.9, -3.55, -1.2, 1.15, 3.5, 5.85]
        y0 = 0.35
        nodes = VGroup(
            box_node(r"x", BLUE, w=1.2),
            box_node(r"z = wx + b", TEAL),
            box_node(r"a = \text{ReLU}(z)", PURPLE, w=2.0),
            box_node(r"s = va + c", TEAL),
            box_node(r"\hat y = \sigma(s)", YELLOW),
            box_node(r"L = -\log \hat y", RED),
        )
        for n, x in zip(nodes, xs):
            n.move_to([x, y0, 0])
        arrows = VGroup(*[Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.06,
                                color=GRID, stroke_width=4, max_tip_length_to_length_ratio=0.25)
                          for i in range(5)])
        params = VGroup(
            MathTex(rf"w = {bp['w']:g},\; b = {bp['b']:g}", font_size=28, color=TEAL
                    ).next_to(nodes[1], UP, buff=0.35),
            MathTex(rf"v = {bp['v']:g},\; c = {bp['c']:g}", font_size=28, color=TEAL
                    ).next_to(nodes[3], UP, buff=0.35),
            MathTex(r"y = 1", font_size=28, color=MUTED).next_to(nodes[5], UP, buff=0.35),
        )
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.1),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1))
        self.play(FadeIn(params))
        fwd_vals = [rf"{bp['x']:g}", rf"{bp['z']:.2f}", rf"{bp['a']:.2f}", rf"{bp['s']:.2f}",
                    rf"{bp['yh']:.2f}", rf"{bp['L']:.2f}"]
        fwd = VGroup(*[MathTex(v, font_size=34, color=INK).next_to(n, DOWN, buff=0.25)
                       for v, n in zip(fwd_vals, nodes)])
        pulse = Dot(nodes[0].get_center(), radius=0.1, color=YELLOW)
        self.play(FadeIn(pulse), FadeIn(fwd[0]))
        for i in range(1, 6):
            self.play(pulse.animate.move_to(nodes[i].get_center()), FadeIn(fwd[i]),
                      run_time=0.5)
        self.play(FadeOut(pulse))
        tag_f = safe_text("hacia adelante →", size=22, color=INK).next_to(fwd, DOWN, buff=0.3)
        self.play(FadeIn(tag_f))

        self.beat("44–46 min. HACIA ATRÁS. Empiece por el final: ∂L/∂s = ŷ − y = −0.56 (la "
                  "sigmoide con entropía cruzada da esa resta tan limpia). Cada caja multiplica "
                  "por su derivada local: ×v en la suma, ×1 en la ReLU (porque z > 0; sería ×0 "
                  "si estuviera apagada), ×x para llegar a w. El signo negativo dice: subir w "
                  "BAJA la pérdida.")
        head = self.swap_heading(self.cur_head, "Retropropagación", "Hacia atrás: la regla de la "
                                 "cadena", RED)
        grads = [
            (4, r"\frac{\partial L}{\partial s} = \hat y - y", bp["dL_ds"]),
            (3, r"\frac{\partial L}{\partial a} = v\,\frac{\partial L}{\partial s}", bp["dL_da"]),
            (2, r"\frac{\partial L}{\partial z} = 1\cdot\frac{\partial L}{\partial a}",
             bp["dL_dz"]),
            (1, r"\frac{\partial L}{\partial w} = x\,\frac{\partial L}{\partial z}", bp["dL_dw"]),
        ]
        back = VGroup()
        pulse = Dot(nodes[5].get_center(), radius=0.1, color=RED)
        self.play(FadeIn(pulse))
        for i, (k, formula, val) in enumerate(grads):
            g = VGroup(MathTex(formula, font_size=26, color=RED),
                       MathTex(f"= {val:.2f}", font_size=30, color=RED)).arrange(DOWN, buff=0.08)
            g.move_to([xs[k] + 1.1, -1.95, 0])
            if i % 2:
                g.shift(DOWN * 0.0)
            back.add(g)
            self.play(pulse.animate.move_to(nodes[k].get_center()), FadeIn(g, shift=LEFT * 0.1),
                      run_time=0.7)
        self.play(FadeOut(pulse))
        extra = MathTex(rf"\tfrac{{\partial L}}{{\partial v}} = a\,\tfrac{{\partial L}}{{\partial s}}"
                        rf" = {bp['dL_dv']:.2f}", font_size=26, color=RED).move_to([4.4, -3.0, 0])
        tag_b = safe_text("← hacia atrás", size=22, color=RED).move_to([-3.5, -3.05, 0])
        self.play(FadeIn(tag_b), FadeOut(tag_f), FadeIn(extra))
        self.bp_scene = VGroup(nodes, arrows, params, fwd, back, tag_b, extra)

        self.beat("46–47 min. Actualizamos con η = 0.5: w = 0.5 − 0.5(−1.69) = 1.34; b, v y c "
                  "también se mueven. Volvemos a pasar hacia adelante: ŷ sube de 0.44 a 0.97 y "
                  "la pérdida baja de 0.83 a 0.03. Eso, repetido miles de veces con miles de "
                  "puntos, es entrenar.")
        upd = VGroup(
            tex(rf"w: {bp['w']:g} \to {bp['w2']:.2f} \qquad b: {bp['b']:g} \to {bp['b2']:.2f}"
                rf"\qquad v: {bp['v']:g} \to {bp['v2']:.2f} \qquad c: {bp['c']:g} \to "
                rf"{bp['c2']:.2f}", size=30, color=TEAL, max_width=12.6),
            tex(rf"\hat y: {bp['yh']:.2f} \to {bp['yh2']:.2f} \qquad L: {bp['L']:.2f} \to "
                rf"{bp['L2']:.2f}", size=36, color=YELLOW),
        ).arrange(DOWN, buff=0.25).move_to([0, 2.0, 0])
        self.play(FadeOut(params))
        self.swap_heading(self.cur_head, "Retropropagación", "Un paso de descenso", YELLOW)
        self.play(FadeIn(upd[0], shift=DOWN * 0.1))
        self.play(FadeIn(upd[1], shift=DOWN * 0.1))

    # ------------------------------------------------------------------
    def chain_rule(self) -> None:
        self.beat("47–48 min. Resumen. El gradiente de cualquier peso es un producto de "
                  "derivadas locales a lo largo del camino hacia la pérdida. Un solo recorrido "
                  "hacia atrás calcula los 337 gradientes (o mil millones) con el costo de un "
                  "recorrido hacia adelante. Las ReLU actúan como compuertas: dejan pasar el "
                  "gradiente (×1) o lo bloquean (×0). PyTorch y JAX lo hacen solos: "
                  "loss.backward(), como vimos en la clase de PyTorch y JAX.")
        self.start("Retropropagación", "Lo que hay que llevarse", RED)
        chain = tex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat y}\cdot"
                    r"\frac{\partial \hat y}{\partial s}\cdot\frac{\partial s}{\partial a}\cdot"
                    r"\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}", size=46,
                    color=INK).move_to([0, 1.2, 0])
        self.play(Write(chain))
        ideas = VGroup(
            VGroup(pill("adelante", TEAL, width=2.6), safe_text("guarda z, a, s, ŷ de cada capa",
                                                                size=24)).arrange(RIGHT, buff=0.4),
            VGroup(pill("atrás", RED, width=2.6), safe_text("multiplica derivadas locales",
                                                             size=24)).arrange(RIGHT, buff=0.4),
            VGroup(pill("ReLU", PURPLE, width=2.6), safe_text("compuerta: deja pasar ×1 o "
                                                               "bloquea ×0",
                                                               size=24)).arrange(RIGHT, buff=0.4),
            VGroup(pill("costo", YELLOW, width=2.6), safe_text("≈ un recorrido hacia adelante",
                                                                size=24)).arrange(RIGHT, buff=0.4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0, -1.35, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.2))
