"""Acto 4 · Una red parte el mapa en polígonos (29–38 min).

La red 2 → 16 → 16 → 1 entrenada sobre Baarle: sus 16 rectas de la primera
capa, las regiones lineales que forman (y las que agregan los pliegues de la
segunda capa), la frontera aprendida y, al final, la representación interna en
la que una recta basta: la lectura geométrica del aprendizaje profundo.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (LOGISTIC, MLP, SHOW_X, SHOW_Y, accuracy, first_layer_lines,
                           logit_grid, n_params, region_ids, representation)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, marker, pill, safe_text, tex)
from sections.viz import MapView, hidden_fields

MAP_CENTER = (-2.75, -0.62, 0)
MAP_W = 7.3


def symlog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def layer_column(n: int, x: float, color, *, height=4.6, r=0.12) -> VGroup:
    ys = np.linspace(height / 2, -height / 2, n) if n > 1 else [0.0]
    return VGroup(*[Circle(radius=r, stroke_color=color, stroke_width=2.5, fill_color=PANEL,
                           fill_opacity=1).move_to([x, y - 0.45, 0]) for y in ys])


class NetworkSection:
    def network(self) -> None:
        self.architecture()
        self.first_layer()
        self.regions()
        self.decision()
        self.representation_morph()

    # ------------------------------------------------------------------
    def architecture(self) -> None:
        self.beat("29–30 min. La red que vamos a mirar por dentro: 2 entradas (x, y), dos "
                  "capas ocultas de 16 neuronas ReLU y una salida sigmoide. 337 parámetros. "
                  "Se entrenó con 4 000 puntos del mapa (Adam, 6 000 épocas, entropía cruzada). "
                  "Cada línea gris es un peso.")
        self.start("La red", "2 → 16 → 16 → 1 con ReLU", YELLOW)
        cols = [layer_column(2, -4.6, BLUE, height=1.4, r=0.2),
                layer_column(16, -1.6, TEAL),
                layer_column(16, 1.4, PURPLE),
                layer_column(1, 4.4, YELLOW, r=0.22)]
        edges = VGroup()
        for a, b in zip(cols[:-1], cols[1:]):
            for u in a:
                for v in b:
                    edges.add(Line(u.get_center(), v.get_center(), stroke_width=0.8,
                                   color=GRID, stroke_opacity=0.55))
        labels = VGroup(
            tex(r"(x, y)", size=30, color=BLUE).next_to(cols[0], DOWN, buff=0.3),
            safe_text("16 ReLU", size=22, color=TEAL).next_to(cols[1], DOWN, buff=0.2),
            safe_text("16 ReLU", size=22, color=PURPLE).next_to(cols[2], DOWN, buff=0.2),
            tex(r"\sigma", size=34, color=YELLOW).next_to(cols[3], DOWN, buff=0.3),
        )
        self.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.2))
        self.play(Create(edges), run_time=1.5)
        self.bring_to_front(*cols)
        self.play(FadeIn(labels))
        info = VGroup(
            safe_text(f"{n_params()} parámetros", size=26, color=YELLOW, weight=BOLD),
            safe_text("2·16+16 + 16·16+16 + 16+1", size=20, color=MUTED),
            safe_text("4 000 puntos · Adam · 6 000 épocas", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        info.scale_to_fit_width(min(info.width, 3.6)).move_to([4.75, 1.75, 0])
        self.play(FadeIn(info))

    # ------------------------------------------------------------------
    def first_layer(self) -> None:
        self.beat("30–32 min. Primera capa: cada una de las 16 neuronas es una recta "
                  "w·x + b = 0, exactamente como al inicio de la clase. Estas son las 16 rectas "
                  "que la red APRENDIÓ. Pregunte: ¿ven rectas que siguen bordes de los "
                  "enclaves? (varias pasan muy cerca de lados de los polígonos).")
        self.start("Dentro de la red · capa 1", "16 neuronas = 16 rectas aprendidas", TEAL)
        mv = MapView(MAP_W, MAP_CENTER)
        self.nmv = mv
        img = mv.truth_image(alpha=0.22)
        borders = mv.borders(color=INK, width=1.5).set_stroke(opacity=0.6)
        dots = mv.dots(SHOW_X, SHOW_Y, r=0.04).set_opacity(0.55)
        self.play(FadeIn(img), Create(mv.frame()), FadeIn(borders), FadeIn(dots))
        lines = VGroup(*[l for l in (mv.line(*abc, TEAL, 3) for abc in first_layer_lines())
                         if l is not None])
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.12), run_time=3)
        side = VGroup(
            tex(r"h^{(1)}_k = \text{ReLU}(\mathbf{w}_k\cdot\mathbf{x} + b_k)", size=32,
                color=TEAL),
            safe_text(f"{len(lines)} rectas cruzan el mapa", size=24),
            safe_text("cada una «enciende» su neurona", size=22, color=MUTED),
            safe_text("de un lado y la apaga del otro", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.95, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.1) for s in side], lag_ratio=0.2))
        self.n_base = Group(img, borders, dots)
        self.n_lines = lines
        self.n_side = side

    # ------------------------------------------------------------------
    def regions(self) -> None:
        self.beat("32–34 min. Las 16 rectas parten el mapa en polígonos: dentro de cada uno, "
                  "el conjunto de neuronas encendidas es el mismo, así que la red es una "
                  "función LINEAL (afín) distinta en cada polígono. Con la capa 1 hay 100 "
                  "regiones.")
        mv = self.nmv
        ids1 = region_ids(layers=1)
        ids2 = region_ids(layers=2)
        n1, n2 = ids1.max() + 1, ids2.max() + 1
        r1 = mv.region_image(ids1)
        head = self.heading("Dentro de la red · regiones", "Las rectas parten el mapa en "
                            "polígonos", PURPLE)
        self.play(FadeOut(self.cur_head), FadeIn(head), FadeOut(self.n_side),
                  FadeOut(self.n_base[2]), FadeOut(self.n_base[1]))
        self.cur_head = head
        self.play(FadeIn(r1), run_time=1.2)
        self.bring_to_front(self.n_lines)
        c1 = VGroup(safe_text("capa 1", size=24, color=TEAL, weight=BOLD),
                    safe_text(f"{n1} regiones lineales", size=26)).arrange(DOWN, buff=0.1,
                                                                          aligned_edge=LEFT)
        c1.move_to([3.95, 1.4, 0])
        self.play(FadeIn(c1, shift=LEFT * 0.1))

        self.beat("34–35 min. La segunda capa no agrega rectas: agrega PLIEGUES. Cada neurona "
                  "de la capa 2 corta a lo largo de una línea quebrada (morado), que se dobla "
                  "cada vez que cruza una recta de la capa 1. Regiones: 455. La profundidad "
                  "multiplica las regiones; el ancho sólo las suma.")
        fields = hidden_fields(MLP, 1, 360, 252)
        bent = VGroup(*[mv.contour(F, PURPLE, 2.5) for F in fields])
        r2 = mv.region_image(ids2, seed=9)
        self.play(LaggedStart(*[Create(b) for b in bent], lag_ratio=0.08), run_time=3)
        self.play(FadeOut(r1), FadeIn(r2))
        self.bring_to_front(self.n_lines, bent)
        c2 = VGroup(safe_text("capa 1 + capa 2", size=24, color=PURPLE, weight=BOLD),
                    safe_text(f"{n2} regiones lineales", size=26)).arrange(DOWN, buff=0.1,
                                                                          aligned_edge=LEFT)
        c2.next_to(c1, DOWN, buff=0.45, aligned_edge=LEFT)
        rule = VGroup(
            safe_text("en cada región:", size=22, color=MUTED),
            tex(r"\text{red}(\mathbf{x}) = A_r\,\mathbf{x} + c_r", size=32, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(c2, DOWN, buff=0.5,
                                                             aligned_edge=LEFT)
        self.play(FadeIn(c2, shift=LEFT * 0.1))
        self.play(FadeIn(rule, shift=LEFT * 0.1))
        self.n_regions = Group(r2, bent)
        self.n_cards = VGroup(c1, c2, rule)

    # ------------------------------------------------------------------
    def decision(self) -> None:
        self.beat("35–36 min. La salida decide, polígono por polígono, de qué lado está. La "
                  "frontera amarilla está hecha de segmentos rectos (es lineal a trozos) y "
                  "reproduce los enclaves y el contraenclave: 99.8 % de exactitud contra 75 % "
                  "de la mejor recta. Señale el hueco dentro del enclave grande.")
        mv = self.nmv
        head = self.heading("Dentro de la red · salida", "La frontera aprendida", YELLOW)
        old = [m for m in self.mobjects if m is not self.branding and m is not self.n_base[0]]
        img = mv.prob_image(MLP, sharp=0.25)
        Z = logit_grid(MLP, 700, 490)
        cont = mv.contour(Z, YELLOW, 5)
        truth = mv.borders(color=INK, width=2)
        truth = VGroup(*[DashedVMobject(t, num_dashes=40) for t in truth])
        self.play(*[FadeOut(m) for m in old], FadeIn(head), FadeIn(img), Create(mv.frame()))
        self.play(Create(cont), run_time=2.5)
        self.play(Create(truth))
        cmp = VGroup(
            VGroup(safe_text("una recta", size=24, color=MUTED),
                   safe_text(f"{accuracy(LOGISTIC):.1%}", size=30, color=MUTED, weight=BOLD)
                   ).arrange(RIGHT, buff=0.3),
            VGroup(safe_text("red ReLU", size=24, color=YELLOW),
                   safe_text(f"{accuracy(MLP):.1%}", size=30, color=YELLOW, weight=BOLD)
                   ).arrange(RIGHT, buff=0.3),
            safe_text("amarillo: frontera de la red", size=20, color=YELLOW),
            safe_text("punteado: frontera real", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.95, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in cmp], lag_ratio=0.2))

    # ------------------------------------------------------------------
    def representation_morph(self) -> None:
        self.beat("36–38 min. La idea central del aprendizaje profundo. Tomamos cada punto y lo "
                  "llevamos a lo que «ve» la última capa oculta (eje horizontal: la salida antes "
                  "de la sigmoide; vertical: la dirección principal restante, en escala "
                  "logarítmica). Pida que miren cómo los enclaves salen de su lugar y se juntan "
                  "con el sur: en el espacio aprendido, una RECTA basta. La red aprendió una "
                  "representación donde el problema es lineal.")
        self.start("Geometría del aprendizaje profundo", "La red deforma el plano hasta que una "
                   "recta basta", GREEN)
        src = MapView(6.0, (-3.4, -0.35, 0))
        R = representation(MLP, SHOW_X)
        u, v = symlog(R[:, 0]), symlog(R[:, 1])
        ulim = max(abs(u).max(), 1e-6)
        vlo, vhi = v.min(), v.max()
        box_w, box_h = 6.0, 3.9
        center = np.array([3.4, -0.35, 0])
        tgt = np.column_stack([u / ulim * box_w / 2, (v - (vlo + vhi) / 2) / (vhi - vlo) * box_h])
        frame_t = Rectangle(width=box_w + 0.3, height=box_h + 0.3, stroke_color=GRID,
                            stroke_width=2).move_to(center)
        dots = src.dots(SHOW_X, SHOW_Y, r=0.045)
        lab_s = safe_text("espacio de entrada (x, y)", size=20, color=MUTED).next_to(
            src.frame(), UP, buff=0.15)
        lab_t = safe_text("espacio aprendido (capa oculta 2)", size=20, color=MUTED).next_to(
            frame_t, UP, buff=0.15)
        self.play(Create(src.frame()), FadeIn(dots), FadeIn(lab_s))
        self.play(Create(frame_t), FadeIn(lab_t))
        moving = dots.copy()
        self.play(*[d.animate.move_to(center + np.array([p[0], p[1], 0]))
                    for d, p in zip(moving, tgt)], run_time=3.5, rate_func=smooth)
        sep = DashedLine(center + UP * (box_h / 2 + 0.1), center + DOWN * (box_h / 2 + 0.1),
                         color=YELLOW, stroke_width=5)
        self.play(Create(sep))
        axes_l = VGroup(
            safe_text("salida de la red (logit) →", size=18, color=MUTED).next_to(frame_t, DOWN,
                                                                                    buff=0.12),
        )
        msg = safe_text("Aprender = encontrar una representación donde las clases se separan "
                        "con un hiperplano", size=23, color=GREEN, weight=BOLD, max_width=12.8)
        msg.move_to([0, -3.08, 0])
        self.play(FadeIn(axes_l), FadeIn(msg, shift=UP * 0.1))
