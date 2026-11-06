"""Acto 6 · De redes densas a redes convolucionales (48–56 min).

Por qué una red densa no escala a imágenes (aplanar destruye la vecindad y los
pesos explotan); la convolución como un filtro pequeño que se desliza con pesos
compartidos; la jerarquía de una CNN; y una línea de tiempo breve.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import CONV_PARAMS, DENSE_HIDDEN, DENSE_IN, DENSE_PARAMS, FEATURE, IMG, KERNEL
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE, RED,
                            TEAL, YELLOW, card, pill, safe_text, tex)

SMILE = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
])


def fmt_int(v: int) -> str:
    return f"{v:,}".replace(",", " ")


def pixel_grid(arr, cell=0.42, *, numbers=False, on=INK, size=22) -> VGroup:
    g = VGroup()
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            v = arr[i, j]
            sq = Square(side_length=cell, stroke_color=GRID, stroke_width=1.2,
                        fill_color=on if v else PANEL, fill_opacity=0.85 if v else 1.0)
            sq.move_to([j * cell, -i * cell, 0])
            if numbers:
                t = MathTex(f"{v}", font_size=size, color=PANEL if v else MUTED).move_to(sq)
                g.add(VGroup(sq, t))
            else:
                g.add(VGroup(sq))
    return g


def prism(w: float, h: float, d: float, color) -> VGroup:
    """Caja en perspectiva simple: frente w × h, profundidad d."""
    off = np.array([d * 0.6, d * 0.45, 0])
    front = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2,
                      fill_color=color, fill_opacity=0.18)
    c = front.get_corner(UL), front.get_corner(UR), front.get_corner(DR)
    top = Polygon(c[0], c[1], c[1] + off, c[0] + off, stroke_color=color, stroke_width=2,
                  fill_color=color, fill_opacity=0.28)
    side = Polygon(c[1], c[2], c[2] + off, c[1] + off, stroke_color=color, stroke_width=2,
                   fill_color=color, fill_opacity=0.10)
    return VGroup(side, top, front)


class CnnSection:
    def cnn(self) -> None:
        self.flatten_problem()
        self.convolution()
        self.cnn_stack()
        self.timeline()

    # ------------------------------------------------------------------
    def flatten_problem(self) -> None:
        self.beat("48–50 min. Baarle tenía dos entradas. ¿Y una foto? Una red densa necesita un "
                  "VECTOR: hay que aplanar la imagen fila por fila. Pregunte: ¿qué se pierde? "
                  "(los píxeles vecinos de arriba y abajo quedan lejos). Y el costo: una foto "
                  "224 × 224 a color tiene 150 528 números; una capa densa de 1 000 neuronas ya "
                  "son 150 millones de pesos.")
        self.start("¿Y si la entrada es una imagen?", "Aplanar destruye la vecindad", PINK)
        g = pixel_grid(SMILE, 0.42).move_to([-4.3, 0.0, 0])
        self.play(LaggedStart(*[FadeIn(c) for c in g], lag_ratio=0.01))
        lab = safe_text("8 × 8 = 64 píxeles", size=22, color=MUTED).next_to(g, DOWN, buff=0.25)
        self.play(FadeIn(lab))
        cell = 0.17
        targets = [np.array([-1.5 + (k % 32) * cell, 1.0 - (k // 32) * 0.55, 0]) for k in
                   range(64)]
        flat = g.copy()
        self.play(*[c.animate.scale(cell / 0.42).move_to(t) for c, t in zip(flat, targets)],
                  run_time=2.2)
        g = flat
        a, b = g[3], g[11]
        hl = VGroup(SurroundingRectangle(a, color=YELLOW, buff=0.03),
                    SurroundingRectangle(b, color=YELLOW, buff=0.03))
        note = safe_text("vecinos arriba/abajo → a 8 posiciones de distancia", size=21,
                         color=YELLOW).move_to([1.5, -0.25, 0])
        self.play(Create(hl), FadeIn(note))
        nums = VGroup(
            VGroup(safe_text("foto 224 × 224 × 3", size=24, color=MUTED),
                   safe_text(f"{fmt_int(DENSE_IN)} entradas", size=28, color=INK, weight=BOLD)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(safe_text(f"capa densa de {fmt_int(DENSE_HIDDEN)} neuronas", size=24,
                             color=MUTED),
                   safe_text(f"{fmt_int(DENSE_PARAMS)} pesos", size=28, color=RED,
                             weight=BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            safe_text("y un gato movido 10 píxeles es otro vector", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([1.6, -2.0, 0])
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.1) for n in nums], lag_ratio=0.3))

    # ------------------------------------------------------------------
    def convolution(self) -> None:
        self.beat("50–52 min. La convolución: en lugar de un peso por píxel, un FILTRO pequeño "
                  "(3 × 3 = 9 pesos) que se desliza por toda la imagen; en cada posición hace "
                  "lo mismo que una neurona: suma ponderada. Este filtro (−1, 0, 1) detecta "
                  "bordes verticales: vale 0 en zonas planas y 3 donde la imagen pasa de oscuro "
                  "a claro. Pida que calculen la primera casilla antes de animar.")
        self.start("Convolución", "Un filtro pequeño que se desliza", PINK)
        cell = 0.62
        img = pixel_grid(IMG, cell, numbers=True, on=INK).move_to([-4.2, -0.35, 0])
        ker = VGroup()
        for i in range(3):
            for j in range(3):
                v = KERNEL[i, j]
                sq = Square(side_length=cell, stroke_color=PINK, stroke_width=2,
                            fill_color=PANEL, fill_opacity=1).move_to([j * cell, -i * cell, 0])
                t = MathTex(f"{v}", font_size=28, color=RED if v < 0 else (GREEN if v > 0 else
                                                                            MUTED)).move_to(sq)
                ker.add(VGroup(sq, t))
        ker.move_to([0.0, 1.0, 0])
        kl = safe_text("filtro (9 pesos)", size=20, color=PINK).next_to(ker, UP, buff=0.15)
        out_cells = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(side_length=cell, stroke_color=GRID, stroke_width=1.2,
                            fill_color=PANEL, fill_opacity=1).move_to([j * cell, -i * cell, 0])
                out_cells.add(sq)
        out_cells.move_to([4.1, -0.35, 0])
        il = safe_text("imagen 6 × 6", size=20, color=MUTED).next_to(img, UP, buff=0.15)
        ol = safe_text("mapa de rasgos 4 × 4", size=20, color=MUTED).next_to(out_cells, UP,
                                                                          buff=0.15)
        self.play(FadeIn(img), FadeIn(il))
        self.play(FadeIn(ker), FadeIn(kl), FadeIn(out_cells), FadeIn(ol))

        def window(i, j):
            cells = VGroup(*[img[(i + a) * 6 + (j + b)] for a in range(3) for b in range(3)])
            return SurroundingRectangle(cells, color=PINK, buff=0.0, stroke_width=5)

        win = window(0, 0)
        self.play(Create(win))
        vals = VGroup()
        for k in range(16):
            i, j = divmod(k, 4)
            v = int(FEATURE[i, j])
            t = MathTex(f"{v}", font_size=30, color=YELLOW if v else MUTED).move_to(
                out_cells[k])
            fill = out_cells[k].copy().set_fill(YELLOW if v else PANEL, 0.35 if v else 1)
            rt = 0.45 if k < 4 else 0.18
            self.play(win.animate.move_to(window(i, j)), run_time=rt)
            self.play(FadeIn(fill), FadeIn(t), run_time=rt)
            vals.add(fill, t)
        calc = VGroup(safe_text("en el borde:", size=20, color=MUTED),
                      tex(r"-1(0{+}0{+}0) + 0 + 1(1{+}1{+}1) = 3", size=28, color=INK,
                          max_width=3.4)).arrange(DOWN, buff=0.12).move_to([0.0, -1.0, 0])
        share = safe_text("los mismos 9 pesos en todas partes", size=24, color=PINK,
                          weight=BOLD).move_to([0, -2.6, 0])
        self.play(FadeIn(calc), FadeIn(share))

    # ------------------------------------------------------------------
    def cnn_stack(self) -> None:
        self.beat("52–54 min. Una CNN apila: convolución + ReLU (muchos filtros, cada uno un "
                  "mapa de rasgos), pooling que reduce el tamaño, y repetir. Las primeras capas "
                  "ven bordes; las siguientes, texturas y partes; las últimas, objetos. Al final "
                  "hay una capa densa: HIPERPLANOS, como en Baarle, pero en un espacio de rasgos "
                  "aprendido. La primera capa de 64 filtros 3×3×3 tiene 1 728 pesos, no 150 "
                  "millones.")
        self.start("Redes convolucionales", "Rasgos cada vez más abstractos", PINK)
        specs = [  # (ancho, alto, profundidad, color, etiqueta, abajo)
            (2.2, 2.2, 0.15, INK, "imagen", "224×224×3"),
            (1.8, 1.8, 0.55, PINK, "conv + ReLU", "bordes"),
            (1.2, 1.2, 0.55, PURPLE, "pooling", "÷2"),
            (1.0, 1.0, 0.9, PINK, "conv + ReLU", "texturas, partes"),
            (0.6, 0.6, 0.9, PURPLE, "pooling", "÷2"),
            (0.25, 1.9, 0.0, YELLOW, "densa", "hiperplanos"),
        ]
        xs = [-5.6, -3.3, -1.25, 0.6, 2.4, 4.2]
        blocks = VGroup()
        for (w, h, d, col, top, bot), x in zip(specs, xs):
            p = prism(w, h, d, col).move_to([x, 0.0, 0])
            t1 = safe_text(top, size=20, color=col, weight=BOLD).next_to(p, UP, buff=0.25)
            t2 = safe_text(bot, size=18, color=MUTED).next_to(p, DOWN, buff=0.25)
            blocks.add(VGroup(p, t1, t2))
        out = VGroup(*[safe_text(t, size=20, color=c) for t, c in
                       [("gato", GREEN), ("perro", MUTED), ("auto", MUTED)]]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2).move_to([6.05, 0.0, 0])
        arrows = VGroup(*[Arrow(blocks[i][0].get_right(), blocks[i + 1][0].get_left(), buff=0.12,
                                color=GRID, stroke_width=3, max_tip_length_to_length_ratio=0.3)
                          for i in range(5)])
        arrows.add(Arrow(blocks[5][0].get_right(), out.get_left(), buff=0.12, color=GRID,
                         stroke_width=3, max_tip_length_to_length_ratio=0.3))
        for i, b in enumerate(blocks):
            self.play(FadeIn(b, shift=RIGHT * 0.1), *([GrowArrow(arrows[i - 1])] if i else []),
                      run_time=0.5)
        self.play(GrowArrow(arrows[5]), FadeIn(out))
        facts = VGroup(
            VGroup(safe_text("64 filtros 3×3×3:", size=22, color=MUTED),
                   safe_text(f"{fmt_int(CONV_PARAMS)} pesos", size=24, color=PINK, weight=BOLD)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(safe_text("vs. densa:", size=22, color=MUTED),
                   safe_text(f"{fmt_int(DENSE_PARAMS)} pesos", size=24, color=RED, weight=BOLD)
                   ).arrange(RIGHT, buff=0.25),
            safe_text("pesos compartidos · conexiones locales · invariancia a traslaciones",
                      size=21, color=INK),
        ).arrange(DOWN, buff=0.18).move_to([0, -2.55, 0])
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.1) for f in facts], lag_ratio=0.25))

    # ------------------------------------------------------------------
    def timeline(self) -> None:
        self.beat("54–56 min. Una historia corta. 1958: perceptrón (una neurona, una recta). "
                  "1969: Minsky y Papert muestran que no resuelve XOR, y el campo se enfría. "
                  "1986: retropropagación popularizada. 1998: LeNet lee cheques. 2012: AlexNet "
                  "(CNN + GPU + ReLU) gana ImageNet por mucho. 2015: ResNet, 152 capas. 2017 y "
                  "2020: Transformers y ViT. La idea geométrica de hoy sigue debajo de todo.")
        self.start("De dónde viene y a dónde fue", "Setenta años en una recta", PINK)
        ln = Line([-6.4, -0.2, 0], [6.4, -0.2, 0], color=GRID, stroke_width=4)
        events = [
            (1958, "Perceptrón", "una neurona = una recta", TEAL),
            (1969, "XOR", "límite de lo lineal", RED),
            (1986, "Backprop", "entrenar capas ocultas", RED),
            (1998, "LeNet-5", "CNN lee dígitos", PINK),
            (2012, "AlexNet", "CNN + GPU + ReLU", PINK),
            (2015, "ResNet", "152 capas", PURPLE),
            (2017, "Transformer", "atención", YELLOW),
            (2020, "ViT", "imágenes como parches", YELLOW),
        ]
        x0, x1 = -5.7, 5.7
        self.play(Create(ln))
        items = VGroup()
        for k, (yr, name, desc, col) in enumerate(events):
            x = x0 + k * (x1 - x0) / (len(events) - 1)
            up = k % 2 == 0
            d = Dot([x, -0.2, 0], radius=0.09, color=col)
            yr_t = safe_text(str(yr), size=20, color=col, weight=BOLD)
            nm = safe_text(name, size=22, color=INK, weight=BOLD)
            ds = safe_text(desc, size=17, color=MUTED, max_width=1.9)
            block = VGroup(yr_t, nm, ds).arrange(DOWN, buff=0.08)
            stem = Line([x, -0.2, 0], [x, 0.55 if up else -0.95, 0], color=col, stroke_width=2)
            if up:
                block.next_to(stem, UP, buff=0.08)
            else:
                block.next_to(stem, DOWN, buff=0.08)
            items.add(VGroup(d, stem, block))
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.1) for i in items], lag_ratio=0.18),
                  run_time=3)
