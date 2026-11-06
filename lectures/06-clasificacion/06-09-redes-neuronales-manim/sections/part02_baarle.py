"""Acto 2 · Baarle: una frontera imposible para una recta (11–19 min).

El mapa estilizado de Baarle-Hertog (Bélgica) dentro de Baarle-Nassau (Países
Bajos). Clasificar (x, y) → país. La mejor recta (regresión logística) sólo
captura el sur; apilar capas lineales no ayuda: su composición vuelve a ser una
recta. Hace falta algo que «doble» el plano.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (DEEP_LINEAR, LOGISTIC, SHOW_X, SHOW_Y, TRAIN_Y, accuracy, collapse,
                           unit_line_to_map)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, tex)
from sections.viz import MapView, map_legend

MAP_CENTER = (-2.75, -0.62, 0)
MAP_W = 7.3


class BaarleSection:
    def baarle(self) -> None:
        self.story()
        self.points()
        self.logistic()
        self.collapse_layers()

    # ------------------------------------------------------------------
    def story(self) -> None:
        self.beat("11–13 min. Baarle: Baarle-Nassau es neerlandés y Baarle-Hertog es belga, "
                  "pero Baarle-Hertog está hecho de 20 enclaves belgas dentro de Países Bajos, "
                  "y dentro de algunos de ellos hay 7 contraenclaves neerlandeses. La frontera "
                  "cruza calles, cafés y hasta casas (la nacionalidad la decide la puerta). "
                  "Aclare que el mapa es ESTILIZADO: pocos pedazos, pero con la misma rareza: "
                  "Bélgica está partida y un pedazo tiene un hueco.")
        self.start("El ejemplo de hoy", "Baarle: la frontera más rara de Europa", BLUE)
        mv = MapView(MAP_W, MAP_CENTER)
        self.mv = mv
        img = mv.truth_image()
        borders = mv.borders()
        self.play(FadeIn(img), Create(mv.frame()), run_time=0.8)
        self.play(Create(borders), run_time=2.0)
        tags = VGroup(
            safe_text("Bélgica (sur)", size=20, color=BLUE, weight=BOLD).move_to(mv.p(5, 0.55)),
            safe_text("enclave belga", size=17, color=BLUE).move_to(mv.p(2.4, 5.85)),
            safe_text("contraenclave", size=17, color=ORANGE).move_to(mv.p(2.4, 2.15)),
            safe_text("Países Bajos", size=20, color=ORANGE, weight=BOLD).move_to(mv.p(4.6, 6.4)),
        )
        arrow = Arrow(mv.p(2.5, 2.4), mv.p(2.8, 3.75), buff=0.02, color=ORANGE, stroke_width=3,
                      max_tip_length_to_length_ratio=0.3)
        self.play(FadeIn(tags), GrowArrow(arrow))
        facts = VGroup(
            safe_text("Baarle-Hertog (Bélgica)", size=24, color=BLUE, weight=BOLD),
            safe_text("20 enclaves dentro de", size=22),
            safe_text("Baarle-Nassau (Países Bajos)", size=24, color=ORANGE, weight=BOLD),
            safe_text("7 contraenclaves neerlandeses", size=22),
            safe_text("dentro de los enclaves belgas", size=22),
            safe_text("la frontera cruza casas", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        box = card(facts.width + 0.6, facts.height + 0.5, GRID)
        panel = VGroup(box, facts.move_to(box)).move_to([3.95, 0.35, 0])
        note = safe_text("Mapa estilizado: misma topología, menos pedazos", size=18,
                         color=MUTED).next_to(panel, DOWN, buff=0.3)
        self.play(FadeIn(panel, shift=LEFT * 0.1), FadeIn(note))
        self.baarle_map = Group(img, borders)
        self.baarle_tags = VGroup(tags, arrow)

    # ------------------------------------------------------------------
    def points(self) -> None:
        self.beat("13–14 min. Lo convertimos en un problema de clasificación: cada punto es "
                  "una coordenada (x, y) y su etiqueta es el país. Es un clasificador como los "
                  "de clases pasadas, sólo que con una frontera endiablada. 65 % del territorio "
                  "es neerlandés: decir siempre «Países Bajos» ya acierta 65 %. Pregunte: "
                  "¿cuántas rectas necesitarían para separarlo?")
        self.start("Planteamiento", "Entrada (x, y) → ¿Bélgica o Países Bajos?", BLUE)
        mv = self.mv
        frame = mv.frame()
        dots = mv.dots(SHOW_X, SHOW_Y)
        self.play(Create(frame))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.002),
                  run_time=1.6)
        leg = map_legend().move_to([3.9, 1.5, 0])
        base = 1 - np.mean(TRAIN_Y == 0)
        info = VGroup(
            tex(r"(x, y) \;\longmapsto\; \hat y \in \{\text{BE}, \text{NL}\}", size=34),
            safe_text(f"siempre «Países Bajos»: {base:.0%} de acierto", size=22, color=MUTED),
            safe_text("¿cuántas rectas hacen falta?", size=26, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.95, -0.6, 0])
        self.play(FadeIn(leg), LaggedStart(*[FadeIn(i, shift=LEFT * 0.1) for i in info],
                                            lag_ratio=0.3))
        self.dots = dots
        self.frame = frame
        self.side = VGroup(leg, info)

    # ------------------------------------------------------------------
    def logistic(self) -> None:
        self.beat("14–16 min. Primer intento: una sola neurona con sigmoide, es decir, "
                  "regresión logística. Entrenada con 4 000 puntos encuentra la mejor recta "
                  "posible: corta el sur y acierta 75 %. Señale los enclaves: TODOS quedan del "
                  "lado neerlandés. No es mala optimización: ninguna recta puede hacerlo mejor "
                  "por mucho.")
        mv = self.mv
        head = self.heading("Primer intento", "Una neurona: regresión logística", YELLOW)
        old = [m for m in self.mobjects if m not in (self.branding, self.dots, self.frame)]
        img = mv.prob_image(LOGISTIC, sharp=0.6)
        a, c, d = unit_line_to_map(LOGISTIC[0][:, 0], LOGISTIC[1][0])
        ln = mv.line(a, c, d, YELLOW, 6)
        self.play(*[FadeOut(m) for m in old], FadeIn(head))
        self.play(FadeIn(img), self.dots.animate.set_opacity(0.85))
        self.bring_to_front(self.dots)
        self.play(Create(ln))
        acc = accuracy(LOGISTIC)
        res = VGroup(
            tex(r"\hat y = \sigma(w_1 x + w_2 y + b)", size=34),
            VGroup(safe_text("exactitud:", size=26, color=MUTED),
                   safe_text(f"{acc:.1%}", size=34, color=YELLOW, weight=BOLD)
                   ).arrange(RIGHT, buff=0.2),
            safe_text("acierta el sur…", size=24, color=BLUE),
            safe_text("…y pierde todos los enclaves", size=24, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([3.95, 0.2, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in res], lag_ratio=0.25))
        mv_b = mv.borders(color=RED, width=3.5)
        self.play(Create(mv_b), run_time=1.5)
        self.logit_img, self.logit_line = img, ln

    # ------------------------------------------------------------------
    def collapse_layers(self) -> None:
        self.beat("16–18 min. Segundo intento, ingenuo: apilar capas, 2 → 16 → 16 → 1, pero "
                  "SIN activación. Pregunte antes de revelar: ¿mejorará? Haga el álgebra en "
                  "voz alta: W₃(W₂(W₁x + b₁) + b₂) + b₃ = W x + b′. Por profunda que sea, una "
                  "red sin no linealidad es UNA neurona disfrazada.")
        self.start("Segundo intento", "¿Y si apilamos muchas capas lineales?", RED)
        algebra = VGroup(
            tex(r"h_1 = W_1 x + b_1", size=36),
            tex(r"h_2 = W_2 h_1 + b_2", size=36),
            tex(r"\hat z = W_3 h_2 + b_3", size=36),
            tex(r"\hat z = \underbrace{W_3 W_2 W_1}_{W} \, x \;+\; "
                r"\underbrace{W_3 W_2 b_1 + W_3 b_2 + b_3}_{b'}", size=36, color=YELLOW,
                max_width=6.4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([-3.35, 0.3, 0])
        self.play(LaggedStart(*[FadeIn(a, shift=RIGHT * 0.1) for a in algebra[:3]],
                              lag_ratio=0.3))
        self.play(TransformFromCopy(VGroup(*algebra[:3]), algebra[3]))
        verdict = safe_text("una composición de funciones lineales es lineal", size=24,
                            color=YELLOW, max_width=6.4).next_to(algebra, DOWN, buff=0.45).align_to(algebra,
                                                                                      LEFT)
        self.play(FadeIn(verdict))

        self.beat("18–19 min. Lo comprobamos: entrenamos la red lineal profunda (337 "
                  "parámetros). Su frontera es otra recta y acierta lo mismo, 75 %. Pregunta "
                  "topológica: un semiplano es UNA pieza sin huecos; Bélgica en Baarle tiene "
                  "cinco piezas y un hueco. Necesitamos una operación que doble el plano.")
        mv = MapView(5.6, (3.55, -0.35, 0))
        img = mv.prob_image(DEEP_LINEAR, linear=True, sharp=0.6)
        w, b = collapse(DEEP_LINEAR)
        a, c, d = unit_line_to_map(w, b)
        ln = mv.line(a, c, d, YELLOW, 6)
        a2, c2, d2 = unit_line_to_map(LOGISTIC[0][:, 0], LOGISTIC[1][0])
        ln2 = mv.line(a2, c2, d2, MUTED, 3)
        ln2 = DashedVMobject(ln2, num_dashes=30)
        dots = mv.dots(SHOW_X[::3], SHOW_Y[::3], r=0.045)
        self.play(FadeIn(img), Create(mv.frame()), FadeIn(dots))
        self.play(Create(ln), Create(ln2))
        acc = accuracy(DEEP_LINEAR, linear=True)
        cap = VGroup(
            safe_text(f"red lineal 2→16→16→1: {acc:.1%}", size=22, color=YELLOW, weight=BOLD),
            safe_text("(punteada: la logística)", size=18, color=MUTED),
        ).arrange(DOWN, buff=0.08).next_to(mv.frame(), UP, buff=0.18)
        self.play(FadeIn(cap))
        need = pill("Necesitamos una no linealidad", RED, size=24).move_to([-3.35, -2.85, 0])
        self.play(FadeIn(need, shift=UP * 0.1))
