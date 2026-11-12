"""Acto 7 · UMAP: confiar sólo en lo cercano (45–54 min).

Un rollo de papel en 3D: la recta por el aire miente, el camino sobre la hoja
no. UMAP sólo cree en distancias cortas (grafo k-NN), da a cada punto su
propia regla (ρ_i, σ_i) y busca un dibujo en 2D que conserve el grafo, no las
distancias. El ejemplo numérico de la recta A–E reproduce el del capítulo.
"""
from __future__ import annotations

import math

from manim import *

from sections.data import (LINE_POS, ROLL_3D, ROLL_COLOR_T, ROLL_EDGES, ROLL_EUC_AB, ROLL_FLAT,
                           ROLL_GEO_AB, ROLL_H, ROLL_PA, ROLL_PB, ROLL_T, H_MAX, K_NN, UMAP_K,
                           UMAP_RHO, UMAP_SIGMA, UMAP_W)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, gradient_color, pill, safe_text, tex)

SC3 = 0.17
SCF = 0.112
FLAT_SHIFT = np.array([0.0, -0.35, 0.0])
PHI_R, THETA_R = 72 * DEGREES, -62 * DEGREES


def roll_pos(i: int) -> np.ndarray:
    return ROLL_3D[i] * SC3


def flat_pos(i: int) -> np.ndarray:
    return np.array([ROLL_FLAT[i, 0] * SCF, ROLL_FLAT[i, 1] * SCF, 0.0]) + FLAT_SHIFT


class UmapSection:
    def umap(self) -> None:
        self.beat("45–46 min. Los datos reales suelen vivir sobre una superficie doblada dentro de "
                  "un espacio grande. Este rollo de papel tiene puntos coloreados desde el centro "
                  "(azul) hasta el borde (amarillo). Pregunte: ¿estos dos puntos marcados son "
                  "vecinos?")
        head = self.start("UMAP", "Una hoja enrollada en 3D", PINK, fixed=True)
        self.move_camera(phi=PHI_R, theta=THETA_R, zoom=1.05, run_time=1.2)
        dots = VGroup(*[Dot3D(roll_pos(i), radius=0.05, color=gradient_color(ROLL_COLOR_T[i]),
                              resolution=(4, 4)) for i in range(len(ROLL_3D))])
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.002), run_time=1.6)
        pa, pb = roll_pos(ROLL_PA), roll_pos(ROLL_PB)
        ma = Dot3D(pa, radius=0.12, color=WHITE, resolution=(8, 8))
        mb = Dot3D(pb, radius=0.12, color=WHITE, resolution=(8, 8))
        self.play(GrowFromCenter(ma), GrowFromCenter(mb))

        self.beat("46–47 min. Por el aire (rojo) están cerca: 6.3. Caminando sobre la hoja (verde) "
                  "están a 68.6, porque hay que dar una vuelta completa. La distancia euclidiana "
                  "en 3D mezcla capas distintas; la que importa es la que se recorre SOBRE la "
                  "hoja. Y en una distancia corta, aire y hoja casi coinciden.")
        air = Line(pa, pb, color=RED, stroke_width=6)
        h_mid = 0.5 * (ROLL_H[ROLL_PA] + ROLL_H[ROLL_PB]) - H_MAX / 2
        ta, tb = ROLL_T[ROLL_PA], ROLL_T[ROLL_PB]
        geo = ParametricFunction(lambda t: SC3 * np.array([t * np.cos(t), h_mid, t * np.sin(t)]),
                                 t_range=[ta, tb, 0.03], color=GREEN, stroke_width=6)
        self.play(Create(air))
        self.play(Create(geo), run_time=2.0)
        panel = VGroup(
            VGroup(safe_text("por el aire", size=24, color=RED, weight=BOLD),
                   tex(rf"{ROLL_EUC_AB:.1f}", size=36, color=RED)).arrange(RIGHT, buff=0.3),
            VGroup(safe_text("sobre la hoja", size=24, color=GREEN, weight=BOLD),
                   tex(rf"{ROLL_GEO_AB:.1f}", size=36, color=GREEN)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        pbg = card(panel.width + 0.5, panel.height + 0.45, PINK)
        pblock = VGroup(pbg, panel.move_to(pbg)).to_corner(DR, buff=0.4).shift(UP * 0.55)
        self.fixed(pblock)
        self.play(FadeIn(pblock))

        self.beat(f"47–48 min. Idea central de UMAP: sólo creer en las distancias CORTAS. Cada "
                  f"punto se conecta con sus {K_NN} vecinos más cercanos. Gire la escena: las "
                  "aristas siguen la hoja y ninguna salta a la otra capa. El grafo ya 'sabe' que "
                  "los dos puntos blancos están lejos.")
        edges = VGroup(*[Line(roll_pos(i), roll_pos(j), stroke_width=1.3, color=PINK,
                              stroke_opacity=0.55) for i, j in ROLL_EDGES])
        self.play(FadeOut(air), FadeOut(geo), FadeOut(pblock), Create(edges), run_time=2.0)
        tag = safe_text(f"grafo de {K_NN} vecinos más cercanos", size=24, color=PINK, weight=BOLD)
        tbg = card(tag.width + 0.5, 0.75, PINK)
        tblock = VGroup(tbg, tag.move_to(tbg)).to_corner(DR, buff=0.4).shift(UP * 0.55)
        self.fixed(tblock)
        self.play(FadeIn(tblock))
        self.next_slide(notes="Giro del grafo. Señale que las aristas sólo unen puntos de la misma "
                              "capa.", loop=True)
        self.move_camera(theta=THETA_R + TAU, run_time=9, rate_func=linear)
        self.set_camera_orientation(theta=THETA_R)

        self.beat("48–50 min. ¿Qué tan fuerte es cada arista? Cada punto usa su PROPIA regla. "
                  "Ejemplo del capítulo: cinco puntos en 0, 1, 3, 5, 6 y k = 3. Para A, ρ = 1 "
                  "(distancia a su vecino más cercano): ese vecino recibe peso 1 sin importar qué "
                  "tan lejos esté. Después el peso decae con un ritmo σ elegido para que la suma "
                  "sea log₂ 3 = 1.585.")
        self.saved_roll = (dots, ma, mb, edges)
        head = self.start("UMAP", "Cada punto tiene su propia regla", PINK)
        nl = NumberLine(x_range=[0, 6, 1], length=10.0, include_numbers=True, font_size=26,
                        color=MUTED).move_to([0, -2.2, 0])
        pts = VGroup(*[Dot(nl.n2p(v), radius=0.12, color=YELLOW if n == "A" else BLUE)
                       for n, v in LINE_POS.items()])
        names = VGroup(*[safe_text(n, size=26, color=YELLOW if n == "A" else BLUE, weight=BOLD)
                         .next_to(d, UP, buff=0.18) for n, d in zip(LINE_POS, pts)])
        self.play(Create(nl), FadeIn(pts), FadeIn(names))
        ax = Axes(x_range=[0, 6, 1], y_range=[0, 1.1, 0.5], x_length=10.0, y_length=2.4, tips=False,
                  axis_config={"color": GRID, "stroke_width": 1.2},
                  y_axis_config={"include_numbers": True, "font_size": 22}).move_to([0, 0.05, 0])
        ax.shift((nl.n2p(0)[0] - ax.c2p(0, 0)[0]) * RIGHT)
        rho = Brace(Line(nl.n2p(0), nl.n2p(UMAP_RHO)), DOWN, color=PINK, buff=0.55)
        rho_l = tex(r"\rho_A = 1", size=30, color=PINK).next_to(rho, DOWN, buff=0.06)
        self.play(GrowFromCenter(rho), FadeIn(rho_l))
        curve = ax.plot(lambda d: 1.0 if d <= UMAP_RHO else math.exp(-(d - UMAP_RHO) / UMAP_SIGMA),
                        x_range=[0, 6, 0.02], color=PINK, stroke_width=5)
        self.play(Create(ax), Create(curve), run_time=1.4)
        bars = VGroup()
        for n in ("B", "C", "D"):
            x = LINE_POS[n]
            w = UMAP_W[n]
            bar = Line(ax.c2p(x, 0), ax.c2p(x, w), color=YELLOW, stroke_width=10)
            val = tex(f"{w:.3f}" if w < 1 else "1", size=28, color=YELLOW).next_to(bar, UR, buff=0.05)
            bars.add(VGroup(bar, val))
        self.play(LaggedStart(*[Create(b[0]) for b in bars], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(b[1]) for b in bars], lag_ratio=0.3))
        eq = tex(rf"1 + e^{{-2/\sigma}} + e^{{-4/\sigma}} = \log_2 3 \;\Rightarrow\; "
                 rf"\sigma_A = {UMAP_SIGMA:.2f}", size=34, color=INK).move_to([0, 1.95, 0])
        self.play(Write(eq))

        self.beat("50–51 min. Formalización del grafo. Restar ρ_i garantiza que nadie quede aislado "
                  "(robusto en zonas dispersas); σ_i se ajusta para que todos tengan el mismo "
                  "número efectivo de vecinos. Como el peso de i a j puede diferir del de j a i, se "
                  "combinan como una unión difusa: probabilidad de que exista al menos una de las "
                  "dos aristas.")
        head = self.swap_heading(head, "Formalización", "El grafo difuso de UMAP", PINK)
        self.clear_except(head)
        f = VGroup(
            tex(r"v_{j\mid i} = \exp\!\left(-\frac{d_{ij}-\rho_i}{\sigma_i}\right),\qquad"
                r"\sum_{j\in \mathrm{kNN}(i)} v_{j\mid i} = \log_2 k", size=40, color=INK),
            tex(r"v_{ij} = v_{j\mid i} + v_{i\mid j} - v_{j\mid i}\,v_{i\mid j}", size=40,
                color=PINK),
        ).arrange(DOWN, buff=0.55).move_to([0, 0.9, 0])
        notes = VGroup(
            safe_text("ρᵢ: distancia al vecino más cercano → nadie queda aislado", size=22,
                      color=MUTED),
            safe_text("σᵢ: regla local, se estira en zonas dispersas y se encoge en densas",
                      size=22, color=MUTED),
            safe_text("unión difusa: simetriza el grafo", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, -1.5, 0])
        self.play(Write(f[0]))
        self.play(Write(f[1]))
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.2) for n in notes], lag_ratio=0.3))

        self.beat("51–52 min. Ahora el dibujo. UMAP busca posiciones en 2D cuyo grafo se parezca "
                  "al grafo difuso: vecinos se atraen, no vecinos se repelen. Aquí mostramos el "
                  "objetivo ideal: la hoja desenrollada, con el color en orden. Aclare que en datos "
                  "reales el resultado es aproximado y depende de la semilla.")
        head = self.start("UMAP", "Desenrollar conservando vecinos", PINK, fixed=True)
        dots, ma, mb, edges = self.saved_roll
        self.set_camera_orientation(phi=PHI_R, theta=THETA_R, zoom=1.05)
        self.play(FadeIn(dots), FadeIn(edges), FadeIn(ma), FadeIn(mb), run_time=0.8)
        self.play(FadeOut(edges), run_time=0.5)
        moves = [d.animate.move_to(flat_pos(i)) for i, d in enumerate(dots)]
        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=3.2,
                         added_anims=moves + [ma.animate.move_to(flat_pos(ROLL_PA)),
                                              mb.animate.move_to(flat_pos(ROLL_PB))])
        flat_edges = VGroup(*[Line(flat_pos(i), flat_pos(j), stroke_width=1.3, color=PINK,
                                   stroke_opacity=0.5) for i, j in ROLL_EDGES])
        self.play(Create(flat_edges), run_time=1.4)
        far = safe_text("los puntos blancos quedan lejos, como sobre la hoja", size=24,
                        color=INK).move_to([0, -2.55, 0])
        self.play(FadeIn(far))

        self.beat("52–53 min. Formalización del dibujo. En 2D la similitud usa una curva de cola "
                  "pesada con parámetros a y b (fijados por min_dist). El costo es una entropía "
                  "cruzada: el primer término atrae a los vecinos (v alto), el segundo repele a los "
                  "no vecinos (1 − v alto). n_neighbors controla qué tan local es la regla.")
        head = self.start("Formalización", "El dibujo: atraer vecinos, repeler al resto", PINK)
        w = tex(r"w_{ij} = \big(1 + a\,\lVert \mathbf{y}_i-\mathbf{y}_j\rVert^{2b}\big)^{-1}",
                size=40, color=INK).move_to([0, 1.55, 0])
        ce = tex([r"\sum_{i\neq j}\Big[", r"v_{ij}\log\frac{v_{ij}}{w_{ij}}", r"\;+\;",
                  r"(1-v_{ij})\log\frac{1-v_{ij}}{1-w_{ij}}", r"\Big]"], size=40, color=INK)
        ce.move_to([0, 0.15, 0])
        ce[1].set_color(GREEN)
        ce[3].set_color(RED)
        att = safe_text("atrae vecinos", size=22, color=GREEN, weight=BOLD)
        att.next_to(ce[1], DOWN, buff=0.3)
        rep = safe_text("repele no vecinos", size=22, color=RED, weight=BOLD)
        rep.next_to(ce[3], DOWN, buff=0.3)
        self.play(Write(w))
        self.play(Write(ce))
        self.play(FadeIn(att), FadeIn(rep))
        hp = VGroup(pill("n_neighbors: qué tan local es la regla", PINK, size=22),
                    pill("min_dist: qué tan apretado queda el dibujo", PURPLE, size=22)
                    ).arrange(RIGHT, buff=0.3).move_to([0, -2.0, 0])
        self.play(FadeIn(hp, shift=UP * 0.15))

        self.beat("53–54 min. Cierre la idea con los axiomas: UMAP no construye una métrica global; "
                  "cada punto tiene su regla y el dibujo sólo promete conservar vecindarios. Por "
                  "eso los tamaños de los grupos y las distancias entre grupos del mapa no se "
                  "interpretan. Para agrupar con UMAP, se reduce a 5–10 dimensiones y se usa un "
                  "método de densidad.")
        head = self.swap_heading(head, "Cómo leer un mapa de UMAP", "UMAP no mide: dibuja vecindarios",
                                 PINK)
        self.clear_except(head)
        items = [("conserva", "quién es vecino de quién", GREEN),
                 ("no conserva", "tamaños de los grupos", RED),
                 ("no conserva", "distancias entre grupos", RED),
                 ("no es", "una métrica global: cada punto usa su regla", RED)]
        rows = VGroup()
        for verb, what, col in items:
            v = safe_text(verb, size=26, color=col, weight=BOLD)
            t = safe_text(what, size=26, color=INK)
            rows.add(VGroup(v, t).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.3))
