"""Acto 8 · Volver con Ana y elegir una distancia (54–60 min).

Gower combina variables mixtas; cambiar pesos cambia al más parecido. Reto en
parejas: elegir distancia para cuatro conjuntos de datos. Mapa del capítulo:
representación → escala → ¿métrica? → dimensión → algoritmo → desempeño.
"""
from __future__ import annotations

from manim import *

from sections.data import GOWER_EQ, GOWER_W
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, fmt, safe_text, tex)


class CloseSection:
    def close_gower(self) -> None:
        self.beat("54–55 min. Volvemos con Ana. Para variables mixtas, Gower promedia contribuciones "
                  "en [0, 1]: diferencia sobre rango para las numéricas, 0/1 para las categóricas. "
                  "Con pesos iguales gana Carla. Pregunte: ¿qué decisión implícita tomó Gower? Un "
                  "desacuerdo de canal pesa lo mismo que 27 años de edad.")
        head = self.start("Volver con Ana", "Gower: medir con variables mixtas", YELLOW)
        g = tex(r"d_G(\mathbf{x},\mathbf{y}) = \frac{\sum_j w_j\, d_j(x_j,y_j)}{\sum_j w_j},"
                r"\quad d_j = \begin{cases} |x_j-y_j|/R_j & \text{num\'erica}\\"
                r"\mathbf{1}(x_j\neq y_j) & \text{categ\'orica}\end{cases}", size=34, color=INK)
        g.move_to([0, 1.45, 0])
        self.play(Write(g))
        names = list(GOWER_EQ)
        ax_x0, ax_len = -2.2, 7.5
        rows = VGroup()
        for k, n in enumerate(names):
            y = 0.1 - 0.72 * k
            lab = safe_text(n, size=26, color=INK).move_to([ax_x0 - 0.9, y, 0])
            bar = Rectangle(width=GOWER_EQ[n] * ax_len, height=0.42, stroke_width=0,
                            fill_color=TEAL, fill_opacity=0.85)
            bar.move_to([ax_x0, y, 0], aligned_edge=LEFT)
            val = safe_text(fmt(GOWER_EQ[n], 3), size=22, color=INK).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, val))
        cap = safe_text("pesos (1, 1, 1, 1)", size=24, color=TEAL, weight=BOLD)
        cap.move_to([4.6, 0.15, 0])
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                for r in rows], lag_ratio=0.2), FadeIn(cap))
        best = min(GOWER_EQ, key=GOWER_EQ.get)
        hl = SurroundingRectangle(rows[names.index(best)], color=YELLOW, buff=0.08)
        self.play(Create(hl))

        self.beat("55–56 min. El negocio decide que canal y tarjeta pesen la mitad: pesos "
                  "(1, 1, ½, ½). Las barras se mueven y ahora gana Daniel. Ninguna respuesta es la "
                  "correcta en abstracto: medir es DECIDIR, y la decisión debe ser explícita y "
                  "justificada con el problema.")
        new_rows = VGroup()
        for k, n in enumerate(names):
            r = rows[k]
            bar = Rectangle(width=GOWER_W[n] * ax_len, height=0.42, stroke_width=0,
                            fill_color=PURPLE, fill_opacity=0.85)
            bar.move_to(r[1].get_left(), aligned_edge=LEFT)
            val = safe_text(fmt(GOWER_W[n], 3), size=22, color=INK).next_to(bar, RIGHT, buff=0.15)
            new_rows.add(VGroup(r[0].copy(), bar, val))
        cap2 = safe_text("pesos (1, 1, ½, ½)", size=24, color=PURPLE, weight=BOLD).move_to(cap)
        best2 = min(GOWER_W, key=GOWER_W.get)
        hl2 = SurroundingRectangle(new_rows[names.index(best2)], color=YELLOW, buff=0.08)
        self.play(*[Transform(rows[k][1], new_rows[k][1]) for k in range(4)],
                  *[Transform(rows[k][2], new_rows[k][2]) for k in range(4)],
                  Transform(cap, cap2), Transform(hl, hl2), run_time=1.8)
        msg = safe_text("Medir es decidir", size=34, color=YELLOW, weight=BOLD).move_to([4.6, -0.7, 0])
        self.play(Write(msg))

    def challenge(self) -> None:
        self.beat("56–58 min. Reto en parejas, 90 segundos. Para cada conjunto, elijan una distancia "
                  "o similitud y digan qué axioma o qué transformación les preocupa. No revele "
                  "hasta que cada pareja haya escrito sus cuatro respuestas.")
        head = self.start("Reto en parejas", "¿Con qué medirías el parecido?", GREEN)
        cases = [
            ("Canastas de compra", "qué compró cada cliente (sí/no)", "Jaccard",
             "la coausencia no informa", BLUE),
            ("Reseñas de texto", "conteos de palabras", "Coseno",
             "la longitud no es el tema", PURPLE),
            ("Sensores correlacionados", "temperatura, presión, humedad", "Mahalanobis",
             "corrige escala y correlación", TEAL),
            ("Clientes mixtos", "edad, ingreso, canal, tarjeta", "Gower + pesos",
             "decidir los pesos de forma explícita", YELLOW),
        ]
        cards = VGroup()
        for title, desc, ans, why, col in cases:
            box = card(5.9, 1.95, col)
            t = safe_text(title, size=26, color=col, weight=BOLD, max_width=5.4)
            d = safe_text(desc, size=19, color=MUTED, max_width=5.4)
            q = safe_text("?", size=34, color=INK, weight=BOLD)
            a = safe_text(ans, size=28, color=INK, weight=BOLD)
            w = safe_text(why, size=18, color=col, max_width=5.4)
            top = VGroup(t, d).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            top.move_to(box).align_to(box, UL).shift(RIGHT * 0.3 + DOWN * 0.25)
            q.move_to(box).shift(DOWN * 0.45)
            a.move_to(box).shift(DOWN * 0.35)
            w.next_to(a, DOWN, buff=0.08)
            cards.add(VGroup(box, top, q, a, w))
        cards.arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.3)).move_to([0, -0.55, 0])
        for c in cards:
            c[3].move_to(c[0]).shift(DOWN * 0.3)
            c[4].next_to(c[3], DOWN, buff=0.08)
            c[2].move_to(c[0]).shift(DOWN * 0.4)
        self.play(LaggedStart(*[FadeIn(VGroup(c[0], c[1], c[2]), shift=UP * 0.15) for c in cards],
                              lag_ratio=0.2))

        self.beat("Revele una por una y pida a una pareja distinta que justifique cada respuesta. "
                  "Errores frecuentes: euclidiana sobre conteos de texto (confunde longitud con "
                  "tema) y coincidencia simple en canastas (las coausencias inflan la similitud).")
        for c in cards:
            self.play(FadeOut(c[2]), FadeIn(c[3], shift=UP * 0.1), FadeIn(c[4]), run_time=0.6)

    def closing(self) -> None:
        self.beat("58–59 min. El sentido del capítulo en una línea: antes de agrupar hay que "
                  "decidir la representación, la escala, verificar si la medida es una métrica (si "
                  "el algoritmo lo necesita), revisar la dimensión y sólo entonces elegir algoritmo. "
                  "Al final, el desempeño (silueta, estabilidad) se evalúa con OTRA medida.")
        head = self.start("El sentido del capítulo", "Antes de agrupar, decidir cómo medir", TEAL)
        steps = [("representar", "tipos de variable", BLUE), ("escalar", "pesos explícitos", ORANGE),
                 ("¿métrica?", "4 axiomas", GREEN), ("¿dimensión?", "concentración", RED),
                 ("algoritmo", "k-medias · UMAP", PINK), ("evaluar", "silueta ≠ distancia", YELLOW)]
        chips = VGroup()
        for verb, what, col in steps:
            v = safe_text(verb, size=24, color=col, weight=BOLD, max_width=1.72)
            w = safe_text(what, size=17, color=MUTED, max_width=1.72)
            box = card(1.92, 1.35, col)
            VGroup(v, w).arrange(DOWN, buff=0.14).move_to(box)
            chips.add(VGroup(box, v, w))
        chips.arrange(RIGHT, buff=0.2).move_to([0, 0.55, 0])
        arrows = VGroup(*[Arrow(chips[i].get_right(), chips[i + 1].get_left(), buff=0.02,
                                color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.5)
                          for i in range(len(chips) - 1)])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.15),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15))
        ideas = VGroup(
            safe_text("1. Una métrica cumple 4 axiomas; similitud y desempeño son otra cosa.",
                      size=23, color=INK),
            safe_text("2. Transformar el espacio (escala, Σ, dirección) cambia quién es vecino.",
                      size=23, color=INK),
            safe_text("3. En alta dimensión las distancias se concentran; UMAP confía en lo local.",
                      size=23, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -1.75, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in ideas], lag_ratio=0.3))

        self.beat("59–60 min. Ticket de salida en una hoja: (1) un ejemplo de similitud que no sea "
                  "métrica y qué axioma falla; (2) por qué no se interpretan las distancias entre "
                  "grupos en un mapa de UMAP. Siguiente clase: k-medias, que usa la euclidiana al "
                  "cuadrado.")
        head = self.swap_heading(head, "Ticket de salida", "Antes de irte, responde", YELLOW)
        self.clear_except(head)
        qs = VGroup(
            safe_text("1. Da una similitud que no sea métrica. ¿Qué axioma falla?", size=28,
                      color=INK),
            safe_text("2. ¿Por qué no se leen las distancias entre grupos en UMAP?", size=28,
                      color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(q, shift=RIGHT * 0.2) for q in qs], lag_ratio=0.4))
        nxt = VGroup(safe_text("Siguiente clase", size=22, color=MUTED),
                     safe_text("k-medias: agrupar con la euclidiana al cuadrado", size=28,
                               color=TEAL, weight=BOLD)).arrange(DOWN, buff=0.12)
        nxt.move_to([0, -1.8, 0])
        self.play(FadeIn(nxt, shift=UP * 0.15))
