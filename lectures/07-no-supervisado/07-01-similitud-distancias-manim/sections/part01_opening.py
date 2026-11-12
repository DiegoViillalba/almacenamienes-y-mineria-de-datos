"""Acto 1 · ¿Quién se parece más a Ana? (0–5 min).

Portada; cinco clientes con variables mixtas; tres reglas distintas dan tres
respuestas distintas. Todo algoritmo no supervisado consume una distancia:
antes de agrupar hay que decidir cómo medir.
"""
from __future__ import annotations

from manim import *

from sections.data import CLIENTS, GOWER_EQ, GOWER_W, NUM_ONLY
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, TEAL,
                            YELLOW, card, fmt, safe_text, tex)


class OpeningSection:
    def opening(self) -> None:
        self.beat("0–1 min. Portada. Hoy la pregunta no es qué algoritmo usar, sino algo previo: "
                  "qué significa que dos casos estén 'cerca'. La fórmula llegará después de la "
                  "intuición, nunca antes.")
        eyebrow = safe_text("APRENDIZAJE NO SUPERVISADO · FUNDAMENTOS", size=22, color=TEAL,
                            weight=BOLD)
        title = safe_text("¿Qué tan lejos está lo parecido?", size=56, weight=BOLD,
                          max_width=12.6)
        subtitle = safe_text("Similitud y distancias: de los axiomas a UMAP", size=30,
                             color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Una tienda quiere recomendarle a Ana lo que compran los clientes más "
                  "parecidos a ella. Pida a cada estudiante que escriba un nombre en papel: "
                  "¿quién se parece más a Ana? No comente todavía.")
        self.clear()
        self.add_branding()
        head = self.heading("La pregunta", "¿Quién se parece más a Ana?", YELLOW)
        self.play(FadeIn(head), run_time=0.35)
        table = self.client_table().move_to([0, -0.35, 0])
        self.play(FadeIn(table.header), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in table.rows],
                              lag_ratio=0.12))
        hint = safe_text("dos numéricas · una categórica · una binaria", size=22, color=MUTED)
        hint.next_to(table, DOWN, buff=0.35)
        self.play(FadeIn(hint))

        self.beat("2–3 min. Revele tres respuestas. Con edad e ingreso (normalizados) gana Daniel. "
                  "Con Gower, que también cuenta canal y tarjeta, gana Carla. Si el negocio decide "
                  "que canal y tarjeta pesen la mitad, vuelve a ganar Daniel. Pregunte: ¿cuál es "
                  "la correcta? Ninguna en abstracto.")
        self.play(FadeOut(hint), FadeOut(table))
        rules = [
            ("sólo edad e ingreso", NUM_ONLY, BLUE),
            ("Gower, pesos iguales", GOWER_EQ, TEAL),
            ("Gower, canal y tarjeta × ½", GOWER_W, PURPLE),
        ]
        cols = VGroup()
        for label, scores, col in rules:
            order = sorted(scores, key=scores.get)
            name = safe_text(label, size=22, color=col, weight=BOLD, max_width=3.3)
            items = VGroup(*[
                VGroup(safe_text(f"{k + 1}. {n}", size=20, color=INK if k else YELLOW,
                                 weight=BOLD if k == 0 else NORMAL),
                       safe_text(fmt(scores[n], 3), size=18, color=MUTED)).arrange(RIGHT, buff=0.2)
                for k, n in enumerate(order)]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            box = card(3.6, 2.9, col)
            VGroup(name, items).arrange(DOWN, buff=0.25).move_to(box)
            cols.add(VGroup(box, name, items))
        cols.arrange(RIGHT, buff=0.35).move_to([0, -0.1, 0])
        for c in cols:
            self.play(FadeIn(c, shift=UP * 0.15), run_time=0.55)
        msg = safe_text("El parecido no se observa: se define", size=30, color=YELLOW, weight=BOLD)
        msg.next_to(cols, DOWN, buff=0.4)
        self.play(Write(msg))

        self.beat("3–4 min. Por qué importa: TODO método de esta unidad empieza por una distancia. "
                  "k-medias usa la euclidiana al cuadrado, el jerárquico y DBSCAN aceptan "
                  "cualquier matriz de distancias, LOF compara densidades de vecinos y UMAP "
                  "construye un grafo de vecinos. Si la regla está mal, todo lo de abajo hereda "
                  "el error.")
        head = self.swap_heading(head, "El cimiento", "Todo algoritmo consume una distancia", TEAL)
        self.play(FadeOut(cols), FadeOut(msg))
        core = tex(r"d(\mathbf{x},\mathbf{y})", size=64, color=TEAL)
        ring = Circle(radius=1.0, color=TEAL, stroke_width=3).move_to(core)
        hub = VGroup(ring, core).move_to([0, -0.4, 0])
        algos = [("k-medias", BLUE), ("jerárquico", BLUE), ("DBSCAN", BLUE),
                 ("k-NN", GREEN), ("anomalías (LOF)", ORANGE), ("t-SNE / UMAP", PINK)]
        nodes = VGroup()
        for k, (name, col) in enumerate(algos):
            ang = PI / 2 - k * TAU / len(algos) + PI / 6
            pos = hub.get_center() + 2.75 * np.array([1.45 * np.cos(ang), 0.95 * np.sin(ang), 0])
            nodes.add(VGroup(card(2.7, 0.75, col).move_to(pos),
                             safe_text(name, size=22, color=col, weight=BOLD).move_to(pos)))
        hc = hub.get_center()
        links = VGroup(*[Line(hc + 1.05 * normalize(n.get_center() - hc), n.get_center(),
                              color=GRID, stroke_width=2.5) for n in nodes])
        self.play(DrawBorderThenFill(ring), Write(core))
        self.play(LaggedStart(*[AnimationGroup(Create(l), FadeIn(n))
                                for l, n in zip(links, nodes)], lag_ratio=0.15))

        self.beat("4–5 min. Ruta de la clase. Primero, qué significa medir y por qué una distancia "
                  "debe cumplir reglas. Luego veremos cómo cambia la distancia al transformar el "
                  "espacio en 2D y 3D, qué pasa cuando hay muchas dimensiones (la esfera) y cómo "
                  "UMAP responde. Al final volvemos con Ana.")
        head = self.swap_heading(head, "Ruta", "De la regla al mapa", PURPLE)
        self.play(FadeOut(hub), FadeOut(nodes), FadeOut(links))
        steps = [("medir", "¿qué es?", TEAL), ("axiomas", "¿por qué?", GREEN),
                 ("geometría", "2D y 3D", BLUE), ("dimensión", "la esfera", RED),
                 ("UMAP", "vecindarios", PINK), ("Ana", "decidir", YELLOW)]
        chips = VGroup()
        for verb, what, col in steps:
            v = safe_text(verb, size=25, color=col, weight=BOLD, max_width=1.75)
            w = safe_text(what, size=19, color=MUTED)
            box = card(2.0, 1.4, col)
            VGroup(v, w).arrange(DOWN, buff=0.14).move_to(box)
            chips.add(VGroup(box, v, w))
        chips.arrange(RIGHT, buff=0.22).move_to([0, -0.3, 0])
        arrows = VGroup(*[Arrow(chips[i].get_right(), chips[i + 1].get_left(), buff=0.03,
                                color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.5)
                          for i in range(len(chips) - 1)])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.14),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.14))

    # ── utilidades ──────────────────────────────────────────────────────────
    def client_table(self) -> VGroup:
        heads = ["Cliente", "Edad", "Ingreso (miles)", "Canal", "Tarjeta"]
        widths = [2.0, 1.3, 2.4, 1.6, 1.5]
        xs = np.cumsum([0] + widths[:-1]) + np.array(widths) / 2
        xs = xs - xs.mean()
        header = VGroup(*[safe_text(h, size=22, color=MUTED, weight=BOLD).move_to([x, 0, 0])
                          for h, x in zip(heads, xs)])
        rule = Line([xs[0] - 1.1, -0.32, 0], [xs[-1] + 0.85, -0.32, 0], color=GRID, stroke_width=2)
        rows = VGroup()
        for r, c in enumerate(CLIENTS):
            col = YELLOW if r == 0 else INK
            y = -0.75 - 0.58 * r
            row = VGroup(*[safe_text(str(v), size=26, color=col,
                                     weight=BOLD if r == 0 else NORMAL).move_to([x, y, 0])
                           for v, x in zip(c, xs)])
            rows.add(row)
        group = VGroup(VGroup(header, rule), rows)
        group.header = group[0]
        group.rows = rows
        return group
