"""Acto 5 · De métricas a decisiones: red de asociación, mapa soporte–lift y fichas.

La red resume relaciones simétricas (lift); el mapa coloca cada regla por
volumen (soporte) y fuerza (lift en escala logarítmica); las fichas traducen
cada regla en una acción con su justificación y su riesgo.
"""
from __future__ import annotations

import math

from manim import *

from sections.data import CAFE, HOOK, STORE_RULES, TE, VINO, fmt
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE,
                            RED, YELLOW, bullet_list, card, pill, safe_text, tex)

NODE_POS = {
    "café": np.array([-2.0, 0.2, 0]),
    "pan dulce": np.array([0.9, 1.35, 0]),
    "leche": np.array([3.9, 0.6, 0]),
    "té": np.array([-4.9, -1.2, 0]),
    "vino": np.array([0.4, -2.0, 0]),
    "queso": np.array([3.6, -2.2, 0]),
}


def edge_color(lift: float) -> str:
    return GREEN if lift > 1.05 else (RED if lift < 0.95 else MUTED)


class DecisionsSection:
    def decisions(self) -> None:
        self.beat("40–42 min. Vista de asociación: cada producto es un nodo y cada regla una "
                  "arista. Color = signo (verde > 1, gris ≈ 1, rojo < 1); grosor = lift. Como el "
                  "lift es simétrico, las aristas no tienen flecha. Pregunte qué esperan ver.")
        head = self.start("4 · Decidir", "Vista de asociación: la red de la tienda", GREEN)
        nodes = VGroup()
        for name, pos in NODE_POS.items():
            lab = safe_text(name, size=24, weight=BOLD)
            box = RoundedRectangle(corner_radius=0.25, width=lab.width + 0.5, height=0.62,
                                   stroke_color=INK, stroke_width=2, fill_color=PANEL,
                                   fill_opacity=1).move_to(pos)
            lab.move_to(box)
            nodes.add(VGroup(box, lab))
        self.play(LaggedStart(*[FadeIn(n, scale=0.8) for n in nodes], lag_ratio=0.12))
        edges, tags = VGroup(), VGroup()
        for r in STORE_RULES:
            a, b = NODE_POS[r.a], NODE_POS[r.b]
            width = 3 + 4 * min(2.0, abs(math.log2(r.lift)))
            line = Line(a, b, color=edge_color(r.lift), stroke_width=width)
            if r.lift < 1.05:
                line = DashedVMobject(line, num_dashes=18)
            edges.add(line)
            tag = safe_text(f"lift {fmt(r.lift)}", size=20, color=edge_color(r.lift), weight=BOLD)
            tag.move_to((a + b) / 2 + rotate_vector(normalize(b - a), PI / 2) * 0.32)
            tag.add_background_rectangle(color="#07111F", opacity=0.85, buff=0.05)
            tags.add(tag)
        edges.set_z_index(-1)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.2), run_time=1.6)
        caution = safe_text("La red muestra fuerza, no volumen ni dirección.", size=22,
                            color=MUTED).move_to([0, -3.0, 0])
        self.play(FadeIn(caution))

        self.decision_map()
        self.decision_cards()
        self.checklist()

    # ------------------------------------------------------------------
    def decision_map(self) -> None:
        self.beat("42–44 min. Mapa de decisión: horizontal = soporte (volumen), vertical = lift "
                  "en escala logarítmica (fuerza). El tamaño de cada burbuja es |leverage|. "
                  "Pida ubicar cada regla antes de que aparezca.")
        head = self.start("4 · Decidir", "Mapa soporte × lift", GREEN)
        def ly(lift: float) -> float:          # lift en escala log2, desplazada a y ≥ 0
            return math.log2(lift) + 2.3

        ax = Axes(x_range=[0, 0.25, 0.05], y_range=[0, 6.1, 1], x_length=8.2, y_length=4.6,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 2},
                  y_axis_config={"include_ticks": False})
        ax.move_to([-1.0, -0.2, 0])
        xt = VGroup(*[safe_text(f"{x:.2f}", size=18, color=MUTED).next_to(ax.c2p(x, 0), DOWN, buff=0.12)
                      for x in [0, 0.05, 0.10, 0.15, 0.20, 0.25]])
        yt = VGroup(*[safe_text(lab, size=18, color=MUTED).next_to(ax.c2p(0, ly(v)), LEFT, buff=0.12)
                      for v, lab in [(0.25, "0.25"), (0.5, "0.5"), (1, "1"), (2, "2"), (4, "4"),
                                     (8, "8")]])
        xl = safe_text("soporte P(A ∪ B)", size=20, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.42)
        yl = safe_text("lift (log)", size=20, color=MUTED).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.65)
        one = DashedLine(ax.c2p(0, ly(1)), ax.c2p(0.25, ly(1)), color=YELLOW, stroke_width=3)
        min_s = DashedLine(ax.c2p(0.05, 0), ax.c2p(0.05, 6.1), color=GRID, stroke_width=3)
        min_lab = safe_text("5 tickets", size=17, color=MUTED).next_to(ax.c2p(0.05, 6.1), RIGHT, buff=0.08)
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl))
        self.play(Create(one), Create(min_s), FadeIn(min_lab))
        zones = VGroup(
            safe_text("Acción amplia", size=20, color=GREEN, weight=BOLD).move_to(ax.c2p(0.175, 5.6)),
            safe_text("Nicho: validar", size=20, color=YELLOW, weight=BOLD).move_to(ax.c2p(0.03, 4.2)),
            safe_text("Sin información", size=20, color=MUTED, weight=BOLD).move_to(ax.c2p(0.12, ly(1) + 0.35)),
            safe_text("Sustitutos", size=20, color=RED, weight=BOLD).move_to(ax.c2p(0.175, 0.5)),
        )
        bubbles = VGroup()
        for r in STORE_RULES:
            p = ax.c2p(r.pab, ly(r.lift))
            rad = 0.12 + 2.2 * abs(r.lev) ** 0.5 * 0.55
            dot = Circle(radius=rad, stroke_color=edge_color(r.lift), stroke_width=3,
                         fill_color=edge_color(r.lift), fill_opacity=0.35).move_to(p)
            lab = safe_text(f"{r.a} ⇒ {r.b}", size=18, weight=BOLD)
            lab.next_to(dot, DOWN if r is HOOK else RIGHT, buff=0.12)
            bubbles.add(VGroup(dot, lab))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in bubbles], lag_ratio=0.3), run_time=1.4)
        self.play(FadeIn(zones))
        side = VGroup(
            safe_text("Volumen × fuerza", size=24, weight=BOLD),
            safe_text("tamaño = tickets extra", size=20, color=MUTED),
            safe_text("línea amarilla = lift 1", size=20, color=YELLOW),
            safe_text("vertical gris = conteo mínimo", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.0, 1.2, 0])
        self.play(FadeIn(side))

    def decision_cards(self) -> None:
        self.beat("44–47 min. Una ficha por regla: acción, métrica que la justifica y riesgo. "
                  "Pida a cuatro equipos defender una ficha cada uno en 30 segundos antes de "
                  "revelarla. Insista: la acción se valida con un experimento (A/B).")
        head = self.start("4 · Decidir", "De la métrica a la acción", GREEN)
        specs = [
            (CAFE, GREEN, "Combo café + pan dulce",
             f"lift {fmt(CAFE.lift)} · 8 tickets extra / 100", "Riesgo: canibalizar ventas que ya ocurrían"),
            (HOOK, MUTED, "No mover la leche",
             "lift 1.00 · leverage 0", "La leche se vende sola: conf 0.60 = tasa base"),
            (VINO, YELLOW, "Recomendar queso con vino",
             f"lift {fmt(VINO.lift, 1)} · conviction {fmt(VINO.conv, 1)}", "Sólo 4 tickets: validar con más días"),
            (TE, RED, "No hacer combo: sustitutos",
             f"lift {fmt(TE.lift)} · leverage −6 tickets", "Útil para surtido: si falta café, ofrecer té"),
        ]
        cards = VGroup()
        for r, color, action, why, risk in specs:
            title = safe_text(f"{r.a} ⇒ {r.b}", size=20, color=MUTED, weight=BOLD)
            act = safe_text(action, size=27, color=color, weight=BOLD, max_width=5.6)
            w = safe_text(why, size=21, max_width=5.6)
            rk = safe_text(risk, size=19, color=MUTED, max_width=5.6)
            body = VGroup(title, act, w, rk).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            box = card(6.15, 1.95, color).move_to(body)
            body.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.25)
            cards.add(VGroup(box, body))
        cards.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.3)).move_to([0, -0.55, 0])
        for c in cards:
            self.play(FadeIn(c, shift=UP * 0.1), run_time=0.5)

    def checklist(self) -> None:
        self.beat("47–49 min. El filtro mental, en orden: primero ¿hay asociación? (lift o "
                  "Zhang), luego ¿hay datos suficientes?, ¿cuánto impacto?, ¿qué dirección?, y "
                  "sólo al final ¿es causal? La confianza sola nunca va primero.")
        head = self.start("4 · Decidir", "Filtro mental antes de actuar", GREEN)
        steps = [
            ("1", "¿Hay asociación?", "lift > 1, Zhang > 0", PURPLE),
            ("2", "¿Hay datos suficientes?", "conteo de A ∪ B, estabilidad en el tiempo", YELLOW),
            ("3", "¿Cuánto mueve?", "leverage × n = tickets de más", GREEN),
            ("4", "¿En qué dirección actúo?", "confianza y conviction de A ⇒ B y B ⇒ A", ORANGE),
            ("5", "¿Es causal?", "sólo un experimento (A/B) lo responde", RED),
        ]
        rows = VGroup()
        for num, q, how, color in steps:
            badge = VGroup(Circle(radius=0.26, stroke_color=color, stroke_width=3,
                                  fill_color=color, fill_opacity=0.15),
                           safe_text(num, size=24, color=color, weight=BOLD))
            qq = safe_text(q, size=27, weight=BOLD)
            hh = safe_text(how, size=22, color=MUTED, max_width=5.3)
            qq.next_to(badge, RIGHT, buff=0.3)
            hh.move_to([0, 0, 0]).next_to(badge, RIGHT, buff=5.0)
            rows.add(VGroup(badge, qq, hh))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -0.15, 0])
        for row in rows:
            row[2].set_x(rows.get_left()[0] + 6.4 + row[2].width / 2)
        rows.set_x(0)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows], lag_ratio=0.2),
                  run_time=1.6)

        self.beat("49–50 min. Cuándo NO confiar en las reglas: pocos datos, un ítem dominante "
                  "(la leche), preguntas causales y cuando importa el orden temporal (patrones "
                  "secuenciales). Enlace con el capítulo 5.3c del libro.")
        nots = VGroup(*[pill(t, RED, size=19) for t in [
            "Pocos tickets: lift inestable", "Ítem dominante: conf inflada",
            "Pregunta causal: usar experimento", "Importa el orden: secuencias"]])
        nots.arrange_in_grid(rows=2, cols=2, buff=(0.4, 0.25)).move_to([0, -2.85, 0])
        self.play(rows.animate.shift(UP * 0.35), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.1) for n in nots], lag_ratio=0.15))
