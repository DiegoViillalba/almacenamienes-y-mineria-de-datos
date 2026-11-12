"""Acto 7 · Reducir columnas (43–53 min).

Encuesta de 1,000 pasajeros. (1) Relevancia: χ², p y V de cada predictora
contra Satisfacción; asiento y momento de compra no pasan p < 0.05. (2)
Redundancia: V entre predictoras útiles; tipo de viaje y frecuencia repiten
lo que ya dice la clase. (3) Decisión y receta. (4) Precauciones.
"""
from __future__ import annotations

from manim import *

from sections.data import AIR_RANK, USEFUL, pair_v
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL,
                            YELLOW, card, pill, safe_text, tex)


def p_str(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


class ReductionSection:
    def reduction(self) -> None:
        self.beat("43–45 min. Regresamos al problema del inicio con la encuesta completa: "
                  "1,000 pasajeros, seis predictoras. Para cada una construimos su tabla contra "
                  "Satisfacción y calculamos χ², g. l., p y V. Recuerde las elecciones que el "
                  "grupo escribió al inicio: ¿acertaron?")
        head = self.start("Reducción, paso 1", "¿Qué variables se relacionan con el objetivo?", TEAL)
        rows = VGroup()
        bar_x0 = -2.6
        scale = 15.0
        for k, r in enumerate(AIR_RANK):
            y = 1.55 - k * 0.72
            name = safe_text(r["var"], size=24, color=INK, weight=BOLD)
            name.move_to([bar_x0 - 0.25, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(r["V"] * scale, 0.02), height=0.42, stroke_width=0,
                            fill_color=TEAL, fill_opacity=0.85)
            bar.move_to([bar_x0, y, 0], aligned_edge=LEFT)
            v = safe_text(f"V = {r['V']:.3f}", size=20, color=TEAL, weight=BOLD)
            v.next_to(bar, RIGHT, buff=0.15)
            stats = safe_text(f"χ² = {r['chi2']:.1f} · g.l. {r['dof']} · {p_str(r['p'])}",
                              size=19, color=MUTED)
            stats.move_to([3.4, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(name, bar, v, stats))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.08))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(r[2]) for r in rows], lag_ratio=0.1))
        self.play(FadeIn(VGroup(*[r[3] for r in rows])))

        self.beat("45–46 min. Filtro de RELEVANCIA: con α = 0.05, asiento (p = 0.73) y momento "
                  "de compra (p = 0.74) no muestran evidencia de relación con la satisfacción. "
                  "Sus χ² son los que el azar produce todo el tiempo. Se van: son ruido para "
                  "el modelo.")
        noise = [r for r, d in zip(rows, AIR_RANK) if d["p"] >= 0.05]
        self.play(*[r[1].animate.set_fill(RED) for r in noise],
                  *[r[3].animate.set_color(RED) for r in noise],
                  *[Circumscribe(r, color=RED, buff=0.08) for r in noise])
        out = safe_text("ruido → eliminar", size=24, color=RED, weight=BOLD)
        out.move_to([0.5, (noise[0].get_y() + noise[-1].get_y()) / 2, 0])
        self.play(FadeIn(out))
        self.play(*[r.animate.shift(RIGHT * 14) for r in noise], out.animate.shift(RIGHT * 14),
                  run_time=1.0)

        self.beat("46–49 min. Filtro de REDUNDANCIA. Cuatro sobreviven, pero ¿dicen cosas "
                  "distintas? Aplicamos la misma V entre pares de predictoras. Clase–tipo de "
                  "viaje (0.49) y clase–frecuencia (0.37) son altas: quien viaja en business "
                  "suele viajar por negocio y con frecuencia. Pregunte: si ya tengo la clase, "
                  "¿cuánto aprendo además del tipo de viaje?")
        head2 = self.swap_heading(head, "Reducción, paso 2", "¿Las sobrevivientes repiten información?",
                                  YELLOW)
        self.play(FadeOut(VGroup(*[r for r in rows if r not in noise])))
        names = USEFUL
        m = len(names)
        cs = 1.25
        heat = VGroup()
        cells = {}
        origin = [-3.6, 1.0, 0]
        for i in range(m):
            for j in range(m):
                v = 1.0 if i == j else pair_v(names[i], names[j])
                col = interpolate_color(ManimColor(PANEL), ManimColor(TEAL), min(v / 0.6, 1.0))
                sq = Square(cs, stroke_color=GRID, stroke_width=1.5, fill_color=col,
                            fill_opacity=1 if i != j else 0.25)
                sq.move_to([origin[0] + j * cs, origin[1] - i * cs, 0])
                t = safe_text("—" if i == j else f"{v:.2f}", size=22,
                              color=INK if v < 0.3 or i == j else "#07111F", weight=BOLD).move_to(sq)
                cells[i, j] = VGroup(sq, t)
                heat.add(cells[i, j])
        short = {"Tipo de viaje": "Tipo viaje", "Frecuencia": "Frecuencia"}
        rl = VGroup(*[safe_text(short.get(n, n), size=20, color=MUTED)
                      .next_to(cells[i, 0], LEFT, buff=0.15) for i, n in enumerate(names)])
        cl = VGroup(*[safe_text(short.get(n, n), size=17, color=MUTED, max_width=cs - 0.12)
                      .next_to(cells[0, j], UP, buff=0.12) for j, n in enumerate(names)])
        grid = VGroup(heat, rl, cl).move_to([-2.3, -0.4, 0])
        self.play(FadeIn(rl), FadeIn(cl), LaggedStart(*[FadeIn(c) for c in heat], lag_ratio=0.03))
        ic = names.index("Clase")
        hot = [(ic, names.index("Tipo de viaje")), (ic, names.index("Frecuencia"))]
        rects = VGroup(*[SurroundingRectangle(cells[a, b], color=YELLOW, buff=0.03, stroke_width=5)
                         for a, b in hot])
        self.play(Create(rects))
        target_v = {d["var"]: d["V"] for d in AIR_RANK}
        rule = VGroup(
            safe_text("De cada par redundante,", size=24, color=INK),
            safe_text("conservar la de mayor V", size=24, color=INK),
            safe_text("con el objetivo:", size=24, color=INK),
            safe_text(f"Clase  {target_v['Clase']:.2f}", size=24, color=GREEN, weight=BOLD),
            safe_text(f"Tipo de viaje  {target_v['Tipo de viaje']:.2f}", size=24, color=ORANGE,
                      weight=BOLD),
            safe_text(f"Frecuencia  {target_v['Frecuencia']:.2f}", size=24, color=ORANGE,
                      weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.0, -0.3, 0])
        self.play(FadeIn(rule[:3]))
        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.1) for x in rule[3:]], lag_ratio=0.3))

        self.beat("49–51 min. Decisión. De seis columnas pasamos a dos que cargan casi toda la "
                  "señal: retraso y clase. Tipo de viaje y frecuencia quedan como OPCIONALES: "
                  "añaden poco si ya está la clase. Insista: la última palabra la da el modelo "
                  "con validación cruzada (el capítulo del libro lo hace con un bosque "
                  "aleatorio). χ² nos da una razón explícita para cada decisión.")
        head3 = self.swap_heading(head2, "Decisión", "De seis columnas a dos", GREEN)
        self.play(FadeOut(VGroup(grid, rects, rule)))
        groups = [
            ("Conservar", GREEN, ["Retraso", "Clase"], "relevantes y no redundantes"),
            ("Opcional", YELLOW, ["Tipo de viaje", "Frecuencia"], "repiten lo que dice la clase"),
            ("Eliminar", RED, ["Asiento", "Momento de compra"], "sin relación: ruido"),
        ]
        cols = VGroup()
        for title, col, vars_, why in groups:
            box = card(3.9, 3.6, col)
            t = safe_text(title, size=32, color=col, weight=BOLD)
            ps = VGroup(*[pill(v, col, width=3.3, size=22) for v in vars_]).arrange(DOWN, buff=0.18)
            w = safe_text(why, size=19, color=MUTED, max_width=3.5)
            VGroup(t, ps, w).arrange(DOWN, buff=0.35).move_to(box)
            cols.add(VGroup(box, t, ps, w))
        cols.arrange(RIGHT, buff=0.35).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.25))

        self.beat("51–52 min. La receta completa, para llevar a cualquier conjunto de datos: "
                  "relevancia contra el objetivo (p y V), redundancia entre sobrevivientes (V "
                  "por pares), decidir y verificar con el modelo. Funciona igual con 6 que con "
                  "600 columnas categóricas.")
        head4 = self.swap_heading(head3, "La receta", "Selección de variables categóricas con χ²", PURPLE)
        self.play(FadeOut(cols))
        steps = [
            ("1", "Tabla de cada X contra Y", BLUE),
            ("2", "χ², g.l. y valor p: ¿relación real?", PURPLE),
            ("3", "V de Cramér: ordenar por fuerza", TEAL),
            ("4", "Descartar las que no pasan α", RED),
            ("5", "V entre sobrevivientes: quitar redundancia", YELLOW),
            ("6", "Validar el subconjunto con el modelo", GREEN),
        ]
        lst = VGroup()
        for num, text, col in steps:
            c = Circle(radius=0.26, color=col, fill_opacity=0.18, stroke_width=2.5)
            n = safe_text(num, size=24, color=col, weight=BOLD).move_to(c)
            t = safe_text(text, size=26, color=INK)
            lst.add(VGroup(VGroup(c, n), t).arrange(RIGHT, buff=0.3))
        lst.arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([0, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.15) for s in lst], lag_ratio=0.2))

        self.beat("52–53 min. Precauciones, una por tarjeta. (1) Si alguna E < 5 la aproximación "
                  "falla: agrupar categorías o usar la prueba exacta de Fisher. (2) Con muchas "
                  "categorías los g. l. crecen: compare con V, no con χ². (3) Con 100 pruebas a "
                  "α = 0.05, unas 5 salen 'significativas' por azar. (4) χ² sólo mira pares: no "
                  "ve interacciones ni causas.")
        self.swap_heading(head4, "Precauciones", "Lo que χ² no hace por ti", RED)
        self.play(FadeOut(lst))
        warns = [
            ("E ≥ 5 en cada celda", "si no: agrupar o Fisher"),
            ("Muchas categorías", "g.l. altos: comparar con V"),
            ("Muchas pruebas", "1 de cada 20 «pasa» por azar"),
            ("Sólo pares", "no ve interacciones ni causas"),
        ]
        cards = VGroup()
        for a, b in warns:
            box = card(5.6, 1.45, RED)
            ta = safe_text(a, size=26, color=INK, weight=BOLD)
            tb = safe_text(b, size=21, color=MUTED)
            VGroup(ta, tb).arrange(DOWN, buff=0.14).move_to(box)
            cards.add(VGroup(box, ta, tb))
        cards.arrange_in_grid(rows=2, cols=2, buff=(0.4, 0.35)).move_to([0, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(c, scale=0.95) for c in cards], lag_ratio=0.2))
