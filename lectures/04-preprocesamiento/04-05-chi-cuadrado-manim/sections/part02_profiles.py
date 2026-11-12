"""Acto 2 · Ver la dependencia (6–14 min).

200 pasajeros (círculo = alta, cuadrado = baja). Separarlos por retraso y
convertir cada grupo en una barra al 100 %. Si las barras de cada fila se
parecen a la barra general, la variable no aporta (asiento); si se separan,
aporta (retraso). Formalización: P(Y | X = x) = P(Y) ⇔ P(X, Y) = P(X) P(Y).
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import SEAT, SEAT_ROWS, TOY, TOY_ROWS
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL, YELLOW,
                            card, person, safe_text, sat_legend, stacked_bar, tex)

STEP = 0.2
SIZE = 0.13


def grid_positions(n: int, cols: int, center) -> list[np.ndarray]:
    rows = (n + cols - 1) // cols
    w, h = (cols - 1) * STEP, (rows - 1) * STEP
    origin = np.array(center) + np.array([-w / 2, h / 2, 0])
    return [origin + RIGHT * (i % cols) * STEP + DOWN * (i // cols) * STEP for i in range(n)]


class ProfilesSection:
    def profiles(self) -> None:
        self.beat("6–8 min. Empezamos con 200 pasajeros (una muestra pequeña para poder "
                  "contar). Círculo verde = satisfacción alta, cuadrado naranja = baja. "
                  "Pregunte: si elijo un pasajero al azar, ¿qué probabilidad hay de que esté "
                  "satisfecho? (100 de 200 = 50 %).")
        head = self.start("Ver antes de calcular", "200 pasajeros: ¿satisfechos o no?", GREEN)
        rng = np.random.default_rng(3)
        pairs = ([(0, 1)] * TOY[0, 0] + [(0, 0)] * TOY[0, 1]
                 + [(1, 1)] * TOY[1, 0] + [(1, 0)] * TOY[1, 1])
        order = rng.permutation(len(pairs))
        pairs = [pairs[k] for k in order]
        dots = VGroup(*[person(s, SIZE) for _, s in pairs])
        for d, p in zip(dots, grid_positions(200, 20, [0, -0.2, 0])):
            d.move_to(p)
        legend = sat_legend()
        legend.next_to(head, DOWN, buff=0.3).to_edge(RIGHT, buff=0.6)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.004,
                              run_time=1.6), FadeIn(legend))
        overall = safe_text("100 de 200 → 50 % satisfechos", size=28, color=INK, weight=BOLD)
        overall.move_to([0, -2.75, 0])
        self.play(Write(overall))

        self.beat("8–9 min. PREDICCIÓN. Ahora los separo según si su vuelo tuvo retraso. "
                  "Antes de mover: ¿el 50 % se mantendrá en los dos grupos? Pida manos "
                  "alzadas: igual / distinto. Después avance.")
        groups = {0: [], 1: []}
        for d, (g, s) in zip(dots, pairs):
            groups[g].append((s, d))
        anims = []
        centers = {0: [-3.6, -0.5, 0], 1: [1.0, -0.5, 0]}
        for g in (0, 1):
            ordered = sorted(groups[g], key=lambda t: -t[0])
            for (s, d), p in zip(ordered, grid_positions(len(ordered), 12, centers[g])):
                anims.append(d.animate.move_to(p))
        self.play(FadeOut(overall), *anims, run_time=1.8)
        names = VGroup(
            safe_text(f"{TOY_ROWS[0]} · 120", size=26, color=INK, weight=BOLD).move_to([-3.6, 1.3, 0]),
            safe_text(f"{TOY_ROWS[1]} · 80", size=26, color=INK, weight=BOLD).move_to([1.0, 1.3, 0]),
        )
        self.play(FadeIn(names))
        counts = VGroup(
            safe_text("80 alta · 40 baja", size=22, color=MUTED).move_to([-3.6, -2.0, 0]),
            safe_text("20 alta · 60 baja", size=22, color=MUTED).move_to([1.0, -2.0, 0]),
        )
        self.play(FadeIn(counts))

        self.beat("9–10 min. Los grupos tienen tamaños distintos (120 y 80), así que no "
                  "comparamos conteos sino PROPORCIONES. Convertimos cada grupo en una barra "
                  "al 100 %: el PERFIL de la fila. La línea punteada es el perfil general "
                  "(50 %). Pregunte qué significa que la barra verde suba o baje de la línea.")
        bar_x = {0: -5.0, 1: -2.4}
        bars = VGroup()
        morphs = []
        for g in (0, 1):
            p = TOY[g, 0] / TOY[g].sum()
            bar = stacked_bar(p, width=1.25, height=3.2).move_to([bar_x[g], -0.45, 0])
            bars.add(bar)
            hi = VGroup(*[d for s, d in groups[g] if s == 1])
            lo = VGroup(*[d for s, d in groups[g] if s == 0])
            morphs += [ReplacementTransform(hi, bar[0]), ReplacementTransform(lo, bar[1])]
        self.play(*morphs, FadeOut(counts),
                  names[0].animate.scale(0.7).move_to([bar_x[0], -2.35, 0]),
                  names[1].animate.scale(0.7).move_to([bar_x[1], -2.35, 0]), run_time=1.6)
        self.play(*[FadeIn(b[2:]) for b in bars])
        base_y = bars[0][0].get_bottom()[1]
        mid = base_y + 1.6
        line_ret = DashedLine([-6.2, mid, 0], [-1.2, mid, 0], color=INK, stroke_width=2.5)
        tag = safe_text("general\n50 %", size=17, color=INK).next_to(line_ret, RIGHT, buff=0.1)
        ret_title = safe_text("Retraso", size=30, color=INK, weight=BOLD).move_to([-3.7, 1.65, 0])
        self.play(Create(line_ret), FadeIn(tag), FadeIn(ret_title))

        self.beat("10–11 min. Ahora otra variable: el ASIENTO preferido (mismos 200 pasajeros). "
                  "Pregunte antes de mostrar: ¿creen que el asiento cambia la satisfacción? "
                  "Las barras de ventana y pasillo quedan casi pegadas a la línea general: "
                  "conocer el asiento NO cambia lo que esperamos.")
        seat_bars = VGroup()
        seat_x = [2.3, 4.9]
        for k in range(2):
            p = SEAT[k, 0] / SEAT[k].sum()
            b = stacked_bar(p, width=1.25, height=3.2).move_to([seat_x[k], -0.45, 0])
            seat_bars.add(b)
        seat_names = VGroup(*[
            safe_text(f"{SEAT_ROWS[k]} · {SEAT[k].sum()}", size=19, color=INK, weight=BOLD)
            .move_to([seat_x[k], -2.35, 0]) for k in range(2)])
        line_seat = DashedLine([1.1, mid, 0], [6.1, mid, 0], color=INK, stroke_width=2.5)
        seat_title = safe_text("Asiento", size=30, color=INK, weight=BOLD).move_to([3.6, 1.65, 0])
        self.play(FadeIn(seat_title), *[GrowFromEdge(b, DOWN) for b in seat_bars],
                  FadeIn(seat_names), Create(line_seat))
        verdict = VGroup(
            safe_text("se separan → aporta", size=24, color=GREEN, weight=BOLD).move_to([-3.7, -2.95, 0]),
            safe_text("pegadas → ruido", size=24, color=RED, weight=BOLD).move_to([3.6, -2.95, 0]),
        )
        gaps = VGroup()
        for k, x in enumerate(bar_x.values()):
            top = bars[k][0].get_top()[1]
            gaps.add(DoubleArrow([x + 0.85, mid, 0], [x + 0.85, top, 0], buff=0, color=PURPLE,
                                 stroke_width=3, tip_length=0.14))
        self.play(LaggedStart(*[GrowFromCenter(g) for g in gaps], lag_ratio=0.2),
                  FadeIn(verdict))

        self.beat("11–13 min. Formalización. Lo que acabamos de ver tiene nombre: dos variables "
                  "son INDEPENDIENTES si el perfil de cada fila es igual al perfil general. "
                  "En probabilidad: P(Y | X = x) = P(Y) para toda x; multiplicando por P(X) "
                  "se obtiene la regla del producto. Señale que la segunda forma será la que "
                  "nos permita CALCULAR la tabla del mundo independiente.")
        keep = VGroup(bars, line_ret, tag, ret_title, names, gaps, verdict[0])
        self.play(FadeOut(VGroup(seat_bars, seat_names, line_seat, seat_title, verdict[1])),
                  keep.animate.scale(0.66).move_to([-4.2, -0.35, 0]))
        panel = card(7.4, 4.4, PURPLE).move_to([2.75, -0.45, 0])
        t1 = safe_text("Independencia", size=30, color=PURPLE, weight=BOLD)
        f1 = tex(r"P(Y \mid X = x) = P(Y)", size=44)
        t2 = safe_text("cada fila tiene el perfil general", size=22, color=MUTED)
        f2 = tex(r"P(X = x,\; Y = y) = P(X = x)\,P(Y = y)", size=40, max_width=6.8)
        t3 = safe_text("regla del producto", size=22, color=MUTED)
        col = VGroup(t1, f1, t2, f2, t3).arrange(DOWN, buff=0.22).move_to(panel)
        self.play(FadeIn(panel), Write(t1))
        self.play(Write(f1), FadeIn(t2))
        self.play(Write(f2), FadeIn(t3))

        self.beat("13–14 min. Puente: los ojos ya decidieron que el retraso importa y el "
                  "asiento no. Pero 'se ven distintas' no es un criterio que un algoritmo "
                  "pueda aplicar a 50 columnas. Necesitamos UN número que diga cuánto se "
                  "aleja la tabla real del mundo independiente.")
        bridge = safe_text("Necesitamos un número: ¿qué tan lejos estamos de la independencia?",
                           size=26, color=YELLOW, weight=BOLD, max_width=12.6)
        bridge.move_to([0, -3.05, 0])
        self.play(FadeOut(gaps), Write(bridge))
