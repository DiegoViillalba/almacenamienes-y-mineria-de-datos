"""Acto 6 · Significancia no es fuerza (38–43 min).

La misma tabla multiplicada por 10: perfiles idénticos, χ² diez veces mayor.
χ² mide evidencia y crece con n. La V de Cramér divide entre n y entre el
tamaño de la tabla para dar una fuerza en [0, 1].
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import STRENGTH, STRENGTH_ROWS, TOY_V, chi2_sf, chi2_stat, cramer_v
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL, YELLOW,
                            Table2, card, safe_text, stacked_bar, tex)


class StrengthSection:
    def strength(self) -> None:
        small = STRENGTH
        big = STRENGTH * 10
        x_s, x_b = chi2_stat(small), chi2_stat(big)
        p_s, p_b = chi2_sf(x_s, 1), chi2_sf(x_b, 1)
        v_s, v_b = cramer_v(small), cramer_v(big)

        self.beat("38–40 min. PREDICCIÓN. Dos encuestas con EXACTAMENTE los mismos porcentajes "
                  "(60 % vs 40 % de satisfechos); una con 100 pasajeros y otra con 1,000. "
                  "¿Tendrán el mismo χ²? Deje votar. Respuesta: χ² se multiplica por 10. "
                  "La relación es igual de fuerte; lo que creció es la EVIDENCIA.")
        head = self.start("Significancia ≠ fuerza", "Mismos perfiles, distinto n", TEAL)
        rows = STRENGTH_ROWS
        cols = ["Alta", "Baja"]
        panels = VGroup()
        for k, (tab, x2, p, n) in enumerate([(small, x_s, p_s, 100), (big, x_b, p_b, 1000)]):
            cx = -3.4 if k == 0 else 3.4
            title = safe_text(f"n = {n:,}".replace(",", " "), size=30, color=INK, weight=BOLD)
            title.move_to([cx, 1.9, 0])
            t = Table2(tab, rows, cols, cw=1.3, ch=0.68, color=BLUE, size=26, row_w=2.2,
                       label_size=19)
            t.move_to([cx - 0.9, 0.55, 0])
            bars = VGroup(*[stacked_bar(tab[i, 0] / tab[i].sum(), width=0.6, height=1.6, size=16)
                            for i in range(2)]).arrange(RIGHT, buff=0.18)
            bars.next_to(t, RIGHT, buff=0.4)
            x2t = tex(rf"\chi^2 = {x2:.0f}", size=40, color=PURPLE).move_to([cx, -1.0, 0])
            pt = safe_text(f"p = {p:.3f}" if p > 1e-3 else "p ≈ 2.5 × 10⁻¹⁰", size=26,
                           color=YELLOW, weight=BOLD).next_to(x2t, DOWN, buff=0.2)
            panels.add(VGroup(title, t, bars, x2t, pt))
        self.play(FadeIn(panels[0][:3]), FadeIn(panels[1][:3]))
        self.play(Write(panels[0][3]), FadeIn(panels[0][4]))
        self.play(TransformFromCopy(panels[0][3], panels[1][3]), FadeIn(panels[1][4]))
        times = safe_text("× 10", size=34, color=PURPLE, weight=BOLD).move_to([0, -1.0, 0])
        self.play(FadeIn(times, scale=1.4))
        lesson = safe_text("χ² crece con n: mide evidencia, no fuerza", size=28, color=RED,
                           weight=BOLD).move_to([0, -2.75, 0])
        self.play(Write(lesson))

        self.beat("40–42 min. Para medir FUERZA quitamos el efecto de n (dividir entre n) y el "
                  "del tamaño de la tabla (dividir entre min(r − 1, c − 1)); la raíz regresa a "
                  "una escala parecida a una correlación, entre 0 y 1. Ambas encuestas: V = 0.20. "
                  "El ejemplo del retraso: V = 0.41.")
        self.play(FadeOut(lesson), FadeOut(times),
                  *[p[4].animate.set_opacity(0.35) for p in panels])
        formula = tex([r"V", r"=", r"\sqrt{\frac{\chi^2 / n}{\min(r-1,\,c-1)}}"], size=50)
        formula[0].set_color(TEAL)
        formula.scale(0.8).move_to([0, -2.6, 0])
        box = SurroundingRectangle(formula, color=TEAL, buff=0.18, corner_radius=0.1)
        self.play(Write(formula), Create(box))
        vt = VGroup(*[safe_text(f"V = {v:.2f}", size=28, color=TEAL, weight=BOLD)
                      .next_to(panels[k][3], RIGHT, buff=0.6) for k, v in enumerate([v_s, v_b])])
        self.play(FadeIn(vt))

        self.beat("42–43 min. Escala orientativa (no es ley): V < 0.1 débil, 0.1–0.3 moderada, "
                  "> 0.3 fuerte. Úsela para COMPARAR variables dentro del mismo conjunto de "
                  "datos. Resumen para el pizarrón: valor p responde '¿es real?'; V responde "
                  "'¿qué tanto?'. Para reducir variables necesitamos las dos.")
        self.play(FadeOut(VGroup(panels, vt)), VGroup(formula, box).animate.move_to([0, 1.3, 0]))
        line = NumberLine(x_range=[0, 1, 0.1], length=11, color=GRID, include_numbers=True,
                          decimal_number_config={"num_decimal_places": 1}, font_size=22)
        line.move_to([0, -0.6, 0])
        zones = VGroup()
        for a, b, col, name in [(0, 0.1, RED, "débil"), (0.1, 0.3, YELLOW, "moderada"),
                                (0.3, 1.0, GREEN, "fuerte")]:
            r = Rectangle(width=line.n2p(b)[0] - line.n2p(a)[0], height=0.35, stroke_width=0,
                          fill_color=col, fill_opacity=0.35)
            r.move_to((line.n2p(a) + line.n2p(b)) / 2 + UP * 0.32)
            lab = safe_text(name, size=20, color=col, weight=BOLD).next_to(r, UP, buff=0.08)
            zones.add(VGroup(r, lab))
        self.play(Create(line), FadeIn(zones))
        marks = VGroup()
        for k, (v, name, col) in enumerate([(v_s, "encuestas: 0.20", BLUE),
                                            (TOY_V, "retraso (juguete): 0.41", GREEN)]):
            tri = Triangle(color=col, fill_opacity=1).scale(0.12).move_to(line.n2p(v) + DOWN * 0.72)
            lab = safe_text(name, size=20, color=col, weight=BOLD).next_to(tri, DOWN, buff=0.1 + 0.4 * k)
            marks.add(VGroup(tri, lab))
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in marks], lag_ratio=0.3))
        qa = VGroup(
            safe_text("valor p → ¿es real?", size=26, color=YELLOW, weight=BOLD),
            safe_text("V de Cramér → ¿qué tan fuerte?", size=26, color=TEAL, weight=BOLD),
        ).arrange(RIGHT, buff=1.0).move_to([0, -2.6, 0])
        self.play(FadeIn(qa))
