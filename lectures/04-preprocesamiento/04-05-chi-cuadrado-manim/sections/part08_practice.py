"""Acto 8 · Reto en parejas y cierre (53–60 min).

Wi-Fi a bordo × Satisfacción, n = 120: calcular E, χ², decidir con 3.84 y
calcular V. Solución: χ² = 1.23, p = 0.27, V = 0.10 → candidata a eliminar.
Cierre con la idea de la clase y ticket de salida.
"""
from __future__ import annotations

from manim import *

from sections.data import (CRIT_1, PRACTICE, PRACTICE_CHI2, PRACTICE_CONTRIB, PRACTICE_E,
                           PRACTICE_P, PRACTICE_ROWS, PRACTICE_V)
from sections.theme import (BLUE, GREEN, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW,
                            Table2, card, safe_text, stacked_bar, tex)


class PracticeSection:
    def practice(self) -> None:
        self.beat("53–57 min. RETO EN PAREJAS (4 minutos). Otra variable candidata: si el "
                  "pasajero usó el Wi-Fi a bordo. Antes de calcular, pida que DIBUJEN las dos "
                  "barras de perfil y apuesten: ¿aporta o no? Luego: E, χ², decisión con 3.84 "
                  "y V. Camine por el salón; error típico: olvidar dividir entre E.")
        head = self.start("Tu turno", "¿El Wi-Fi a bordo aporta algo?", YELLOW)
        t = Table2(PRACTICE, PRACTICE_ROWS, ["Alta", "Baja"], cw=1.5, ch=0.82, color=BLUE,
                   totals=True, row_w=2.2)
        t.move_to([-3.4, 0.5, 0])
        n_t = safe_text("n = 120", size=24, color=INK, weight=BOLD)
        n_t.next_to(t.col_tot, RIGHT, buff=0.4).align_to(t.col_tot, DOWN)
        self.play(FadeIn(t), FadeIn(n_t))
        tasks = VGroup(
            safe_text("1. Dibuja las barras de perfil", size=26, color=INK),
            safe_text("2. Calcula la tabla E", size=26, color=INK),
            safe_text("3. Calcula χ² (g.l. = ?)", size=26, color=INK),
            safe_text("4. Compara con 3.84: ¿rechazas?", size=26, color=INK),
            safe_text("5. Calcula V: ¿la conservas?", size=26, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.3, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.1) for x in tasks], lag_ratio=0.15))
        clock = safe_text("4 minutos", size=28, color=YELLOW, weight=BOLD).move_to([0, -2.7, 0])
        self.play(FadeIn(clock))

        self.beat("57–58 min. Solución, paso a paso. Perfiles: 47 % vs 37 % de satisfechos; "
                  "se separan un poco. E = 25 y 35 en cada fila (60 × 50 / 120 = 25). "
                  "Residuos ±3. Contribuciones 0.36 y 0.26.")
        self.play(FadeOut(tasks), FadeOut(clock))
        bars = VGroup(*[stacked_bar(PRACTICE[i, 0] / PRACTICE[i].sum(), width=0.9, height=2.4,
                                    size=18) for i in range(2)]).arrange(RIGHT, buff=0.3)
        bars.move_to([-3.6, -1.9, 0])
        bars_l = VGroup(*[safe_text(r, size=16, color=MUTED).next_to(b, DOWN, buff=0.08)
                          for r, b in zip(PRACTICE_ROWS, bars)])
        self.play(t.animate.scale(0.8).move_to([-3.5, 1.15, 0]), FadeOut(n_t))
        self.play(GrowFromEdge(bars, DOWN), FadeIn(bars_l))
        e = Table2(PRACTICE_E, PRACTICE_ROWS, ["Alta", "Baja"], cw=1.3, ch=0.7, color=YELLOW,
                   dashed=True, row_w=2.0, size=26, label_size=19)
        e.move_to([2.6, 1.0, 0])
        e_tag = safe_text("E · esperado", size=22, color=YELLOW, weight=BOLD).next_to(e.row_labels, UP, buff=0.35)
        self.play(FadeIn(e), FadeIn(e_tag))
        c = PRACTICE_CONTRIB
        sum_tex = tex(rf"\chi^2 = \tfrac{{9}}{{25}} + \tfrac{{9}}{{35}} + \tfrac{{9}}{{25}} + "
                      rf"\tfrac{{9}}{{35}} = {PRACTICE_CHI2:.2f}", size=36, color=PURPLE)
        sum_tex.move_to([2.4, -0.3, 0])
        self.play(Write(sum_tex))

        self.beat("58–59 min. Decisión: 1.23 < 3.84 con 1 g. l. → NO rechazamos la "
                  "independencia (p = 0.27). V = 0.10, en el límite de 'débil'. Conclusión: "
                  "no hay evidencia de que el Wi-Fi aporte; candidata a eliminar. Pregunte "
                  "quién apostó bien con sólo mirar las barras.")
        dec = VGroup(
            tex(rf"{PRACTICE_CHI2:.2f} < {CRIT_1:.2f} \;\Rightarrow\; p = {PRACTICE_P:.2f}",
                size=34, color=YELLOW),
            tex(rf"V = \sqrt{{{PRACTICE_CHI2:.2f}/120}} = {PRACTICE_V:.2f}", size=34, color=TEAL),
        ).arrange(DOWN, buff=0.25).next_to(sum_tex, DOWN, buff=0.35)
        self.play(Write(dec[0]))
        self.play(Write(dec[1]))
        verdict = card(6.6, 0.85, RED).next_to(dec, DOWN, buff=0.3)
        vt = safe_text("Sin evidencia de relación → eliminar", size=24, color=RED,
                       weight=BOLD, max_width=6.2).move_to(verdict)
        self.play(FadeIn(verdict), Write(vt))

    def closing(self) -> None:
        self.beat("59–60 min. Cierre. Lea la idea en voz alta. Ticket de salida (en papel, "
                  "1 minuto): 'Explica con dibujos de barras por qué χ² = 0 cuando dos "
                  "variables son independientes, y por qué un p muy pequeño no garantiza que "
                  "valga la pena conservar la variable'.")
        self.start("Para llevarse", "Una variable aporta si cambia el perfil del objetivo", PURPLE)
        lines = VGroup(
            safe_text("Ver: barras por fila contra la barra general", size=27, color=GREEN),
            safe_text("Medir: χ² = Σ (O − E)² / E, distancia a la independencia", size=27,
                      color=PURPLE),
            safe_text("Decidir: comparar con el azar (g.l., valor p)", size=27, color=YELLOW),
            safe_text("Graduar: V de Cramér, sin efecto del tamaño n", size=27, color=TEAL),
            safe_text("Reducir: quitar ruido y luego redundancia", size=27, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.15, 0])
        self.play(LaggedStart(*[FadeIn(x, shift=RIGHT * 0.15) for x in lines], lag_ratio=0.2))
        ticket = card(11.6, 1.0, YELLOW).move_to([0, -2.65, 0])
        tt = safe_text("Ticket de salida: ¿por qué χ² = 0 si las barras coinciden?  "
                       "¿Basta un p pequeño para conservar una variable?", size=20, color=YELLOW,
                       max_width=11.2).move_to(ticket)
        self.play(FadeIn(ticket), Write(tt))
