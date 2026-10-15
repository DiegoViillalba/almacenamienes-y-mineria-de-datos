"""Acto 6 · Reto en parejas con datos nuevos, solución escalonada, cierre y ticket."""
from __future__ import annotations

from manim import *

from sections.base import UnitSquare
from sections.data import PRACTICE, fmt
from sections.theme import (BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, bullet_list, card, pill, safe_text, tex)


class PracticeSection:
    def practice(self) -> None:
        r = PRACTICE
        self.beat("50–54 min. Parejas, 4 minutos. Datos nuevos: 200 tickets de una tienda "
                  "escolar. Pida primero el dibujo del cuadrado (ancho, alturas, nivel) y "
                  "después las cifras. Pista: lev × 200 da tickets.")
        self.start("Reto en parejas", "Papas y refresco en 200 tickets", YELLOW)
        facts = VGroup(
            safe_text(f"{r.na} tickets con papas", size=30, color=BLUE, weight=BOLD),
            safe_text(f"{r.nb} tickets con refresco", size=30, color=ORANGE, weight=BOLD),
            safe_text(f"{r.nab} tickets con ambos", size=30, color=GREEN, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.9, 0.3, 0])
        tasks = bullet_list([
            "1. Dibuja el cuadrado y sus alturas.",
            "2. conf, lift y leverage (en tickets).",
            "3. conviction en ambas direcciones.",
            "4. ¿Qué harías y en qué dirección?",
        ], size=26, width=6.4).move_to([2.9, 0.4, 0])
        timer = pill("4 min · conf = altura en A · lift = altura / nivel", MUTED, size=19)
        timer.move_to([0, -2.6, 0])
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.1) for f in facts], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tasks], lag_ratio=0.15))
        self.play(FadeIn(timer))

        self.beat("54–56 min. Solución (1/2): el cuadrado. Ancho 0.25; nivel 0.40; dentro "
                  "30/50 = 0.60; fuera 50/150 ≈ 0.33. Pida a una pareja describirlo antes de "
                  "animarlo.")
        self.start("Solución", "Primero el dibujo", GREEN)
        sq = UnitSquare(4.6, r.pa, r.conf, r.h_not_a).move_to([-3.4, -0.5, 0])
        level = sq.level_line(r.pb)
        labs = VGroup(
            tex(rf"P(A)={fmt(r.pa)}", size=26, color=BLUE).next_to(sq.p(r.pa / 2, 0), DOWN, buff=0.15),
            sq.level_label(rf"P(B)={fmt(r.pb)}", r.pb),
            tex(rf"{fmt(r.conf)}", size=28, color=ORANGE).next_to(sq.p(r.pa / 2, r.conf), UP, buff=0.08),
            tex(rf"{fmt(r.h_not_a)}", size=28, color=BG).next_to(sq.p((1 + r.pa) / 2, r.h_not_a), DOWN, buff=0.1),
        )
        self.play(FadeIn(sq), Create(level))
        self.play(FadeIn(labs))

        self.beat("56–57 min. Solución (2/2): cifras y decisión. La dirección importa: papas ⇒ "
                  "refresco (conf 0.60, conv 1.50) es más útil que refresco ⇒ papas (0.375, "
                  "1.20), aunque el lift (1.5) y el leverage (+10 tickets) sean los mismos.")
        rv = r.reverse
        sol = VGroup(
            tex(rf"\mathrm{{conf}}=\tfrac{{30}}{{50}}={fmt(r.conf)},\quad"
                rf"\mathrm{{lift}}=\tfrac{{{fmt(r.conf)}}}{{{fmt(r.pb)}}}={fmt(r.lift)}", size=32),
            tex(rf"\mathrm{{lev}}={fmt(r.pab)}-{fmt(r.pa)}\cdot{fmt(r.pb)}={fmt(r.lev)}"
                rf"\;\Rightarrow\;+{r.lev * r.n:.0f}\text{{ tickets}}", size=32, color=GREEN),
            tex(rf"\mathrm{{conv}}(A\Rightarrow B)=\tfrac{{0.60}}{{0.40}}={fmt(r.conv)},\quad"
                rf"\mathrm{{conv}}(B\Rightarrow A)={fmt(rv.conv)}", size=30, color=RED),
            pill("Ofrecer refresco a quien lleva papas, y probarlo", GREEN, size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.15, -0.3, 0])
        for s in sol:
            self.play(FadeIn(s, shift=UP * 0.1), run_time=0.6)

    def closing(self) -> None:
        self.beat("57–59 min. Síntesis en cuatro verbos; pida a un estudiante reformular cada uno "
                  "con el ejemplo de la leche y el pan dulce.")
        self.start("Cierre", "Lo que una regla dice y lo que no", BLUE)
        ideas = VGroup(*[
            VGroup(pill(k, c, width=2.7, size=22), safe_text(t, size=26, max_width=8.6)).arrange(RIGHT, buff=0.4)
            for k, c, t in [
                ("Contar", BLUE, "Soporte = área: ¿qué tan común es el patrón?"),
                ("Comparar", PURPLE, "Lift = altura / nivel: ¿cambia B al saber A?"),
                ("Medir", GREEN, "Leverage = volumen; conviction = dirección; Zhang = escala fija."),
                ("Decidir", RED, "Asociación ≠ causa: actuar y medir con un experimento."),
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.2, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.2),
                  run_time=1.4)

        self.beat("59–60 min. Ticket individual. Respuesta: lift = 0.90 / 0.90 = 1, no aporta "
                  "nada aunque la confianza sea 90 %. Bonus: lift máximo = 1 / 0.5 = 2.")
        self.start("Ticket de salida", "Decide y justifica", PURPLE)
        q = VGroup(
            tex(r"\mathrm{conf}(X\Rightarrow\text{leche})=0.90,\qquad P(\text{leche})=0.90",
                size=44),
            safe_text("¿Usarías esta regla para una promoción? Justifica con el cuadrado.",
                      size=28, max_width=11.5),
            safe_text("Bonus: si P(A) = P(B) = 0.5, ¿cuál es el lift más alto posible?", size=24,
                      color=MUTED, max_width=11.5),
        ).arrange(DOWN, buff=0.45).move_to([0, -0.1, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in q], lag_ratio=0.25))
        final = safe_text("Siguiente: Apriori y FP-Growth a escala", size=22, color=BLUE)
        final.move_to([0, -2.75, 0])
        self.play(FadeIn(final))
