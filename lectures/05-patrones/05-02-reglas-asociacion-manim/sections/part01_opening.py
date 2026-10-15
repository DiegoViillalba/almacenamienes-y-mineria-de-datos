"""Acto 1 · Portada, objetivos y el dilema del gerente (intuición).

Cien tickets en una cuadrícula de 10 × 10. Cada columna son 10 tickets; las
tres primeras columnas contienen pan dulce y en *todas* las columnas los seis
tickets de abajo llevan leche. Así 18 de 30 (60 %) y 42 de 70 (60 %): la
regla pan dulce ⇒ leche tiene confianza 0.60 y lift 1. Al cerrar los huecos,
la cuadrícula se convierte en el cuadrado unitario que se usa el resto de la
clase.
"""
from __future__ import annotations

from manim import *

from sections.base import UnitSquare
from sections.data import HOOK, fmt
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, RED,
                            YELLOW, bullet_list, pill, safe_text, tex)

PITCH = 0.48
CELL = 0.40
GRID_CENTER = np.array([-3.3, -0.42, 0])
SQUARE_SIDE = 4.8


def cell_pos(col: int, row: int, pitch: float = PITCH) -> np.ndarray:
    return GRID_CENTER + np.array([(col - 4.5) * pitch, (row - 4.5) * pitch, 0])


class OpeningSection:
    def opening(self) -> None:
        self.beat("0–2 min. Portada. Pregunte: ¿qué decisión tomarían en una tienda si supieran "
                  "qué productos se compran juntos? Recoja dos ejemplos (combos, acomodo, "
                  "recomendaciones) y anótelos: se retoman en el acto final.")
        eyebrow = safe_text("EXTRACCIÓN DE PATRONES · REGLAS DE ASOCIACIÓN", size=22,
                            color=BLUE, weight=BOLD)
        title = safe_text("¿Qué se compra con qué?", size=60, weight=BOLD, max_width=12.4)
        subtitle = safe_text("Construir reglas, medirlas como áreas y decidir con ellas",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("2–3 min. Objetivos como acciones verificables. Anuncie el reto en parejas "
                  "(minuto 50) y el ticket de salida.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "¿Qué sabremos hacer al terminar?", YELLOW)
        goals = bullet_list([
            "1. Construir reglas A ⇒ B a partir de tickets: soporte, confianza, Apriori.",
            "2. Leer soporte, confianza y lift como áreas y alturas en un cuadrado.",
            "3. Calcular e interpretar leverage, conviction y Zhang.",
            "4. Convertir métricas en una decisión justificada (y saber cuándo no).",
        ], size=27).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.15) for g in goals], lag_ratio=0.15),
                  run_time=1.0)
        self.manager_dilemma()

    # ------------------------------------------------------------------
    def manager_dilemma(self) -> None:
        self.beat("3–4 min. Presente la tienda: cada cuadrito es un ticket de un día, 100 en "
                  "total. Todavía no hay fórmulas. Pregunte: ¿qué tendríamos que contar para "
                  "saber si dos productos van juntos?")
        head = self.start("Intuición", "La tienda de la esquina: 100 tickets", ORANGE)
        cells = VGroup()
        for col in range(10):
            for row in range(10):
                sq = Square(side_length=CELL, stroke_color=GRID, stroke_width=1.5,
                            fill_color=PANEL, fill_opacity=1).move_to(cell_pos(col, row))
                sq.col, sq.row = col, row
                cells.add(sq)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.006),
                  run_time=1.2)

        self.beat("4–5 min. Resalte las tres columnas: 30 tickets con pan dulce. Haga notar que "
                  "el orden de la cuadrícula es una elección nuestra para poder contar.")
        pan = VGroup(*[c for c in cells if c.col < 3])
        pan_bg = Rectangle(width=3 * PITCH + 0.06, height=10 * PITCH + 0.06,
                           stroke_color=BLUE, stroke_width=3, fill_color=BLUE,
                           fill_opacity=0.12).move_to((cell_pos(0, 0) + cell_pos(2, 9)) / 2)
        pan_lab = safe_text("pan dulce · 30", size=22, color=BLUE, weight=BOLD)
        pan_lab.next_to(pan_bg, UP, buff=0.12)
        self.play(FadeIn(pan_bg), pan.animate.set_stroke(BLUE, width=2.5), FadeIn(pan_lab))

        self.beat("5–6 min. Lea la frase del gerente. Pida votar a mano alzada: ¿acomodamos la "
                  "leche junto al pan dulce? No revele nada más todavía.")
        leche_pan = VGroup(*[c for c in cells if c.col < 3 and c.row < 6])
        self.play(leche_pan.animate.set_fill(ORANGE, opacity=0.85), run_time=0.9)
        quote = VGroup(
            safe_text("El gerente:", size=22, color=MUTED),
            safe_text("«De los 30 que llevan pan dulce,", size=27, max_width=6.2),
            safe_text("18 también llevan leche: 60 %.»", size=27, color=ORANGE,
                      weight=BOLD, max_width=6.2),
            safe_text("¿Ponemos la leche junto al pan dulce?", size=26, color=YELLOW,
                      max_width=6.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.3, 1.0, 0])
        legend = VGroup(Square(0.28, stroke_width=0, fill_color=ORANGE, fill_opacity=0.85),
                        safe_text("ticket con leche", size=20, color=MUTED)).arrange(RIGHT, buff=0.15)
        legend.next_to(quote, DOWN, buff=0.4).align_to(quote, LEFT)
        self.play(LaggedStart(*[FadeIn(q, shift=RIGHT * 0.1) for q in quote], lag_ratio=0.25),
                  FadeIn(legend))

        self.beat("6–7 min. Ahora sí: pinte la leche en el resto de la tienda. Pregunte antes: "
                  "¿qué porcentaje esperan fuera del pan dulce? Luego cuenten juntos: 42 de 70.")
        leche_rest = VGroup(*[c for c in cells if c.col >= 3 and c.row < 6])
        self.play(LaggedStart(*[c.animate.set_fill(ORANGE, opacity=0.85) for c in leche_rest],
                              lag_ratio=0.01), run_time=1.3)
        inside = tex(rf"\frac{{18}}{{30}} = {fmt(HOOK.conf)}", size=36, color=BLUE)
        outside = tex(rf"\frac{{42}}{{70}} = {fmt(HOOK.h_not_a)}", size=36, color=MUTED)
        total = tex(rf"\frac{{60}}{{100}} = {fmt(HOOK.pb)}", size=36, color=ORANGE)
        labs = VGroup(safe_text("con pan dulce", size=20, color=BLUE),
                      safe_text("sin pan dulce", size=20, color=MUTED),
                      safe_text("toda la tienda", size=20, color=ORANGE))
        cols = VGroup(*[VGroup(l, m).arrange(DOWN, buff=0.15)
                        for l, m in zip(labs, [inside, outside, total])]).arrange(RIGHT, buff=0.55)
        cols.move_to([3.3, -1.65, 0])
        self.play(FadeOut(legend), FadeIn(cols, shift=UP * 0.1))

        self.beat("7–8 min. Conclusión verbal: la leche está en 60 % de TODOS los tickets. "
                  "Saber que alguien llevó pan dulce no cambia nada. Error a anticipar: «60 % es "
                  "alto». Alto ¿comparado con qué? Con la tasa base.")
        verdict = pill("Saber «pan dulce» no cambia nada", RED, size=21)
        verdict.move_to([3.3, -2.95, 0])
        self.play(FadeIn(verdict, shift=UP * 0.1))

        self.beat("8–9 min. Cerramos los huecos: cada ticket es 1 % del área. La leche forma una "
                  "franja plana a la altura 0.60 en ambas zonas: así se ve la independencia. "
                  "Anuncie que este cuadrado será nuestra herramienta.")
        self.play(FadeOut(quote), FadeOut(cols), FadeOut(verdict), FadeOut(pan_lab),
                  FadeOut(pan_bg), run_time=0.5)
        side = SQUARE_SIDE / 10
        self.play(*[c.animate.set_width(side).move_to(cell_pos(c.col, c.row, side))
                    .set_stroke(GRID, width=0.6) for c in cells], run_time=1.4)
        square = UnitSquare(SQUARE_SIDE, HOOK.pa, HOOK.conf, HOOK.h_not_a)
        square.move_to(GRID_CENTER)
        self.play(FadeIn(square), FadeOut(cells), run_time=0.9)
        level = square.level_line(HOOK.pb)
        level_lab = square.level_label(r"P(\text{leche})=0.60", HOOK.pb)
        a_lab = safe_text("pan dulce", size=20, color=BLUE, weight=BOLD)
        a_lab.next_to(square.p(HOOK.pa / 2, 1), UP, buff=0.12)
        area_note = VGroup(
            safe_text("1 ticket = 1 % del área", size=26),
            safe_text("área total = 1", size=26, color=MUTED),
            safe_text("franja plana = independencia", size=26, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.55, 1.0, 0])
        self.play(Create(level), FadeIn(level_lab), FadeIn(a_lab),
                  LaggedStart(*[FadeIn(a, shift=RIGHT * 0.1) for a in area_note], lag_ratio=0.3))

        self.beat("9–10 min. Pregunta guía de la clase (léala en voz alta). Muestre la ruta: "
                  "construir, ver, medir, decidir. Si alguien propone «comparar con el 60 %», "
                  "felicítelo: acaba de inventar el lift.")
        guide = VGroup(
            safe_text("Pregunta guía", size=20, color=MUTED, weight=BOLD),
            safe_text("¿Cuándo una regla A ⇒ B", size=30, weight=BOLD),
            safe_text("nos dice algo que no sabíamos?", size=30, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        route = VGroup(*[pill(t, c, size=19) for t, c in [
            ("1 Construir", BLUE), ("2 Ver", ORANGE), ("3 Medir", PURPLE), ("4 Decidir", GREEN)]]
        ).arrange(RIGHT, buff=0.15)
        block = VGroup(guide, route).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        block.move_to([3.55, -1.95, 0])
        self.play(FadeIn(block, shift=UP * 0.1))
