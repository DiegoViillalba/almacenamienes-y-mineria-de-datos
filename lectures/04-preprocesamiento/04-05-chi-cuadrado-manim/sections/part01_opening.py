"""Acto 1 · El problema (0–6 min).

Portada; seis columnas categóricas para predecir la satisfacción de 1,000
pasajeros. ¿Cuáles sobran? Con categorías no hay promedio ni correlación de
Pearson: hace falta otra herramienta.
"""
from __future__ import annotations

from manim import *

from sections.data import PREDICTORS
from sections.theme import (BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, TEAL, YELLOW,
                            card, pill, safe_text, tex)


class OpeningSection:
    def opening(self) -> None:
        self.beat("0–1 min. Portada. Hoy no vamos a empezar con la fórmula. Vamos a empezar "
                  "mirando barras de colores y la fórmula aparecerá sola cuando la necesitemos.")
        eyebrow = safe_text("PREPROCESAMIENTO · REDUCCIÓN DE VARIABLES", size=22, color=PURPLE,
                            weight=BOLD)
        title = safe_text("¿Esta variable aporta algo?", size=58, weight=BOLD, max_width=12.6)
        subtitle = safe_text("La prueba χ² para elegir variables categóricas", size=30,
                             color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–3 min. Una aerolínea encuestó a 1,000 pasajeros y quiere predecir su "
                  "SATISFACCIÓN. Tiene seis columnas, todas categóricas. Pida a cada estudiante "
                  "que escriba en papel cuáles dos eliminaría y por qué. No comente todavía.")
        self.clear()
        self.add_branding()
        head = self.heading("El problema", "1,000 pasajeros, seis columnas categóricas", BLUE)
        self.play(FadeIn(head), run_time=0.35)
        target = pill("Satisfacción: baja · media · alta", YELLOW, size=26)
        target.move_to([3.6, 0.2, 0])
        examples = {
            "Clase": "económica · business · primera",
            "Tipo de viaje": "negocio · vacaciones",
            "Frecuencia": "frecuente · ocasional · 1ª vez",
            "Retraso": "sin · leve · grave",
            "Asiento": "ventana · pasillo · centro",
            "Momento de compra": "anticipado · último momento",
        }
        rows = VGroup()
        for v in PREDICTORS:
            name = safe_text(v, size=24, color=INK, weight=BOLD)
            levels = safe_text(examples[v], size=18, color=MUTED)
            row = VGroup(name, levels).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-3.3, -0.35, 0])
        arrows = VGroup(*[Arrow(r.get_right() + RIGHT * 0.15, target.get_left(), buff=0.1,
                                color=MUTED, stroke_width=2, stroke_opacity=0.5,
                                max_tip_length_to_length_ratio=0.04) for r in rows])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.1))
        self.play(FadeIn(target), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05))
        q = safe_text("¿Cuáles dos eliminarías?", size=30, color=YELLOW, weight=BOLD)
        q.move_to([3.6, -1.4, 0])
        self.play(Write(q))

        self.beat("3–5 min. Con números haríamos una matriz de correlación. Pregunte: ¿cuál es "
                  "el promedio de ventana y pasillo? No existe. Las categorías no tienen "
                  "distancia ni orden: la correlación de Pearson no aplica. Lo único que podemos "
                  "hacer con categorías es CONTAR.")
        self.play(FadeOut(arrows), FadeOut(q), FadeOut(target),
                  rows.animate.scale(0.8).to_edge(LEFT, buff=0.6).set_opacity(0.35))
        absurd = tex(r"\frac{\text{ventana} + \text{pasillo}}{2} = \; ?", size=50)
        absurd.move_to([2.4, 1.1, 0])
        pearson = tex(r"r = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\cdots}", size=40, color=MUTED)
        pearson.move_to([2.4, -0.4, 0])
        cross = Cross(pearson, stroke_color=RED, stroke_width=5)
        self.play(Write(absurd))
        self.play(FadeIn(pearson), Create(cross))
        count = card(7.4, 1.1, TEAL).move_to([2.4, -2.1, 0])
        count_t = safe_text("Con categorías sólo podemos contar", size=28, color=TEAL,
                            weight=BOLD, max_width=6.9).move_to(count)
        self.play(FadeIn(count), Write(count_t))

        self.beat("5–6 min. Ruta de la clase en una frase: una variable aporta si conocerla "
                  "CAMBIA lo que esperamos del objetivo. Vamos a ver ese cambio con barras, "
                  "medirlo con un número (χ²), decidir si es real (valor p), medir su fuerza "
                  "(V de Cramér) y usarlo para reducir columnas.")
        self.clear()
        head = self.heading("La pregunta de hoy", "Conocer X, ¿cambia lo que espero de Y?", PURPLE)
        self.play(FadeIn(head), run_time=0.35)
        steps = [("ver", "perfiles por fila", GREEN), ("medir", "χ² = distancia", PURPLE),
                 ("decidir", "valor p", YELLOW), ("graduar", "V de Cramér", TEAL),
                 ("reducir", "eliminar columnas", RED)]
        chips = VGroup()
        for verb, what, col in steps:
            v = safe_text(verb, size=30, color=col, weight=BOLD)
            w = safe_text(what, size=20, color=MUTED)
            box = card(2.25, 1.55, col)
            VGroup(v, w).arrange(DOWN, buff=0.16).move_to(box)
            chips.add(VGroup(box, v, w))
        chips.arrange(RIGHT, buff=0.32).move_to([0, -0.2, 0])
        links = VGroup(*[Arrow(chips[i].get_right(), chips[i + 1].get_left(), buff=0.04,
                               color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.5)
                         for i in range(4)])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.18),
                  LaggedStart(*[GrowArrow(a) for a in links], lag_ratio=0.18))
