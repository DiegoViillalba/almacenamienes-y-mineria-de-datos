"""Acto 1 · Portada, objetivos y el resorte observado por tres cámaras."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (CAMERAS, CAMERA_BASES, SPRING_ANCHOR, SPRING_COLS,
                           mass_position)
from sections.theme import (BLUE, CAM_COLORS, GREEN, GRID, INK, MUTED, ORANGE,
                            PANEL, YELLOW, bullet_list, card, pill, safe_text)

CAM_RADIUS = 3.7
PANEL_CENTERS = {"A": np.array([5.35, 1.78, 0]), "B": np.array([5.35, -0.02, 0]),
                 "C": np.array([5.35, -1.82, 0])}
PANEL_SCALE = 0.62


def spring_curve(mass_x: float) -> ParametricFunction:
    start = SPRING_ANCHOR[0]
    end = mass_x - 0.32
    coils, r = 9, 0.22

    def f(s: float) -> np.ndarray:
        taper = min(1.0, 8 * s, 8 * (1 - s))
        return np.array([start + s * (end - start),
                         r * taper * np.cos(TAU * coils * s),
                         r * taper * np.sin(TAU * coils * s)])

    return ParametricFunction(f, t_range=[0, 1, 1 / 220], color=MUTED, stroke_width=3.2)


def reading(cam: str, phase: float) -> np.ndarray:
    p = mass_position(phase)
    u, v = CAMERA_BASES[cam]
    return np.array([p @ u, p @ v])


class OpeningSection:
    def opening(self) -> None:
        self.beat("0–2 min. Portada. Pregunte: ¿cuántas columnas necesita realmente una tabla? "
                  "Deje dos o tres respuestas sin corregir; se retomarán al final del resorte.")
        eyebrow = safe_text("PREPROCESAMIENTO · REDUCCIÓN DE DIMENSIONALIDAD", size=22,
                            color=BLUE, weight=BOLD)
        title = safe_text("Análisis de Componentes Principales", size=58, weight=BOLD,
                          max_width=12.4)
        subtitle = safe_text("Seis columnas, un solo movimiento: girar los ejes para ver menos y entender más",
                             size=26, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("2–3 min. Lea los objetivos como acciones verificables. Anuncie un reto en "
                  "parejas (minuto 52) y un ticket de salida.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "¿Qué sabremos hacer al terminar?", YELLOW)
        goals = bullet_list([
            "1. Explicar PCA como un giro de ejes que sigue la mayor variación.",
            "2. Centrar, construir S y obtener autovalores y autovectores a mano.",
            "3. Leer varianza explicada, puntuaciones (scores) y loadings.",
            "4. Decidir cuándo escalar y reconocer límites: atípicos, clases, no linealidad.",
        ], size=27).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.15) for g in goals], lag_ratio=0.15),
                  run_time=1.0)
        self.spring_story()

    # ------------------------------------------------------------------
    def spring_story(self) -> None:
        self.beat("3–4 min. Presente el experimento: una masa unida a un resorte que sólo oscila "
                  "a lo largo de una línea. Pregunte cuántas cantidades describen su estado.")
        head = self.start("Experimento", "Una masa, un resorte", ORANGE, fixed=True)
        self.move_camera(phi=68 * DEGREES, theta=-38 * DEGREES, zoom=0.86, run_time=1.2)

        wall = Prism(dimensions=[0.14, 1.9, 1.9], fill_color=GRID, fill_opacity=0.9,
                     stroke_width=0).move_to(SPRING_ANCHOR + LEFT * 0.08)
        floor = Rectangle(width=7.4, height=4.2, stroke_color=GRID, stroke_width=1.5,
                          fill_color=GRID, fill_opacity=0.10).move_to([0, 0, -0.95])
        axis = DashedLine([-2.6, 0, 0], [2.2, 0, 0], color=YELLOW, stroke_width=2,
                          dash_length=0.12)
        phase = ValueTracker(0.0)
        mass = Sphere(radius=0.32, resolution=(14, 14)).set_color(ORANGE)
        mass.set_fill(ORANGE, opacity=1).set_stroke(width=0)
        mass.move_to(mass_position(0))
        mass.add_updater(lambda m: m.move_to(mass_position(phase.get_value())))
        spring = spring_curve(mass_position(0)[0])
        spring.add_updater(lambda m: m.become(spring_curve(mass_position(phase.get_value())[0])))
        self.play(FadeIn(floor), FadeIn(wall), Create(spring), FadeIn(mass), run_time=1.2)

        self.beat("4–5 min. Deje ver dos oscilaciones. Pida que señalen la única dirección en la "
                  "que se mueve la masa (la línea amarilla punteada).", loop=True)
        self.play(phase.animate.set_value(2 * TAU), run_time=4, rate_func=linear)
        phase.set_value(0)

        self.beat("5–6 min. Ahora no vemos la masa directamente: tres cámaras la filman desde "
                  "ángulos arbitrarios, que no conocemos. Cada cámara entrega (x, y) en su pantalla.")
        self.play(FadeIn(axis))
        cams = VGroup()
        labels = VGroup()
        for key, normal in CAMERAS.items():
            n = normal / np.linalg.norm(normal)
            pos = n * CAM_RADIUS
            body = Cube(side_length=0.34, fill_color=CAM_COLORS[key], fill_opacity=1,
                        stroke_width=0).move_to(pos + n * 0.18)
            lens = Cone(base_radius=0.2, height=0.34, direction=n, fill_color=CAM_COLORS[key],
                        fill_opacity=0.9, stroke_width=0).move_to(pos - n * 0.12)
            sight = DashedLine(pos, ORIGIN, color=CAM_COLORS[key], stroke_width=1.6,
                               stroke_opacity=0.6, dash_length=0.1)
            cams.add(VGroup(body, lens, sight))
            lab = safe_text(key, size=30, color=CAM_COLORS[key], weight=BOLD)
            lab.move_to(pos + n * 0.62 + OUT * 0.35)
            labels.add(lab)
        self.oriented(*labels)
        self.play(LaggedStart(*[FadeIn(c) for c in cams], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.25), run_time=1.4)

        self.beat("6–7 min. Cada panel muestra lo que registra una cámara. Pregunte: ¿qué figura "
                  "dibuja el punto en cada pantalla? (una línea). Las seis cifras de abajo son "
                  "una fila de la tabla en cada instante.")
        panels = Group()
        dots = VGroup()
        for key in "ABC":
            c = PANEL_CENTERS[key]
            frame = card(2.55, 1.6, CAM_COLORS[key]).move_to(c)
            name = safe_text(f"Cámara {key}", size=17, color=CAM_COLORS[key], weight=BOLD)
            name.move_to(frame.get_corner(UL) + RIGHT * 0.62 + DOWN * 0.2)
            center = np.array([reading(key, a) for a in np.linspace(0, TAU, 60)]).mean(axis=0)

            def to_panel(r, c=c, center=center):
                d = (r - center) * PANEL_SCALE
                return c + np.array([d[0], d[1], 0]) + DOWN * 0.1

            ends = [to_panel(reading(key, 0)), to_panel(reading(key, PI))]
            trace = Line(ends[0], ends[1], color=CAM_COLORS[key], stroke_width=2,
                         stroke_opacity=0.45)
            dot = Dot(to_panel(reading(key, 0)), radius=0.08, color=ORANGE)
            dot.add_updater(lambda m, f=to_panel, k=key: m.move_to(f(reading(k, phase.get_value()))))
            panels.add(frame, name, trace)
            dots.add(dot)

        row = VGroup()
        for j, col in enumerate(SPRING_COLS):
            cam = col[-1]
            lab = safe_text(col, size=20, color=CAM_COLORS[cam], weight=BOLD)
            num = DecimalNumber(0, num_decimal_places=2, font_size=30, color=INK,
                                include_sign=True)
            cell = VGroup(lab, num).arrange(DOWN, buff=0.1)
            row.add(cell)
        row.arrange(RIGHT, buff=0.42).move_to([-2.2, -2.95, 0])
        row_box = card(row.width + 0.5, row.height + 0.3).move_to(row)
        for j, cell in enumerate(row):
            cam, idx = SPRING_COLS[j][-1], 0 if SPRING_COLS[j][0] == "x" else 1
            cell[1].set_value(reading(cam, 0)[idx])
        self.fixed(panels, dots, row_box, row)
        for j, cell in enumerate(row):
            cam, idx = SPRING_COLS[j][-1], 0 if SPRING_COLS[j][0] == "x" else 1
            self.keep_fixed(cell[1], lambda m, c=cam, i=idx: m.set_value(reading(c, phase.get_value())[i]))
        self.play(FadeIn(panels), FadeIn(dots), FadeIn(row_box), FadeIn(row), run_time=0.9)

        self.beat("7–8 min. Observe en silencio una oscilación completa. Pregunte: ¿alguna de las "
                  "seis cifras se mueve por su cuenta?", loop=True)
        self.play(phase.animate.set_value(2 * TAU), run_time=4, rate_func=linear)
        phase.set_value(0)

        self.beat("8–9 min. Giramos la cámara del salón para comprobar que la masa sigue sobre una "
                  "sola línea. Las pantallas no cambian: el experimento es el mismo.", loop=True)
        self.move_camera(theta=-38 * DEGREES + TAU, run_time=8, rate_func=linear,
                         added_anims=[phase.animate(rate_func=linear).set_value(4 * TAU)])
        phase.set_value(0)
        self.set_camera_orientation(theta=-38 * DEGREES)

        self.beat("9–10 min. Pida la respuesta: seis números por instante, pero un solo grado de "
                  "libertad (posición sobre el eje). El resto es ruido de las cámaras. Anuncie la "
                  "pregunta que guía la clase: ¿cómo lo descubre una máquina sin saber física?")
        q = pill("6 columnas → ¿cuántas dimensiones?", YELLOW, size=18)
        q.to_edge(LEFT, buff=0.35).set_y(1.5)
        ans = pill("1 grado de libertad + ruido", ORANGE, size=18).next_to(q, DOWN, buff=0.18)
        ans.align_to(q, LEFT)
        self.fixed(q, ans)
        self.play(FadeIn(q, shift=DOWN * 0.1))
        self.play(FadeIn(ans, shift=DOWN * 0.1))
