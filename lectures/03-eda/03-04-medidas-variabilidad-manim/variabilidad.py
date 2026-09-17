"""Repaso animado y numérico de medidas de variabilidad.

La escena está pensada para presentarse en vivo con Manim Slides. Cada llamada
a ``next_slide`` es una pausa controlada por el docente, no una nueva escena.
Los ejemplos siguen la convención de cuartiles del PDF de referencia: mediana
de la mitad inferior y de la mitad superior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manim import *
from manim_slides import Slide


# Paleta oscura y de alto contraste, inspirada en explicaciones matemáticas
# construidas sobre una pizarra. Cada concepto conserva su color en toda la clase.
BG = "#07111F"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"
YELLOW = "#F7C948"
GREEN = "#6FCF97"
RED = "#FF6B6B"
PURPLE = "#B28DFF"
ORANGE = "#FFB86B"

# Generic Pango family: portable across macOS/Linux and available in Manim's
# headless renderers without bundling a proprietary font.
FONT = "Sans"
LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "Logo_FC_Blanco.png"


def safe_text(
    content: str,
    *,
    size: float = 34,
    color: str = INK,
    weight: str = NORMAL,
    max_width: float | None = None,
) -> Text:
    """Create portable text and constrain it to the 16:9 frame when needed."""

    mob = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def number_card(value: object, color: str = BLUE, width: float = 0.82) -> VGroup:
    """Small numeric tile used to make sorting and grouping visible."""

    box = RoundedRectangle(
        corner_radius=0.10,
        width=width,
        height=0.62,
        stroke_color=color,
        stroke_width=2,
        fill_color=BG,
        fill_opacity=0.96,
    )
    label = safe_text(str(value), size=25, color=INK)
    label.scale_to_fit_width(min(width - 0.12, label.width))
    label.move_to(box)
    return VGroup(box, label)


def pill(content: str, color: str, width: float = 2.25) -> VGroup:
    """Compact colored label."""

    box = RoundedRectangle(
        corner_radius=0.20,
        width=width,
        height=0.58,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.13,
        stroke_width=2,
    )
    label = safe_text(content, size=25, color=color, weight=BOLD, max_width=width - 0.25)
    label.move_to(box)
    return VGroup(box, label)


class MedidasVariabilidad(Slide):
    """Presentación completa: una narrativa, una escena, muchos cortes."""

    def construct(self) -> None:
        # Manim Slides reads ``to_hex()`` from the renderer background while it
        # writes its manifest, so keep the value as ManimColor (not a raw str).
        self.camera.background_color = ManimColor(BG)
        self.branding: Group | None = None
        self.opening()
        self.range_story()
        self.quartile_story()
        self.outlier_and_boxplot_story()
        self.variance_story()
        self.choice_story()
        self.transfer_challenge()
        self.closing()

    # ------------------------------------------------------------------
    # Utilidades de composición
    # ------------------------------------------------------------------
    def pause(self, notes: str) -> None:
        self.next_slide(notes=notes)

    def clear(self, run_time: float = 0.55) -> None:
        removable = [m for m in self.mobjects if m is not self.branding]
        if removable:
            self.play(*[FadeOut(m) for m in removable], run_time=run_time)

    def logo_lockup(self, *, width: float, height: float) -> Group:
        """Place the shared white Facultad de Ciencias–UNAM lockup."""

        card = RoundedRectangle(
            corner_radius=0.10,
            width=width,
            height=height,
            stroke_color=GRID,
            stroke_width=1.2,
            stroke_opacity=0.65,
            fill_color=BG,
            fill_opacity=0.35,
        )
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width - 0.26)
        if logo.height > height - 0.18:
            logo.scale_to_fit_height(height - 0.18)
        logo.move_to(card)
        return Group(card, logo)

    def add_course_branding(self) -> None:
        """Keep the same institutional signature on every content slide."""

        bar = Rectangle(
            width=config.frame_width,
            height=0.40,
            stroke_width=0,
            fill_color="#050D18",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0)
        rule = Line(
            LEFT * config.frame_width / 2,
            RIGHT * config.frame_width / 2,
            color=GRID,
            stroke_width=1.5,
        ).move_to([0, -3.59, 0])
        signature = safe_text(
            "Diego Villalba  ·  Almacenes y Minería de Datos",
            size=13,
            color=MUTED,
        ).move_to([-4.75, -3.80, 0])
        logo = self.logo_lockup(width=1.42, height=0.34).move_to([6.12, -3.80, 0])
        self.branding = Group(bar, rule, signature, logo).set_z_index(100)
        self.add(self.branding)

    def heading(self, kicker: str, title: str, color: str = BLUE) -> VGroup:
        kicker_mob = safe_text(kicker.upper(), size=20, color=color, weight=BOLD)
        title_mob = safe_text(title, size=43, color=INK, weight=BOLD, max_width=12.4)
        block = VGroup(kicker_mob, title_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        block.to_corner(UL, buff=0.45)
        rule = Line(LEFT * 0.55, RIGHT * 0.55, color=color, stroke_width=4)
        rule.next_to(block, DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(block, rule)

    def footer(self, label: str) -> VGroup:
        line = Line(LEFT, RIGHT, color=GRID, stroke_width=2).set_width(13.3)
        line.to_edge(DOWN, buff=0.32)
        text = safe_text(label, size=16, color=MUTED)
        text.next_to(line, DOWN, buff=0.07).align_to(line, LEFT)
        return VGroup(line, text)

    def dot_plot(
        self,
        values: list[float],
        *,
        y: float,
        label: str,
        color: str,
        axis_range: tuple[float, float, float] = (0, 90, 10),
        length: float = 10.5,
    ) -> tuple[VGroup, NumberLine, VGroup]:
        axis = NumberLine(
            x_range=list(axis_range),
            length=length,
            include_numbers=False,
            include_tip=False,
            color=GRID,
            stroke_width=3,
        ).move_to([0.55, y, 0])
        label_mob = pill(label, color, width=1.35).next_to(axis, LEFT, buff=0.35)

        duplicates: dict[float, int] = {}
        dots = VGroup()
        for value in values:
            level = duplicates.get(value, 0)
            duplicates[value] = level + 1
            dot = Dot(
                axis.n2p(value) + UP * (0.15 + level * 0.19),
                radius=0.085,
                color=color,
            )
            dots.add(dot)
        ticks = VGroup()
        for value in (axis_range[0], 42, axis_range[1]):
            if axis_range[0] <= value <= axis_range[1]:
                tick = safe_text(str(value), size=18, color=MUTED)
                tick.next_to(axis.n2p(value), DOWN, buff=0.16)
                ticks.add(tick)
        return VGroup(axis, label_mob, dots, ticks), axis, dots

    def table_row(
        self,
        label: str,
        values: list[object],
        *,
        y: float,
        color: str,
        cell_width: float = 1.02,
    ) -> VGroup:
        row_label = safe_text(label, size=24, color=color, weight=BOLD)
        row_label.move_to([-5.75, y, 0])
        cells = VGroup()
        xs = np.linspace(-4.65, 4.65, len(values))
        for x, value in zip(xs, values):
            card = number_card(value, color=color, width=cell_width)
            card.move_to([x, y, 0])
            cells.add(card)
        return VGroup(row_label, cells)

    # ------------------------------------------------------------------
    # 1. Gancho: igual centro, distinta historia
    # ------------------------------------------------------------------
    def opening(self) -> None:
        eyebrow = safe_text(
            "ANÁLISIS EXPLORATORIO · REPASO NUMÉRICO",
            size=24,
            color=BLUE,
            weight=BOLD,
        )
        title = safe_text(
            "La forma de la dispersión",
            size=63,
            color=INK,
            weight=BOLD,
            max_width=12.4,
        )
        subtitle = safe_text(
            "del dato individual a una decisión defendible",
            size=31,
            color=MUTED,
            max_width=11.5,
        )
        formula = MathTex(
            r"R\;\longrightarrow\;IQR\;\longrightarrow\;s^2\;\longrightarrow\;s",
            font_size=48,
            color=INK,
        )
        formula.set_color_by_tex("R", ORANGE)
        formula.set_color_by_tex("IQR", GREEN)
        formula.set_color_by_tex("s^2", PURPLE)
        formula.set_color_by_tex("s", YELLOW)

        course = safe_text(
            "Almacenes y Minería de Datos",
            size=25,
            color=BLUE,
            weight=BOLD,
        )
        author = safe_text(
            "Diego Villalba · Facultad de Ciencias · UNAM",
            size=20,
            color=MUTED,
        )
        identity = VGroup(course, author).arrange(DOWN, buff=0.08)
        logo = self.logo_lockup(width=4.05, height=1.18)

        group = Group(eyebrow, title, subtitle, formula, identity, logo).arrange(
            DOWN, buff=0.20
        )
        self.play(FadeIn(eyebrow, shift=UP * 0.2), run_time=0.5)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.18), run_time=1.1)
        self.play(Write(formula), run_time=0.9)
        self.play(FadeIn(identity, shift=UP * 0.12), FadeIn(logo), run_time=0.7)
        self.pause(
            "Portada. Enfatice que no se memorizarán fórmulas: se reconstruirá "
            "qué información conserva y qué información pierde cada medida."
        )
        self.clear()
        self.add_course_branding()

        header = self.heading("Pregunta guía", "¿Cuál ruta es más confiable?", YELLOW)
        route_a = [38, 40, 41, 42, 43, 44, 45, 43]
        route_b = [10, 20, 30, 40, 44, 50, 60, 82]
        plot_a, axis_a, _ = self.dot_plot(route_a, y=1.15, label="RUTA A", color=BLUE)
        plot_b, axis_b, _ = self.dot_plot(route_b, y=-1.35, label="RUTA B", color=PURPLE)
        question = safe_text(
            "Predice antes de calcular: ¿A, B o da lo mismo?",
            size=29,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.62)

        self.play(FadeIn(header), run_time=0.5)
        self.play(Create(plot_a[0]), FadeIn(plot_a[1]), LaggedStart(*[GrowFromCenter(d) for d in plot_a[2]], lag_ratio=0.08))
        self.play(Create(plot_b[0]), FadeIn(plot_b[1]), LaggedStart(*[GrowFromCenter(d) for d in plot_b[2]], lag_ratio=0.08))
        self.play(FadeIn(plot_a[3]), FadeIn(plot_b[3]), Write(question))
        self.pause(
            "Haga una votación rápida. No revele todavía la media. Pida que la "
            "elección se justifique observando la anchura de cada nube de puntos."
        )

        mean_x_a = axis_a.n2p(42)[0]
        mean_line = DashedLine(
            [mean_x_a, -2.05, 0],
            [mean_x_a, 2.05, 0],
            dash_length=0.11,
            color=YELLOW,
            stroke_width=4,
        )
        mean_label = MathTex(r"\bar{x}_A=\bar{x}_B=42\ \text{min}", font_size=42, color=YELLOW)
        mean_label.next_to(header, DOWN, buff=0.28)
        answer = safe_text(
            "Misma media. Historias radicalmente distintas.",
            size=31,
            color=INK,
            weight=BOLD,
        ).move_to(question)
        self.play(Create(mean_line), Write(mean_label), Transform(question, answer), run_time=1.0)
        self.play(Circumscribe(plot_a[2], color=GREEN), Circumscribe(plot_b[2], color=RED), run_time=1.2)
        self.pause(
            "Ambas medias son 42 minutos. La ruta A es estrecha; la B puede tomar "
            "de 10 a 82. Ese contraste instala la necesidad de medir variabilidad."
        )
        self.clear()

        header = self.heading("Mapa del repaso", "Cuatro lentes para una misma nube", BLUE)
        center = Circle(radius=1.15, color=BLUE, stroke_width=4, fill_color=BLUE, fill_opacity=0.10)
        center_text = safe_text("DATOS", size=34, color=INK, weight=BOLD).move_to(center)
        center_group = VGroup(center, center_text)
        concepts = [
            ("RANGO", "extremos", ORANGE, UL * 2.15),
            ("IQR", "centro 50%", GREEN, UR * 2.15),
            ("VARIANZA", "cuadrados", PURPLE, DL * 2.15),
            ("DESV. EST.", "unidades", YELLOW, DR * 2.15),
        ]
        nodes = VGroup()
        connectors = VGroup()
        for name, detail, color, direction in concepts:
            node = VGroup(
                pill(name, color, width=2.5),
                safe_text(detail, size=21, color=MUTED),
            ).arrange(DOWN, buff=0.12)
            node.move_to(direction)
            connector = Line(center.get_center(), node.get_center(), color=color, stroke_opacity=0.55)
            connector.set_z_index(-1)
            nodes.add(node)
            connectors.add(connector)

        rule = safe_text(
            "Cada lente responde una pregunta distinta.",
            size=30,
            color=INK,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(header), GrowFromCenter(center_group))
        self.play(LaggedStart(*[Create(line) for line in connectors], lag_ratio=0.12))
        self.play(LaggedStart(*[FadeIn(node, scale=0.9) for node in nodes], lag_ratio=0.12))
        self.play(Write(rule))
        self.pause(
            "Mapa de ruta. Anticipe las preguntas: extremos, mitad central, "
            "distancias cuadráticas y una escala interpretable."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 2. Rango: una medida de extremos
    # ------------------------------------------------------------------
    def range_story(self) -> None:
        header = self.heading("1 · Rango", "¿Qué tan separados están los extremos?", ORANGE)
        axis_close = NumberLine(
            x_range=[10, 16, 1],
            length=10.6,
            include_numbers=True,
            font_size=22,
            include_tip=False,
            color=GRID,
        ).shift(DOWN * 0.15)
        base_values = [12, 13, 13.5, 14, 15]
        dots = VGroup(*[Dot(axis_close.n2p(v), radius=0.10, color=BLUE) for v in base_values])
        min_tag = pill("mín = 12", BLUE, width=1.75).next_to(axis_close.n2p(12), UP, buff=0.35)
        max_tag = pill("máx = 15", BLUE, width=1.75).next_to(axis_close.n2p(15), UP, buff=0.35)
        brace = BraceBetweenPoints(axis_close.n2p(12), axis_close.n2p(15), DOWN, color=ORANGE)
        calculation = MathTex(r"R=15-12=3", font_size=46, color=ORANGE)
        calculation.next_to(brace, DOWN, buff=0.35)
        units = safe_text("sueldos en miles de pesos", size=22, color=MUTED).to_edge(DOWN, buff=0.42)

        self.play(FadeIn(header), Create(axis_close), FadeIn(units))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.13))
        self.play(FadeIn(min_tag), FadeIn(max_tag), GrowFromCenter(brace), Write(calculation))
        self.pause(
            "Con los cinco sueldos habituales, el rango es 3 mil pesos. Señale que "
            "el cálculo sólo consulta mínimo y máximo; ignora los tres puntos internos."
        )

        axis_wide = NumberLine(
            x_range=[0, 125, 25],
            length=11.6,
            include_numbers=True,
            font_size=22,
            include_tip=False,
            color=GRID,
        ).shift(DOWN * 0.15)
        wide_dots = VGroup(*[Dot(axis_wide.n2p(v), radius=0.10, color=BLUE) for v in base_values])
        outlier = Dot(axis_wide.n2p(120), radius=0.13, color=RED)
        new_min_tag = pill("mín = 12", BLUE, width=1.75).next_to(axis_wide.n2p(12), UP, buff=0.35)
        new_max_tag = pill("máx = 120", RED, width=2.05).next_to(axis_wide.n2p(120), UP, buff=0.35)
        new_brace = BraceBetweenPoints(axis_wide.n2p(12), axis_wide.n2p(120), DOWN, color=ORANGE)
        new_calc = MathTex(r"R=120-12=108", font_size=46, color=ORANGE)
        new_calc.next_to(new_brace, DOWN, buff=0.32)

        self.play(
            ReplacementTransform(axis_close, axis_wide),
            Transform(dots, wide_dots),
            Transform(min_tag, new_min_tag),
            FadeOut(max_tag),
            Transform(brace, new_brace),
            TransformMatchingTex(calculation, new_calc),
            run_time=1.45,
        )
        self.play(GrowFromCenter(outlier), FadeIn(new_max_tag, shift=LEFT * 0.25))
        warning = safe_text(
            "Un solo dato cambió el rango: 3 → 108",
            size=31,
            color=RED,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.42)
        self.play(Transform(units, warning), Flash(outlier, color=RED, flash_radius=0.45))
        self.pause(
            "El sueldo de 120 mil comprime visualmente a toda la mayoría. El rango "
            "describe la separación total, pero es muy sensible a un extremo."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 3. Cuartiles e IQR: construir el centro 50 %
    # ------------------------------------------------------------------
    def quartile_story(self) -> None:
        header = self.heading("2 · Orden e IQR", "Primero ordenamos; después cortamos", GREEN)
        raw_values = [42, 18, 25, 30, 22, 19, 28, 24, 21, 26, 20, 90]
        sorted_values = sorted(raw_values)
        cards = VGroup(*[number_card(v, color=BLUE, width=0.78) for v in raw_values])
        xs = np.linspace(-5.35, 5.35, len(cards))
        for card, x in zip(cards, xs):
            card.move_to([x, 0.80, 0])

        prompt = safe_text(
            "12 tiempos de espera (minutos)", size=27, color=MUTED
        ).next_to(header, DOWN, buff=0.28).align_to(header, LEFT)
        shuffled = safe_text("orden de llegada", size=21, color=ORANGE).next_to(cards, UP, buff=0.22)
        self.play(FadeIn(header), FadeIn(prompt), FadeIn(shuffled))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in cards], lag_ratio=0.06))
        self.pause(
            "Muestre la lista en el orden de llegada. Pregunte qué operación debe "
            "ocurrir antes de hablar de percentiles o cuartiles."
        )

        target_xs = np.linspace(-5.35, 5.35, len(sorted_values))
        position_by_value = {value: x for value, x in zip(sorted_values, target_xs)}
        sort_animations = [
            card.animate.move_to([position_by_value[value], 0.80, 0])
            for card, value in zip(cards, raw_values)
        ]
        sorted_label = safe_text("menor  →  mayor", size=23, color=GREEN).move_to(shuffled)
        self.play(*sort_animations, Transform(shuffled, sorted_label), run_time=1.55)

        ordered_cards = sorted(zip(raw_values, cards), key=lambda item: item[0])
        index_labels = VGroup()
        for index, (_, card) in enumerate(ordered_cards, start=1):
            idx = safe_text(str(index), size=15, color=MUTED)
            idx.next_to(card, DOWN, buff=0.09)
            index_labels.add(idx)
        self.play(FadeIn(index_labels))
        self.pause(
            "Lea los datos ordenados. La posición, no la distancia al extremo, "
            "determina los cortes. Usaremos mediana de mitades como convención."
        )

        sorted_cards = [card for _, card in ordered_cards]
        divider = DashedLine(
            [0, 0.25, 0], [0, 1.55, 0], color=MUTED, dash_length=0.10
        )
        lower_box = SurroundingRectangle(
            VGroup(*sorted_cards[:6]), color=BLUE, buff=0.10, corner_radius=0.08
        ).set_fill(BLUE, opacity=0.06)
        upper_box = SurroundingRectangle(
            VGroup(*sorted_cards[6:]), color=PURPLE, buff=0.10, corner_radius=0.08
        ).set_fill(PURPLE, opacity=0.06)
        convention = safe_text(
            "convención: mediana de cada mitad",
            size=21,
            color=MUTED,
        ).to_edge(DOWN, buff=0.38)
        q2_formula = MathTex(r"Q_2=\frac{24+25}{2}=24.5", font_size=43, color=YELLOW)
        q2_formula.move_to([0, -0.75, 0])
        mid_highlights = VGroup(
            SurroundingRectangle(sorted_cards[5], color=YELLOW, buff=0.04),
            SurroundingRectangle(sorted_cards[6], color=YELLOW, buff=0.04),
        )

        self.play(Create(divider), Create(lower_box), Create(upper_box), FadeIn(convention))
        self.play(Create(mid_highlights), Write(q2_formula))
        self.pause(
            "Hay 12 observaciones: Q2 es el promedio de las posiciones 6 y 7, "
            "24 y 25. El resultado es 24.5 minutos."
        )

        q1_highlights = VGroup(
            SurroundingRectangle(sorted_cards[2], color=BLUE, buff=0.04),
            SurroundingRectangle(sorted_cards[3], color=BLUE, buff=0.04),
        )
        q3_highlights = VGroup(
            SurroundingRectangle(sorted_cards[8], color=PURPLE, buff=0.04),
            SurroundingRectangle(sorted_cards[9], color=PURPLE, buff=0.04),
        )
        q1_formula = MathTex(r"Q_1=\frac{20+21}{2}=20.5", font_size=39, color=BLUE)
        q3_formula = MathTex(r"Q_3=\frac{28+30}{2}=29", font_size=39, color=PURPLE)
        quartile_formulas = VGroup(q1_formula, q3_formula).arrange(RIGHT, buff=1.0)
        quartile_formulas.move_to([0, -1.85, 0])
        self.play(
            ReplacementTransform(mid_highlights, VGroup(q1_highlights, q3_highlights)),
            q2_formula.animate.shift(UP * 0.25),
            Write(quartile_formulas),
            run_time=1.15,
        )
        self.pause(
            "Dentro de cada mitad, promedie su pareja central: Q1=20.5 y Q3=29. "
            "Aclare que algunas bibliotecas interpolan con otra convención."
        )

        iqr_formula = MathTex(r"IQR=Q_3-Q_1=29-20.5=8.5\ \text{min}", font_size=45)
        iqr_formula.set_color_by_tex("IQR", GREEN)
        iqr_formula.set_color_by_tex("8.5", GREEN)
        iqr_formula.move_to([0, -2.90, 0])
        self.play(Write(iqr_formula))
        self.pause(
            "IQR mide la anchura de la mitad central: desde 20.5 hasta 29 minutos. "
            "La interpretación debe incluir siempre qué porcentaje y qué unidades."
        )

        # Compara explícitamente las dos anchuras sin pedir que una sola escala
        # represente a la vez el centro compacto y el valor 90.
        self.play(
            FadeOut(cards),
            FadeOut(index_labels),
            FadeOut(prompt),
            FadeOut(shuffled),
            FadeOut(divider),
            FadeOut(lower_box),
            FadeOut(upper_box),
            FadeOut(q1_highlights),
            FadeOut(q3_highlights),
            FadeOut(q2_formula),
            FadeOut(quartile_formulas),
            FadeOut(convention),
            FadeOut(iqr_formula),
            run_time=0.65,
        )

        scale = 0.135
        range_line = Line(LEFT * 4.86, RIGHT * 4.86, color=ORANGE, stroke_width=12)
        range_line.move_to([0, 0.35, 0])
        iqr_line = Line(LEFT * (8.5 * scale / 2), RIGHT * (8.5 * scale / 2), color=GREEN, stroke_width=12)
        iqr_line.move_to([0, -1.35, 0])
        range_label = VGroup(
            safe_text("RANGO", size=22, color=ORANGE, weight=BOLD),
            MathTex(r"90-18=72\ \text{min}", font_size=34, color=INK),
        ).arrange(DOWN, buff=0.08).next_to(range_line, UP, buff=0.18)
        iqr_label = VGroup(
            safe_text("MITAD CENTRAL", size=22, color=GREEN, weight=BOLD),
            MathTex(r"29-20.5=8.5\ \text{min}", font_size=34, color=INK),
        ).arrange(DOWN, buff=0.08).next_to(iqr_line, UP, buff=0.18)
        range_ends = VGroup(
            safe_text("18", size=23, color=ORANGE).next_to(range_line.get_left(), DOWN, buff=0.15),
            safe_text("90", size=23, color=ORANGE).next_to(range_line.get_right(), DOWN, buff=0.15),
        )
        iqr_ends = VGroup(
            safe_text("20.5", size=23, color=GREEN).next_to(iqr_line.get_left(), DOWN, buff=0.15),
            safe_text("29", size=23, color=GREEN).next_to(iqr_line.get_right(), DOWN, buff=0.15),
        )
        takeaway = safe_text(
            "Rango ve los extremos · IQR ve el centro",
            size=31,
            color=INK,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.48)
        self.play(Create(range_line), FadeIn(range_label), FadeIn(range_ends))
        self.play(Create(iqr_line), FadeIn(iqr_label), FadeIn(iqr_ends), Write(takeaway))
        self.pause(
            "Ambas barras usan la misma escala. El rango mide 72 minutos; el IQR "
            "sólo 8.5. El valor 90 domina una medida pero no la otra."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 4. Cercas de Tukey y construcción geométrica del boxplot
    # ------------------------------------------------------------------
    def outlier_and_boxplot_story(self) -> None:
        header = self.heading("3 · Atípicos", "La caja propone una frontera, no un veredicto", RED)
        axis = NumberLine(
            x_range=[5, 45, 5],
            length=11.4,
            include_numbers=True,
            font_size=20,
            include_tip=False,
            color=GRID,
        ).shift(DOWN * 0.35)
        q1, median, q3 = 20.5, 24.5, 29.0
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        box_band = Rectangle(
            width=axis.n2p(q3)[0] - axis.n2p(q1)[0],
            height=0.95,
            color=GREEN,
            fill_color=GREEN,
            fill_opacity=0.14,
            stroke_width=3,
        ).move_to([(axis.n2p(q1)[0] + axis.n2p(q3)[0]) / 2, axis.get_y(), 0])
        q1_mark = Line(UP * 0.65, DOWN * 0.65, color=GREEN, stroke_width=4).move_to(axis.n2p(q1))
        q3_mark = Line(UP * 0.65, DOWN * 0.65, color=GREEN, stroke_width=4).move_to(axis.n2p(q3))
        iqr_brace = BraceBetweenPoints(axis.n2p(q1) + UP * 0.60, axis.n2p(q3) + UP * 0.60, UP, color=GREEN)
        iqr_tag = MathTex(r"IQR=8.5", font_size=31, color=GREEN).next_to(iqr_brace, UP, buff=0.10)

        formulas = VGroup(
            MathTex(r"LI=Q_1-1.5(IQR)=7.75", font_size=34, color=BLUE),
            MathTex(r"LS=Q_3+1.5(IQR)=41.75", font_size=34, color=PURPLE),
        ).arrange(RIGHT, buff=0.75)
        formulas.next_to(header, DOWN, buff=0.32)

        self.play(FadeIn(header), FadeIn(formulas), Create(axis))
        self.play(GrowFromCenter(box_band), Create(q1_mark), Create(q3_mark), GrowFromCenter(iqr_brace), Write(iqr_tag))
        self.pause(
            "Parta de la caja central. Cada cerca queda a 1.5 IQR de su cuartil: "
            "LI=7.75 y LS=41.75. Son umbrales de exploración."
        )

        lower_mark = DashedLine(
            axis.n2p(lower) + DOWN * 0.65,
            axis.n2p(lower) + UP * 0.65,
            color=BLUE,
            dash_length=0.09,
        )
        upper_mark = DashedLine(
            axis.n2p(upper) + DOWN * 0.65,
            axis.n2p(upper) + UP * 0.65,
            color=PURPLE,
            dash_length=0.09,
        )
        lower_span = Line(axis.n2p(q1), axis.n2p(lower), color=BLUE, stroke_width=7)
        upper_span = Line(axis.n2p(q3), axis.n2p(upper), color=PURPLE, stroke_width=7)
        lower_label = safe_text("7.75", size=21, color=BLUE).next_to(lower_mark, DOWN, buff=0.12)
        upper_label = safe_text("41.75", size=21, color=PURPLE).next_to(upper_mark, DOWN, buff=0.12)
        self.play(
            GrowFromPoint(lower_span, axis.n2p(q1)),
            GrowFromPoint(upper_span, axis.n2p(q3)),
            Create(lower_mark),
            Create(upper_mark),
            FadeIn(lower_label),
            FadeIn(upper_label),
            run_time=1.15,
        )

        values_in_view = [18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 42]
        dots = VGroup()
        duplicate_levels: dict[int, int] = {}
        for value in values_in_view:
            level = duplicate_levels.get(value, 0)
            duplicate_levels[value] = level + 1
            color = RED if value > upper else INK
            dots.add(Dot(axis.n2p(value) + UP * (0.16 + 0.16 * level), radius=0.085, color=color))
        ninety_arrow = Arrow(
            axis.n2p(44.1) + UP * 0.65,
            axis.n2p(44.1) + RIGHT * 0.80 + UP * 0.65,
            color=RED,
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25,
        )
        ninety_label = safe_text("90, mucho más allá", size=22, color=RED, weight=BOLD)
        ninety_label.next_to(ninety_arrow, LEFT, buff=0.12)
        comparison = MathTex(r"42>41.75\qquad 90>41.75", font_size=40, color=RED)
        comparison.to_edge(DOWN, buff=0.45)
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.06))
        self.play(GrowArrow(ninety_arrow), FadeIn(ninety_label), Write(comparison))
        self.play(Flash(dots[-1], color=RED, flash_radius=0.32), Flash(ninety_arrow.get_end(), color=RED))
        self.pause(
            "42 queda apenas por encima de 41.75; 90 queda muy lejos. Ambos son "
            "candidatos atípicos. Candidato no significa error ni dato eliminable."
        )

        investigate = VGroup(
            safe_text("ATÍPICO", size=22, color=RED, weight=BOLD),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3),
            safe_text("investigar contexto", size=25, color=INK, weight=BOLD),
        ).arrange(RIGHT, buff=0.28)
        investigate.move_to([0, -2.45, 0])
        self.play(Transform(comparison, investigate))
        self.pause(
            "Detenga la clase aquí: la regla de Tukey es una alarma descriptiva. "
            "La acción correcta es revisar procedencia y reglas de negocio."
        )

        # Reconstrucción del boxplot sobre la misma escala.
        self.play(
            FadeOut(formulas),
            FadeOut(lower_span),
            FadeOut(upper_span),
            FadeOut(lower_mark),
            FadeOut(upper_mark),
            FadeOut(lower_label),
            FadeOut(upper_label),
            FadeOut(iqr_brace),
            FadeOut(iqr_tag),
            FadeOut(q1_mark),
            FadeOut(q3_mark),
            FadeOut(dots),
            FadeOut(ninety_arrow),
            FadeOut(ninety_label),
            FadeOut(comparison),
            run_time=0.55,
        )
        new_header = self.heading("3 · Boxplot", "Construyámoslo, pieza por pieza", GREEN)
        self.play(Transform(header, new_header), axis.animate.shift(DOWN * 0.35), box_band.animate.shift(DOWN * 0.35))
        axis_y = axis.get_y()
        box_band.move_to([(axis.n2p(q1)[0] + axis.n2p(q3)[0]) / 2, axis_y, 0])

        median_line = Line(
            [axis.n2p(median)[0], axis_y - 0.48, 0],
            [axis.n2p(median)[0], axis_y + 0.48, 0],
            color=YELLOW,
            stroke_width=5,
        )
        left_whisker = Line(axis.n2p(18), axis.n2p(q1), color=INK, stroke_width=4)
        right_whisker = Line(axis.n2p(q3), axis.n2p(30), color=INK, stroke_width=4)
        left_cap = Line(UP * 0.34, DOWN * 0.34, color=INK, stroke_width=4).move_to(axis.n2p(18))
        right_cap = Line(UP * 0.34, DOWN * 0.34, color=INK, stroke_width=4).move_to(axis.n2p(30))
        outlier_42 = Dot(axis.n2p(42), radius=0.11, color=RED)
        outlier_90 = VGroup(
            Dot(axis.n2p(44.4), radius=0.11, color=RED),
            safe_text("→ 90", size=23, color=RED).next_to(axis.n2p(44.4), RIGHT, buff=0.10),
        )
        labels = VGroup(
            pill("Q1 20.5", GREEN, 1.65).next_to(axis.n2p(q1), UP, buff=0.78),
            pill("Q2 24.5", YELLOW, 1.65).next_to(axis.n2p(median), DOWN, buff=0.72),
            pill("Q3 29", GREEN, 1.55).next_to(axis.n2p(q3), UP, buff=0.78),
        )
        self.play(GrowFromCenter(box_band), FadeIn(labels[0]), FadeIn(labels[2]))
        self.play(Create(median_line), FadeIn(labels[1]))
        self.play(Create(left_whisker), Create(right_whisker), Create(left_cap), Create(right_cap))
        self.play(GrowFromCenter(outlier_42), FadeIn(outlier_90))
        note = safe_text(
            "Los bigotes terminan en 18 y 30: los datos no atípicos más extremos.",
            size=25,
            color=MUTED,
            max_width=11.9,
        ).to_edge(DOWN, buff=0.46)
        self.play(Write(note))
        self.pause(
            "Enumere las piezas: caja Q1–Q3, línea de mediana, bigotes hasta 18 y "
            "30, puntos en 42 y 90. Los bigotes no son automáticamente mínimo y máximo."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 5. Varianza y desviación estándar: distancias que se hacen área
    # ------------------------------------------------------------------
    def variance_story(self) -> None:
        header = self.heading("4 · Distancias a la media", "La varianza se construye", PURPLE)
        route_a = [38, 40, 41, 42, 43, 44, 45, 43]
        deviations_a = [-4, -2, -1, 0, 1, 2, 3, 1]
        squares_a = [16, 4, 1, 0, 1, 4, 9, 1]
        route_b = [10, 20, 30, 40, 44, 50, 60, 82]
        deviations_b = [-32, -22, -12, -2, 2, 8, 18, 40]
        squares_b = [1024, 484, 144, 4, 4, 64, 324, 1600]

        axis = NumberLine(
            x_range=[36, 48, 1],
            length=10.4,
            include_numbers=True,
            font_size=20,
            include_tip=False,
            color=GRID,
        ).shift(DOWN * 0.15)
        mean_point = axis.n2p(42)
        mean_line = DashedLine(
            mean_point + DOWN * 1.0,
            mean_point + UP * 1.65,
            color=YELLOW,
            dash_length=0.10,
            stroke_width=4,
        )
        mean_tag = pill("media = 42", YELLOW, width=2.05).next_to(mean_line, UP, buff=0.15)
        duplicate_levels: dict[int, int] = {}
        dots = VGroup()
        for value in route_a:
            level = duplicate_levels.get(value, 0)
            duplicate_levels[value] = level + 1
            dots.add(Dot(axis.n2p(value) + UP * (0.12 + level * 0.20), radius=0.09, color=BLUE))

        self.play(FadeIn(header), Create(axis), Create(mean_line), FadeIn(mean_tag))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.08))
        self.pause(
            "Recupere la ruta A. La media es 42. Pregunte: ¿cómo convertir cada "
            "separación visual en un número que luego podamos combinar?"
        )

        # Una liga por dato: el signo depende del lado de la media.
        segments = VGroup()
        deviation_labels = VGroup()
        for index, (value, deviation) in enumerate(zip(route_a, deviations_a)):
            offset = UP * (0.42 + index * 0.11)
            start = mean_point + offset
            end = axis.n2p(value) + offset
            segment = Line(start, end, color=PURPLE, stroke_width=5)
            label = MathTex(
                f"{deviation:+d}" if deviation else "0",
                font_size=25,
                color=PURPLE,
            )
            midpoint = (start + end) / 2
            if abs(deviation) < 0.5:
                label.next_to(start, RIGHT, buff=0.10)
            else:
                label.move_to(midpoint + UP * 0.12)
            segments.add(segment)
            deviation_labels.add(label)
        self.play(
            LaggedStart(*[Create(segment) for segment in segments], lag_ratio=0.08),
            LaggedStart(*[FadeIn(label) for label in deviation_labels], lag_ratio=0.08),
            run_time=1.45,
        )
        signed_sum = MathTex(
            r"-4-2-1+0+1+2+3+1=0",
            font_size=39,
            color=PURPLE,
        ).to_edge(DOWN, buff=0.42)
        self.play(Write(signed_sum))
        zero_question = safe_text(
            "¿La dispersión es cero?",
            size=28,
            color=RED,
            weight=BOLD,
        ).next_to(signed_sum, UP, buff=0.16)
        self.play(FadeIn(zero_question, shift=UP * 0.15))
        self.pause(
            "Las desviaciones con signo siempre suman cero alrededor de la media. "
            "Pida al grupo detectar por qué esa suma no puede medir dispersión."
        )

        self.play(
            FadeOut(axis),
            FadeOut(mean_line),
            FadeOut(mean_tag),
            FadeOut(dots),
            FadeOut(segments),
            FadeOut(deviation_labels),
            FadeOut(signed_sum),
            FadeOut(zero_question),
            run_time=0.55,
        )
        new_header = self.heading("4 · Distancias a la media", "Elevar al cuadrado evita la cancelación", PURPLE)
        self.play(Transform(header, new_header))

        values_row = self.table_row("xᵢ", route_a, y=1.35, color=BLUE)
        deviations_row = self.table_row("dᵢ", deviations_a, y=0.30, color=PURPLE)
        squares_row = self.table_row("dᵢ²", squares_a, y=-0.75, color=YELLOW)
        self.play(FadeIn(values_row[0]), LaggedStart(*[FadeIn(c) for c in values_row[1]], lag_ratio=0.06))
        self.play(
            FadeIn(deviations_row[0]),
            LaggedStart(
                *[TransformFromCopy(src, dst) for src, dst in zip(values_row[1], deviations_row[1])],
                lag_ratio=0.06,
            ),
            run_time=1.15,
        )
        self.pause(
            "Lea la segunda fila como x_i menos 42. Conserve el signo un instante: "
            "es información de dirección, aunque la suma se cancele."
        )

        square_rule = MathTex(r"d_i\longmapsto d_i^2", font_size=37, color=YELLOW)
        square_rule.next_to(squares_row, DOWN, buff=0.32)
        self.play(
            FadeIn(squares_row[0]),
            LaggedStart(
                *[TransformFromCopy(src, dst) for src, dst in zip(deviations_row[1], squares_row[1])],
                lag_ratio=0.06,
            ),
            Write(square_rule),
            run_time=1.35,
        )
        sum_box = VGroup(
            RoundedRectangle(
                corner_radius=0.15,
                width=3.35,
                height=0.82,
                color=PURPLE,
                fill_color=PURPLE,
                fill_opacity=0.10,
            ),
            MathTex(r"\sum d_i^2=36", font_size=40, color=INK),
        )
        sum_box[1].move_to(sum_box[0])
        sum_box.to_edge(DOWN, buff=0.35)
        self.play(Transform(square_rule, sum_box))
        self.pause(
            "Los cuadrados impiden cancelación y hacen que una distancia grande pese "
            "mucho. La suma de cuadrados de la ruta A es 36 min²."
        )

        self.play(
            FadeOut(values_row),
            FadeOut(deviations_row),
            squares_row.animate.shift(UP * 1.55),
            FadeOut(square_rule),
            run_time=0.55,
        )
        divisor_question = MathTex(r"s^2=\frac{36}{\square}", font_size=58, color=INK)
        divisor_question.move_to([-3.15, -1.25, 0])
        choices = VGroup(pill("¿8?", BLUE, 1.45), pill("¿7?", PURPLE, 1.45)).arrange(RIGHT, buff=0.65)
        choices.move_to([3.15, -1.25, 0])
        self.play(Write(divisor_question), FadeIn(choices))
        self.pause(
            "Pregunta predictiva: para varianza muestral, ¿el divisor es 8 o 7? "
            "No revele la respuesta hasta que alguien explique el n-1."
        )

        constraint = MathTex(
            r"d_8=-\left(d_1+d_2+\cdots+d_7\right)",
            font_size=39,
            color=PURPLE,
        )
        constraint.move_to([0, -2.40, 0])
        free_brace = Brace(squares_row[1][:7], DOWN, color=BLUE)
        free_label = safe_text("7 desviaciones libres", size=23, color=BLUE, weight=BOLD)
        free_label.next_to(free_brace, DOWN, buff=0.10)
        chosen = pill("n − 1 = 7", PURPLE, 2.25).move_to(choices)
        self.play(GrowFromCenter(free_brace), FadeIn(free_label), Write(constraint))
        self.play(Transform(choices, chosen))
        self.pause(
            "La propia muestra estimó la media. Una vez elegidas siete desviaciones, "
            "la octava queda forzada por suma cero: quedan n-1 grados de libertad."
        )

        variance = MathTex(
            r"s^2=\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}"
            r"=\frac{36}{7}=5.14\ \text{min}^2",
            font_size=42,
            color=INK,
        )
        variance.set_color_by_tex("s^2", PURPLE)
        variance.set_color_by_tex("5.14", PURPLE)
        variance.move_to([0, -0.50, 0])
        self.play(
            FadeOut(squares_row),
            FadeOut(free_brace),
            FadeOut(free_label),
            FadeOut(constraint),
            FadeOut(choices),
            TransformMatchingTex(divisor_question, variance),
            run_time=1.2,
        )
        unit_square = VGroup(
            Square(side_length=1.10, color=PURPLE, fill_color=PURPLE, fill_opacity=0.18),
            MathTex(r"\text{min}^2", font_size=35, color=PURPLE),
        )
        unit_square[1].move_to(unit_square[0])
        unit_square.next_to(variance, DOWN, buff=0.48)
        self.play(GrowFromCenter(unit_square))
        self.pause(
            "La varianza cuantifica separación cuadrática. Su unidad es min²: útil "
            "para el cálculo, pero todavía incómoda para interpretar tiempos."
        )

        std_formula = MathTex(r"s=\sqrt{5.14}=2.27\ \text{min}", font_size=55, color=YELLOW)
        std_formula.move_to(variance)
        unit_segment = VGroup(
            Line(LEFT * 0.75, RIGHT * 0.75, color=YELLOW, stroke_width=8),
            safe_text("min", size=28, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, buff=0.17)
        unit_segment.move_to(unit_square)
        bridge = safe_text(
            "raíz cuadrada = regreso a la escala original",
            size=27,
            color=MUTED,
        ).to_edge(DOWN, buff=0.40)
        self.play(TransformMatchingTex(variance, std_formula), Transform(unit_square, unit_segment))
        self.play(Write(bridge))
        self.pause(
            "La raíz devuelve minutos. Interprete 2.27 como una escala típica de "
            "alejamiento respecto de la media, no como una garantía o un intervalo."
        )

        self.clear()

        # Vuelve al gancho y hace visible la diferencia de órdenes de magnitud.
        header = self.heading("5 · Comparación", "La misma media no implica la misma consistencia", YELLOW)
        plot_a, axis_a, _ = self.dot_plot(route_a, y=1.40, label="RUTA A", color=BLUE)
        plot_b, axis_b, _ = self.dot_plot(route_b, y=-1.05, label="RUTA B", color=PURPLE)
        mean_x = axis_a.n2p(42)[0]
        mean_line = DashedLine(
            [mean_x, -1.78, 0], [mean_x, 2.07, 0], color=YELLOW, dash_length=0.10, stroke_width=4
        )
        same_mean = MathTex(r"\bar{x}_A=\bar{x}_B=42", font_size=38, color=YELLOW)
        same_mean.next_to(header, DOWN, buff=0.18)
        self.play(FadeIn(header), Create(plot_a[0]), FadeIn(plot_a[1]), FadeIn(plot_a[3]), FadeIn(plot_a[2]))
        self.play(Create(plot_b[0]), FadeIn(plot_b[1]), FadeIn(plot_b[3]), FadeIn(plot_b[2]))
        self.play(Create(mean_line), Write(same_mean))
        self.pause(
            "Regrese a la predicción inicial. Ambas nubes comparten la marca 42; "
            "ahora calcularemos cuánto cuesta su distinta anchura."
        )

        # Áreas proporcionales a las sumas de cuadrados: razón aproximada 101:1.
        small_square = Square(
            side_length=0.50,
            color=BLUE,
            fill_color=BLUE,
            fill_opacity=0.30,
        ).move_to([-3.1, -2.45, 0])
        large_square = Square(
            side_length=5.02,
            color=PURPLE,
            fill_color=PURPLE,
            fill_opacity=0.13,
        ).move_to([2.50, -0.05, 0])
        # Antes de mostrar el área grande, despejamos el dotplot inferior para que
        # el cambio de escala sea legible y no se convierta en una tabla saturada.
        self.play(FadeOut(plot_a), FadeOut(plot_b), FadeOut(mean_line), FadeOut(same_mean))
        area_title = safe_text(
            "área total de las desviaciones cuadradas",
            size=27,
            color=MUTED,
        ).next_to(header, DOWN, buff=0.28)
        small_label = VGroup(
            pill("A", BLUE, 0.75),
            MathTex(r"\sum d_i^2=36", font_size=34, color=BLUE),
        ).arrange(DOWN, buff=0.15).next_to(small_square, UP, buff=0.18)
        large_label = VGroup(
            pill("B", PURPLE, 0.75),
            MathTex(r"\sum d_i^2=3648", font_size=34, color=PURPLE),
        ).arrange(DOWN, buff=0.15).move_to([2.50, 1.45, 0])
        self.play(FadeIn(area_title), GrowFromCenter(small_square), FadeIn(small_label))
        self.play(GrowFromCenter(large_square), FadeIn(large_label), run_time=1.25)
        self.pause(
            "Las áreas son proporcionales: 36 frente a 3648. La ruta B acumula "
            "más de cien veces la separación cuadrática de A."
        )

        results = VGroup(
            VGroup(
                pill("RUTA A", BLUE, 1.55),
                MathTex(r"s_A^2=5.14\ \text{min}^2", font_size=35, color=INK),
                MathTex(r"s_A=2.27\ \text{min}", font_size=41, color=BLUE),
            ).arrange(DOWN, buff=0.16),
            VGroup(
                pill("RUTA B", PURPLE, 1.55),
                MathTex(r"s_B^2=521.14\ \text{min}^2", font_size=35, color=INK),
                MathTex(r"s_B=22.83\ \text{min}", font_size=41, color=PURPLE),
            ).arrange(DOWN, buff=0.16),
        ).arrange(RIGHT, buff=1.65)
        results.move_to([0, -0.15, 0])
        verdict = safe_text(
            "A es mucho más constante: su desviación estándar es ≈ 10 veces menor.",
            size=28,
            color=GREEN,
            weight=BOLD,
            max_width=12.0,
        ).to_edge(DOWN, buff=0.45)
        self.play(
            FadeOut(small_square),
            FadeOut(large_square),
            FadeOut(small_label),
            FadeOut(large_label),
            FadeOut(area_title),
            FadeIn(results, shift=UP * 0.25),
        )
        self.play(Write(verdict))
        self.pause(
            "Cierre el gancho: A y B tardan lo mismo en promedio, pero A es más "
            "predecible. La unidad minutos permite defender esa interpretación."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 6. Forma y robustez: la distribución elige la pareja
    # ------------------------------------------------------------------
    def choice_story(self) -> None:
        header = self.heading("6 · Elegir", "¿Qué historia queremos conservar?", BLUE)
        clean = [4, 5, 5, 6, 6, 7, 8]
        contaminated = [4, 5, 5, 6, 6, 7, 8, 24]
        axis = NumberLine(
            x_range=[0, 25, 5],
            length=11.4,
            include_numbers=True,
            font_size=20,
            include_tip=False,
            color=GRID,
        ).shift(UP * 0.50)

        clean_dots = VGroup()
        levels: dict[int, int] = {}
        for value in clean:
            level = levels.get(value, 0)
            levels[value] = level + 1
            clean_dots.add(Dot(axis.n2p(value) + UP * (0.16 + level * 0.20), radius=0.10, color=BLUE))
        base_label = safe_text(
            "datos compactos",
            size=25,
            color=BLUE,
            weight=BOLD,
        ).next_to(axis, UP, buff=0.65)
        prompt = safe_text(
            "Añadiremos una observación: 24",
            size=29,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.48)
        self.play(FadeIn(header), Create(axis), FadeIn(base_label))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in clean_dots], lag_ratio=0.08), Write(prompt))
        self.pause(
            "Antes de añadir 24, pida predecir cuáles medidas cambiarán mucho: "
            "media, mediana, rango, IQR y desviación estándar."
        )

        outlier = Dot(axis.n2p(24) + UP * 0.16, radius=0.12, color=RED)
        outlier_tag = pill("24", RED, 0.95).next_to(outlier, UP, buff=0.28)
        arrow = Arrow(
            axis.n2p(8) + UP * 1.30,
            axis.n2p(24) + UP * 0.40,
            color=RED,
            buff=0.18,
            max_tip_length_to_length_ratio=0.08,
        )
        self.play(GrowArrow(arrow), GrowFromCenter(outlier), FadeIn(outlier_tag))
        self.play(FadeOut(arrow), Flash(outlier, color=RED, flash_radius=0.38))

        before = ["5.86", "6", "4", "2", "1.35"]
        after = ["8.13", "6", "20", "2.5", "6.53"]
        row_names = ["media", "mediana", "rango", "IQR", "s"]
        row_colors = [YELLOW, GREEN, ORANGE, GREEN, PURPLE]
        table = VGroup()
        table_header = VGroup(
            safe_text("medida", size=21, color=MUTED),
            safe_text("sin 24", size=21, color=BLUE),
            safe_text("con 24", size=21, color=RED),
        )
        table_header.arrange(RIGHT, buff=0.76).move_to([-0.1, -0.66, 0])
        table.add(table_header)
        for index, (name, old, new, color) in enumerate(zip(row_names, before, after, row_colors)):
            y = -1.20 - index * 0.42
            name_mob = safe_text(name, size=21, color=color, weight=BOLD).move_to([-1.65, y, 0])
            old_mob = safe_text(old, size=21, color=INK).move_to([0.10, y, 0])
            new_mob = safe_text(new, size=21, color=INK).move_to([1.67, y, 0])
            table.add(VGroup(name_mob, old_mob, new_mob))
        sensitive = VGroup(
            pill("sensibles", RED, 1.75),
            safe_text("media · rango · s", size=23, color=INK),
        ).arrange(RIGHT, buff=0.22).move_to([4.42, -1.45, 0])
        robust = VGroup(
            pill("robustas", GREEN, 1.75),
            safe_text("mediana · IQR", size=23, color=INK),
        ).arrange(RIGHT, buff=0.22).move_to([4.42, -2.35, 0])
        self.play(FadeOut(prompt), FadeIn(table, shift=UP * 0.15), run_time=0.8)
        self.play(FadeIn(sensitive), FadeIn(robust))
        self.pause(
            "Compare cambios: la mediana queda en 6 y el IQR pasa de 2 a 2.5; "
            "media, rango y s se desplazan mucho. Robustez no significa invariancia."
        )
        self.clear()

        header = self.heading("6 · Elegir", "Primero mira la forma; después resume", BLUE)
        left_panel = RoundedRectangle(
            corner_radius=0.22,
            width=5.75,
            height=4.55,
            color=GRID,
            fill_color=BLUE,
            fill_opacity=0.035,
        ).move_to([-3.25, -0.10, 0])
        right_panel = RoundedRectangle(
            corner_radius=0.22,
            width=5.75,
            height=4.55,
            color=GRID,
            fill_color=RED,
            fill_opacity=0.035,
        ).move_to([3.25, -0.10, 0])
        left_axis = NumberLine(
            x_range=[0, 10, 1], length=4.6, include_numbers=False, include_tip=False, color=GRID
        ).move_to([-3.25, 0.40, 0])
        right_axis = NumberLine(
            x_range=[0, 10, 1], length=4.6, include_numbers=False, include_tip=False, color=GRID
        ).move_to([3.25, 0.40, 0])
        symmetric_values = [2, 3, 4, 5, 6, 7, 8]
        skewed_values = [1.8, 2.4, 2.8, 3.2, 3.6, 4.2, 9.2]
        left_dots = VGroup(*[Dot(left_axis.n2p(v) + UP * 0.13, color=BLUE, radius=0.09) for v in symmetric_values])
        right_dots = VGroup(*[
            Dot(right_axis.n2p(v) + UP * 0.13, color=RED if v > 8 else PURPLE, radius=0.09)
            for v in skewed_values
        ])
        left_title = safe_text("aprox. simétrica", size=27, color=BLUE, weight=BOLD).next_to(left_panel, UP, buff=-0.52)
        right_title = safe_text("sesgo / extremos", size=27, color=RED, weight=BOLD).next_to(right_panel, UP, buff=-0.52)
        cards = VGroup(
            VGroup(
                pill("CENTRO", YELLOW, 1.55),
                MathTex(r"\bar{x}", font_size=46, color=YELLOW),
                pill("DISPERSIÓN", PURPLE, 2.15),
                MathTex(r"s", font_size=46, color=PURPLE),
            ).arrange(RIGHT, buff=0.20),
            VGroup(
                pill("CENTRO", YELLOW, 1.55),
                safe_text("mediana", size=28, color=YELLOW, weight=BOLD),
                pill("DISPERSIÓN", GREEN, 2.15),
                MathTex(r"IQR", font_size=40, color=GREEN),
            ).arrange(RIGHT, buff=0.20),
        )
        cards[0].scale(0.84).move_to([-3.25, -1.25, 0])
        cards[1].scale(0.78).move_to([3.25, -1.25, 0])
        caveat = safe_text(
            "Heurística: valida siempre con la gráfica y el propósito.",
            size=24,
            color=MUTED,
        ).to_edge(DOWN, buff=0.45)

        self.play(FadeIn(header), FadeIn(left_panel), FadeIn(right_panel))
        self.play(FadeIn(left_title), Create(left_axis), LaggedStart(*[GrowFromCenter(d) for d in left_dots], lag_ratio=0.08))
        self.play(FadeIn(right_title), Create(right_axis), LaggedStart(*[GrowFromCenter(d) for d in right_dots], lag_ratio=0.08))
        self.pause(
            "Antes de revelar las tarjetas, pida asignar cada pareja. Evite presentar "
            "la regla como automatismo: el propósito analítico también importa."
        )
        self.play(FadeIn(cards[0], shift=UP * 0.16), FadeIn(cards[1], shift=UP * 0.16))
        self.play(Write(caveat))
        self.pause(
            "Aproximadamente simétrica y sin extremos influyentes: media y s. Con "
            "sesgo o extremos: mediana e IQR suelen conservar mejor la historia central."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 7. Transferencia: el alumnado ejecuta el flujo completo
    # ------------------------------------------------------------------
    def transfer_challenge(self) -> None:
        header = self.heading("Reto de transferencia", "Corta → cerca → decide", YELLOW)
        raw = [6, 24, 5, 8, 4, 6, 7, 5]
        ordered = sorted(raw)
        cards = VGroup(*[number_card(v, color=BLUE, width=0.92) for v in raw])
        cards.arrange(RIGHT, buff=0.24).move_to([0, 0.75, 0])
        task = VGroup(
            safe_text("1  Ordena y calcula Q1, mediana, Q3 e IQR", size=27, color=INK),
            safe_text("2  Construye las cercas y detecta atípicos", size=27, color=INK),
            safe_text("3  Elige una pareja de medidas e interpreta", size=27, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        task.move_to([0, -1.25, 0])
        pause_badge = pill("PAUSA · 45 s", YELLOW, 2.65).to_edge(DOWN, buff=0.40)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in cards], lag_ratio=0.08))
        self.play(LaggedStart(*[FadeIn(line, shift=RIGHT * 0.18) for line in task], lag_ratio=0.12), FadeIn(pause_badge))
        self.pause(
            "Pausa real de 45 segundos. El conjunto representa una variable numérica "
            "genérica. Pida escribir cálculos y una frase con unidades ficticias."
        )

        positions = [card.get_center() for card in cards]
        sorted_position_by_index: dict[int, np.ndarray] = {}
        used: dict[int, int] = {}
        ordered_indices_by_value: dict[int, list[int]] = {}
        for idx, value in enumerate(ordered):
            ordered_indices_by_value.setdefault(value, []).append(idx)
        for original_index, value in enumerate(raw):
            occurrence = used.get(value, 0)
            used[value] = occurrence + 1
            target_index = ordered_indices_by_value[value][occurrence]
            sorted_position_by_index[original_index] = positions[target_index]
        self.play(
            *[
                card.animate.move_to(sorted_position_by_index[index])
                for index, card in enumerate(cards)
            ],
            FadeOut(task),
            FadeOut(pause_badge),
            run_time=1.35,
        )
        ordered_cards = [card for _, card in sorted(zip(raw, cards), key=lambda pair: pair[0])]
        convention = safe_text(
            "cuartiles por mediana de mitades",
            size=22,
            color=MUTED,
        ).next_to(cards, UP, buff=0.30)
        split = DashedLine([0, 0.20, 0], [0, 1.35, 0], color=MUTED, dash_length=0.09)
        lower = SurroundingRectangle(VGroup(*ordered_cards[:4]), color=BLUE, buff=0.10, corner_radius=0.08)
        upper = SurroundingRectangle(VGroup(*ordered_cards[4:]), color=PURPLE, buff=0.10, corner_radius=0.08)
        self.play(FadeIn(convention), Create(split), Create(lower), Create(upper))

        q1_marks = VGroup(*[SurroundingRectangle(ordered_cards[i], color=BLUE, buff=0.04) for i in (1, 2)])
        q2_marks = VGroup(*[SurroundingRectangle(ordered_cards[i], color=YELLOW, buff=0.04) for i in (3, 4)])
        q3_marks = VGroup(*[SurroundingRectangle(ordered_cards[i], color=PURPLE, buff=0.04) for i in (5, 6)])
        quartiles = VGroup(
            MathTex(r"Q_1=5", font_size=41, color=BLUE),
            MathTex(r"Q_2=6", font_size=41, color=YELLOW),
            MathTex(r"Q_3=\frac{7+8}{2}=7.5", font_size=41, color=PURPLE),
        ).arrange(RIGHT, buff=0.80).move_to([0, -0.70, 0])
        iqr = MathTex(r"IQR=7.5-5=2.5", font_size=48, color=GREEN).move_to([0, -1.72, 0])
        self.play(Create(q1_marks), Create(q2_marks), Create(q3_marks), Write(quartiles))
        self.play(Write(iqr))
        self.pause(
            "Solución del corte: Q1=5, mediana=6, Q3=7.5 e IQR=2.5. "
            "Vuelva a verbalizar que el IQR mide la mitad central."
        )

        fences = VGroup(
            MathTex(r"LI=5-1.5(2.5)=1.25", font_size=39, color=BLUE),
            MathTex(r"LS=7.5+1.5(2.5)=11.25", font_size=39, color=PURPLE),
        ).arrange(RIGHT, buff=0.72).move_to([0, -2.68, 0])
        self.play(Write(fences))
        outlier_card = ordered_cards[-1]
        outlier_ring = SurroundingRectangle(
            outlier_card,
            color=RED,
            buff=0.09,
            corner_radius=0.12,
            stroke_width=5,
        )
        outlier_test = MathTex(r"24>11.25", font_size=40, color=RED).next_to(outlier_card, UP, buff=0.70)
        self.play(Create(outlier_ring), Write(outlier_test), Flash(outlier_card, color=RED, flash_radius=0.55))
        self.pause(
            "24 supera la cerca superior 11.25. Es un candidato atípico. Pregunte qué "
            "harían después: investigar, no borrar automáticamente."
        )

        report = VGroup(
            pill("REPORTE", GREEN, 1.70),
            safe_text(
                "mediana = 6 · IQR = 2.5 · 24 es atípico alto",
                size=30,
                color=INK,
                weight=BOLD,
                max_width=10.0,
            ),
        ).arrange(RIGHT, buff=0.35)
        report.move_to([0, -1.60, 0])
        contrast = MathTex(
            r"\bar{x}=8.13\qquad s=6.53",
            font_size=36,
            color=MUTED,
        ).next_to(report, DOWN, buff=0.32)
        reason = safe_text(
            "La forma con extremo favorece mediana + IQR.",
            size=27,
            color=GREEN,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.35)
        self.play(
            FadeOut(split),
            FadeOut(lower),
            FadeOut(upper),
            FadeOut(q1_marks),
            FadeOut(q2_marks),
            FadeOut(q3_marks),
            FadeOut(quartiles),
            FadeOut(iqr),
            FadeOut(fences),
            cards.animate.shift(UP * 0.50),
            outlier_ring.animate.shift(UP * 0.50),
            outlier_test.animate.shift(UP * 0.50),
            convention.animate.shift(UP * 0.50),
            run_time=0.75,
        )
        self.play(FadeIn(report, shift=UP * 0.15), FadeIn(contrast), Write(reason))
        self.pause(
            "Modele el reporte final: número, unidad y evidencia de forma. La media "
            "8.13 y s=6.53 quedan como contraste de sensibilidad, no como errores."
        )
        self.clear()

    # ------------------------------------------------------------------
    # 8. Cierre: un procedimiento transferible
    # ------------------------------------------------------------------
    def closing(self) -> None:
        title = safe_text(
            "De una nube a una decisión",
            size=50,
            color=INK,
            weight=BOLD,
        ).to_edge(UP, buff=0.48)
        steps = [
            ("1", "PREGUNTA", "¿qué quiero describir?", BLUE),
            ("2", "DIBUJA", "ordena + grafica", GREEN),
            ("3", "MIRA", "forma y extremos", YELLOW),
            ("4", "ELIGE", "elige una pareja", PURPLE),
            ("5", "INTERPRETA", "valor + unidad + contexto", ORANGE),
        ]
        nodes = VGroup()
        arrows = VGroup()
        xs = np.linspace(-5.35, 5.35, len(steps))
        for x, (number, verb, detail, color) in zip(xs, steps):
            circle = Circle(radius=0.38, color=color, fill_color=color, fill_opacity=0.16, stroke_width=3)
            numeral = safe_text(number, size=25, color=color, weight=BOLD).move_to(circle)
            verb_mob = safe_text(verb, size=23, color=color, weight=BOLD)
            detail_mob = safe_text(detail, size=18, color=MUTED, max_width=2.05)
            node = VGroup(VGroup(circle, numeral), verb_mob, detail_mob).arrange(DOWN, buff=0.16)
            node.move_to([x, -0.25, 0])
            nodes.add(node)
        for left, right in zip(nodes[:-1], nodes[1:]):
            arrows.add(
                Arrow(
                    left.get_right() + RIGHT * 0.05,
                    right.get_left() + LEFT * 0.05,
                    color=GRID,
                    buff=0.10,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.18,
                )
            )
        principle = safe_text(
            "No existe una medida “mejor” fuera de contexto.",
            size=35,
            color=INK,
            weight=BOLD,
        )
        principle.move_to([0, -2.15, 0])
        subprinciple = safe_text(
            "Existe la medida que responde a la pregunta y respeta la forma de los datos.",
            size=26,
            color=GREEN,
            max_width=12.2,
        ).next_to(principle, DOWN, buff=0.22)

        self.play(Write(title))
        for index, node in enumerate(nodes):
            if index:
                self.play(GrowArrow(arrows[index - 1]), FadeIn(node, shift=UP * 0.12), run_time=0.42)
            else:
                self.play(FadeIn(node, shift=UP * 0.12), run_time=0.42)
        self.play(Write(principle), FadeIn(subprinciple, shift=UP * 0.12))
        self.pause(
            "Cierre con el procedimiento transferible. Solicite un exit ticket: una "
            "decisión de medida acompañada por una justificación de una sola frase."
        )
