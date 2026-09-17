"""Repaso aplicado de medidas de heterogeneidad con Manim Slides.

La presentación usa los dos materiales de referencia del curso como fuentes de
contenido. Cada ``next_slide`` es una pausa docente dentro de una sola escena
continua; la construcción visual precede siempre a la fórmula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manim import *
from manim_slides import Slide


BG = "#07111F"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"
ORANGE = "#FFB86B"
PURPLE = "#B28DFF"
YELLOW = "#F7C948"
GREEN = "#6FCF97"
RED = "#FF6B6B"
CAT_COLORS = [BLUE, PURPLE, GREEN, ORANGE, RED]
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
    """Create portable, frame-safe Pango text."""

    mob = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def pill(content: str, color: str, width: float = 2.3, height: float = 0.58) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.16,
        width=width,
        height=height,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.12,
    )
    label = safe_text(content, size=23, color=color, weight=BOLD, max_width=width - 0.25)
    label.move_to(box)
    return VGroup(box, label)


def metric_card(label: str, value: str, color: str, width: float = 2.7) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.16,
        width=width,
        height=1.05,
        stroke_color=color,
        stroke_width=2.2,
        fill_color=color,
        fill_opacity=0.10,
    )
    label_mob = safe_text(label, size=18, color=MUTED, weight=BOLD)
    value_mob = safe_text(value, size=28, color=color, weight=BOLD, max_width=width - 0.25)
    content = VGroup(label_mob, value_mob).arrange(DOWN, buff=0.10).move_to(box)
    return VGroup(box, content)


def distribution_chart(
    probs: list[float],
    labels: list[str],
    *,
    colors: list[str] | None = None,
    width: float = 7.2,
    height: float = 3.2,
    show_values: bool = True,
) -> VGroup:
    """Simple categorical bar chart whose objects can transform continuously."""

    colors = colors or CAT_COLORS[: len(probs)]
    baseline = Line(LEFT * width / 2, RIGHT * width / 2, color=GRID, stroke_width=3)
    bars = VGroup()
    category_labels = VGroup()
    values = VGroup()
    slot = width / len(probs)
    bar_width = min(0.92, slot * 0.56)
    for idx, (prob, label, color) in enumerate(zip(probs, labels, colors)):
        x = -width / 2 + slot * (idx + 0.5)
        bar_height = max(0.025, height * prob)
        bar = Rectangle(
            width=bar_width,
            height=bar_height,
            stroke_color=color,
            stroke_width=2,
            fill_color=color,
            fill_opacity=0.72,
        )
        bar.move_to([x, bar_height / 2, 0])
        bars.add(bar)
        category_labels.add(
            safe_text(label, size=19, color=color, weight=BOLD).move_to([x, -0.30, 0])
        )
        values.add(
            safe_text(f"{prob:.2f}", size=18, color=INK).next_to(bar, UP, buff=0.08)
        )
    group = VGroup(baseline, bars, category_labels)
    if show_values:
        group.add(values)
    return group


def token_row(labels: list[str], colors: dict[str, str], y: float = 0.0) -> VGroup:
    tokens = VGroup()
    xs = np.linspace(-4.0, 4.0, len(labels))
    for x, label in zip(xs, labels):
        token = VGroup(
            Circle(radius=0.28, color=colors[label], fill_color=colors[label], fill_opacity=0.22),
            safe_text(label, size=22, color=INK, weight=BOLD),
        )
        token[1].move_to(token[0])
        token.move_to([x, y, 0])
        tokens.add(token)
    return tokens


def probability_grid(
    probs: list[float],
    labels: list[str],
    colors: list[str],
    *,
    side: float = 3.8,
) -> tuple[VGroup, VGroup]:
    """Partition a square; diagonal cells have areas p_k²."""

    lower_left = np.array([-side / 2, -side / 2, 0.0])
    background = Square(
        side_length=side,
        stroke_color=INK,
        stroke_width=2.5,
        fill_color=ORANGE,
        fill_opacity=0.12,
    )
    lines = VGroup()
    diagonal = VGroup()
    labels_mob = VGroup()
    cumulative = 0.0
    for prob, label, color in zip(probs, labels, colors):
        center = cumulative + prob / 2
        cell = Rectangle(
            width=side * prob,
            height=side * prob,
            stroke_color=color,
            stroke_width=2,
            fill_color=color,
            fill_opacity=0.72,
        )
        cell.move_to(lower_left + [side * center, side * center, 0])
        diagonal.add(cell)
        label_mob = safe_text(label + label, size=17, color=INK, weight=BOLD)
        if label_mob.width < cell.width * 0.8 and label_mob.height < cell.height * 0.8:
            label_mob.move_to(cell)
            labels_mob.add(label_mob)
        cumulative += prob
        if cumulative < 1 - 1e-9:
            x = lower_left[0] + side * cumulative
            y = lower_left[1] + side * cumulative
            lines.add(Line([x, lower_left[1], 0], [x, lower_left[1] + side, 0], color=GRID))
            lines.add(Line([lower_left[0], y, 0], [lower_left[0] + side, y, 0], color=GRID))
    grid = VGroup(background, lines, diagonal, labels_mob)
    return grid, diagonal


class MedidasHeterogeneidad(Slide):
    """Una narrativa continua: reparto categórico, separación y transferencia."""

    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.branding: Group | None = None
        self.opening_and_distribution()
        self.gini_simpson_story()
        self.shannon_story()
        self.comparison_and_transition()
        self.group_separation_story()
        self.transfer_challenge()
        self.closing()

    # ------------------------------------------------------------------
    # Composición compartida
    # ------------------------------------------------------------------
    def pause(self, notes: str) -> None:
        self.next_slide(notes=notes)

    def clear_content(self, run_time: float = 0.42) -> None:
        removable = [mob for mob in self.mobjects if mob is not self.branding]
        if removable:
            self.play(*[FadeOut(mob) for mob in removable], run_time=run_time)

    def logo_lockup(self, width: float, height: float) -> Group:
        card = RoundedRectangle(
            corner_radius=0.10,
            width=width,
            height=height,
            stroke_color=GRID,
            stroke_width=1.2,
            fill_color=BG,
            fill_opacity=0.30,
        )
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width - 0.25)
        if logo.height > height - 0.16:
            logo.scale_to_fit_height(height - 0.16)
        logo.move_to(card)
        return Group(card, logo)

    def add_branding(self) -> None:
        bar = Rectangle(
            width=config.frame_width,
            height=0.40,
            stroke_width=0,
            fill_color="#050D18",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0)
        rule = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2, color=GRID)
        rule.move_to([0, -3.58, 0])
        signature = safe_text(
            "Diego Villalba  ·  Almacenes y Minería de Datos",
            size=13,
            color=MUTED,
        ).move_to([-4.72, -3.80, 0])
        logo = self.logo_lockup(1.42, 0.34).move_to([6.12, -3.80, 0])
        self.branding = Group(bar, rule, signature, logo).set_z_index(100)
        self.add(self.branding)

    def heading(self, kicker: str, title: str, color: str) -> VGroup:
        kicker_mob = safe_text(kicker.upper(), size=19, color=color, weight=BOLD)
        title_mob = safe_text(title, size=42, color=INK, weight=BOLD, max_width=12.5)
        text = VGroup(kicker_mob, title_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        text.to_corner(UL, buff=0.43)
        rule = Line(LEFT * 0.55, RIGHT * 0.55, color=color, stroke_width=4)
        rule.next_to(text, DOWN, aligned_edge=LEFT, buff=0.10)
        return VGroup(text, rule)

    # ------------------------------------------------------------------
    # Estados 1–5 · dos formas de heterogeneidad y reparto
    # ------------------------------------------------------------------
    def opening_and_distribution(self) -> None:
        eyebrow = safe_text(
            "ANÁLISIS EXPLORATORIO · REPASO APLICADO",
            size=23,
            color=BLUE,
            weight=BOLD,
        )
        title = safe_text(
            "Medidas de heterogeneidad",
            size=60,
            color=INK,
            weight=BOLD,
            max_width=12.5,
        )
        subtitle = safe_text(
            "del reparto de la masa a la separación entre grupos",
            size=30,
            color=MUTED,
            max_width=11.8,
        )
        visual = MathTex(
            r"p_1,\ldots,p_K\quad\longrightarrow\quad D,\ H,\ \eta^2",
            font_size=48,
            color=INK,
        )
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text(
            "Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED
        )
        identity = VGroup(course, author).arrange(DOWN, buff=0.08)
        logo = self.logo_lockup(4.0, 1.15)
        cover = Group(eyebrow, title, subtitle, visual, identity, logo).arrange(DOWN, buff=0.19)

        self.play(FadeIn(eyebrow, shift=UP * 0.15), Write(title), run_time=0.95)
        self.play(FadeIn(subtitle), Write(visual), FadeIn(identity), FadeIn(logo), run_time=0.85)
        self.pause(
            "Portada. Recuerde que el grupo ya estudió el tema: hoy justificará "
            "cada medida mediante objetos y decisiones, no mediante memorización."
        )

        self.clear_content()
        self.add_branding()
        header = self.heading("Dos preguntas", "¿Reparto o separación?", YELLOW)
        left_box = RoundedRectangle(
            width=5.6, height=3.7, corner_radius=0.18, color=BLUE, fill_color=BLUE, fill_opacity=0.06
        ).move_to([-3.25, -0.15, 0])
        right_box = RoundedRectangle(
            width=5.6, height=3.7, corner_radius=0.18, color=GREEN, fill_color=GREEN, fill_opacity=0.06
        ).move_to([3.25, -0.15, 0])
        left_title = safe_text("Variable categórica", size=26, color=BLUE, weight=BOLD)
        left_title.next_to(left_box, UP, buff=-0.48)
        mini_bars = distribution_chart([0.55, 0.30, 0.15], ["A", "B", "C"], width=3.7, height=2.0)
        mini_bars.move_to(left_box).shift(DOWN * 0.20)
        left_q = safe_text("¿cómo se reparte?", size=27, color=INK, weight=BOLD)
        left_q.next_to(left_box, DOWN, buff=-0.55)
        right_title = safe_text("Respuesta numérica", size=26, color=GREEN, weight=BOLD)
        right_title.next_to(right_box, UP, buff=-0.48)
        lanes = VGroup()
        for idx, (ys, color) in enumerate(zip([0.75, -0.05, -0.85], CAT_COLORS[:3])):
            line = NumberLine(x_range=[0, 10, 2], length=3.9, include_numbers=False, color=GRID)
            line.move_to([3.25, ys, 0])
            dots = VGroup(*[Dot(line.n2p(v), radius=0.07, color=color) for v in ([2, 3], [5, 7], [6, 8])[idx]])
            lanes.add(VGroup(line, dots))
        right_q = safe_text("¿cuánto se separan?", size=27, color=INK, weight=BOLD)
        right_q.next_to(right_box, DOWN, buff=-0.55)
        self.play(FadeIn(header), Create(left_box), Create(right_box))
        self.play(FadeIn(left_title), FadeIn(mini_bars), FadeIn(left_q))
        self.play(FadeIn(right_title), LaggedStart(*[FadeIn(lane) for lane in lanes], lag_ratio=0.15), FadeIn(right_q))
        self.pause(
            "Pida un ejemplo de cada pregunta. No nombre aún Gini–Simpson, "
            "Shannon ni eta cuadrada; haga que distingan composición de respuesta numérica."
        )

        self.clear_content()
        header = self.heading("Predicción", "¿Qué tan fácil es adivinar la categoría?", YELLOW)
        color_map = {"A": BLUE, "B": PURPLE, "C": GREEN}
        labels = ["A", "A", "A", "B", "C", "C"]
        tokens = token_row(labels, color_map, y=-0.05)
        cover_token = RoundedRectangle(
            width=1.05, height=1.05, corner_radius=0.15, color=YELLOW, fill_color=YELLOW, fill_opacity=0.12
        ).move_to([0, -1.55, 0])
        question = safe_text("?", size=42, color=YELLOW, weight=BOLD).move_to(cover_token)
        prompt = safe_text("baja · media · alta", size=29, color=MUTED).next_to(cover_token, DOWN, buff=0.30)
        self.play(FadeIn(header), LaggedStart(*[GrowFromCenter(t) for t in tokens], lag_ratio=0.10))
        self.play(Create(cover_token), Write(question), FadeIn(prompt))
        self.pause(
            "Solicite una predicción cualitativa: diversidad baja, media o alta. "
            "Pida justificarla sin calcular y aclare que adivinar es sólo una intuición."
        )

        bins = VGroup()
        target_positions: list[np.ndarray] = []
        counts = [3, 1, 2]
        for idx, (label, count, color) in enumerate(zip(["A", "B", "C"], counts, CAT_COLORS)):
            x = [-3.2, 0, 3.2][idx]
            box = RoundedRectangle(
                width=2.2, height=2.7, corner_radius=0.15, color=color, fill_color=color, fill_opacity=0.08
            ).move_to([x, -0.25, 0])
            label_mob = safe_text(label, size=28, color=color, weight=BOLD).next_to(box, DOWN, buff=0.16)
            count_mob = safe_text(f"f = {count}", size=24, color=INK).next_to(box, UP, buff=0.12)
            bins.add(VGroup(box, label_mob, count_mob))
            for level in range(count):
                target_positions.append(np.array([x, -1.05 + level * 0.58, 0]))
        ordered_targets = target_positions[:3] + target_positions[3:4] + target_positions[4:]
        self.play(FadeOut(cover_token), FadeOut(question), FadeOut(prompt), FadeIn(bins))
        self.play(*[token.animate.move_to(pos) for token, pos in zip(tokens, ordered_targets)], run_time=1.2)
        self.pause(
            "Lea los conteos 3, 1 y 2. Pida convertirlos mentalmente a proporciones "
            "y comprobar que los conteos suman seis."
        )

        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"])
        chart.shift(DOWN * 0.35)
        proportions = MathTex(
            r"p=\left(\frac12,\frac16,\frac13\right)", font_size=43, color=INK
        ).next_to(header, DOWN, buff=0.30)
        self.play(FadeOut(tokens), FadeOut(bins), FadeIn(chart), Write(proportions))
        uniform_chart = distribution_chart([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"])
        uniform_chart.shift(DOWN * 0.35)
        uniform_formula = MathTex(
            r"(3,1,2)\ \longrightarrow\ (2,2,2)", font_size=38, color=GREEN
        ).move_to(proportions)
        prediction = safe_text(
            "mismo n · mismo K · ¿más o menos diversidad?",
            size=27,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(Transform(chart, uniform_chart), TransformMatchingTex(proportions, uniform_formula), run_time=1.25)
        self.play(Write(prediction))
        self.pause(
            "Pregunte qué cambió si n=6 y K=3 permanecieron fijos. Espere que "
            "identifiquen el mayor equilibrio antes de introducir cualquier índice."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estados 6–12 · Gini–Simpson como área de pares diferentes
    # ------------------------------------------------------------------
    def gini_simpson_story(self) -> None:
        header = self.heading("Gini–Simpson", "Dos extracciones: ¿coinciden o difieren?", ORANGE)
        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], width=5.2, height=2.6)
        chart.move_to([-3.3, -0.35, 0])
        urn_one = VGroup(
            Circle(radius=0.62, color=BLUE, fill_color=BLUE, fill_opacity=0.09),
            safe_text("1", size=29, color=INK, weight=BOLD),
        ).move_to([2.0, 0.25, 0])
        urn_two = VGroup(
            Circle(radius=0.62, color=PURPLE, fill_color=PURPLE, fill_opacity=0.09),
            safe_text("2", size=29, color=INK, weight=BOLD),
        ).move_to([4.05, 0.25, 0])
        urn_one[1].move_to(urn_one[0])
        urn_two[1].move_to(urn_two[0])
        independence = safe_text(
            "independientes · con reemplazo",
            size=21,
            color=MUTED,
        ).move_to([3.02, -0.82, 0])
        choice = VGroup(
            pill("MISMA", RED, 1.9),
            safe_text("o", size=24, color=MUTED),
            pill("DISTINTA", ORANGE, 2.15),
        ).arrange(RIGHT, buff=0.20).move_to([3.0, -1.72, 0])
        self.play(FadeIn(header), FadeIn(chart))
        self.play(GrowFromCenter(urn_one), GrowFromCenter(urn_two), FadeIn(independence), FadeIn(choice))
        self.pause(
            "Pregunte qué evento conviene calcular primero: misma categoría o distinta. "
            "Declare desde ahora dos extracciones independientes, equivalentes a reemplazo."
        )

        self.play(FadeOut(chart), FadeOut(urn_one), FadeOut(urn_two), FadeOut(independence), FadeOut(choice))
        grid, diagonal = probability_grid([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], CAT_COLORS[:3])
        grid.move_to([-2.45, -0.35, 0])
        axes_labels = VGroup(
            safe_text("extracción 1", size=20, color=BLUE, weight=BOLD).next_to(grid, DOWN, buff=0.18),
            safe_text("extracción 2", size=20, color=PURPLE, weight=BOLD).rotate(PI / 2).next_to(grid, LEFT, buff=0.18),
        )
        same = safe_text("diagonal = misma categoría", size=29, color=RED, weight=BOLD)
        same.move_to([3.1, 0.65, 0])
        different = safe_text("resto = categorías distintas", size=27, color=ORANGE, weight=BOLD)
        different.move_to([3.1, -0.25, 0])
        self.play(FadeIn(grid[0]), FadeIn(grid[1]), FadeIn(grid[3]), FadeIn(axes_labels))
        self.play(LaggedStart(*[GrowFromCenter(cell) for cell in diagonal], lag_ratio=0.16), Write(same))
        self.play(FadeIn(different))
        self.pause(
            "Lea el cuadrado como todos los pares posibles. Pida estimar si el área "
            "diagonal —los pares iguales— supera la mitad antes de poner cifras."
        )

        area_labels = VGroup()
        for cell, tex in zip(diagonal, [r"\frac14", r"\frac1{36}", r"\frac19"]):
            label = MathTex(tex, font_size=28, color=INK).move_to(cell)
            if label.width > cell.width * 0.78:
                label.scale_to_fit_width(cell.width * 0.78)
            area_labels.add(label)
        sum_formula = MathTex(
            r"\sum_{k=1}^{3}p_k^2=\frac14+\frac1{36}+\frac19=\frac7{18}=0.3889",
            font_size=36,
            color=INK,
        ).move_to([2.85, -1.45, 0])
        if sum_formula.width > 6.0:
            sum_formula.scale_to_fit_width(6.0)
        self.play(FadeOut(same), FadeOut(different), FadeIn(area_labels), Write(sum_formula))
        self.pause(
            "Nombre los sucesos AA, BB y CC. Como son excluyentes, sus áreas se "
            "suman: 1/4 + 1/36 + 1/9 = 7/18."
        )

        d_formula = MathTex(
            r"D=1-\sum_{k=1}^{K}p_k^2=1-\frac7{18}=\frac{11}{18}=0.6111",
            font_size=42,
            color=ORANGE,
        ).move_to([2.75, 0.55, 0])
        if d_formula.width > 6.2:
            d_formula.scale_to_fit_width(6.2)
        interpretation = safe_text(
            "61.11% de probabilidad de categorías distintas",
            size=26,
            color=GREEN,
            weight=BOLD,
            max_width=6.1,
        ).move_to([2.75, -0.35, 0])
        orange_frame = SurroundingRectangle(grid[0], color=ORANGE, buff=0.03, stroke_width=4)
        self.play(Create(orange_frame), Write(d_formula), FadeIn(interpretation))
        self.pause(
            "Modele la frase completa: D=0.6111 es la probabilidad de que dos "
            "extracciones independientes pertenezcan a categorías distintas. "
            "No diga simplemente ‘61.11% de diversidad’."
        )

        self.clear_content()
        header = self.heading("Extremo 1", "Toda la masa en una categoría", RED)
        chart = distribution_chart([1.0, 0.0, 0.0], ["A", "B", "C"], width=5.2, height=2.8)
        chart.move_to([-3.2, -0.25, 0])
        full_square = Square(
            side_length=3.0,
            color=RED,
            fill_color=RED,
            fill_opacity=0.32,
        ).move_to([3.1, -0.35, 0])
        full_label = MathTex(r"\sum p_k^2=1", font_size=38, color=INK).move_to(full_square)
        zero = MathTex(r"D=1-1=0", font_size=45, color=RED).next_to(full_square, DOWN, buff=0.28)
        self.play(FadeIn(header), FadeIn(chart), GrowFromCenter(full_square), Write(full_label))
        self.play(Write(zero))
        self.pause(
            "Pregunte por qué tres nombres de categoría no garantizan diversidad "
            "observada. Mantenga K=3: B y C existen, pero tienen frecuencia cero."
        )

        uniform_chart = distribution_chart([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"], width=5.2, height=2.8)
        uniform_chart.move_to([-3.2, -0.25, 0])
        uniform_grid, _ = probability_grid([1 / 3, 1 / 3, 1 / 3], ["A", "B", "C"], CAT_COLORS[:3], side=3.0)
        uniform_grid.move_to([3.1, -0.35, 0])
        uniform_header = self.heading("Extremo 2", "Misma masa en cada categoría", ORANGE)
        max_formula = MathTex(
            r"D_{\max}=1-\frac1K=1-\frac13=\frac23",
            font_size=43,
            color=ORANGE,
        ).next_to(uniform_grid, DOWN, buff=0.25)
        self.play(
            Transform(header, uniform_header),
            Transform(chart, uniform_chart),
            ReplacementTransform(full_square, uniform_grid),
            FadeOut(full_label),
            FadeOut(zero),
        )
        self.play(Write(max_formula))
        self.pause(
            "Aclare que el máximo ocurre bajo uniformidad y depende de K. Para "
            "K finito D no llega a 1; con K=3, el máximo es 2/3."
        )

        self.clear_content()
        header = self.heading("Escala interpretable", "¿Cuántas categorías igualmente frecuentes?", GREEN)
        diagonal_card = metric_card("PROBABILIDAD DE COINCIDIR", "Σp² = 7/18", ORANGE, 3.5)
        diagonal_card.move_to([-3.5, -0.10, 0])
        arrow = Arrow([-1.35, -0.10, 0], [0.65, -0.10, 0], color=MUTED, stroke_width=4)
        inverse = metric_card("NÚMERO EFECTIVO · SIMPSON", "1 / Σp² = 18/7 = 2.57", GREEN, 4.4)
        inverse.move_to([3.25, -0.10, 0])
        warning = safe_text(
            "No es 1/D · tampoco es el Gini de desigualdad",
            size=23,
            color=RED,
            weight=BOLD,
            max_width=9.2,
        ).move_to([0, -1.55, 0])
        self.play(FadeIn(header), FadeIn(diagonal_card), GrowArrow(arrow), FadeIn(inverse))
        self.play(FadeIn(warning))
        self.pause(
            "Traduzca el índice: esta composición equivale a 2.57 categorías "
            "igualmente frecuentes según Simpson. Prevenga el error 1/D: se "
            "invierte la probabilidad de coincidencia, suma de p al cuadrado."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estados 13–19 · Shannon: sorpresa ponderada, normalización y efectivos
    # ------------------------------------------------------------------
    def shannon_story(self) -> None:
        header = self.heading("Shannon", "¿Qué revelación produciría mayor sorpresa?", PURPLE)
        chart = distribution_chart([1 / 2, 1 / 6, 1 / 3], ["A", "B", "C"], width=6.6, height=2.7)
        chart.move_to([-2.8, -0.35, 0])
        hidden = VGroup()
        for idx, color in enumerate(CAT_COLORS[:3]):
            card = RoundedRectangle(
                width=1.25,
                height=1.25,
                corner_radius=0.16,
                color=color,
                fill_color=color,
                fill_opacity=0.10,
            )
            qmark = safe_text("?", size=38, color=color, weight=BOLD).move_to(card)
            hidden.add(VGroup(card, qmark))
        hidden.arrange(RIGHT, buff=0.35).move_to([3.65, -0.25, 0])
        intuition = safe_text(
            "intuición: menos probable → más sorpresa",
            size=25,
            color=MUTED,
        ).move_to([3.65, -1.35, 0])
        self.play(FadeIn(header), FadeIn(chart), LaggedStart(*[FadeIn(card) for card in hidden], lag_ratio=0.13))
        self.play(FadeIn(intuition))
        self.pause(
            "Pregunte cuál revelación sorprendería más: A, B o C. Espere B porque "
            "es la categoría menos probable. Presente la adivinanza sólo como intuición."
        )

        self.clear_content()
        header = self.heading("Sorpresa", "Lo raro aporta más información por evento", PURPLE)
        probs = [1 / 2, 1 / 6, 1 / 3]
        surprises = [0.6931, 1.7918, 1.0986]
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2.1, 0.5],
            x_length=8.1,
            y_length=4.1,
            axis_config={"color": GRID, "include_tip": False},
        ).move_to([-0.7, -0.20, 0])
        bars = VGroup()
        labels = VGroup()
        values = VGroup()
        for idx, (label, value, color) in enumerate(zip(["A", "B", "C"], surprises, CAT_COLORS[:3]), start=1):
            bar = Rectangle(
                width=1.05,
                height=axes.y_axis.unit_size * value,
                color=color,
                fill_color=color,
                fill_opacity=0.70,
            )
            bar.move_to([axes.c2p(idx, 0)[0], axes.c2p(idx, value / 2)[1], 0])
            bars.add(bar)
            labels.add(safe_text(label, size=23, color=color, weight=BOLD).next_to(bar, DOWN, buff=0.15))
            values.add(MathTex(rf"-\ln p={value:.3f}", font_size=28, color=INK).next_to(bar, UP, buff=0.10))
        side_formula = MathTex(r"-\ln(p_k)", font_size=43, color=PURPLE).move_to([4.55, 0.50, 0])
        rare = pill("B es rara", PURPLE, 2.5).move_to([4.55, -0.60, 0])
        self.play(FadeIn(header), Create(axes), LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.15))
        self.play(FadeIn(labels), FadeIn(values), Write(side_formula), FadeIn(rare))
        self.pause(
            "Compare el factor -ln(p) que ya vive dentro de la fórmula de Shannon: "
            "-ln(1/6) es mayor que -ln(1/2). Pida explicar por qué B sorprende más."
        )

        contribution_values = [0.3466, 0.2986, 0.3662]
        contribution_bars = VGroup()
        contribution_labels = VGroup()
        for idx, (value, color) in enumerate(zip(contribution_values, CAT_COLORS[:3]), start=1):
            bar = Rectangle(
                width=1.05,
                height=axes.y_axis.unit_size * value,
                color=color,
                fill_color=color,
                fill_opacity=0.75,
            )
            bar.move_to([axes.c2p(idx, 0)[0], axes.c2p(idx, value / 2)[1], 0])
            contribution_bars.add(bar)
            contribution_labels.add(
                MathTex(rf"-p\ln p={value:.4f}", font_size=26, color=INK).next_to(bar, UP, buff=0.10)
            )
        balance = safe_text(
            "sorpresa × frecuencia",
            size=27,
            color=YELLOW,
            weight=BOLD,
        ).move_to([4.55, -0.55, 0])
        self.play(Transform(bars, contribution_bars), Transform(values, contribution_labels), FadeOut(rare), Transform(side_formula, MathTex(r"p_k[-\ln(p_k)]", font_size=41, color=PURPLE).move_to(side_formula)), run_time=1.2)
        self.play(FadeIn(balance))
        self.pause(
            "Antes de elegir el término mayor, haga comparar rareza y frecuencia. "
            "B sorprende mucho, pero ocurre poco; C produce la mayor contribución ponderada."
        )

        self.clear_content()
        header = self.heading("Paisaje continuo", "El punto dulce de la sorpresa: 1/e", PURPLE)
        curve_axes = Axes(
            x_range=[0, 1.05, 0.2],
            y_range=[0, 0.45, 0.1],
            x_length=7.4,
            y_length=3.7,
            axis_config={"color": GRID, "include_tip": False},
        ).move_to([-1.2, -0.30, 0])
        x_lbl = safe_text("probabilidad p", size=18, color=MUTED).next_to(curve_axes.x_axis, DOWN, buff=0.15)
        y_lbl = MathTex(r"-p\ln p", font_size=24, color=PURPLE).next_to(curve_axes.y_axis, UP, buff=0.12)

        curve = curve_axes.plot(
            lambda p: -p * np.log(p) if p > 1e-5 else 0.0,
            x_range=[0.001, 1.0, 0.005],
            color=PURPLE,
            stroke_width=4,
        )

        inv_e = 1.0 / np.e
        peak_dot = Dot(curve_axes.c2p(inv_e, inv_e), radius=0.09, color=YELLOW)
        peak_v_line = DashedLine(curve_axes.c2p(inv_e, 0), curve_axes.c2p(inv_e, inv_e), color=YELLOW, dash_length=0.06)
        peak_label = MathTex(r"p=\frac{1}{e}\approx 0.368", font_size=25, color=YELLOW).next_to(peak_dot, UP, buff=0.12)

        dots_on_curve = VGroup()
        dot_labels = VGroup()
        for label, prob, col in zip(["A", "B", "C"], [0.5, 1 / 6, 1 / 3], CAT_COLORS[:3]):
            val = -prob * np.log(prob)
            pt = curve_axes.c2p(prob, val)
            d = Dot(pt, radius=0.10, color=col)
            d_line = DashedLine(curve_axes.c2p(prob, 0), pt, color=col, stroke_opacity=0.6, dash_length=0.05)
            lbl = safe_text(label, size=20, color=col, weight=BOLD).next_to(d, UP if label != "C" else RIGHT, buff=0.10)
            dots_on_curve.add(VGroup(d_line, d))
            dot_labels.add(lbl)

        card_peak = metric_card("PICO DE INFORMACIÓN", "p = 1/e ≈ 0.368", YELLOW, 3.5).move_to([4.55, 0.70, 0])
        card_zero = metric_card("LÍMITE NATURAL", "0 ln 0 := 0", BLUE, 3.5).move_to([4.55, -0.65, 0])

        self.play(FadeIn(header), Create(curve_axes), FadeIn(x_lbl), FadeIn(y_lbl), Create(curve), run_time=1.1)
        self.play(Create(peak_v_line), GrowFromCenter(peak_dot), Write(peak_label), FadeIn(card_peak))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_on_curve], lag_ratio=0.14), FadeIn(dot_labels), FadeIn(card_zero))
        self.pause(
            "Revele la colina continua f(p) = -p ln(p). Señale el punto crítico en 1/e ≈ 0.368. "
            "C (1/3) está casi en la cima, por eso aporta más información que A o B. "
            "Observe que la curva aterriza suavemente en el origen: 0 ln 0 = 0 es continuidad pura."
        )

        self.clear_content()
        header = self.heading("Entropía", "Apilar las contribuciones", PURPLE)
        block_width = 5.7
        x_left = -block_width / 2
        blocks = VGroup()
        block_labels = VGroup()
        cumulative = 0.0
        total = sum(contribution_values)
        for label, value, color in zip(["A  0.3466", "B  0.2986", "C  0.3662"], contribution_values, CAT_COLORS[:3]):
            width = block_width * value / total
            block = Rectangle(
                width=width,
                height=1.05,
                color=color,
                fill_color=color,
                fill_opacity=0.68,
            ).move_to([x_left + cumulative + width / 2, 0.45, 0])
            blocks.add(block)
            label_mob = safe_text(label, size=18, color=INK, weight=BOLD, max_width=width - 0.10)
            label_mob.move_to(block)
            block_labels.add(label_mob)
            cumulative += width
        formula = MathTex(
            r"H=-\sum_{k=1}^{K}p_k\ln(p_k)", font_size=48, color=PURPLE
        ).move_to([0, -0.95, 0])
        result = MathTex(r"H=1.0114\ \text{nats}", font_size=48, color=GREEN).move_to([0, -2.05, 0])
        self.play(FadeIn(header), LaggedStart(*[GrowFromEdge(block, LEFT) for block in blocks], lag_ratio=0.15), FadeIn(block_labels))
        self.play(Write(formula), Write(result))
        self.pause(
            "Reconstruya 0.3466 + 0.2986 + 0.3662 de izquierda a derecha. "
            "Exija la unidad: con logaritmo natural, H=1.0114 nats."
        )

        self.clear_content()
        header = self.heading("Convenciones", "Declare la escala antes de comparar", YELLOW)
        chips = VGroup(
            metric_card("BASE", "ln  →  nats", PURPLE, 3.2),
            metric_card("CATEGORÍA VACÍA", "0 ln 0 := 0", BLUE, 3.2),
            metric_card("UNIVERSO", "K = niveles declarados", ORANGE, 3.6),
        ).arrange(RIGHT, buff=0.35).move_to([0, -0.20, 0])
        note = safe_text(
            "Cambiar la base cambia la unidad, no el orden.",
            size=29,
            color=INK,
            weight=BOLD,
        ).move_to([0, -1.65, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(chip, shift=UP * 0.15) for chip in chips], lag_ratio=0.15))
        self.play(Write(note))
        self.pause(
            "Declare que toda la clase usa ln y nats. Verbalice 0 ln 0 = 0 por "
            "continuidad. Defina K como los niveles fijados antes del análisis."
        )

        self.clear_content()
        header = self.heading("Normalización", "¿Qué fracción del máximo posible?", PURPLE)
        max_h = np.log(3)
        observed_h = 1.0114042647
        bar_bg = RoundedRectangle(
            width=9.0, height=0.85, corner_radius=0.18, color=GRID, fill_color=GRID, fill_opacity=0.25
        ).move_to([0, -0.10, 0])
        bar_fill = RoundedRectangle(
            width=9.0 * observed_h / max_h,
            height=0.85,
            corner_radius=0.18,
            color=PURPLE,
            fill_color=PURPLE,
            fill_opacity=0.78,
        )
        bar_fill.align_to(bar_bg, LEFT).move_to([
            bar_bg.get_left()[0] + bar_fill.width / 2,
            bar_bg.get_y(),
            0,
        ])
        min_label = safe_text("0", size=21, color=MUTED).next_to(bar_bg, DOWN, aligned_edge=LEFT)
        max_label = MathTex(r"H_{\max}=\ln3=1.0986", font_size=29, color=MUTED).next_to(bar_bg, DOWN, aligned_edge=RIGHT)
        norm = MathTex(
            r"H_{\mathrm{norm}}=\frac{1.0114}{\ln3}=0.9206",
            font_size=48,
            color=GREEN,
        ).move_to([0, -1.62, 0])
        self.play(FadeIn(header), FadeIn(bar_bg), FadeIn(min_label), FadeIn(max_label))
        self.play(GrowFromEdge(bar_fill, LEFT), Write(norm))
        self.pause(
            "Pregunte qué permite comparar la normalización. Precise que numerador "
            "y denominador deben usar la misma base y que K debe estar fijado."
        )

        self.clear_content()
        header = self.heading("Número efectivo", "Volver a una escala de categorías", GREEN)
        source = metric_card("SHANNON", "H = 1.0114 nats", PURPLE, 3.5).move_to([-3.7, 0.15, 0])
        arrow = Arrow([-1.55, 0.15, 0], [0.15, 0.15, 0], color=MUTED, stroke_width=4)
        target = metric_card("CATEGORÍAS EFECTIVAS", "exp(H) = 2.75", GREEN, 4.2).move_to([2.65, 0.15, 0])
        full_one = Circle(radius=0.34, color=BLUE, fill_color=BLUE, fill_opacity=0.25)
        full_two = Circle(radius=0.34, color=PURPLE, fill_color=PURPLE, fill_opacity=0.25)
        partial_outline = Circle(radius=0.34, color=GREEN)
        partial_fill = Sector(
            radius=0.31,
            angle=3 * PI / 2,
            start_angle=PI / 2,
            stroke_width=0,
            fill_color=GREEN,
            fill_opacity=0.28,
        )
        partial = VGroup(partial_outline, partial_fill)
        fraction = safe_text("¾", size=18, color=GREEN, weight=BOLD).move_to(partial)
        partial.add(fraction)
        equal_tokens = VGroup(full_one, full_two, partial).arrange(RIGHT, buff=0.28).move_to([2.65, -1.25, 0])
        comparison = safe_text(
            "Simpson: 2.57  ·  Shannon: 2.75",
            size=27,
            color=INK,
            weight=BOLD,
        ).move_to([-1.5, -1.25, 0])
        self.play(FadeIn(header), FadeIn(source), GrowArrow(arrow), FadeIn(target))
        self.play(FadeIn(equal_tokens), FadeIn(comparison))
        self.pause(
            "Interprete 2.75 como el número de categorías igualmente frecuentes "
            "con la misma entropía. Compare con 2.57 de Simpson: ponderan de forma distinta."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estados 20–23 · sensibilidad común y transición a grupos numéricos
    # ------------------------------------------------------------------
    def comparison_and_transition(self) -> None:
        header = self.heading("Comparación", "Mismo K; movamos la masa", YELLOW)
        labels = ["A", "B", "C", "D"]
        dominant = [0.82, 0.08, 0.06, 0.04]
        chart = distribution_chart(dominant, labels, width=7.1, height=3.15)
        chart.move_to([-2.6, -0.25, 0])
        hidden_metrics = VGroup(
            metric_card("GINI–SIMPSON", "?", ORANGE, 3.0),
            metric_card("SHANNON NORM.", "?", PURPLE, 3.0),
            metric_card("EFECTIVAS · SHANNON", "?", GREEN, 3.0),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        prompt = safe_text(
            "Predice: ¿cómo cambiará cada medidor?",
            size=26,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), FadeIn(chart), FadeIn(hidden_metrics), Write(prompt))
        self.pause(
            "Antes de revelar cifras, pida decidir si cada medidor sube, baja o queda "
            "igual cuando el reparto se equilibre. Compare cada medida consigo misma."
        )

        intermediate = [0.50, 0.25, 0.15, 0.10]
        intermediate_chart = distribution_chart(intermediate, labels, width=7.1, height=3.15)
        intermediate_chart.move_to([-2.6, -0.25, 0])
        metrics_mid = VGroup(
            metric_card("GINI–SIMPSON", "0.316  →  0.655", ORANGE, 3.1),
            metric_card("SHANNON NORM.", "0.478  →  0.871", PURPLE, 3.1),
            metric_card("EFECTIVAS · SHANNON", "1.94  →  3.35", GREEN, 3.1),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        self.play(Transform(chart, intermediate_chart), Transform(hidden_metrics, metrics_mid), FadeOut(prompt), run_time=1.35)
        moved = safe_text(
            "masa: dominante → categorías minoritarias",
            size=25,
            color=INK,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(Write(moved))
        self.pause(
            "Pida señalar la redistribución dentro del dibujo: salió masa de A "
            "y llegó a categorías minoritarias. Los tres resúmenes aumentan."
        )

        uniform = [0.25, 0.25, 0.25, 0.25]
        uniform_chart = distribution_chart(uniform, labels, width=7.1, height=3.15)
        uniform_chart.move_to([-2.6, -0.25, 0])
        metrics_uniform = VGroup(
            metric_card("GINI–SIMPSON", "D = 0.750", ORANGE, 3.1),
            metric_card("SHANNON NORM.", "Hnorm = 1.000", PURPLE, 3.1),
            metric_card("EFECTIVAS · SHANNON", "exp(H) = 4.00", GREEN, 3.1),
        ).arrange(DOWN, buff=0.28).move_to([4.55, -0.25, 0])
        conclusion = safe_text(
            "equilibrio máximo para K = 4",
            size=27,
            color=GREEN,
            weight=BOLD,
        ).move_to(moved)
        self.play(Transform(chart, uniform_chart), Transform(hidden_metrics, metrics_uniform), Transform(moved, conclusion), run_time=1.30)
        self.pause(
            "Concluya: con el mismo K, uniformidad maximiza ambos índices y ambos "
            "números efectivos valen cuatro. Un índice alto no significa mejor desempeño."
        )

        self.clear_content()
        header = self.heading("Cambio de pregunta", "De composición a respuesta numérica", GREEN)
        categorical = distribution_chart(uniform, labels, width=6.2, height=2.8)
        categorical.move_to([-3.25, -0.25, 0])
        divider = Arrow([-0.10, -0.15, 0], [1.30, -0.15, 0], color=MUTED, stroke_width=4)
        lanes = VGroup()
        for idx, (label, vals, color) in enumerate(zip(["A", "B", "C"], [[4, 5], [8, 9], [5, 5]], CAT_COLORS[:3])):
            y = 0.85 - idx * 0.95
            axis = NumberLine(x_range=[3, 10, 1], length=4.0, include_numbers=False, color=GRID)
            axis.move_to([4.0, y, 0])
            lane_label = safe_text(label, size=22, color=color, weight=BOLD).next_to(axis, LEFT, buff=0.18)
            dots = VGroup(*[Dot(axis.n2p(v), radius=0.10, color=color) for v in vals])
            lanes.add(VGroup(axis, lane_label, dots))
        left_q = safe_text("reparto de categorías", size=24, color=ORANGE, weight=BOLD).move_to([-3.25, -2.10, 0])
        right_q = safe_text("separación entre grupos", size=24, color=GREEN, weight=BOLD).move_to([4.0, -2.10, 0])
        self.play(FadeIn(header), FadeIn(categorical), FadeIn(left_q))
        self.play(GrowArrow(divider), LaggedStart(*[FadeIn(lane) for lane in lanes], lag_ratio=0.13), FadeIn(right_q))
        self.pause(
            "Nombre el cambio: ahora las categorías definen grupos y la respuesta es "
            "numérica. Ya no preguntamos cómo se reparte la masa, sino cuánto se separan."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estados 24–32 · descomposición total, entre y dentro de grupos
    # ------------------------------------------------------------------
    def group_separation_story(self) -> None:
        header = self.heading("Tres grupos", "La misma respuesta, tres carriles", GREEN)
        group_data = {"A": [4, 5], "B": [8, 9], "C": [5, 5]}
        axes = VGroup()
        all_dots = VGroup()
        group_dots: dict[str, VGroup] = {}
        for idx, (label, vals) in enumerate(group_data.items()):
            y = 1.20 - idx * 1.25
            color = CAT_COLORS[idx]
            axis = NumberLine(
                x_range=[3, 10, 1],
                length=9.6,
                include_numbers=True,
                font_size=18,
                include_tip=False,
                color=GRID,
            ).move_to([0.55, y, 0])
            lane_label = pill(f"GRUPO {label}", color, 1.65).next_to(axis, LEFT, buff=0.25)
            dots = VGroup()
            duplicate: dict[int, int] = {}
            for value in vals:
                level = duplicate.get(value, 0)
                duplicate[value] = level + 1
                dot = Dot(axis.n2p(value) + UP * (0.15 + 0.18 * level), radius=0.11, color=color)
                dots.add(dot)
                all_dots.add(dot)
            group_dots[label] = dots
            axes.add(VGroup(axis, lane_label))
        data_formula = MathTex(
            r"A=(4,5)\qquad B=(8,9)\qquad C=(5,5)",
            font_size=39,
            color=INK,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(axis) for axis in axes], lag_ratio=0.13))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in all_dots], lag_ratio=0.10), Write(data_formula))
        self.pause(
            "Pida estimar la media global sin calcular. Haga que señalen visualmente "
            "qué grupo está más lejos del centro conjunto."
        )

        mean_lines = VGroup()
        mean_labels = VGroup()
        for lane in axes:
            axis = lane[0]
            x = axis.n2p(6)[0]
            line = DashedLine([x, axis.get_y() - 0.42, 0], [x, axis.get_y() + 0.42, 0], color=YELLOW, dash_length=0.08)
            mean_lines.add(line)
        global_mean = MathTex(r"\bar y=\frac{36}{6}=6", font_size=45, color=YELLOW)
        global_mean.next_to(header, DOWN, buff=0.27)
        distance_lines = VGroup()
        for label, vals in group_data.items():
            axis = axes[["A", "B", "C"].index(label)][0]
            for value in vals:
                distance_lines.add(
                    Line(axis.n2p(value), axis.n2p(6), color=YELLOW, stroke_width=5, stroke_opacity=0.55)
                )
        self.play(LaggedStart(*[Create(line) for line in mean_lines], lag_ratio=0.12), Write(global_mean))
        self.play(LaggedStart(*[GrowFromPoint(line, line.get_start()) for line in distance_lines], lag_ratio=0.08))
        self.pause(
            "Compruebe 36/6=6. Lea las seis distancias respecto de una sola media "
            "y pida localizar los puntos que aportarán los cuadrados mayores."
        )

        squared_values = [4, 1, 4, 9, 1, 1]
        squares_visual = VGroup()
        for label, vals in group_data.items():
            axis_idx = ["A", "B", "C"].index(label)
            axis = axes[axis_idx][0]
            col = CAT_COLORS[axis_idx]
            for val in vals:
                p_val = axis.n2p(val)
                p_mean = axis.n2p(6)
                dist = abs(p_val[0] - p_mean[0])
                if dist > 0.05:
                    sq = Square(
                        side_length=dist,
                        stroke_color=col,
                        stroke_width=1.6,
                        fill_color=col,
                        fill_opacity=0.18,
                    )
                    sq_center_x = (p_val[0] + p_mean[0]) / 2
                    sq_center_y = axis.get_y() + dist / 2
                    sq.move_to([sq_center_x, sq_center_y, 0])
                    squares_visual.add(sq)

        cards = VGroup(*[
            metric_card(f"({value}−6)²", str(square), color, 1.65)
            for value, square, color in zip([4, 5, 8, 9, 5, 5], squared_values, [BLUE, BLUE, PURPLE, PURPLE, GREEN, GREEN])
        ]).arrange(RIGHT, buff=0.16).scale(0.88).move_to([0, -2.35, 0])
        total = MathTex(r"SS_T=4+1+4+9+1+1=20", font_size=41, color=YELLOW)
        total.move_to([0, 2.02, 0])
        self.play(
            FadeOut(data_formula),
            FadeIn(cards, shift=UP * 0.18),
            LaggedStart(*[FadeIn(sq) for sq in squares_visual], lag_ratio=0.10),
            Transform(global_mean, total)
        )
        self.pause(
            "Convierta cada distancia en su cuadrado y sume 20. Insista: SS_T mide "
            "la variación total respecto de la media global, representada por el área "
            "acumulada de los cuadrados geométricos."
        )

        self.play(FadeOut(cards), FadeOut(squares_visual), FadeOut(distance_lines), FadeOut(mean_lines), FadeOut(global_mean))
        hidden_means = VGroup()
        for idx, label in enumerate(["A", "B", "C"]):
            axis = axes[idx][0]
            marker = Triangle(color=CAT_COLORS[idx], fill_color=CAT_COLORS[idx], fill_opacity=0.75).scale(0.14)
            marker.next_to(axis, UP, buff=0.28)
            q = safe_text("?", size=23, color=YELLOW, weight=BOLD).next_to(marker, UP, buff=0.05)
            hidden_means.add(VGroup(marker, q))
        question = safe_text(
            "Predice las medias y la contribución mayor",
            size=28,
            color=YELLOW,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(hidden_means), Write(question))
        self.pause(
            "Pida predecir 4.5, 8.5 y 5, y decidir qué grupo contribuirá más a "
            "la separación entre medias. No revele todavía el factor n_g."
        )

        group_means = [4.5, 8.5, 5.0]
        revealed_means = VGroup()
        for idx, (label, mean) in enumerate(zip(["A", "B", "C"], group_means)):
            axis = axes[idx][0]
            marker = Triangle(color=CAT_COLORS[idx], fill_color=CAT_COLORS[idx], fill_opacity=0.82).scale(0.16)
            direction = DOWN if label == "C" else UP
            marker.next_to(axis.n2p(mean), direction, buff=0.27)
            if label == "C":
                marker.rotate(PI)
            value = MathTex(rf"\bar y_{label}={mean:g}", font_size=29, color=CAT_COLORS[idx]).next_to(marker, direction, buff=0.05)
            revealed_means.add(VGroup(marker, value))
        preserved = safe_text(
            "cada grupo se resume por centro + tamaño",
            size=27,
            color=INK,
            weight=BOLD,
        ).move_to(question)
        self.play(Transform(hidden_means, revealed_means), Transform(question, preserved), run_time=1.1)
        self.pause(
            "Revele las medias. Pregunte qué se conserva al sustituir cada dato por "
            "su media grupal: la posición del centro y cuántas observaciones representa."
        )

        self.clear_content()
        header = self.heading("Entre grupos", "Mover los centros hasta la media global", GREEN)
        mean_axis = NumberLine(
            x_range=[3, 10, 1], length=9.5, include_numbers=True, font_size=20, include_tip=False, color=GRID
        ).move_to([0, 0.65, 0])
        global_mark = DashedLine(
            mean_axis.n2p(6) + DOWN * 0.75,
            mean_axis.n2p(6) + UP * 0.75,
            color=YELLOW,
            dash_length=0.09,
        )
        global_label = pill("media global = 6", YELLOW, 2.65).next_to(global_mark, UP, buff=0.14)
        center_dots = VGroup()
        center_labels = VGroup()
        arrows = VGroup()
        for label, mean, color in zip(["A", "B", "C"], group_means, CAT_COLORS[:3]):
            dot = Dot(mean_axis.n2p(mean), radius=0.12, color=color)
            dot.shift(UP * ({"A": 0.18, "B": 0.18, "C": -0.18}[label]))
            center_dots.add(dot)
            center_labels.add(safe_text(label, size=20, color=color, weight=BOLD).next_to(dot, UP if label != "C" else DOWN, buff=0.10))
            arrows.add(Arrow(dot.get_center(), mean_axis.n2p(6), color=color, buff=0.12, stroke_width=4))
        general_ssb = MathTex(
            r"SS_B=\sum_g n_g(\bar y_g-\bar y)^2",
            font_size=29,
            color=MUTED,
        ).move_to([3.35, 1.72, 0])
        ssb = MathTex(
            r"SS_B=2(4.5-6)^2+2(8.5-6)^2+2(5-6)^2",
            font_size=37,
            color=GREEN,
        ).move_to([0, -0.90, 0])
        ssb_result = MathTex(r"SS_B=4.5+12.5+2=19", font_size=43, color=GREEN).move_to([0, -1.85, 0])
        self.play(FadeIn(header), FadeIn(general_ssb), Create(mean_axis), Create(global_mark), FadeIn(global_label))
        self.play(FadeIn(center_dots), FadeIn(center_labels), LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.16))
        self.play(Write(ssb), Write(ssb_result))
        self.pause(
            "Señale el factor n_g=2: cada centro representa dos observaciones. "
            "No permita sumar sólo tres distancias; el componente entre grupos es 19."
        )

        self.clear_content()
        header = self.heading("Dentro de grupos", "Lo que las medias grupales no capturan", PURPLE)
        residual_cards = VGroup(
            metric_card("GRUPO A", "0.25 + 0.25 = 0.5", BLUE, 3.2),
            metric_card("GRUPO B", "0.25 + 0.25 = 0.5", PURPLE, 3.2),
            metric_card("GRUPO C", "0 + 0 = 0", GREEN, 3.2),
        ).arrange(RIGHT, buff=0.35).move_to([0, 0.05, 0])
        ssw = MathTex(r"SS_W=0.5+0.5+0=1", font_size=48, color=PURPLE).move_to([0, -1.50, 0])
        cue = safe_text(
            "distancias cortas: dato → media de su grupo",
            size=25,
            color=MUTED,
        ).move_to([0, 1.62, 0])
        residual_visual = VGroup()
        for center_x, color, coincident in zip([-3.35, 0, 3.35], CAT_COLORS[:3], [False, False, True]):
            center = Dot([center_x, 0.86, 0], radius=0.07, color=YELLOW)
            if coincident:
                points = VGroup(
                    Dot([center_x, 0.86, 0], radius=0.10, color=color),
                    Dot([center_x, 1.05, 0], radius=0.10, color=color),
                )
                segments = VGroup(Line([center_x, 1.05, 0], [center_x, 0.86, 0], color=color, stroke_width=4))
            else:
                points = VGroup(
                    Dot([center_x - 0.42, 0.86, 0], radius=0.10, color=color),
                    Dot([center_x + 0.42, 0.86, 0], radius=0.10, color=color),
                )
                segments = VGroup(
                    Line([center_x - 0.42, 0.86, 0], [center_x, 0.86, 0], color=color, stroke_width=4),
                    Line([center_x + 0.42, 0.86, 0], [center_x, 0.86, 0], color=color, stroke_width=4),
                )
            residual_visual.add(VGroup(segments, center, points))
        self.play(FadeIn(header), FadeIn(cue))
        self.play(LaggedStart(*[FadeIn(item) for item in residual_visual], lag_ratio=0.16))
        self.play(LaggedStart(*[FadeIn(card) for card in residual_cards], lag_ratio=0.14))
        self.play(Write(ssw))
        self.pause(
            "Verifique directamente los seis residuos dentro de grupos. A y B aportan "
            "0.5 cada uno; C no aporta porque sus dos valores coinciden con su media."
        )

        self.clear_content()
        header = self.heading("Partición", "El total se divide sin perder nada", YELLOW)
        total_bar = RoundedRectangle(
            width=10.5, height=1.0, corner_radius=0.17, color=INK, fill_color=GRID, fill_opacity=0.22
        ).move_to([0, 0.20, 0])
        between_bar = Rectangle(
            width=10.5 * 19 / 20,
            height=0.96,
            stroke_width=0,
            fill_color=GREEN,
            fill_opacity=0.80,
        )
        between_bar.align_to(total_bar, LEFT).move_to([total_bar.get_left()[0] + between_bar.width / 2, total_bar.get_y(), 0])
        within_bar = Rectangle(
            width=10.5 / 20,
            height=0.96,
            stroke_width=0,
            fill_color=PURPLE,
            fill_opacity=0.88,
        )
        within_bar.align_to(total_bar, RIGHT).move_to([total_bar.get_right()[0] - within_bar.width / 2, total_bar.get_y(), 0])
        between_label = safe_text("ENTRE = 19", size=24, color=BG, weight=BOLD).move_to(between_bar)
        within_label = safe_text("1", size=20, color=INK, weight=BOLD).move_to(within_bar)
        identity = MathTex(r"SS_T=SS_B+SS_W\qquad 20=19+1", font_size=43, color=INK).move_to([0, -1.05, 0])
        eta = MathTex(r"\eta^2=\frac{SS_B}{SS_T}=\frac{19}{20}=0.95", font_size=52, color=GREEN).move_to([0, -2.05, 0])
        self.play(FadeIn(header), FadeIn(total_bar))
        self.play(GrowFromEdge(between_bar, LEFT), GrowFromEdge(within_bar, LEFT), FadeIn(between_label), FadeIn(within_label))
        self.play(Write(identity), Write(eta))
        self.pause(
            "Interprete primero la barra: 19 partes entre y una dentro. Después lea "
            "eta cuadrada como 95% de la suma de cuadrados total observada asociada a medias grupales."
        )

        self.clear_content()
        header = self.heading("Límites", "Un cociente no sustituye la inspección", RED)
        cautions = VGroup(
            metric_card("CENTRO", "medias y medianas", YELLOW, 2.8),
            metric_card("FORMA", "dispersión y colas", PURPLE, 2.8),
            metric_card("TAMAÑO", "n por grupo", BLUE, 2.8),
            metric_card("ALCANCE", "descriptivo ≠ causal", RED, 3.0),
        ).arrange(RIGHT, buff=0.25).scale(0.92).move_to([0, -0.05, 0])
        footer = safe_text(
            "η² no prueba causalidad ni sustituye la inspección",
            size=28,
            color=RED,
            weight=BOLD,
        ).move_to([0, -1.55, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in cautions], lag_ratio=0.13))
        self.play(Write(footer))
        self.pause(
            "Modele la cautela: eta cuadrada es descriptiva; no demuestra causalidad, "
            "y no reemplaza forma, dispersión ni tamaños grupales."
        )
        self.clear_content()

        header = self.heading("Dos agrupadores", "Cruzar variables exige mirar cada celda", BLUE)
        row_labels = ["REGIÓN 1", "REGIÓN 2"]
        col_labels = ["TIPO A", "TIPO B", "TIPO C"]
        means = [[52, 60, 68], [48, 57, 73]]
        counts = [[40, 38, 3], [35, 32, 2]]
        cells = VGroup()
        for row in range(2):
            for col in range(3):
                x = -2.7 + col * 2.7
                y = 0.75 - row * 1.45
                small = counts[row][col] <= 3
                color = RED if small else CAT_COLORS[col]
                opacity = 0.10 + (means[row][col] - 45) / 55
                cell = RoundedRectangle(
                    width=2.25,
                    height=1.15,
                    corner_radius=0.12,
                    color=color,
                    fill_color=color,
                    fill_opacity=min(0.60, opacity),
                    stroke_width=3 if small else 1.8,
                )
                value = safe_text(f"media = {means[row][col]}", size=21, color=INK, weight=BOLD)
                n_label = safe_text(f"n = {counts[row][col]}", size=18, color=RED if small else MUTED, weight=BOLD)
                content = VGroup(value, n_label).arrange(DOWN, buff=0.10).move_to(cell)
                cells.add(VGroup(cell, content).move_to([x, y, 0]))
        rows = VGroup(*[
            safe_text(label, size=20, color=MUTED, weight=BOLD).move_to([-5.25, 0.75 - idx * 1.45, 0])
            for idx, label in enumerate(row_labels)
        ])
        cols = VGroup(*[
            safe_text(label, size=20, color=CAT_COLORS[idx], weight=BOLD).move_to([-2.7 + idx * 2.7, 1.65, 0])
            for idx, label in enumerate(col_labels)
        ])
        warning = safe_text(
            "heatmap + n por celda · una celda pequeña exige cautela",
            size=27,
            color=RED,
            weight=BOLD,
        ).move_to([0, -2.10, 0])
        self.play(FadeIn(header), FadeIn(rows), FadeIn(cols))
        self.play(LaggedStart(*[FadeIn(cell, scale=0.95) for cell in cells], lag_ratio=0.10))
        self.play(Write(warning), Circumscribe(cells[-1], color=RED))
        self.pause(
            "Presente el heatmap como heterogeneidad cruzada. Haga comparar medias, "
            "pero pida leer también n: las celdas con 3 y 2 observaciones requieren cautela."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estados 33–35 · reto nuevo, pausa real y solución construida
    # ------------------------------------------------------------------
    def transfer_challenge(self) -> None:
        header = self.heading("Reto de transferencia", "Canales de atención · datos nuevos", YELLOW)
        data = [
            ("WEB", 9, BLUE),
            ("APP", 6, PURPLE),
            ("TELÉFONO", 3, GREEN),
            ("TIENDA", 2, ORANGE),
        ]
        cards = VGroup(*[
            metric_card(label, f"f = {count}", color, 2.6) for label, count, color in data
        ]).arrange(RIGHT, buff=0.30).move_to([0, 0.65, 0])
        tasks = VGroup(
            safe_text("1  D de Gini–Simpson", size=25, color=ORANGE, weight=BOLD),
            safe_text("2  H normalizada", size=25, color=PURPLE, weight=BOLD),
            safe_text("3  categorías efectivas de Shannon e interpretación", size=25, color=GREEN, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -0.85, 0])
        timer = pill("PAUSA REAL · 3 min", YELLOW, 3.35, 0.68).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.16) for card in cards], lag_ratio=0.13))
        self.play(FadeIn(tasks), FadeIn(timer))
        self.pause(
            "Deje tres minutos reales. No revele proporciones ni fórmulas resueltas. "
            "Pida primero dibujar el reparto, fijar K=4 y anotar que se usa ln."
        )

        self.clear_content()
        header = self.heading("Solución · 1/2", "Conteos → proporciones → pares", ORANGE)
        probs = [0.45, 0.30, 0.15, 0.10]
        labels = ["WEB", "APP", "TEL", "TIENDA"]
        chart = distribution_chart(probs, labels, width=6.7, height=3.0)
        chart.move_to([-3.15, -0.25, 0])
        p_formula = MathTex(
            r"p=(0.45,\ 0.30,\ 0.15,\ 0.10)", font_size=37, color=INK
        ).move_to([3.45, 1.05, 0])
        check = MathTex(r"\sum p_k=1", font_size=34, color=GREEN).move_to([3.45, 0.35, 0])
        squares = MathTex(
            r"\sum p_k^2=0.2025+0.09+0.0225+0.01=0.325",
            font_size=33,
            color=INK,
        ).move_to([3.45, -0.55, 0])
        if squares.width > 6.0:
            squares.scale_to_fit_width(6.0)
        d_result = MathTex(r"D=1-0.325=0.675", font_size=47, color=ORANGE).move_to([3.45, -1.55, 0])
        interp = safe_text(
            "67.5%: dos extracciones independientes difieren",
            size=23,
            color=GREEN,
            weight=BOLD,
            max_width=6.0,
        ).move_to([3.45, -2.35, 0])
        self.play(FadeIn(header), FadeIn(chart), Write(p_formula))
        self.play(Write(check), Write(squares))
        self.play(Write(d_result), FadeIn(interp))
        self.pause(
            "Compruebe que las proporciones suman uno. Construya 0.325 término a "
            "término y solicite interpretar D=0.675 como probabilidad, con el esquema de muestreo."
        )

        self.clear_content()
        header = self.heading("Solución · 2/2", "Sorpresa → escala comparable → significado", PURPLE)
        contributions = VGroup(
            metric_card("WEB", "0.35933", BLUE, 2.35),
            metric_card("APP", "0.36119", PURPLE, 2.35),
            metric_card("TELÉFONO", "0.28457", GREEN, 2.35),
            metric_card("TIENDA", "0.23026", ORANGE, 2.35),
        ).arrange(RIGHT, buff=0.28).move_to([0, 1.10, 0])
        h_formula = MathTex(
            r"H=\sum[-p_k\ln(p_k)]\approx1.2353\ \text{nats}",
            font_size=42,
            color=PURPLE,
        ).move_to([0, -0.15, 0])
        summary = VGroup(
            metric_card("H NORMALIZADA", "1.2353 / ln 4 = 0.8911", PURPLE, 3.8),
            metric_card("EFECTIVAS · SHANNON", "exp(H) = 3.44", GREEN, 3.8),
            metric_card("EFECTIVAS · SIMPSON", "1 / 0.325 = 3.08", ORANGE, 3.8),
        ).arrange(RIGHT, buff=0.25).move_to([0, -1.25, 0])
        workflow = VGroup(
            pill("dibujar", BLUE, 1.65),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("calcular", ORANGE, 1.75),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("declarar", YELLOW, 1.75),
            Arrow(LEFT, RIGHT, color=MUTED, stroke_width=3).scale(0.35),
            pill("interpretar", GREEN, 2.05),
        ).arrange(RIGHT, buff=0.18).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.12) for card in contributions], lag_ratio=0.12))
        self.play(Write(h_formula), FadeIn(summary))
        self.play(FadeIn(workflow))
        self.pause(
            "Sume con valores no redondeados: H≈1.2353 nats. Pida una frase final: "
            "el reparto equivale a 3.44 categorías igualmente frecuentes según Shannon "
            "y 3.08 según Simpson. Observe cómo Simpson penaliza más a las categorías minoritarias."
        )
        self.clear_content()

    # ------------------------------------------------------------------
    # Estado 37 · cierre y guía de decisión
    # ------------------------------------------------------------------
    def closing(self) -> None:
        header = self.heading("Cierre", "Una pregunta decide qué objeto mirar", BLUE)
        cards = VGroup(
            metric_card("UNA CATEGORÍA", "barras → D, H → efectivos", ORANGE, 3.75),
            metric_card("RESPUESTA POR GRUPO", "puntos → SS → η²", GREEN, 3.75),
            metric_card("DOS AGRUPADORES", "heatmap + n por celda", PURPLE, 3.75),
        ).arrange(RIGHT, buff=0.35).scale(0.94).move_to([0, 0.55, 0])
        conventions = VGroup(
            pill("fijar K", ORANGE, 1.75),
            pill("declarar ln", PURPLE, 2.0),
            pill("nombrar el evento", BLUE, 2.65),
            pill("acotar η²", RED, 1.85),
            pill("SIMULADOR EN VIVO", YELLOW, 2.75),
        ).arrange(RIGHT, buff=0.18).move_to([0, -0.85, 0])
        takeaway = safe_text(
            "Primero mire el reparto; después resúmalo.",
            size=34,
            color=INK,
            weight=BOLD,
        ).move_to([0, -1.95, 0])
        self.play(FadeIn(header), LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in cards], lag_ratio=0.14))
        self.play(LaggedStart(*[FadeIn(chip) for chip in conventions], lag_ratio=0.12))
        self.play(Write(takeaway))
        self.pause(
            "Cierre metodológico. Recuerde conectar estas medidas con Minería de Datos: "
            "D es la impureza de Gini en CART; H es la ganancia de información en C4.5; "
            "y SS_T = SS_B + SS_W es el criterio de agrupamiento en k-means. "
            "Invite al grupo a explorar el simulador interactivo en interactivo.html."
        )
        self.wait(0.8)
