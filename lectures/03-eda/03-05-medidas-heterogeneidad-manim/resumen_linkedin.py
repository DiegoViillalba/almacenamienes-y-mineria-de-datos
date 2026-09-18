"""Video corto y autocontenido: la geometría de Gini–Simpson y Shannon.

Un solo clip (sin pausas de docente) para redes sociales. Reutiliza el
ejemplo de las seis canicas de
``unidades/intro_mineria/2-6-heterogeneidad.qmd``: A=3 (azul), B=1 (morada),
C=2 (verde), p=(1/2, 1/6, 1/3). Muestra dos geometrías construidas sobre la
misma partición de [0,1]:

- Gini–Simpson: área de un cuadrado unitario particionado en 9 celdas.
- Shannon: área bajo un histograma de "sorpresa" -log2(p_k).

Render rápido:
    ../../../.venv-manim/bin/manim -pql resumen_linkedin.py ResumenHeterogeneidad

Render final (usa manim.cfg: 1920x1080 @30fps):
    ../../../.venv-manim/bin/manim -pqh resumen_linkedin.py ResumenHeterogeneidad
"""

from __future__ import annotations

import numpy as np
from manim import *

from sections.theme import (
    BG, INK, MUTED, GRID, ORANGE, PURPLE, GREEN, CAT_COLORS, LOGO_PATH, safe_text,
)

AUTHOR = "Diego Villalba"

# ---------------------------------------------------------------------------
# El mismo ejemplo del capítulo: A=3, B=1, C=2 canicas sobre 6.
# ---------------------------------------------------------------------------
LABELS = ["A", "B", "C"]
COUNTS = [3, 1, 2]
PROBS = [3 / 6, 1 / 6, 2 / 6]
FRACS = ["1/2", "1/6", "1/3"]
COLORS = CAT_COLORS[:3]  # BLUE, PURPLE, GREEN -> A, B, C (convención del curso)

SUM_SQ = sum(p * p for p in PROBS)                      # 7/18 ≈ 0.3889
D_VALUE = 1 - SUM_SQ                                     # 11/18 ≈ 0.6111
SURPRISES = [-np.log2(p) for p in PROBS]                 # bits
CONTRIB = [p * s for p, s in zip(PROBS, SURPRISES)]
H_VALUE = sum(CONTRIB)                                   # ≈ 1.4591 bits


def partition_bar(width: float = 9.0, height: float = 0.9) -> VGroup:
    """Segmento [0,1] partido en A, B, C — la base geométrica compartida."""
    x_left = -width / 2
    segments, names, fracs = VGroup(), VGroup(), VGroup()
    cumulative = 0.0
    for label, p, color, frac in zip(LABELS, PROBS, COLORS, FRACS):
        w = width * p
        seg = Rectangle(
            width=w, height=height, stroke_color=BG, stroke_width=2,
            fill_color=color, fill_opacity=0.85,
        ).move_to([x_left + cumulative + w / 2, 0, 0])
        segments.add(seg)
        names.add(safe_text(label, size=22, color=BG, weight=BOLD).move_to(seg))
        fracs.add(
            safe_text(frac, size=19, color=color, weight=BOLD).next_to(seg, DOWN, buff=0.16)
        )
        cumulative += w
    return VGroup(segments, names, fracs)


def gini_grid(side: float = 4.4, diff_color: str = ORANGE):
    """Cuadrado unitario particionado en 9 celdas: diagonal = misma clase."""
    lower_left = np.array([-side / 2, -side / 2, 0.0])
    diagonal, off_diagonal = VGroup(), VGroup()
    cum_i = 0.0
    for i, (pi, ci) in enumerate(zip(PROBS, COLORS)):
        cum_j = 0.0
        for j, (pj, _cj) in enumerate(zip(PROBS, COLORS)):
            w, h = side * pj, side * pi
            same = i == j
            cell = Rectangle(
                width=w, height=h, stroke_color=BG, stroke_width=1.5,
                fill_color=ci if same else diff_color,
                fill_opacity=0.85 if same else 0.55,
            )
            cell.move_to(lower_left + np.array([side * cum_j + w / 2, side * cum_i + h / 2, 0]))
            (diagonal if same else off_diagonal).add(cell)
            cum_j += pj
        cum_i += pi
    frame = Square(side_length=side, stroke_color=INK, stroke_width=2.5, fill_opacity=0)
    return frame, diagonal, off_diagonal


def entropy_bars(total_width: float = 8.6, y_scale: float = 1.05) -> VGroup:
    """Barras de ancho p_k y alto -log2(p_k): el área es la sorpresa ponderada."""
    x_left = -total_width / 2
    bars = VGroup()
    cumulative = 0.0
    for p, color, s in zip(PROBS, COLORS, SURPRISES):
        w, h = total_width * p, s * y_scale
        bar = Rectangle(
            width=w, height=h, stroke_color=BG, stroke_width=1.5,
            fill_color=color, fill_opacity=0.22,
        )
        bar.move_to([x_left + cumulative + w / 2, h / 2, 0])
        bars.add(bar)
        cumulative += w
    return bars


class ResumenHeterogeneidad(Scene):
    """Clip único: misma bolsa, dos geometrías (Gini–Simpson y Shannon)."""

    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.intro()
        self.setup_example()
        self.gini_simpson_geometry()
        self.transition()
        self.shannon_geometry()
        self.comparison()
        self.closing()

    # ------------------------------------------------------------------
    def intro(self) -> None:
        logo = ImageMobject(str(LOGO_PATH)).scale_to_fit_width(3.4)
        title = safe_text("Heterogeneidad: dos geometrías", size=48, color=INK, weight=BOLD)
        subtitle = safe_text(
            "Gini–Simpson y Shannon, con el mismo ejemplo", size=27, color=MUTED
        )
        stack = Group(logo, title, subtitle).arrange(DOWN, buff=0.45).move_to(ORIGIN)
        credit = safe_text(AUTHOR, size=22, color=MUTED).to_corner(DR, buff=0.45)

        self.play(FadeIn(logo), run_time=0.5)
        self.play(Write(title), run_time=0.9)
        self.play(FadeIn(subtitle, shift=UP * 0.15), FadeIn(credit), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(Group(logo, title, subtitle, credit)), run_time=0.45)

    def setup_example(self) -> None:
        header = safe_text("Una bolsa con 6 canicas", size=32, color=INK, weight=BOLD)
        header.to_edge(UP, buff=0.6)

        xs = np.linspace(-3.5, 3.5, 6)
        seq_colors = [COLORS[0]] * COUNTS[0] + [COLORS[1]] * COUNTS[1] + [COLORS[2]] * COUNTS[2]
        seq_labels = [LABELS[0]] * COUNTS[0] + [LABELS[1]] * COUNTS[1] + [LABELS[2]] * COUNTS[2]
        marbles = VGroup()
        for x, color, label in zip(xs, seq_colors, seq_labels):
            dot = Circle(radius=0.34, color=color, fill_color=color, fill_opacity=0.9, stroke_color=BG)
            tag = safe_text(label, size=19, color=BG, weight=BOLD).move_to(dot)
            marbles.add(VGroup(dot, tag).move_to([x, 0.35, 0]))
        counts_label = safe_text(
            "3 azules · 1 morada · 2 verdes", size=23, color=MUTED
        ).move_to([0, -0.85, 0])

        self.play(FadeIn(header), run_time=0.35)
        self.play(LaggedStart(*[GrowFromCenter(m) for m in marbles], lag_ratio=0.1), run_time=0.9)
        self.play(FadeIn(counts_label), run_time=0.4)
        self.wait(0.4)

        bar = partition_bar(width=9.0, height=0.9).move_to([0, -0.4, 0])
        prop_formula = MathTex(
            r"p=\left(\tfrac12,\ \tfrac16,\ \tfrac13\right)", font_size=38, color=INK
        ).move_to([0, 1.1, 0])
        self.play(
            FadeOut(marbles), FadeOut(counts_label),
            FadeIn(bar), Write(prop_formula),
            run_time=1.0,
        )
        self.wait(0.6)
        self.play(FadeOut(header), FadeOut(prop_formula), FadeOut(bar), run_time=0.5)

    # ------------------------------------------------------------------
    def gini_simpson_geometry(self) -> None:
        header = safe_text("Gini–Simpson", size=36, color=ORANGE, weight=BOLD)
        header.to_edge(UP, buff=0.55)
        subheader = safe_text(
            "¿coinciden dos canicas extraídas al azar?", size=23, color=MUTED
        ).next_to(header, DOWN, buff=0.16)
        self.play(FadeIn(header), FadeIn(subheader), run_time=0.4)

        frame, diagonal, off_diagonal = gini_grid(side=4.5, diff_color=ORANGE)
        grid_group = VGroup(frame, diagonal, off_diagonal).move_to([-2.7, -0.55, 0])
        axis1 = safe_text("extracción 1", size=17, color=MUTED).next_to(grid_group, LEFT, buff=0.2).rotate(PI / 2)
        axis2 = safe_text("extracción 2", size=17, color=MUTED).next_to(grid_group, DOWN, buff=0.2)

        same_label = safe_text("misma categoría", size=22, color=INK, weight=BOLD).move_to([2.9, 1.3, 0])
        self.play(Create(frame), FadeIn(axis1), FadeIn(axis2), run_time=0.5)
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in diagonal], lag_ratio=0.18),
            FadeIn(same_label),
            run_time=1.0,
        )

        sq_formula = MathTex(
            r"\sum p_k^2=\tfrac14+\tfrac1{36}+\tfrac19=\tfrac7{18}\approx0.3889",
            font_size=30, color=INK,
        ).move_to([2.9, 0.35, 0])
        if sq_formula.width > 6.0:
            sq_formula.scale_to_fit_width(6.0)
        self.play(Write(sq_formula), run_time=0.9)
        self.wait(0.5)

        diff_label = safe_text("categorías distintas", size=22, color=ORANGE, weight=BOLD).move_to([2.9, -0.6, 0])
        self.play(FadeOut(same_label), FadeOut(sq_formula), run_time=0.3)
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in off_diagonal], lag_ratio=0.05),
            FadeIn(diff_label),
            run_time=0.9,
        )

        d_formula = MathTex(
            rf"D=1-\sum p_k^2={D_VALUE:.4f}", font_size=38, color=ORANGE,
        ).move_to([2.9, 0.5, 0])
        interpretation = safe_text(
            f"{D_VALUE * 100:.1f}% de pares con categorías distintas",
            size=21, color=GREEN, weight=BOLD, max_width=5.6,
        ).move_to([2.9, -1.35, 0])
        self.play(FadeOut(diff_label), Write(d_formula), run_time=0.7)
        self.play(FadeIn(interpretation), run_time=0.4)
        self.wait(0.9)

        self.play(
            FadeOut(VGroup(header, subheader, grid_group, axis1, axis2, d_formula, interpretation)),
            run_time=0.5,
        )

    # ------------------------------------------------------------------
    def transition(self) -> None:
        line = safe_text(
            "Misma bolsa, otra pregunta:", size=28, color=INK, weight=BOLD
        )
        line2 = safe_text(
            "¿cuánta sorpresa trae, en promedio, una sola extracción?",
            size=26, color=MUTED,
        )
        group = VGroup(line, line2).arrange(DOWN, buff=0.3)
        self.play(Write(line), run_time=0.6)
        self.play(FadeIn(line2, shift=UP * 0.1), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(group), run_time=0.4)

    # ------------------------------------------------------------------
    def shannon_geometry(self) -> None:
        header = safe_text("Shannon", size=36, color=PURPLE, weight=BOLD)
        header.to_edge(UP, buff=0.55)
        subheader = safe_text(
            "sorpresa −log₂(p) ponderada por su frecuencia", size=23, color=MUTED
        ).next_to(header, DOWN, buff=0.16)
        self.play(FadeIn(header), FadeIn(subheader), run_time=0.4)

        total_width = 8.6
        bars = entropy_bars(total_width=total_width, y_scale=1.05).shift(DOWN * 1.9)
        baseline = Line(
            LEFT * total_width / 2, RIGHT * total_width / 2, color=GRID, stroke_width=3
        ).move_to([0, -1.9, 0])

        surprise_labels = VGroup()
        class_labels = VGroup()
        for bar, label, color, s in zip(bars, LABELS, COLORS, SURPRISES):
            surprise_labels.add(
                MathTex(rf"-\log_2 p={s:.3f}", font_size=22, color=INK).next_to(bar, UP, buff=0.12)
            )
            class_labels.add(
                safe_text(label, size=20, color=color, weight=BOLD).next_to(bar, DOWN, buff=0.16)
            )

        self.play(Create(baseline), run_time=0.3)
        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.18),
            FadeIn(class_labels),
            run_time=1.0,
        )
        self.play(FadeIn(surprise_labels), run_time=0.5)
        self.wait(0.6)

        contrib_labels = VGroup()
        for bar, c in zip(bars, CONTRIB):
            contrib_labels.add(
                MathTex(rf"p\cdot\!\left(-\log_2 p\right)={c:.4f}", font_size=20, color=INK)
                .next_to(bar, UP, buff=0.12)
            )
        self.play(
            *[bar.animate.set_fill(opacity=0.82) for bar in bars],
            Transform(surprise_labels, contrib_labels),
            run_time=0.9,
        )
        self.wait(0.4)

        h_formula = MathTex(
            rf"H=-\sum p_k\log_2 p_k={H_VALUE:.4f}\ \text{{bits}}",
            font_size=36, color=PURPLE,
        ).move_to([0, 2.1, 0])
        if h_formula.width > 9.0:
            h_formula.scale_to_fit_width(9.0)
        self.play(Write(h_formula), run_time=0.9)
        self.wait(0.9)

        self.play(
            FadeOut(
                VGroup(
                    header, subheader, bars, baseline, surprise_labels,
                    class_labels, h_formula,
                )
            ),
            run_time=0.5,
        )

    # ------------------------------------------------------------------
    def comparison(self) -> None:
        header = safe_text("Misma bolsa, dos geometrías", size=32, color=INK, weight=BOLD)
        header.to_edge(UP, buff=0.7)

        gini_card = self._metric_card(
            "GINI–SIMPSON", f"D = {D_VALUE:.4f}",
            "área de un cuadrado", ORANGE,
        ).move_to([-3.1, -0.2, 0])
        shannon_card = self._metric_card(
            "SHANNON", f"H = {H_VALUE:.4f} bits",
            "área bajo un histograma", PURPLE,
        ).move_to([3.1, -0.2, 0])

        self.play(FadeIn(header), run_time=0.4)
        self.play(FadeIn(gini_card, shift=UP * 0.15), FadeIn(shannon_card, shift=UP * 0.15), run_time=0.7)
        self.wait(1.1)
        self.play(FadeOut(VGroup(header, gini_card, shannon_card)), run_time=0.5)

    @staticmethod
    def _metric_card(label: str, value: str, caption: str, color: str) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.18, width=4.6, height=2.0,
            stroke_color=color, stroke_width=2.4, fill_color=color, fill_opacity=0.10,
        )
        label_mob = safe_text(label, size=20, color=MUTED, weight=BOLD)
        value_mob = safe_text(value, size=32, color=color, weight=BOLD)
        caption_mob = safe_text(caption, size=19, color=INK)
        content = VGroup(label_mob, value_mob, caption_mob).arrange(DOWN, buff=0.18).move_to(box)
        return VGroup(box, content)

    # ------------------------------------------------------------------
    def closing(self) -> None:
        title = safe_text("Heterogeneidad es geometría de la probabilidad", size=32, color=INK, weight=BOLD, max_width=11.5)
        subtitle = safe_text("Almacenes y Minería de Datos", size=24, color=MUTED)
        group = VGroup(title, subtitle).arrange(DOWN, buff=0.4)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle), run_time=0.5)
        self.wait(1.1)
        self.play(FadeOut(group), run_time=0.5)


if __name__ == "__main__":
    pass
