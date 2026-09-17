"""Constantes de estilo, colores, fuentes y funciones auxiliares visuales."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from manim import *

# Paleta institucional del curso
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

# ==============================================================================
# CONFIGURACIÓN DE TIPOGRAFÍA (FUENTE)
# ------------------------------------------------------------------------------
# FONT = "" utiliza la tipografía por defecto de Manim (Pango system default).
# Para cambiar la tipografía en toda la presentación, define aquí el nombre de
# la fuente que tengas instalada en tu sistema operativo.
#
# Ejemplos:
#   FONT = ""                   # Fuente por defecto de Manim (default)
#   FONT = "Helvetica"          # Sans-serif estándar en macOS
#   FONT = "Arial"              # Sans-serif multiplataforma
#   FONT = "Fira Sans"          # Fuente técnica moderna
#   FONT = "CMU Serif"          # Estilo clásico LaTeX Computer Modern
#   FONT = "Latin Modern Roman" # Tipografía académica formal
# ==============================================================================
FONT = ""

LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


def safe_text(
    content: str,
    *,
    size: float = 34,
    color: str = INK,
    weight: str = NORMAL,
    max_width: float | None = None,
    font: str = FONT,
) -> Text:
    """Crea texto seguro y portable para Manim con Pango.
    
    Si font="", Manim utiliza su tipografía por defecto.
    """
    mob = Text(content, font=font, font_size=size, color=color, weight=weight)
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
    """Gráfico de barras categórico simple cuyos objetos se transforman continuamente."""
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
    """Particiona el cuadrado unitario [0,1]^2; las celdas diagonales tienen áreas p_k^2."""
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
