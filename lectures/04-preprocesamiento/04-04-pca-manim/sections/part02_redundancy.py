"""Acto 2 · La tabla del resorte: seis columnas que se mueven juntas."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import SPRING_COLS, SPRING_CORR, SPRING_T, SPRING_X
from sections.theme import (BLUE, CAM_COLORS, GRID, INK, MUTED, ORANGE, RED,
                            YELLOW, card, pill, safe_text, tex)


def _heat_color(r: float) -> ManimColor:
    base = ManimColor(BLUE) if r >= 0 else ManimColor(RED)
    return interpolate_color(ManimColor("#0D1E34"), base, min(1.0, abs(r)))


class RedundancySection:
    def redundancy(self) -> None:
        self.beat("10–11 min. Cambiamos de la escena a la tabla: 120 instantes × 6 columnas. "
                  "Pregunte qué esperan ver si graficamos cada columna contra el tiempo.")
        self.start("La tabla del resorte", "120 filas × 6 columnas", ORANGE)
        n_show = 60
        t = SPRING_T[:n_show]
        plots = VGroup()
        for j, col in enumerate(SPRING_COLS):
            y = SPRING_X[:n_show, j] - SPRING_X[:, j].mean()
            ax = Axes(x_range=[0, t[-1], t[-1]], y_range=[-1.3, 1.3, 1.3], x_length=3.6,
                      y_length=1.35, tips=False,
                      axis_config={"color": GRID, "stroke_width": 1.5, "include_ticks": False})
            curve = ax.plot_line_graph(t, y, add_vertex_dots=False,
                                       line_color=CAM_COLORS[col[-1]], stroke_width=3)
            lab = safe_text(col, size=22, color=CAM_COLORS[col[-1]], weight=BOLD)
            lab.next_to(ax, LEFT, buff=0.15)
            plots.add(VGroup(ax, curve, lab))
        plots.arrange_in_grid(rows=2, cols=3, buff=(0.75, 0.55)).move_to([0.2, -0.35, 0])
        self.play(LaggedStart(*[Create(p[1]) for p in plots], lag_ratio=0.12),
                  *[FadeIn(VGroup(p[0], p[2])) for p in plots], run_time=2.2)

        self.beat("11–12 min. Pida describirlas: todas oscilan con el mismo ritmo; unas en fase, "
                  "otras en espejo, con amplitudes distintas. yA se mueve poco: la cámara A ve el eje "
                  "casi de lado, así que esa columna tiene más ruido relativo.")
        msg = pill("mismo ritmo · distinta escala · a veces en espejo", YELLOW, size=24)
        msg.move_to([0, 2.05, 0])
        self.play(FadeIn(msg, shift=DOWN * 0.1))

        self.beat("12–13 min. Resuma la redundancia con correlaciones. Pregunte qué valor esperan "
                  "entre xA y xC. Lectura: |r| cerca de 1 = columnas casi intercambiables.")
        self.play(FadeOut(plots), FadeOut(msg))
        cells = VGroup()
        size = 0.72
        for i in range(6):
            for j in range(6):
                r = SPRING_CORR[i, j]
                sq = Square(size, stroke_color="#07111F", stroke_width=2,
                            fill_color=_heat_color(r), fill_opacity=1)
                sq.move_to([(j - 2.5) * size, (2.5 - i) * size, 0])
                val = safe_text(f"{r:+.2f}", size=15,
                                color=INK if abs(r) < 0.8 else "#07111F", weight=BOLD)
                val.move_to(sq)
                cells.add(VGroup(sq, val))
        heat = VGroup(cells)
        for k, col in enumerate(SPRING_COLS):
            top = safe_text(col, size=19, color=CAM_COLORS[col[-1]], weight=BOLD)
            top.move_to([(k - 2.5) * size, 3 * size + 0.22, 0])
            left = top.copy().move_to([-3 * size - 0.35, (2.5 - k) * size, 0])
            heat.add(top, left)
        heat.move_to([-3.3, -0.65, 0])
        self.play(LaggedStart(*[FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.015),
                  FadeIn(heat[1:]), run_time=1.6)
        upper = np.abs(SPRING_CORR[np.triu_indices(6, 1)])
        notes = VGroup(
            safe_text(f"{(upper > 0.6).sum()} de 15 pares con |r| > 0.6", size=26, color=INK),
            safe_text(f"{(upper > 0.95).sum()} pares con |r| > 0.95", size=26, color=BLUE),
            safe_text("azul: suben juntas", size=22, color=BLUE),
            safe_text("rojo: una sube, otra baja", size=22, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([3.2, 0.75, 0])
        self.play(FadeIn(notes, shift=LEFT * 0.1))

        self.beat("13–14 min. Formalice la notación: X de n×p, fila = observación, columna = "
                  "variable. Pregunta guía: ¿existe un eje nuevo que resuma las seis columnas? "
                  "Para responderla, bajamos a dos columnas.")
        formal = VGroup(
            tex(r"X\in\mathbb{R}^{n\times p},\qquad n=120,\; p=6", size=34, color=INK),
            safe_text("p columnas medidas ≠ dimensiones con información", size=20, color=ORANGE,
                      max_width=5.6),
        ).arrange(DOWN, buff=0.25)
        frame = card(formal.width + 0.6, formal.height + 0.5, ORANGE)
        box = VGroup(frame, formal).move_to([3.2, -1.85, 0])
        formal.move_to(frame)
        self.play(FadeIn(box, shift=UP * 0.1))
