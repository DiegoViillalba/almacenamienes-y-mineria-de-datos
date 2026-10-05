"""Acto 8 · Cuándo PCA engaña: escala, atípicos y clases."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import CLOUD, pca
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, pill, safe_text, tex)

UNIT = 0.72
OUT_DIR = np.array([np.cos(105 * DEGREES), np.sin(105 * DEGREES)])


def pc1_of(points: np.ndarray) -> tuple[np.ndarray, float]:
    vals, vecs, _ = pca(points)
    return vecs[:, 0], vals[0] / vals.sum()


class LimitsSection:
    def _frame(self, origin: np.ndarray, w: float = 4.3, h: float = 3.2) -> VGroup:
        return VGroup(Line(origin + LEFT * w, origin + RIGHT * w, color=GRID, stroke_width=1.5),
                      Line(origin + DOWN * h, origin + UP * h, color=GRID, stroke_width=1.5))

    def _pc1_line(self, origin, get_pts, length=4.0) -> Mobject:
        def build():
            v, _ = pc1_of(get_pts())
            d = np.array([v[0], v[1], 0])
            return Line(origin - d * length, origin + d * length, color=ORANGE, stroke_width=5)
        return always_redraw(build)

    def limits(self) -> None:
        self.beat("50–51 min. Límite 1: escala. Misma nube; sólo cambiamos la unidad de x (×2). "
                  "Pida predecir hacia dónde gira PC1 antes de animar.")
        self.start("Límites · 1", "Las unidades deciden la varianza", RED)
        o = np.array([-2.6, -0.6, 0])
        axes = self._frame(o)
        dots = VGroup(*[Dot(o + UNIT * np.array([x, y, 0]), radius=0.07, color=BLUE) for x, y in CLOUD])

        def pts():
            return np.array([(d.get_center() - o)[:2] / UNIT for d in dots])

        line = self._pc1_line(o, pts)
        angle = DecimalNumber(0, num_decimal_places=0, font_size=36, color=ORANGE, unit=r"^\circ")
        angle.set_value(np.rad2deg(np.arctan2(*pc1_of(pts())[0][::-1])))
        read = VGroup(safe_text("ángulo de PC1", size=22, color=MUTED), angle).arrange(RIGHT, buff=0.25)
        read.move_to([4.4, 1.4, 0])
        self.play(FadeIn(axes), FadeIn(dots), Create(line), FadeIn(read))
        angle.add_updater(lambda m: m.set_value(np.rad2deg(np.arctan2(*pc1_of(pts())[0][::-1]))))
        self.play(*[d.animate.move_to(o + (d.get_center() - o) * np.array([2.0, 1.0, 1.0])) for d in dots],
                  run_time=2)
        t1 = safe_text("x en otra unidad (×2): PC1 se acuesta", size=22, color=RED, max_width=4.4).move_to([4.4, 0.6, 0])
        self.play(FadeIn(t1))

        self.beat("51–52 min. Estandarizar (dividir cada columna entre su desviación) devuelve la "
                  "comparabilidad. Decisión: escalar si las unidades son distintas o arbitrarias; "
                  "sólo centrar si comparten unidad y su varianza importa.")
        sd = pts().std(axis=0, ddof=1)
        self.play(*[d.animate.move_to(o + (d.get_center() - o) * np.array([1 / sd[0], 1 / sd[1], 1.0]))
                    for d in dots], run_time=2)
        t2 = safe_text("estandarizada: la correlación decide", size=22, color=GREEN, max_width=4.4).move_to([4.4, -0.1, 0])
        self.play(FadeIn(t2))
        line.clear_updaters()
        angle.clear_updaters()

        self.beat("52–53 min. Límite 2: un solo atípico. Movemos un punto lejos de la nube. Pida "
                  "observar el ángulo de PC1 y su porcentaje de varianza.")
        self.start("Límites · 2", "Un punto atípico puede girar PC1", RED)
        o = np.array([-2.6, -1.1, 0])
        axes = self._frame(o, h=2.3)
        base = [o + 0.62 * np.array([x, y, 0]) for x, y in CLOUD]
        bdots = VGroup(*[Dot(p, radius=0.07, color=BLUE) for p in base])
        dist = ValueTracker(0.0)

        def all_pts():
            return np.vstack([CLOUD, dist.get_value() * OUT_DIR])

        outlier = always_redraw(lambda: Dot(o + 0.62 * np.array([*(dist.get_value() * OUT_DIR), 0]),
                                            radius=0.12, color=RED))
        ghost = Line(o + np.array([np.cos(32 * DEGREES), np.sin(32 * DEGREES), 0]) * -4,
                     o + np.array([np.cos(32 * DEGREES), np.sin(32 * DEGREES), 0]) * 4,
                     color=ORANGE, stroke_width=2, stroke_opacity=0.35)
        line = self._pc1_line(o, all_pts)
        ang = DecimalNumber(0, num_decimal_places=0, font_size=36, color=ORANGE, unit=r"^\circ")
        ang.set_value(np.rad2deg(np.arctan2(*pc1_of(all_pts())[0][::-1])))
        pve = DecimalNumber(0, num_decimal_places=1, font_size=36, color=ORANGE, unit=r"\%")
        pve.set_value(100 * pc1_of(all_pts())[1])
        read = VGroup(VGroup(safe_text("ángulo de PC1", size=22, color=MUTED), ang).arrange(RIGHT, buff=0.25),
                      VGroup(safe_text("PVE de PC1", size=22, color=MUTED), pve).arrange(RIGHT, buff=0.25),
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.8, 1.0, 0])
        self.play(FadeIn(axes), FadeIn(bdots), FadeIn(ghost), Create(line), FadeIn(outlier), FadeIn(read))
        ang.add_updater(lambda m: m.set_value(np.rad2deg(np.arctan2(*pc1_of(all_pts())[0][::-1]))))
        pve.add_updater(lambda m: m.set_value(100 * pc1_of(all_pts())[1]))
        self.play(dist.animate.set_value(5.5), run_time=3)
        turn = np.rad2deg(np.arctan2(*pc1_of(all_pts())[0][::-1])) - np.rad2deg(np.arctan2(*pc1_of(CLOUD)[0][::-1]))
        msg = VGroup(safe_text(f"1 de 27 puntos gira el eje {turn:.0f}°", size=24, color=RED),
                     safe_text("revisa atípicos antes · considera PCA robusto", size=22, color=MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.8, -1.2, 0])
        self.play(FadeIn(msg))
        for m in (line, ang, pve, outlier):
            m.clear_updaters()

        self.beat("53–54 min. Límite 3: PCA no mira etiquetas. Dos clases alargadas: la mayor "
                  "varianza va a lo largo de ellas. Proyecte en PC1 (se mezclan) y en PC2 (se "
                  "separan). Si el objetivo es clasificar, compare con LDA. Mencione que para "
                  "estructuras curvas (Swiss roll) existen Isomap, t-SNE y UMAP.")
        self.start("Límites · 3", "Máxima varianza ≠ máxima separación", RED)
        rng = np.random.default_rng(5)
        n = 22
        A = np.column_stack([rng.uniform(-3.2, 3.2, n), rng.normal(0.55, 0.16, n)])
        B = np.column_stack([rng.uniform(-3.2, 3.2, n), rng.normal(-0.55, 0.16, n)])
        o2 = np.array([-2.4, 0.3, 0])
        u = 0.8
        da = VGroup(*[Dot(o2 + u * np.array([x, y, 0]), radius=0.07, color=BLUE) for x, y in A])
        db = VGroup(*[Square(0.13, color=YELLOW, fill_opacity=1, stroke_width=0)
                      .move_to(o2 + u * np.array([x, y, 0])) for x, y in B])
        pc1 = Line(o2 + LEFT * 3.2 * u * 1.1, o2 + RIGHT * 3.2 * u * 1.1, color=ORANGE, stroke_width=4)
        pc2 = Line(o2 + DOWN * 1.6, o2 + UP * 1.6, color=GREEN, stroke_width=4)
        n1 = safe_text("PC1", size=20, color=ORANGE, weight=BOLD).next_to(pc1, RIGHT, buff=0.1)
        n2 = safe_text("PC2", size=20, color=GREEN, weight=BOLD).next_to(pc2, UP, buff=0.1)
        self.play(FadeIn(da), FadeIn(db), Create(pc1), Create(pc2), FadeIn(n1), FadeIn(n2))
        y1 = -2.35
        rail1 = Line([o2[0] - 2.9, y1, 0], [o2[0] + 2.9, y1, 0], color=ORANGE, stroke_width=2)
        lab1 = safe_text("sobre PC1: se mezclan", size=20, color=ORANGE).next_to(rail1, RIGHT, buff=0.3)
        rail2 = Line([3.6, o2[1] - 1.4, 0], [3.6, o2[1] + 1.4, 0], color=GREEN, stroke_width=2)
        lab2 = safe_text("sobre PC2: se separan", size=20, color=GREEN).next_to(rail2, RIGHT, buff=0.2)
        c1 = VGroup(*[m.copy() for m in [*da, *db]])
        c2 = VGroup(*[m.copy() for m in [*da, *db]])
        self.play(Create(rail1), FadeIn(lab1),
                  *[m.animate.move_to([m.get_x(), y1 + (0.1 if i < n else -0.1), 0]) for i, m in enumerate(c1)],
                  run_time=1.5)
        self.play(Create(rail2), FadeIn(lab2),
                  *[m.animate.move_to([3.6, m.get_y(), 0]) for m in c2], run_time=1.5)
