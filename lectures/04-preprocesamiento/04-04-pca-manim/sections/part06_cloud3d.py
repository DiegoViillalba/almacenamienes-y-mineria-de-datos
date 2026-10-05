"""Acto 6 · Una nube 3D que en realidad es casi plana."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import PAN_PVE, PAN_VALS, PAN_VECS, PAN_Z, PANCAKE
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, card, pill, pve_bars, safe_text, tex)

SC = 0.62          # unidades de datos → unidades de escena
PHI0, THETA0 = 70 * DEGREES, -50 * DEGREES


class Cloud3DSection:
    def cloud_3d(self) -> None:
        self.beat("36–37 min. Tres variables medidas. Pida adivinar, mirando sólo esta vista, "
                  "cuántas dimensiones ocupa la nube.")
        head = self.start("Tres dimensiones", "¿Cuántas dimensiones usa esta nube?", PURPLE, fixed=True)
        self.move_camera(phi=PHI0, theta=THETA0, zoom=0.95, run_time=1.2)
        axes = ThreeDAxes(x_range=[-5, 5, 1], y_range=[-4, 4, 1], z_range=[-3, 3, 1],
                          x_length=10 * SC, y_length=8 * SC, z_length=6 * SC,
                          axis_config={"color": GRID, "stroke_width": 1.5, "include_ticks": False},
                          tips=False)
        dots = VGroup(*[Dot3D(p * SC, radius=0.055, color=BLUE, resolution=(5, 5)) for p in PANCAKE])
        self.play(Create(axes), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.01), run_time=1.3)

        self.beat("37–38 min. Giramos la escena. Deje que el grupo cambie su respuesta: vista de "
                  "canto, la nube es casi una hoja. Pregunte en qué dirección es delgada.", loop=True)
        self.move_camera(theta=THETA0 + TAU, run_time=9, rate_func=linear)
        self.set_camera_orientation(theta=THETA0)

        self.beat("38–39 min. PCA responde con tres ejes perpendiculares. Longitud de cada flecha "
                  "= 2 desviaciones estándar de las puntuaciones (2√λ). Lea las barras: dos "
                  "componentes guardan casi toda la variación.")
        colors = [ORANGE, GREEN, PURPLE]
        arrows = VGroup(*[Arrow3D(ORIGIN, PAN_VECS[:, k] * 2 * np.sqrt(PAN_VALS[k]) * SC, color=c,
                                  thickness=0.03, height=0.25, base_radius=0.08)
                          for k, c in enumerate(colors)])
        labels = VGroup(*[safe_text(f"PC{k + 1}", size=26, color=c, weight=BOLD)
                          .move_to(PAN_VECS[:, k] * (2 * np.sqrt(PAN_VALS[k]) * SC + 0.4))
                          for k, c in enumerate(colors)])
        self.oriented(*labels)
        bars = pve_bars(PAN_PVE, height=2.2, width=0.62).move_to([5.0, 0.1, 0])
        bars_cap = safe_text("varianza explicada", size=20, color=MUTED).next_to(bars, UP, buff=0.3)
        bars_card = card(bars.width + 0.8, bars.height + 1.2).move_to(VGroup(bars, bars_cap))
        self.fixed(bars_card, bars, bars_cap)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.3), run_time=1.6)
        self.play(FadeIn(bars_card), FadeIn(bars), FadeIn(bars_cap))

        self.beat(f"39–40 min. Plano generado por PC1 y PC2 ({100 * PAN_PVE[:2].sum():.1f}% de la "
                  "varianza). Cada punto baja perpendicularmente al plano: los segmentos rojos son "
                  "el error de reconstrucción, cortísimo.")
        v1, v2, v3 = (PAN_VECS[:, k] for k in range(3))
        a, b = 4.4 * SC, 2.6 * SC
        plane = Polygon(*[s * a * v1 + t * b * v2 for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1)]],
                        stroke_color=ORANGE, stroke_width=1.5, fill_color=ORANGE, fill_opacity=0.12)
        feet = [(p - (p @ v3) * v3) * SC for p in PANCAKE - PANCAKE.mean(0)]
        drops = VGroup(*[Line(d.get_center(), f, color=RED, stroke_width=2)
                         for d, f in zip(dots, feet)])
        self.play(FadeIn(plane), FadeOut(arrows[2]), FadeOut(labels[2]))
        self.play(Create(drops), run_time=1.0)
        self.play(*[d.animate.move_to(f) for d, f in zip(dots, feet)], FadeOut(drops), run_time=1.6)

        self.beat("40–41 min. La nube proyectada gira con el plano: casi no cambió. Pida explicar "
                  "con la barra morada qué se perdió (≈1%).", loop=True)
        self.move_camera(theta=THETA0 + TAU, run_time=9, rate_func=linear)
        self.set_camera_orientation(theta=THETA0)

        self.beat("41–43 min. Leemos el plano de frente: estas son las puntuaciones Z = X̃V₂, una "
                  "tabla de 70 × 2. Recalque: los ejes nuevos son combinaciones de las variables "
                  "originales (loadings), no variables medidas.")
        self.clear()
        self.play(FadeIn(self.heading("Tres dimensiones", "Las puntuaciones: una tabla de n × 2", PURPLE)),
                  run_time=0.35)
        ax = Axes(x_range=[-5, 5, 1], y_range=[-3, 3, 1], x_length=6.8, y_length=4.1, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 1.5})
        ax.move_to([-2.4, -0.6, 0])
        xl = safe_text("PC1", size=22, color=ORANGE, weight=BOLD).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = safe_text("PC2", size=22, color=GREEN, weight=BOLD).next_to(ax.y_axis, UP, buff=0.1)
        sdots = VGroup(*[Dot(ax.c2p(z[0], z[1]), radius=0.06, color=BLUE) for z in PAN_Z])
        load = VGroup(
            tex(r"Z=\tilde X\,V_2", size=44, color=INK),
            tex(rf"\mathbf v_1=({PAN_VECS[0,0]:.2f},\;{PAN_VECS[1,0]:.2f},\;{PAN_VECS[2,0]:.2f})", size=32, color=ORANGE),
            tex(rf"\mathbf v_2=({PAN_VECS[0,1]:.2f},\;{PAN_VECS[1,1]:.2f},\;{PAN_VECS[2,1]:.2f})", size=32, color=GREEN),
            safe_text("loadings: pesos de x₁, x₂, x₃", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.1, -0.3, 0])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl),
                  LaggedStart(*[FadeIn(d) for d in sdots], lag_ratio=0.01), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(m) for m in load], lag_ratio=0.2))
