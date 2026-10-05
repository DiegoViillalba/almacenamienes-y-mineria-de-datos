"""Acto 7 · PCA sobre el resorte y sobre la hélice del capítulo."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import (HELIX_PVE, HELIX_TRUE, HELIX_X, HELIX_Z, SPRING_COLS,
                           SPRING_PVE, SPRING_T, SPRING_VECS, SPRING_Z, SPRING_TRUE,
                           _SENSORS, _camera_basis)
from sections.theme import (BLUE, CAM_COLORS, GREEN, GRID, INK, MUTED, ORANGE,
                            PURPLE, RED, YELLOW, card, pill, pve_bars, safe_text, tex)

PHI0, THETA0 = 72 * DEGREES, -60 * DEGREES


class SpringPCASection:
    def spring_pca(self) -> None:
        self.beat("43–44 min. Regreso al resorte: aplicamos la receta a la tabla de 120 × 6. "
                  "Pida predecir el porcentaje de PC1 antes de revelar las barras.")
        self.start("De vuelta al resorte", "PCA sobre las seis columnas", ORANGE)
        bars = pve_bars(SPRING_PVE, height=3.0, width=0.6, gap=0.25).move_to([-3.9, -0.5, 0])
        cap = safe_text("varianza explicada", size=20, color=MUTED).next_to(bars, UP, buff=0.35)
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in bars], lag_ratio=0.03),
                  run_time=1.2)

        self.beat("44–45 min. Loadings de PC1: cada columna pesa según cuánto ve cada cámara el eje "
                  "del movimiento; los signos opuestos reproducen el 'espejo' que vimos. yA pesa poco.")
        v1 = SPRING_VECS[:, 0]
        lax = VGroup()
        base = Line([1.0, 0.9, 0], [6.4, 0.9, 0], color=MUTED, stroke_width=2)
        lax.add(base)
        for j, (col, w) in enumerate(zip(SPRING_COLS, v1)):
            x = 1.45 + j * 0.9
            h = abs(w) * 1.4
            r = Rectangle(width=0.55, height=max(h, 0.02), stroke_width=0,
                          fill_color=CAM_COLORS[col[-1]], fill_opacity=0.9)
            r.move_to([x, 0.9 + np.sign(w) * h / 2, 0])
            val = safe_text(f"{w:+.2f}", size=17, color=INK).next_to(r, UP if w > 0 else DOWN, buff=0.06)
            lab = safe_text(col, size=18, color=CAM_COLORS[col[-1]], weight=BOLD)
            lab.move_to([x, 0.9 + (-0.25 if w > 0 else 0.25), 0])
            lax.add(r, val, lab)
        ltitle = safe_text("loadings de PC1", size=20, color=MUTED).move_to([3.7, -0.42, 0])
        self.play(FadeIn(ltitle), Create(base), LaggedStart(*[FadeIn(m) for m in lax[1:]], lag_ratio=0.03))

        self.beat("45–46 min. Puntuaciones de PC1 contra el tiempo (naranja) y la posición real de "
                  "la masa reescalada (línea punteada). PCA recuperó el movimiento sin saber física.")
        ax = Axes(x_range=[0, SPRING_T[-1], np.pi], y_range=[-2.2, 2.2, 1], x_length=5.3,
                  y_length=2.2, tips=False, axis_config={"color": GRID, "include_ticks": False})
        ax.move_to([3.7, -1.9, 0])
        z = SPRING_Z[:, 0]
        truth = SPRING_TRUE[:, 0] - SPRING_TRUE[:, 0].mean()
        truth = truth * np.sign(np.corrcoef(truth, z)[0, 1]) * z.std() / truth.std()
        zc = ax.plot_line_graph(SPRING_T, z, add_vertex_dots=False, line_color=ORANGE, stroke_width=4)
        tc = DashedVMobject(ax.plot_line_graph(SPRING_T, truth, add_vertex_dots=False,
                                               line_color=INK, stroke_width=2)["line_graph"],
                            num_dashes=70)
        r = np.corrcoef(truth, z)[0, 1]
        rtxt = safe_text(f"correlación con la posición real: {abs(r):.3f}", size=20, color=MUTED)
        rtxt.next_to(ax, DOWN, buff=0.15)
        self.play(Create(ax), Create(zc), run_time=1.4)
        self.play(Create(tc), FadeIn(rtxt))

        self.helix()

    def helix(self) -> None:
        self.beat("46–47 min. Ejemplo del capítulo: una partícula recorre una hélice (un resorte "
                  "estirado). Ahora sí hay más de un grado de libertad espacial.")
        self.start("La hélice del capítulo", "Un resorte en el espacio, tres sensores", BLUE, fixed=True)
        self.move_camera(phi=PHI0, theta=THETA0, zoom=0.8, run_time=1.0)
        s = 1.25
        curve = VMobject(color=ORANGE, stroke_width=4).set_points_smoothly([p * s for p in HELIX_TRUE])
        planes = VGroup()
        for n, c in zip(_SENSORS, [BLUE, GREEN, PURPLE]):
            nn = n / np.linalg.norm(n)
            u, v = _camera_basis(n)
            center = nn * 2.5
            planes.add(Polygon(*[center + (a * u + b * v) * 1.0 for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]],
                               stroke_color=c, stroke_width=2, fill_color=c, fill_opacity=0.12))
        self.play(Create(curve), run_time=1.6)
        self.play(LaggedStart(*[FadeIn(p) for p in planes], lag_ratio=0.3))

        self.beat("47–48 min. Cada sensor aplana la hélice sobre su plano y suma ruido: seis "
                  "columnas otra vez. Gire la escena y pregunte cuántas componentes esperan "
                  "necesitar ahora.", loop=True)
        self.move_camera(theta=THETA0 + TAU, run_time=9, rate_func=linear)
        self.set_camera_orientation(theta=THETA0)

        self.beat(f"48–49 min. Tres componentes guardan {100 * HELIX_PVE[:3].sum():.1f}% y las tres "
                  "restantes son ruido: la señal vive en 3 dimensiones. PCA no sabe de hélices: "
                  "sólo encuentra dónde está la varianza.")
        bars = pve_bars(HELIX_PVE, height=2.1, width=0.5, gap=0.2, label_size=17)
        bars.move_to([4.9, -0.3, 0])
        bcap = safe_text("PCA de las 6 columnas", size=19, color=MUTED).next_to(bars, UP, buff=0.3)
        bcard = card(bars.width + 0.6, bars.height + 1.1).move_to(VGroup(bars, bcap))
        self.fixed(bcard, bars, bcap)
        self.play(FadeIn(bcard), FadeIn(bars), FadeIn(bcap))

        self.beat("49–50 min. Graficamos las tres primeras puntuaciones: reaparece la hélice, "
                  "deformada linealmente (escala y orientación distintas). Límite: PC1 sola no "
                  "bastaría.")
        Z = HELIX_Z[:, :3] / HELIX_Z[:, :3].std(axis=0) * np.array([1.4, 0.8, 0.8])
        rec = VGroup(*[Dot3D(p, radius=0.045, color=BLUE, resolution=(4, 4)) for p in Z])
        self.play(FadeOut(planes), FadeOut(curve))
        self.play(LaggedStart(*[FadeIn(d) for d in rec], lag_ratio=0.005), run_time=1.4)
        self.beat("50 min. Deje girar la hélice recuperada mientras resume: señal con más varianza "
                  "que el ruido ⇒ primeros componentes = señal; últimos = ruido.", loop=True)
        self.move_camera(theta=THETA0 + TAU, run_time=9, rate_func=linear)
        self.set_camera_orientation(theta=THETA0)
