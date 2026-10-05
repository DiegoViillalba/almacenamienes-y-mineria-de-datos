"""Acto 5 · S como transformación, cálculo a mano y receta."""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import CLOUD_S, CLOUD_VALS, CLOUD_VECS
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, bullet_list, card, pill, safe_text, tex)

UNIT = 1.05


class EigenSection:
    def eigen_story(self) -> None:
        self.beat("29–30 min. ¿Qué significa Su = λu? Aplicamos S a muchas direcciones unitarias "
                  "(círculo). Pida predecir qué figura resulta.")
        self.start("Autovectores", "¿Qué direcciones no gira la matriz S?", YELLOW)
        origin = np.array([-2.9, -0.6, 0])
        plane = NumberPlane(x_range=[-3, 3, 1], y_range=[-2.5, 2.5, 1], x_length=6 * UNIT,
                            y_length=5 * UNIT,
                            background_line_style={"stroke_color": GRID, "stroke_width": 1},
                            axis_config={"stroke_color": MUTED, "stroke_width": 1.5})
        plane.move_to(origin)
        angles = np.linspace(0, TAU, 16, endpoint=False)
        eig_angles = [np.arctan2(CLOUD_VECS[1, j], CLOUD_VECS[0, j]) for j in range(2)]
        arrows = VGroup()
        for a in angles:
            arrows.add(Arrow(origin, origin + UNIT * np.array([np.cos(a), np.sin(a), 0]), buff=0,
                             color=BLUE, stroke_width=3, max_tip_length_to_length_ratio=0.18))
        eig = VGroup(*[Arrow(origin, origin + UNIT * np.array([np.cos(a), np.sin(a), 0]), buff=0,
                             color=c, stroke_width=6, max_tip_length_to_length_ratio=0.2)
                       for a, c in zip(eig_angles, [ORANGE, GREEN])])
        circle = Circle(radius=UNIT, color=MUTED, stroke_width=2).move_to(origin)
        S_tex = tex(rf"S=\begin{{pmatrix}}{CLOUD_S[0,0]:.2f}&{CLOUD_S[0,1]:.2f}\\"
                    rf"{CLOUD_S[1,0]:.2f}&{CLOUD_S[1,1]:.2f}\end{{pmatrix}}", size=40)
        S_tex.move_to([3.6, 1.35, 0])
        cap = safe_text("covarianza de la nube anterior", size=20, color=MUTED).next_to(S_tex, DOWN, buff=0.15)
        self.play(FadeIn(plane), Create(circle), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.04),
                  Write(S_tex), FadeIn(cap), run_time=1.4)

        self.beat("30–31 min. Transformación: casi todas las flechas cambian de dirección. Pida "
                  "localizar las que sólo se estiran o encogen sobre su propia recta.")
        M = np.eye(3)
        M[:2, :2] = CLOUD_S
        def image(a: float, color: str, width: float) -> Arrow:
            w = CLOUD_S @ np.array([np.cos(a), np.sin(a)])
            return Arrow(origin, origin + UNIT * np.array([w[0], w[1], 0]), buff=0, color=color,
                         stroke_width=width, max_tip_length_to_length_ratio=0.18)

        targets = VGroup(*[image(a, BLUE, 3) for a in angles])
        self.play(ApplyMatrix(M, circle, about_point=origin),
                  *[Transform(a, t) for a, t in zip(arrows, targets)], run_time=2.2)
        self.play(arrows.animate.set_opacity(0.3), FadeIn(eig))
        self.play(*[Transform(e, image(a, c, 6)) for e, a, c in zip(eig, eig_angles, [ORANGE, GREEN])],
                  run_time=1.6)

        self.beat("31–32 min. La naranja y la verde conservan su dirección: son autovectores. El "
                  "factor de estiramiento es λ, y es justo la varianza proyectada que vimos en el "
                  "barrido. La otra dirección es perpendicular (S es simétrica).")
        info = VGroup(
            tex(rf"S\mathbf v_1={CLOUD_VALS[0]:.2f}\,\mathbf v_1", size=40, color=ORANGE),
            tex(rf"S\mathbf v_2={CLOUD_VALS[1]:.2f}\,\mathbf v_2", size=40, color=GREEN),
            tex(rf"\mathrm{{PVE}}_1=\frac{{{CLOUD_VALS[0]:.2f}}}{{{CLOUD_VALS[0]:.2f}+{CLOUD_VALS[1]:.2f}}}"
                rf"={100 * CLOUD_VALS[0] / CLOUD_VALS.sum():.1f}\%", size=38),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.6, -1.35, 0])
        for m in info:
            self.play(Write(m), run_time=0.8)

        self.beat("32–34 min. Volvemos a los dos sensores y hacemos el cálculo completo a mano. "
                  "Pida a la mitad del grupo calcular S y a la otra mitad el determinante.")
        self.start("Cálculo a mano", "Los dos sensores, paso a paso", BLUE)
        rows = [
            (r"S=\tfrac{1}{2}\tilde X^{T}\tilde X=\begin{pmatrix}1&1\\1&1\end{pmatrix}", INK,
             "covarianza (n − 1 = 2)"),
            (r"\det(S-\lambda I)=(1-\lambda)^2-1=\lambda^2-2\lambda=0", INK, "ecuación característica"),
            (r"\lambda_1=2,\qquad\lambda_2=0", YELLOW, "varianzas de PC1 y PC2"),
            (r"(S-2I)\mathbf v=\mathbf 0\;\Rightarrow\;\mathbf v_1=\tfrac{1}{\sqrt2}(1,\,1)", ORANGE,
             "dirección normalizada"),
            (r"\mathrm{PVE}_1=\frac{2}{2+0}=100\%,\qquad z=(-1.41,\;0,\;1.41)", GREEN,
             "información conservada y puntuaciones"),
        ]
        group = VGroup()
        for formula, color, why in rows:
            f = tex(formula, size=36, color=color, max_width=8.6)
            w = safe_text(why, size=19, color=MUTED)
            group.add(VGroup(f, w))
        for f, w in group:
            w.next_to(f, RIGHT, buff=0.5)
        group.arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to([0.1, -0.5, 0])
        for g in group[:3]:
            self.play(Write(g[0]), FadeIn(g[1]), run_time=0.9)
        self.beat("34–35 min. Complete autovector, varianza explicada y puntuaciones. Verifique "
                  "que coinciden con la rotación geométrica del acto anterior.")
        for g in group[3:]:
            self.play(Write(g[0]), FadeIn(g[1]), run_time=0.9)

        self.beat("35–36 min. Receta general. Pregunte qué paso depende de las unidades (el 1). "
                  "Aclare: la biblioteca usa SVD de X̃, que da lo mismo con mejor estabilidad.")
        self.start("Receta", "PCA en cinco pasos", GREEN)
        recipe = VGroup(*[
            VGroup(pill(str(i), c, width=0.62, size=24),
                   tex(f, size=36, max_width=9.2), safe_text(t, size=20, color=MUTED))
            for i, (f, t, c) in enumerate([
                (r"\tilde X_{ij}=\frac{X_{ij}-\bar x_j}{s_j}", "centrar (y escalar si las unidades difieren)", YELLOW),
                (r"S=\frac{1}{n-1}\tilde X^{T}\tilde X", "matriz de covarianza (p × p)", BLUE),
                (r"S\mathbf v_k=\lambda_k\mathbf v_k", "direcciones y varianzas", ORANGE),
                (r"\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_p,\quad\mathrm{PVE}_k=\lambda_k/\textstyle\sum_j\lambda_j",
                 "ordenar y elegir k", GREEN),
                (r"Z=\tilde X V_k", "puntuaciones: n × k", PURPLE),
            ], start=1)
        ])
        for r in recipe:
            r.arrange(RIGHT, buff=0.35)
        recipe.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -0.5, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in recipe], lag_ratio=0.2), run_time=1.6)
