"""Acto 5 · Cambiar el punto de vista en 3D (31–37 min).

Tres documentos como vectores de conteos: la euclidiana confunde longitud con
tema. Proyectarlos sobre la esfera unitaria (mirar sólo la dirección) da el
coseno; 1 − cos no es métrica, el ángulo sí. Después, dos grupos que se
enciman vistos desde arriba y se separan al girar: proyectar nunca aleja,
sólo crea vecinos falsos.
"""
from __future__ import annotations

from manim import *

from sections.data import (ANGLE_13, COS_D13, COS_TRI, DOCS, EUC_12, EUC_13, PROJ_A, PROJ_B)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL,
                            YELLOW, card, safe_text, tex)

SC = 0.6
PHI0, THETA0 = 68 * DEGREES, -38 * DEGREES


def slerp_arc(u: np.ndarray, v: np.ndarray, radius: float, color: str) -> ParametricFunction:
    u, v = normalize(u), normalize(v)
    ang = np.arccos(np.clip(u @ v, -1, 1))

    def f(t):
        return radius * (np.sin((1 - t) * ang) * u + np.sin(t * ang) * v) / np.sin(ang)

    return ParametricFunction(f, t_range=[0, 1, 0.02], color=color, stroke_width=6)


class Views3DSection:
    def views_3d(self) -> None:
        self.beat("31–32 min. Tres textos contados por palabras (datos, minería, fútbol). d1 = "
                  "(3, 2, 0); d2 = (6, 4, 0) es d1 escrito dos veces; d3 = (0, 1, 3) habla de "
                  "fútbol. Pregunte: ¿qué par debería estar más cerca?")
        head = self.start("Cambiar el punto de vista", "Tres documentos en el espacio de palabras",
                          PURPLE, fixed=True)
        self.move_camera(phi=PHI0, theta=THETA0, zoom=1.0, run_time=1.2)
        axes = ThreeDAxes(x_range=[0, 7, 1], y_range=[0, 5, 1], z_range=[0, 4, 1],
                          x_length=7 * SC, y_length=5 * SC, z_length=4 * SC, tips=False,
                          axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False})
        axes.shift(-axes.c2p(0, 0, 0))
        names = VGroup(safe_text("datos", size=24, color=MUTED).move_to(axes.c2p(7.7, 0, -0.4)),
                       safe_text("minería", size=24, color=MUTED).move_to(axes.c2p(0, 5.8, 0)),
                       safe_text("fútbol", size=24, color=MUTED).move_to(axes.c2p(-1.3, 0, 3.7)))
        self.oriented(*names)
        cols = {"d1": BLUE, "d2": TEAL, "d3": ORANGE}
        arrows = {k: Arrow3D(ORIGIN, axes.c2p(*v), color=cols[k], thickness=0.025, height=0.22,
                             base_radius=0.07) for k, v in DOCS.items()}
        offs = {"d1": IN * 0.35, "d2": OUT * 0.35, "d3": RIGHT * 0.45}
        labels = {k: safe_text(k, size=28, color=cols[k], weight=BOLD)
                  .move_to(axes.c2p(*DOCS[k]) + offs[k]) for k in DOCS}
        self.oriented(*labels.values())
        self.play(Create(axes), FadeIn(names))
        self.play(*[Create(a) for a in arrows.values()], *[FadeIn(l) for l in labels.values()],
                  run_time=1.4)
        tips = {k: axes.c2p(*v) for k, v in DOCS.items()}
        l12 = DashedLine(tips["d1"], tips["d2"], color=RED, stroke_width=4)
        l13 = DashedLine(tips["d1"], tips["d3"], color=RED, stroke_width=4)
        panel = VGroup(
            safe_text("euclidiana", size=24, color=RED, weight=BOLD),
            tex(rf"d(d_1, d_2) = {EUC_12:.2f}", size=34, color=INK),
            tex(rf"d(d_1, d_3) = {EUC_13:.2f}", size=34, color=INK),
            safe_text("la copia está casi tan lejos", size=20, color=MUTED),
            safe_text("como el otro tema", size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        pbg = card(panel.width + 0.5, panel.height + 0.45, RED)
        pblock = VGroup(pbg, panel.move_to(pbg)).to_corner(DL, buff=0.4).shift(UP * 0.55)
        self.fixed(pblock)
        self.play(Create(l12), Create(l13), FadeIn(pblock))

        self.beat("32–34 min. Cambiemos la manera de mirar: nos quedamos sólo con la DIRECCIÓN de "
                  "cada vector, llevándolo a la esfera unitaria. d1 y d2 caen en el mismo punto: "
                  "son el mismo tema. El ángulo entre d1 y d3 es de 80°. Gire la escena con el "
                  "grupo y pida describir qué se perdió (la longitud del texto).")
        R = 2.1
        sphere = Sphere(radius=R, resolution=(18, 36), fill_opacity=0.10, stroke_width=0.4,
                        checkerboard_colors=[PURPLE, PURPLE]).set_stroke(PURPLE, opacity=0.25)
        new_arrows = {k: Arrow3D(ORIGIN, R * normalize(v), color=cols[k], thickness=0.025,
                                 height=0.22, base_radius=0.07) for k, v in DOCS.items()}
        self.play(FadeOut(l12), FadeOut(l13), FadeOut(pblock), FadeOut(names), FadeIn(sphere))
        self.play(*[Transform(arrows[k], new_arrows[k]) for k in DOCS],
                  *[labels[k].animate.move_to(R * normalize(DOCS[k]) + offs[k])
                    for k in DOCS], run_time=2.0)
        arc = slerp_arc(DOCS["d1"], DOCS["d3"], R, YELLOW)
        ang = tex(rf"\theta = {ANGLE_13:.0f}^\circ", size=36, color=YELLOW)
        ang.move_to(R * 1.3 * normalize(normalize(DOCS["d1"]) + normalize(DOCS["d3"])))
        self.oriented(ang)
        self.play(Create(arc), FadeIn(ang))
        panel2 = VGroup(
            safe_text("sólo la dirección", size=24, color=PURPLE, weight=BOLD),
            tex(r"d_{\cos}(d_1,d_2) = 0", size=34, color=INK),
            tex(rf"d_{{\cos}}(d_1,d_3) = {COS_D13:.3f}", size=34, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        pbg2 = card(panel2.width + 0.5, panel2.height + 0.45, PURPLE)
        pblock2 = VGroup(pbg2, panel2.move_to(pbg2)).to_corner(DR, buff=0.4).shift(UP * 0.55)
        self.fixed(pblock2)
        self.play(FadeIn(pblock2))
        self.next_slide(notes="Giro completo de la esfera. Deje que la vista haga el trabajo.",
                        loop=True)
        self.move_camera(theta=THETA0 + TAU, run_time=8, rate_func=linear)
        self.set_camera_orientation(theta=THETA0)

        self.beat("34–35 min. Formalización. El coseno compara direcciones. Sobre la esfera, la "
                  "distancia euclidiana entre vectores unitarios y el coseno están atados: "
                  "‖u − v‖² = 2(1 − cos θ). Pero 1 − cos NO es métrica: con tres direcciones a 0°, "
                  "45° y 90°, el rodeo cuesta 0.59 y el camino directo 1. El ángulo θ sí cumple "
                  "los cuatro axiomas.")
        head = self.start("Formalización", "Coseno: similitud, no métrica", PURPLE)
        f1 = tex(r"\cos\theta = \frac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\,"
                 r"\lVert\mathbf{y}\rVert}, \qquad d_{\cos} = 1-\cos\theta", size=40, color=INK)
        f2 = tex(r"\lVert \mathbf{u}-\mathbf{v} \rVert^2 = 2\,(1-\cos\theta)"
                 r"\quad (\lVert\mathbf{u}\rVert=\lVert\mathbf{v}\rVert=1)", size=36, color=PURPLE)
        VGroup(f1, f2).arrange(DOWN, buff=0.4).move_to([0, 1.2, 0])
        self.play(Write(f1))
        self.play(Write(f2))
        circ = Circle(radius=1.35, color=GRID, stroke_width=2).move_to([-4.0, -1.55, 0])
        c0 = circ.get_center()
        vecs = VGroup(*[Arrow(c0, c0 + 1.35 * np.array([np.cos(a), np.sin(a), 0]), buff=0,
                              color=col, stroke_width=5)
                        for a, col in ((0, BLUE), (PI / 4, GREEN), (PI / 2, ORANGE))])
        vl = VGroup(*[tex(n, size=32, color=col).next_to(v.get_end(), d, buff=0.08)
                      for n, col, v, d in (("x", BLUE, vecs[0], RIGHT), ("y", GREEN, vecs[1], UR),
                                           ("z", ORANGE, vecs[2], UP))])
        self.play(Create(circ), LaggedStart(*[GrowArrow(v) for v in vecs], lag_ratio=0.2),
                  FadeIn(vl))
        tri = VGroup(
            tex(rf"d_{{\cos}}(x,y) + d_{{\cos}}(y,z) = {2 * COS_TRI['xy']:.2f}", size=36,
                color=GREEN),
            tex(rf"d_{{\cos}}(x,z) = {COS_TRI['xz']:.0f} \;>\; {2 * COS_TRI['xy']:.2f}", size=36,
                color=RED),
            safe_text("el ángulo θ sí es una métrica", size=24, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([1.6, -1.5, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in tri], lag_ratio=0.4))

        # ── proyectar crea vecinos falsos ───────────────────────────────────
        self.beat("35–36 min. Ahora el caso contrario: mirar MENOS. Desde arriba vemos una sola "
                  "nube. Pregunte: ¿cuántos grupos hay? Casi todos dirán uno.")
        head = self.start("Mirar desde otro ángulo", "¿Cuántos grupos ves?", BLUE, fixed=True)
        dots_a = VGroup(*[Dot3D(p * 1.05, radius=0.07, color=BLUE, resolution=(6, 6)) for p in PROJ_A])
        dots_b = VGroup(*[Dot3D(p * 1.05, radius=0.07, color=BLUE, resolution=(6, 6)) for p in PROJ_B])
        axes = ThreeDAxes(x_range=[-2.5, 2.5, 1], y_range=[-2, 2, 1], z_range=[-2, 2, 1],
                          x_length=5, y_length=4, z_length=4, tips=False,
                          axis_config={"color": GRID, "stroke_width": 1.5, "include_ticks": False})
        self.play(Create(axes), FadeIn(dots_a), FadeIn(dots_b), run_time=1.0)

        self.beat("36–37 min. Giramos: eran dos grupos separados en la tercera dimensión. Lo que "
                  "vimos era una PROYECCIÓN, y proyectar nunca aleja: ‖Px − Py‖ ≤ ‖x − y‖. Al "
                  "reducir dimensión las distancias sólo pueden encogerse y aparecen vecinos "
                  "falsos. Ya lo vimos con la regla 'sólo edad'. Recuérdelo para UMAP.")
        self.move_camera(phi=78 * DEGREES, theta=-20 * DEGREES, run_time=2.4,
                         added_anims=[dots_b.animate.set_color(ORANGE)])
        law = VGroup(
            tex(r"\lVert P\mathbf{x}-P\mathbf{y}\rVert \le \lVert \mathbf{x}-\mathbf{y}\rVert",
                size=40, color=INK),
            safe_text("proyectar sólo junta: vecinos falsos", size=24, color=RED, weight=BOLD),
        ).arrange(DOWN, buff=0.2)
        lbg = card(law.width + 0.5, law.height + 0.45, RED)
        lblock = VGroup(lbg, law.move_to(lbg)).to_corner(DL, buff=0.4).shift(UP * 0.55)
        self.fixed(lblock)
        self.play(FadeIn(lblock))
        self.next_slide(notes="Giro lento para que se vea la separación desde varios ángulos.",
                        loop=True)
        self.move_camera(theta=-20 * DEGREES + TAU, run_time=8, rate_func=linear)
        self.set_camera_orientation(theta=-20 * DEGREES)
