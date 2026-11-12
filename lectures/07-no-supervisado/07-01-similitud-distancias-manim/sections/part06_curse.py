"""Acto 6 · La maldición de la dimensión: la esfera (37–45 min).

Círculo en el cuadrado (78.5 %) → esfera en el cubo (52.4 %) → la fracción
se desploma con p porque el volumen se va a las 2^p esquinas. Dentro de la
propia esfera, casi todo el volumen está en la cáscara. Consecuencia: las
distancias se concentran y "el más cercano" deja de significar algo.
"""
from __future__ import annotations

from manim import *

from sections.data import (BALL_DIMS, BALL_FRAC, CONC_RATIO, HIST, HIST_BINS, HIST_DIMS,
                           MC_INSIDE, MC_SQUARE, SHELL_DIMS, SHELL_FRAC)
from sections.theme import (BG, PANEL, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, TEAL,
                            YELLOW, card, safe_text, tex)


def pct(x: float) -> str:
    if x >= 0.1:
        return f"{100 * x:.1f} %"
    if x >= 1e-3:
        return f"{100 * x:.2f} %"
    return f"{100 * x:.1e} %".replace("e-0", "e-")


class CurseSection:
    def curse(self) -> None:
        self.beat("37–38 min. Un experimento mental. Lanzamos dardos al azar sobre un cuadrado con "
                  "un círculo inscrito. ¿Qué fracción cae dentro? Que lo estimen antes de ver los "
                  "dardos.")
        head = self.start("La maldición de la dimensión", "Dardos sobre un cuadrado", RED)
        U = 2.3
        sq = Square(side_length=2 * U, color=MUTED, stroke_width=3).move_to([-3.0, -0.55, 0])
        c0 = sq.get_center()
        circ = Circle(radius=U, color=TEAL, stroke_width=4).move_to(c0)
        self.play(Create(sq), Create(circ))
        darts = VGroup(*[Dot(c0 + U * np.array([x, y, 0]), radius=0.035,
                             color=TEAL if inside else RED)
                         for (x, y), inside in zip(MC_SQUARE, MC_INSIDE)])
        self.play(LaggedStart(*[FadeIn(d, scale=2) for d in darts], lag_ratio=0.004), run_time=2.2)
        n_in = int(MC_INSIDE.sum())
        res = VGroup(
            tex(rf"\frac{{{n_in}}}{{{len(MC_INSIDE)}}} = {100 * n_in / len(MC_INSIDE):.0f}\,\%",
                size=44, color=TEAL),
            tex(r"\text{exacto: } \frac{\pi}{4} = 78.5\,\%", size=38, color=INK),
            safe_text("las esquinas rojas: 21.5 %", size=24, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.3, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in res], lag_ratio=0.35))

        self.beat("38–40 min. En 3D: una esfera dentro de un cubo. Pregunte: ¿la fracción sube o "
                  "baja? Ahora hay 8 esquinas y la esfera ocupa 52.4 %. Gire la escena y señale "
                  "cuánto espacio queda entre la esfera y cada vértice.")
        head = self.start("La maldición de la dimensión", "Una esfera dentro de un cubo", RED,
                          fixed=True)
        self.move_camera(phi=65 * DEGREES, theta=-40 * DEGREES, run_time=1.0)
        R = 1.55
        cube = Cube(side_length=2 * R, fill_opacity=0.04, fill_color=MUTED, stroke_color=MUTED,
                    stroke_width=2)
        sphere = Sphere(radius=R, resolution=(20, 40), fill_opacity=0.45, stroke_width=0.3,
                        checkerboard_colors=[TEAL, TEAL]).set_stroke(TEAL, opacity=0.4)
        corners = VGroup(*[Dot3D(R * np.array([sx, sy, sz]), radius=0.11, color=RED,
                                 resolution=(8, 8))
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        self.play(Create(cube), run_time=0.8)
        self.play(FadeIn(sphere), run_time=1.0)
        self.play(LaggedStart(*[GrowFromCenter(c) for c in corners], lag_ratio=0.08))
        info = VGroup(tex(r"\frac{\tfrac{4}{3}\pi r^3}{(2r)^3} = \frac{\pi}{6} = 52.4\,\%", size=38,
                          color=TEAL),
                      safe_text("8 esquinas vacías", size=24, color=RED, weight=BOLD)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        ibg = card(info.width + 0.5, info.height + 0.45, TEAL)
        iblock = VGroup(ibg, info.move_to(ibg)).to_corner(DR, buff=0.4).shift(UP * 0.55)
        self.fixed(iblock)
        self.play(FadeIn(iblock))
        self.next_slide(notes="Giro completo: la esfera parece llenar el cubo, pero cada vértice "
                              "está lejísimos de ella.", loop=True)
        self.move_camera(theta=-40 * DEGREES + TAU, run_time=8, rate_func=linear)
        self.set_camera_orientation(theta=-40 * DEGREES)

        self.beat("40–41 min. Sigamos sin poder dibujar: en p dimensiones el cubo tiene 2^p "
                  "esquinas y la bola ocupa una fracción que se desploma. En 10 dimensiones, 0.25 %; "
                  "en 20, dos cienmillonésimas. Casi todo el volumen del cubo está en las "
                  "esquinas, lejos del centro. Muestre la fórmula sólo después de las barras.")
        head = self.start("La maldición de la dimensión", "¿Cuánto del cubo ocupa la bola?", RED)
        ax = Axes(x_range=[0, len(BALL_DIMS), 1], y_range=[0, 1.05, 0.25], x_length=7.2,
                  y_length=3.9, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 1.6},
                  y_axis_config={"include_numbers": False}).move_to([-2.5, -0.45, 0])
        bars = VGroup()
        for k, p in enumerate(BALL_DIMS):
            h = BALL_FRAC[p]
            bar = Rectangle(width=0.78, height=max(h * 3.9 * (1 / 1.05), 0.012), stroke_width=0,
                            fill_color=TEAL if p <= 3 else RED, fill_opacity=0.9)
            bar.move_to(ax.c2p(k + 0.5, 0), aligned_edge=DOWN)
            lab = safe_text(f"p={p}", size=22, color=MUTED).next_to(ax.c2p(k + 0.5, 0), DOWN, buff=0.2)
            val = safe_text(pct(h), size=19, color=INK).next_to(bar, UP, buff=0.1)
            bars.add(VGroup(bar, lab, val))
        self.play(Create(ax))
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(b[0], DOWN), FadeIn(b[1]), FadeIn(b[2]))
                                for b in bars], lag_ratio=0.3), run_time=2.6)
        vol = VGroup(
            tex(r"V_p(r) = \frac{\pi^{p/2}}{\Gamma\!\left(\frac{p}{2}+1\right)}\,r^p", size=40,
                color=INK),
            tex(r"\frac{V_p(1)}{2^p} \xrightarrow[p\to\infty]{} 0", size=40, color=RED),
            safe_text("2^p esquinas se llevan el volumen", size=22, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.95, -0.25, 0])
        self.play(LaggedStart(*[FadeIn(v, shift=LEFT * 0.2) for v in vol], lag_ratio=0.35))

        self.beat("41–42 min. Segundo efecto, dentro de la propia bola. ¿Qué fracción del volumen "
                  "está en la cáscara exterior, del 95 % al 100 % del radio? En 2D, menos del 10 %; "
                  "en 50 dimensiones, 92 %; en 100, 99.4 %. Es como una naranja que es casi pura "
                  "cáscara: en alta dimensión todos los puntos están 'en el borde'.")
        head = self.swap_heading(head, "La maldición de la dimensión", "Casi todo es cáscara", RED)
        self.clear_except(head)
        c0 = np.array([-4.2, -0.55, 0])
        outer = Circle(radius=2.0, stroke_width=0, fill_color=ORANGE, fill_opacity=0.85).move_to(c0)
        inner = Circle(radius=1.9, stroke_width=0, fill_color=PANEL, fill_opacity=1).move_to(c0)
        core_l = safe_text("interior: 95 % del radio", size=20, color=MUTED).move_to(c0)
        shell_l = safe_text("cáscara: 5 % del radio", size=22, color=ORANGE).next_to(outer, DOWN,
                                                                                   buff=0.2)
        self.play(FadeIn(outer), FadeIn(inner), FadeIn(shell_l), FadeIn(core_l))
        rows = VGroup()
        for p in SHELL_DIMS:
            frac = SHELL_FRAC[p]
            lab = safe_text(f"p = {p}", size=24, color=MUTED)
            track = Rectangle(width=5.0, height=0.36, stroke_color=GRID, stroke_width=1.5)
            fill = Rectangle(width=max(5.0 * frac, 0.01), height=0.36, stroke_width=0,
                             fill_color=ORANGE, fill_opacity=0.9).align_to(track, LEFT)
            val = safe_text(f"{100 * frac:.1f} %", size=22, color=INK)
            y = 0.6 - 0.62 * len(rows)
            lab.move_to([-1.1, y, 0], aligned_edge=LEFT)
            track.move_to([0.4, y, 0], aligned_edge=LEFT)
            fill.move_to([0.4, y, 0], aligned_edge=LEFT)
            val.next_to(track, RIGHT, buff=0.25)
            rows.add(VGroup(lab, VGroup(track, fill), val))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], *[Create(r[1][0]) for r in rows],
                              lag_ratio=0.05))
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(r[1][1], LEFT), FadeIn(r[2]))
                                for r in rows], lag_ratio=0.3), run_time=2.4)
        shell_f = tex(r"1 - 0.95^{\,p}", size=40, color=ORANGE).next_to(rows, UP, buff=0.35)
        self.play(Write(shell_f))

        self.beat("42–44 min. Consecuencia para nosotros: las distancias se CONCENTRAN. Distancias "
                  "de un punto a 500 puntos al azar, divididas entre su promedio. En 2D hay "
                  "cercanos y lejanos; en 1000 dimensiones todos están casi a la misma distancia. "
                  "El cociente (d_max − d_min)/d_min pasa de 82 a 0.12: el más lejano está apenas "
                  "12 % más lejos que el más cercano.")
        head = self.swap_heading(head, "Concentración de distancias",
                                 "Todos quedan a la misma distancia", RED)
        self.clear_except(head)
        ax = Axes(x_range=[0, 2, 0.5], y_range=[0, 0.55, 0.1], x_length=7.4, y_length=3.7,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 1.6,
                                           "include_numbers": True, "font_size": 22}
                  ).move_to([-2.2, -0.55, 0])
        xl = tex(r"d \,/\, \bar d", size=30, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(xl))
        width = HIST_BINS[1] - HIST_BINS[0]

        def hist(p):
            g = VGroup()
            for k, h in enumerate(HIST[p]):
                if h <= 0:
                    continue
                x0 = HIST_BINS[k]
                bl, tr = ax.c2p(x0, 0), ax.c2p(x0 + width, min(h, 0.55))
                g.add(Rectangle(width=tr[0] - bl[0], height=tr[1] - bl[1], stroke_width=0.8,
                                stroke_color=BG, fill_color=PINK, fill_opacity=0.85)
                      .move_to((bl + tr) / 2))
            return g

        def ratio_label(p):
            return VGroup(tex(rf"p = {p}", size=40, color=PINK),
                          tex(rf"\frac{{d_{{\max}}-d_{{\min}}}}{{d_{{\min}}}} = {CONC_RATIO[p]:.2f}",
                              size=38, color=INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.3
                                                           ).move_to([4.1, 0.4, 0])

        h = hist(HIST_DIMS[0])
        r = ratio_label(HIST_DIMS[0])
        self.play(FadeIn(h), FadeIn(r))
        for p in HIST_DIMS[1:]:
            self.wait(0.6)
            self.play(Transform(h, hist(p)), Transform(r, ratio_label(p)), run_time=1.4)

        self.beat("44–45 min. Formalización (Beyer et al., 1999): si la dispersión relativa de las "
                  "distancias tiende a cero, el cociente entre el vecino más lejano y el más "
                  "cercano tiende a 1. Remedios: elegir variables, reducir dimensión y, sobre todo, "
                  "notar que los datos reales NO llenan el cubo: viven cerca de una superficie de "
                  "pocas dimensiones. Ahí entra UMAP.")
        head = self.swap_heading(head, "Formalización", "Cuando el más cercano deja de serlo", RED)
        self.clear_except(head)
        beyer = VGroup(
            tex(r"\frac{\operatorname{Var}(D_p)}{\mathbb{E}[D_p]^2} \xrightarrow[p\to\infty]{} 0"
                r"\quad\Longrightarrow\quad \frac{D_{\max}}{D_{\min}} \xrightarrow{\;P\;} 1",
                size=44, color=INK),
            safe_text("Beyer, Goldstein, Ramakrishnan y Shaft (1999)", size=20, color=MUTED),
        ).arrange(DOWN, buff=0.25).move_to([0, 1.1, 0])
        self.play(Write(beyer[0]), FadeIn(beyer[1]))
        cons = [("seleccionar variables con sentido", GREEN), ("reducir dimensión (PCA)", GREEN),
                ("confiar sólo en vecindarios locales", PINK)]
        chips = VGroup(*[VGroup(card(3.9, 1.0, col), safe_text(t, size=22, color=col, weight=BOLD,
                                                                max_width=3.6))
                         for t, col in cons])
        for chip in chips:
            chip[1].move_to(chip[0])
        chips.arrange(RIGHT, buff=0.3).move_to([0, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.3))
        bridge = safe_text("Los datos reales no llenan el cubo: viven sobre una superficie",
                           size=26, color=PINK, weight=BOLD, max_width=12.4).move_to([0, -2.4, 0])
        self.play(Write(bridge))
