"""Acto 1 · Portada, ruta y la neurona (0–12 min).

De la neurona biológica a la artificial; una neurona calculada a mano; las
funciones de activación; y la lectura geométrica: una neurona es un hiperplano
w·x + b = 0 y una decisión sobre de qué lado cae cada punto.
"""
from __future__ import annotations

import numpy as np
from manim import *

from sections.data import NEURON_B, NEURON_RELU, NEURON_SIG, NEURON_W, NEURON_X, NEURON_Z
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE, RED,
                            TEAL, YELLOW, card, marker, pill, safe_text, tex)


def node(label: str, color=TEAL, r=0.36, size=26) -> VGroup:
    c = Circle(radius=r, stroke_color=color, stroke_width=3, fill_color=PANEL, fill_opacity=1)
    t = MathTex(label, font_size=size * 1.25, color=INK) if label else VMobject()
    return VGroup(c, t.move_to(c))


def small_graph(fn, title: str, color, *, y_range=(-1.2, 1.2), center=ORIGIN) -> VGroup:
    ax = Axes(x_range=[-3, 3, 1], y_range=[*y_range, 1], x_length=2.7, y_length=1.9,
              tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                       "include_ticks": False})
    curve = ax.plot(fn, x_range=[-3, 3, 0.01], color=color, stroke_width=5,
                    use_smoothing=False)
    t = safe_text(title, size=22, color=color, weight=BOLD).next_to(ax, UP, buff=0.15)
    return VGroup(ax, curve, t).move_to(center)


class NeuronSection:
    def opening(self) -> None:
        self.beat("0–1 min. Portada. Pregunte: ¿quién ha usado hoy algo que funcione con una red "
                  "neuronal? (filtro de fotos, traductor, desbloqueo facial). Hoy abrimos la caja: "
                  "veremos que una red es, geométricamente, una forma muy inteligente de cortar "
                  "el espacio con rectas.")
        eyebrow = safe_text("CLASIFICACIÓN · REDES NEURONALES", size=22, color=TEAL, weight=BOLD)
        title = safe_text("Cortar el plano con muchas rectas", size=56, weight=BOLD,
                          max_width=12.6)
        subtitle = safe_text("Neuronas, ReLU, hiperplanos, retropropagación y el camino a las CNN",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Ruta. El hilo conductor es UN mapa: el pueblo de Baarle, donde la "
                  "frontera entre Bélgica y Países Bajos está partida en pedazos. Ningún "
                  "clasificador lineal puede con él; veremos por qué una red sí.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "De una neurona a una red profunda", YELLOW)
        route = VGroup(*[pill(t, c, size=23, width=6.0) for t, c in [
            ("1 · ¿Qué es una neurona?", TEAL),
            ("2 · Baarle: una frontera imposible", BLUE),
            ("3 · ReLU: doblar el plano", PURPLE),
            ("4 · Una red parte el mapa en polígonos", YELLOW),
            ("5 · ¿Cómo aprende? Retropropagación", RED),
            ("6 · De redes densas a CNN", PINK),
            ("7 · Cierre", GREEN)]]).arrange(DOWN, buff=0.17)
        route.move_to([-3.3, -0.45, 0])
        promise = VGroup(
            safe_text("Al terminar sabrás", size=26, color=MUTED),
            safe_text("· calcular la salida de una neurona", size=23),
            safe_text("· por qué sin no linealidad no hay red", size=23),
            safe_text("· leer una red ReLU como hiperplanos", size=23),
            safe_text("· la idea de la retropropagación", size=23),
            safe_text("· qué cambia en una CNN", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([3.4, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in route], lag_ratio=0.1))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in promise], lag_ratio=0.12))

        self.biological()
        self.artificial()
        self.activations()
        self.hyperplane()

    # ------------------------------------------------------------------
    def biological(self) -> None:
        self.beat("2–4 min. Una neurona biológica, en caricatura: las dendritas reciben señales "
                  "de otras neuronas; unas sinapsis son fuertes y otras débiles (o inhibidoras); "
                  "el cuerpo celular las acumula y, si superan un umbral, dispara un pulso por el "
                  "axón. No vamos a hacer biología: sólo nos quedamos con esas tres palabras.")
        self.start("¿Qué es una neurona?", "Recibe, pondera y decide si dispara", TEAL)
        soma = Circle(radius=0.75, stroke_color=TEAL, stroke_width=4, fill_color=TEAL,
                      fill_opacity=0.18).move_to([-2.2, -0.5, 0])
        nucleus = Dot(soma.get_center(), radius=0.18, color=TEAL)
        dendrites = VGroup()
        for ang, ln in [(150, 2.2), (180, 2.4), (210, 2.1), (125, 1.8), (235, 1.8)]:
            a = np.deg2rad(ang)
            start = soma.get_center() + 0.75 * np.array([np.cos(a), np.sin(a), 0])
            end = soma.get_center() + ln * np.array([np.cos(a), np.sin(a), 0])
            br = VGroup(Line(start, end, color=TEAL, stroke_width=4))
            for da in (-25, 25):
                b = np.deg2rad(ang + da)
                br.add(Line(end, end + 0.45 * np.array([np.cos(b), np.sin(b), 0]),
                            color=TEAL, stroke_width=3))
            dendrites.add(br)
        axon = Line(soma.get_right(), soma.get_right() + RIGHT * 4.2, color=TEAL, stroke_width=6)
        terminals = VGroup(*[Line(axon.get_end(), axon.get_end() + 0.55 * np.array(
            [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0]), color=TEAL, stroke_width=3)
            for a in (-35, 0, 35)])
        labels = VGroup(
            safe_text("dendritas: reciben", size=22, color=MUTED).move_to([-5.0, 1.55, 0]),
            safe_text("cuerpo: acumula", size=22, color=MUTED).next_to(soma, DR, buff=0.15),
            safe_text("axón: dispara (o no)", size=22, color=MUTED).next_to(axon, UP, buff=0.25),
        )
        self.play(GrowFromCenter(soma), FadeIn(nucleus))
        self.play(LaggedStart(*[Create(d) for d in dendrites], lag_ratio=0.12),
                  Create(axon), Create(terminals), FadeIn(labels))
        pulses = VGroup(*[Dot(d[0].get_end(), radius=0.09, color=YELLOW) for d in dendrites])
        self.play(LaggedStart(*[p.animate.move_to(soma.get_center()) for p in pulses],
                              lag_ratio=0.15), run_time=1.4)
        spike = Dot(soma.get_right(), radius=0.12, color=YELLOW)
        self.play(FadeOut(pulses), soma.animate.set_fill(YELLOW, 0.35), FadeIn(spike))
        self.play(spike.animate.move_to(axon.get_end()), run_time=1.0)
        self.play(FadeOut(spike), soma.animate.set_fill(TEAL, 0.18))
        words = VGroup(pill("recibe", TEAL), pill("pondera", YELLOW), pill("decide", PURPLE)
                       ).arrange(RIGHT, buff=0.35).move_to([0, -2.85, 0])
        self.play(LaggedStart(*[FadeIn(w, shift=UP * 0.1) for w in words], lag_ratio=0.2))

    # ------------------------------------------------------------------
    def artificial(self) -> None:
        self.beat("4–6 min. La neurona artificial. Cada entrada se multiplica por su peso (la "
                  "fuerza de la sinapsis; puede ser negativo), se suma todo más un sesgo b (el "
                  "umbral, movido al otro lado) y el resultado z pasa por una función de "
                  "activación f. Pregunte antes de revelar: ¿cuánto vale z?")
        self.start("La neurona artificial", "Suma ponderada + activación", TEAL)
        xs = VGroup(*[node(f"x_{i + 1}", BLUE) for i in range(3)]).arrange(DOWN, buff=0.75)
        xs.move_to([-5.0, -0.4, 0])
        sigma = node(r"\Sigma", TEAL, r=0.55, size=34).move_to([-1.0, -0.4, 0])
        act = VGroup(RoundedRectangle(corner_radius=0.15, width=1.3, height=1.0,
                                      stroke_color=PURPLE, stroke_width=3, fill_color=PANEL,
                                      fill_opacity=1),
                     MathTex("f", font_size=44, color=PURPLE)).move_to([1.6, -0.4, 0])
        act[1].move_to(act[0])
        out = node(r"\hat y", YELLOW).move_to([4.0, -0.4, 0])
        arrows = VGroup(*[Arrow(x.get_right(), sigma.get_left(), buff=0.08, color=GRID,
                                stroke_width=4, max_tip_length_to_length_ratio=0.08) for x in xs])
        wl = VGroup(*[MathTex(f"w_{i + 1}", font_size=32, color=TEAL).move_to(
            a.point_from_proportion(0.42) + UP * 0.28) for i, a in enumerate(arrows)])
        bias = VGroup(MathTex("b", font_size=34, color=TEAL)).next_to(sigma, UP, buff=0.55)
        barr = Arrow(bias.get_bottom(), sigma.get_top(), buff=0.05, color=GRID, stroke_width=4)
        a2 = Arrow(sigma.get_right(), act.get_left(), buff=0.08, color=GRID, stroke_width=4)
        a3 = Arrow(act.get_right(), out.get_left(), buff=0.08, color=GRID, stroke_width=4)
        zl = MathTex("z", font_size=34, color=INK).next_to(a2, UP, buff=0.1)
        self.play(LaggedStart(*[FadeIn(x) for x in xs], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1), FadeIn(wl),
                  FadeIn(sigma))
        self.play(FadeIn(bias), GrowArrow(barr))
        self.play(GrowArrow(a2), FadeIn(zl), FadeIn(act), GrowArrow(a3), FadeIn(out))
        formula = tex(r"z = w_1x_1 + w_2x_2 + w_3x_3 + b = \mathbf{w}\cdot\mathbf{x} + b,"
                      r"\qquad \hat y = f(z)", size=38).move_to([0, -2.75, 0])
        self.play(Write(formula))

        self.beat("6–7 min. Con números: x = (2, 1, 3), w = (0.5, −1, 0.25), b = 0.5. "
                  "z = 1 − 1 + 0.75 + 0.5 = 1.25. Con ReLU la salida es 1.25; con la sigmoide, "
                  "0.78. Señale que el peso negativo «inhibe»: la entrada x₂ resta.")
        vals = VGroup(*[MathTex(f"{v:g}", font_size=34, color=BLUE).move_to(x) for v, x in
                        zip(NEURON_X, xs)])
        wv = VGroup(*[MathTex(f"{v:g}", font_size=32, color=TEAL if v >= 0 else RED).move_to(w)
                      for v, w in zip(NEURON_W, wl)])
        bv = MathTex(f"{NEURON_B:g}", font_size=34, color=TEAL).move_to(bias)
        self.play(*[Transform(x[1], v) for x, v in zip(xs, vals)],
                  *[Transform(w, v) for w, v in zip(wl, wv)], Transform(bias[0], bv))
        calc = tex(r"z = 0.5(2) - 1(1) + 0.25(3) + 0.5 = " + f"{NEURON_Z:.2f}", size=38,
                   color=INK).move_to([0, -2.75, 0])
        self.play(ReplacementTransform(formula, calc))
        zv = MathTex(f"{NEURON_Z:.2f}", font_size=30, color=YELLOW).move_to(zl)
        self.play(Transform(zl, zv))
        res = VGroup(
            tex(r"\text{ReLU}: \max(0, 1.25) = " + f"{NEURON_RELU:.2f}", size=32, color=PURPLE),
            tex(r"\sigma(1.25) = \frac{1}{1+e^{-1.25}} = " + f"{NEURON_SIG:.2f}", size=32,
                color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.4, 1.6, 0])
        self.play(FadeIn(res, shift=LEFT * 0.1), out[0].animate.set_fill(YELLOW, 0.3))

    # ------------------------------------------------------------------
    def activations(self) -> None:
        self.beat("7–8 min. Cuatro activaciones. El escalón es el perceptrón original (1958): "
                  "no tiene derivada útil. La sigmoide y tanh son suaves pero se aplanan en los "
                  "extremos (gradiente casi cero). ReLU = max(0, z) es la más simple y la que "
                  "domina hoy: es lineal a trozos. Esa palabra, «a trozos», es la clave de hoy.")
        self.start("Funciones de activación", "La f que decide «cuánto dispara»", PURPLE)
        graphs = VGroup(
            small_graph(lambda z: 1.0 if z >= 0 else 0.0, "escalón", MUTED, y_range=(-0.3, 1.3)),
            small_graph(lambda z: 1 / (1 + np.exp(-2 * z)), "sigmoide", BLUE,
                        y_range=(-0.3, 1.3)),
            small_graph(np.tanh, "tanh", TEAL),
            small_graph(lambda z: max(0.0, z), "ReLU", PURPLE, y_range=(-0.5, 3.1)),
        ).arrange(RIGHT, buff=0.55).move_to([0, 0.1, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in graphs], lag_ratio=0.2))
        formulas = VGroup(
            tex(r"\mathbb{1}[z \ge 0]", size=30, color=MUTED),
            tex(r"1/(1+e^{-z})", size=30, color=BLUE),
            tex(r"\tanh z", size=30, color=TEAL),
            tex(r"\max(0, z)", size=30, color=PURPLE),
        )
        for f, g in zip(formulas, graphs):
            f.next_to(g, DOWN, buff=0.25)
        self.play(FadeIn(formulas))
        box = SurroundingRectangle(VGroup(graphs[3], formulas[3]), color=PURPLE, buff=0.18,
                                   corner_radius=0.12)
        msg = safe_text("ReLU: dos rectas pegadas en z = 0 → lineal a trozos", size=26,
                        color=PURPLE).move_to([0, -2.85, 0])
        self.play(Create(box), FadeIn(msg, shift=UP * 0.1))

    # ------------------------------------------------------------------
    def hyperplane(self) -> None:
        self.beat("8–11 min. La lectura geométrica, que usaremos toda la clase. Con dos "
                  "entradas, z = w₁x₁ + w₂x₂ + b es un plano inclinado; donde z = 0 hay una "
                  "RECTA. El vector w es perpendicular a ella y apunta hacia donde z crece. Una "
                  "neurona sólo pregunta: ¿de qué lado de la recta estás y qué tan lejos?")
        self.start("Una neurona es un hiperplano", "w·x + b = 0 parte el plano en dos", YELLOW)
        ax = NumberPlane(x_range=[-4, 4, 1], y_range=[-3, 3, 1], x_length=6.4, y_length=4.8,
                         background_line_style={"stroke_color": GRID, "stroke_width": 1,
                                                "stroke_opacity": 0.5},
                         axis_config={"stroke_color": GRID})
        ax.move_to([-2.6, -0.55, 0])
        theta = ValueTracker(np.deg2rad(35))
        bias = ValueTracker(0.6)

        def wvec():
            return np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])

        box = [(-4, -3), (4, -3), (4, 3), (-4, 3)]

        def half(sign, color):
            def make():
                w, b = wvec(), bias.get_value()
                pts = clip_halfplane(box, w, b, sign)
                if len(pts) < 3:
                    return VMobject()
                return Polygon(*[ax.c2p(*p) for p in pts], stroke_width=0, fill_color=color,
                               fill_opacity=0.2)
            return make

        def line():
            w, b = wvec(), bias.get_value()
            c = -b * w
            d = np.array([-w[1], w[0]])
            pts = clip_segment(c - 9 * d, c + 9 * d, (-4, -3, 4, 3))
            if not pts:
                return VMobject()
            return Line(ax.c2p(*pts[0]), ax.c2p(*pts[1]), color=YELLOW, stroke_width=5)

        def arrow():
            w, b = wvec(), bias.get_value()
            c = -b * w
            return Arrow(ax.c2p(*c), ax.c2p(*(c + 1.3 * w)), buff=0, color=TEAL,
                         stroke_width=6, max_tip_length_to_length_ratio=0.25)

        pos = always_redraw(half(+1, ORANGE))
        neg = always_redraw(half(-1, BLUE))
        ln = always_redraw(line)
        ar = always_redraw(arrow)
        wl = always_redraw(lambda: MathTex(r"\mathbf{w}", font_size=36, color=TEAL).next_to(
            ar.get_end(), UR, buff=0.05))
        rng = np.random.default_rng(2)
        pts = rng.uniform([-3.6, -2.6], [3.6, 2.6], (26, 2))
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.06, color=INK) for p in pts])

        def color_dots(m):
            w, b = wvec(), bias.get_value()
            for d, p in zip(m, pts):
                d.set_color(ORANGE if p @ w + b > 0 else BLUE)

        self.play(Create(ax))
        self.play(FadeIn(neg), FadeIn(pos), Create(ln), GrowArrow(ar), FadeIn(wl))
        dots.add_updater(color_dots)
        self.play(FadeIn(dots))
        eqs = VGroup(
            tex(r"z = \mathbf{w}\cdot\mathbf{x} + b", size=38),
            VGroup(tex(r"z > 0", size=32, color=ORANGE), safe_text("un lado", size=22,
                                                                   color=MUTED)).arrange(
                RIGHT, buff=0.25),
            VGroup(tex(r"z = 0", size=32, color=YELLOW), safe_text("la frontera", size=22,
                                                                   color=MUTED)).arrange(
                RIGHT, buff=0.25),
            VGroup(tex(r"z < 0", size=32, color=BLUE), safe_text("el otro lado", size=22,
                                                                 color=MUTED)).arrange(
                RIGHT, buff=0.25),
            safe_text("|z| / ‖w‖ = distancia a la recta", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.9, 0.3, 0])
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.1) for e in eqs], lag_ratio=0.15))

        self.beat("11 min. Mueva los parámetros: cambiar w GIRA la recta; cambiar b la DESLIZA "
                  "sin girarla. Con d entradas lo mismo ocurre en d dimensiones: un hiperplano. "
                  "Conclusión incómoda: una neurona sola sólo sabe trazar fronteras RECTAS.")
        self.play(theta.animate.set_value(np.deg2rad(120)), run_time=2.0)
        self.play(bias.animate.set_value(-1.2), run_time=1.5)
        self.play(theta.animate.set_value(np.deg2rad(70)), bias.animate.set_value(0.3),
                  run_time=1.5)
        verdict = VGroup(
            safe_text("girar w  ·  deslizar b", size=24, color=TEAL),
            safe_text("Una neurona = una frontera recta", size=28, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, buff=0.15)
        verdict.move_to([3.9, -2.75, 0])
        self.play(FadeIn(verdict, shift=UP * 0.1))
        dots.clear_updaters()
        for m in (pos, neg, ln, ar, wl):
            m.clear_updaters()


def clip_segment(a, b, box):
    """Recorta el segmento ab (2D) a la caja (x0, y0, x1, y1) (Liang–Barsky)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x0, y0, x1, y1 = box
    dx, dy = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return []
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return []
    return [a + t0 * (b - a), a + t1 * (b - a)]


def clip_halfplane(poly, w, b, sign):
    """Sutherland–Hodgman: parte de ``poly`` donde sign·(w·x + b) ≥ 0."""
    out = []
    n = len(poly)
    f = [sign * (w[0] * p[0] + w[1] * p[1] + b) for p in poly]
    for i in range(n):
        p, q = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % n], float)
        fp, fq = f[i], f[(i + 1) % n]
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + fp / (fp - fq) * (q - p))
    return out
