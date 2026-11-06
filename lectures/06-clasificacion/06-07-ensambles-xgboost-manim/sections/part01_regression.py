"""Acto 1 · Del árbol al ensamble: árboles de regresión (0–9 min).

Seis laptops (RAM, precio). La predicción constante deja residuos (segmentos
verticales) cuya suma de cuadrados se ve como una barra; cada corte la encoge.
El árbol de regresión es una función escalonada. Después, un árbol profundo
sobre 40 puntos ruidosos memoriza el ruido: el problema que resuelven los
ensambles.
"""
from __future__ import annotations

from manim import *

from sections.data import (BAG_TR, LAPTOPS, PRICE, RAM, RAM_CUTS, SSE0, f_true, reg_steps,
                           reg_tree)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE, RED,
                            TEAL, YELLOW, card, pill, safe_text, tex)


def money(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def laptop_axes(center=(-2.6, -0.7, 0)) -> tuple[Axes, VGroup]:
    ax = Axes(x_range=[0, 18, 2], y_range=[0, 26000, 5000], x_length=6.6, y_length=4.0,
              tips=False, axis_config={"color": GRID, "stroke_width": 2, "include_ticks": True,
                                       "tick_size": 0.04})
    ax.move_to(center)
    labels = VGroup(
        *[safe_text(f"{v}", size=15, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
          for v in (4, 8, 12, 16)],
        *[safe_text(f"{v // 1000} mil", size=15, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1)
          for v in (5000, 10000, 15000, 20000, 25000)],
        safe_text("RAM (GB)", size=18, color=MUTED).next_to(ax, DOWN, buff=0.45),
        safe_text("precio ($)", size=18, color=MUTED).next_to(ax, UP, buff=0.1).align_to(ax, LEFT),
    )
    return ax, labels


def step_graph(ax: Axes, steps, color=YELLOW, width: float = 5) -> VGroup:
    """Función escalonada: un segmento horizontal por hoja y uniones verticales tenues."""
    g = VGroup()
    for i, (a, b, v) in enumerate(steps):
        g.add(Line(ax.c2p(a, v), ax.c2p(b, v), color=color, stroke_width=width))
        if i:
            g.add(Line(ax.c2p(a, steps[i - 1][2]), ax.c2p(a, v), color=color,
                       stroke_width=width * 0.35, stroke_opacity=0.6))
    return g


class RegressionSection:
    def opening(self) -> None:
        self.beat("0–1 min. Portada. La clase pasada terminó con una debilidad: quitar UN punto "
                  "cambiaba todo el árbol. Pregunte: si un solo experto es inestable, ¿qué hacen "
                  "en la vida real? (piden varias opiniones). Hoy convertimos esa intuición en "
                  "algoritmos.")
        eyebrow = safe_text("CLASIFICACIÓN · ENSAMBLES DE ÁRBOLES", size=22, color=TEAL,
                            weight=BOLD)
        title = safe_text("Muchos árboles piensan mejor que uno", size=56, weight=BOLD,
                          max_width=12.6)
        subtitle = safe_text("Bagging, bosques aleatorios, AdaBoost, gradient boosting y XGBoost",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Ruta: dos familias. Promediar árboles independientes (bagging, "
                  "bosques) reduce la VARIANZA; sumar árboles que corrigen a los anteriores "
                  "(boosting) reduce el SESGO. XGBoost es boosting con una función objetivo "
                  "regularizada de segundo orden. Ejercicio a mano al final.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "Dos maneras de combinar árboles", YELLOW)
        route = VGroup(*[pill(t, c, size=23, width=5.9) for t, c in [
            ("1 · Árboles que predicen números", TEAL),
            ("2 · La sabiduría de muchos", PURPLE),
            ("3 · Bagging y bosques aleatorios", GREEN),
            ("4 · AdaBoost: repesar los errores", ORANGE),
            ("5 · Gradient boosting: ajustar residuos", BLUE),
            ("6 · XGBoost: segundo orden y regularización", PINK),
            ("7 · Comparar, ejercicio y cierre", YELLOW)]]).arrange(DOWN, buff=0.17)
        route.move_to([-3.3, -0.45, 0])
        promise = VGroup(
            safe_text("Al terminar sabrás", size=26, color=MUTED),
            safe_text("· por qué promediar reduce la varianza", size=23),
            safe_text("· calcular una ronda de AdaBoost", size=23),
            safe_text("· hacer gradient boosting a mano", size=23),
            safe_text("· leer los pesos y la ganancia de XGBoost", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([3.45, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in route], lag_ratio=0.1))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in promise], lag_ratio=0.12))
        self.laptops()
        self.deep_tree()

    # ------------------------------------------------------------------
    def laptops(self) -> None:
        self.beat("2–4 min. Los árboles también predicen números. Seis laptops: RAM y precio. "
                  "Sin preguntar nada, la mejor constante es el promedio, 13 533. Los segmentos "
                  "rojos son los errores; la barra de la derecha es su suma de cuadrados (SSE). "
                  "Pregunte: ¿dónde cortarían la RAM para que la barra se encoja más?")
        self.start("Árboles de regresión", "En cada hoja: el promedio", TEAL)
        ax, labels = laptop_axes()
        dots = VGroup()
        for name, r, p in LAPTOPS:
            d = Dot(ax.c2p(r, p), radius=0.09, color=BLUE)
            lab = safe_text(name, size=16, color=INK).next_to(d, LEFT if name in "ACE" else RIGHT,
                                                            buff=0.08)
            dots.add(VGroup(d, lab))
        self.play(Create(ax), FadeIn(labels), LaggedStart(*[FadeIn(d) for d in dots],
                                                         lag_ratio=0.1))
        bar_x, bar_y0, bar_h = 3.4, -2.4, 3.6
        scale = bar_h / SSE0

        def sse_bar(value, color):
            r = Rectangle(width=0.9, height=max(value * scale, 0.02), stroke_width=0,
                          fill_color=color, fill_opacity=0.8)
            r.move_to([bar_x, bar_y0 + r.height / 2, 0])
            t = safe_text(f"SSE = {value / 1e6:.2f}·10⁶", size=21, color=color, weight=BOLD
                          ).next_to(r, RIGHT, buff=0.2).align_to(r, UP)
            return VGroup(r, t)

        def model(steps):
            line = step_graph(ax, steps)
            res = VGroup()
            for name, r, p in LAPTOPS:
                pred = next(v for a, b, v in steps if a <= r <= b)
                res.add(Line(ax.c2p(r, p), ax.c2p(r, pred), color=RED, stroke_width=4))
            return line, res

        mean = float(PRICE.mean())
        line, res = model([(0, 18, mean)])
        base_lab = safe_text(f"promedio {money(mean)}", size=19, color=YELLOW
                             ).next_to(ax.c2p(18, mean), UP, buff=0.08).align_to(ax.c2p(18, 0), RIGHT)
        bar = sse_bar(SSE0, RED)
        frame = Rectangle(width=0.9, height=bar_h, stroke_color=MUTED, stroke_width=1.5
                          ).move_to([bar_x, bar_y0 + bar_h / 2, 0])
        self.play(Create(line), FadeIn(base_lab))
        self.play(LaggedStart(*[Create(r) for r in res], lag_ratio=0.1), FadeIn(frame),
                  GrowFromEdge(bar[0], DOWN), FadeIn(bar[1]))
        self.bring_to_front(dots)
        state = dict(line=line, res=res, bar=bar, lab=base_lab)

        for t, beat in [
            (6.0, "4–5 min. Primer intento: RAM ≤ 6. A la izquierda A y B (6 850), a la derecha "
                  "el resto (16 875). La barra baja a la mitad: 50.3 %."),
            (12.0, "5–6 min. Segundo intento: RAM ≤ 12. Ahora A–D (9 050) contra E y F (22 500): "
                   "la barra baja 90.5 %. Ése es el corte de CART con el error cuadrático: la "
                   "misma receta de la clase pasada con la SSE en lugar de Gini."),
        ]:
            self.beat(beat)
            lm, rm, s = RAM_CUTS[t]
            new_line, new_res = model([(0, t, lm), (t, 18, rm)])
            new_bar = sse_bar(s, YELLOW if t == 12.0 else ORANGE)
            cut = DashedLine(ax.c2p(t, 0), ax.c2p(t, 26000), color=MUTED, stroke_width=2)
            new_lab = VGroup(
                safe_text(f"RAM ≤ {t:g}", size=21, color=YELLOW, weight=BOLD),
                safe_text(f"{money(lm)} | {money(rm)}", size=19, color=YELLOW),
                safe_text(f"reducción {100 * (1 - s / SSE0):.1f} %", size=19, color=GREEN),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            new_lab.move_to([0, 0.75, 0]).align_to([4.15, 0, 0], LEFT)
            self.play(FadeOut(state["lab"]), Create(cut))
            self.play(Transform(state["line"], new_line), Transform(state["res"], new_res),
                      Transform(state["bar"], new_bar), FadeIn(new_lab))
            self.bring_to_front(dots)
            state["lab"] = VGroup(new_lab, cut)

        self.beat("6–7 min. Fórmula: el árbol de regresión es una función ESCALONADA; en cada "
                  "región predice el promedio, que es la constante que minimiza el error "
                  "cuadrático. Cada corte se elige para bajar lo más posible la SSE.")
        f = tex(r"\hat f(x) = \sum_{h} c_h\,\mathbf{1}[x \in R_h], \qquad "
                r"c_h = \overline{y}_{R_h}", size=30, color=INK)
        f.move_to([0, 2.05, 0]).align_to([0.2, 0, 0], LEFT)
        self.play(Write(f))

    # ------------------------------------------------------------------
    def deep_tree(self) -> None:
        self.beat("7–9 min. Ahora 40 puntos con ruido alrededor de una curva suave (línea "
                  "punteada). Un árbol sin límite de profundidad sigue a cada punto: 37 "
                  "escalones. Si cambiamos la muestra, cambia todo: mucha VARIANZA. Pregunte: "
                  "¿cómo obtendrían una curva más estable sin perder detalle? Idea: promediar.")
        self.start("El problema: la varianza", "Un árbol profundo memoriza el ruido", TEAL)
        ax = Axes(x_range=[0, 10, 1], y_range=[-2.5, 6, 1], x_length=10.5, y_length=4.6,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False})
        ax.move_to([0, -0.45, 0])
        truth = DashedVMobject(ax.plot(lambda x: float(f_true(x)), x_range=[0, 10, 0.05],
                                       color=INK, stroke_width=3), num_dashes=70)
        pts = VGroup(*[Dot(ax.c2p(x, y), radius=0.065, color=BLUE) for x, y in BAG_TR])
        one = reg_tree(BAG_TR[:, 0], BAG_TR[:, 1])
        st = step_graph(ax, reg_steps(one), color=ORANGE, width=4)
        self.play(Create(ax), LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.02))
        self.play(Create(truth))
        self.play(Create(st), run_time=2.0)
        lab = VGroup(safe_text(f"un árbol profundo: {len(reg_steps(one))} hojas", size=22,
                               color=ORANGE, weight=BOLD),
                     safe_text("- - - función real  2 sen x + 0.3x", size=19, color=MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_corner(UR, buff=0.5
                                                                              ).shift(DOWN * 1.1)
        self.play(FadeIn(lab))
