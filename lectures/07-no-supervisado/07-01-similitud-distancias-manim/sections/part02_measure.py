"""Acto 2 · ¿Qué significa medir? (5–14 min).

Medir = asignar números que copian la estructura de los objetos (orden y
concatenación de varas). Una distancia mide la separación de PARES. Tres
"reglas rotas" (sólo edad, tiempo de subida, euclidiana al cuadrado) hacen
aparecer, por contraste, los axiomas de métrica. Después se formalizan.
"""
from __future__ import annotations

from manim import *

from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, TEAL,
                            YELLOW, card, check, cross, safe_text, tex)


class MeasureSection:
    def measure(self) -> None:
        self.beat("5–7 min. Antes de distancias: ¿qué hacemos cuando medimos una vara? Le "
                  "asignamos un número. Pero no cualquier número: si una vara es más larga, su "
                  "número debe ser mayor; si pegamos dos varas, sus números deben sumarse. Medir "
                  "es copiar la estructura de los objetos en los números.")
        head = self.start("Medir", "Asignar números que respeten la estructura", TEAL)
        line = NumberLine(x_range=[0, 6, 1], length=9.0, include_numbers=True, color=MUTED,
                          font_size=26).move_to([0, -2.1, 0])
        unit = line.n2p(1)[0] - line.n2p(0)[0]
        a = Rectangle(width=1.5 * unit, height=0.32, stroke_width=0, fill_color=BLUE,
                      fill_opacity=0.9)
        b = Rectangle(width=2.5 * unit, height=0.32, stroke_width=0, fill_color=ORANGE,
                      fill_opacity=0.9)
        a.move_to([-3.2, 0.9, 0])
        b.move_to([1.8, 0.9, 0])
        la = safe_text("vara a", size=22, color=BLUE).next_to(a, UP, buff=0.15)
        lb = safe_text("vara b", size=22, color=ORANGE).next_to(b, UP, buff=0.15)
        self.play(FadeIn(a), FadeIn(b), FadeIn(la), FadeIn(lb))
        self.play(Create(line))
        self.play(a.animate.next_to(line.n2p(0), UP, buff=0.12, aligned_edge=LEFT)
                  .align_to(line.n2p(0), LEFT),
                  la.animate.next_to(line.n2p(0.75) + UP * 0.45, UP, buff=0.05))
        ma = tex(r"m(a)=1.5", size=34, color=BLUE).move_to([-4.4, 0.6, 0])
        self.play(Write(ma))
        self.play(b.animate.next_to(a, RIGHT, buff=0),
                  lb.animate.next_to(line.n2p(2.75) + UP * 0.45, UP, buff=0.05))
        brace = Brace(VGroup(a, b), UP, color=INK, buff=0.55)
        mab = tex(r"m(a \circ b) = m(a) + m(b) = 4", size=34, color=INK).next_to(brace, UP,
                                                                               buff=0.1)
        self.play(GrowFromCenter(brace), Write(mab))
        rule = VGroup(
            tex(r"a \succ b \;\Longleftrightarrow\; m(a) > m(b)", size=34, color=TEAL),
            tex(r"m(a\circ b)=m(a)+m(b)", size=34, color=TEAL),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        rule_box = VGroup(card(rule.width + 0.6, rule.height + 0.9, TEAL), rule)
        rule.move_to(rule_box[0]).shift(DOWN * 0.15)
        cap = safe_text("la medida copia orden y suma", size=19, color=MUTED)
        cap.next_to(rule, UP, buff=0.12)
        rule_box.add(cap)
        rule_box.to_edge(RIGHT, buff=0.5).shift(UP * 1.2)
        self.play(FadeOut(ma), FadeIn(rule_box, shift=LEFT * 0.2))

        self.beat("7–8 min. Una distancia no mide un objeto: mide la SEPARACIÓN de un par. Tres "
                  "ciudades en un mapa. Pregunte al grupo: si inventaran una 'regla de "
                  "separación', ¿qué condiciones le exigirían para confiar en ella? Anote en el "
                  "pizarrón lo que propongan antes de avanzar.")
        head = self.swap_heading(head, "Medir pares", "¿Qué le exigirías a una regla de separación?",
                                 TEAL)
        self.play(*[FadeOut(m) for m in (line, a, b, la, lb, brace, mab, rule_box)])
        A, B, C = np.array([-3.6, -1.6, 0]), np.array([0.2, 1.3, 0]), np.array([3.4, -1.0, 0])
        cities = VGroup(*[Dot(p, radius=0.13, color=c) for p, c in ((A, YELLOW), (B, BLUE),
                                                                     (C, BLUE))])
        names = VGroup(*[safe_text(n, size=28, color=c, weight=BOLD).next_to(d, DOWN, buff=0.15)
                         for n, d, c in zip("ABC", cities, (YELLOW, BLUE, BLUE))])
        ab = Line(A, B, color=TEAL, stroke_width=4)
        bc = Line(B, C, color=TEAL, stroke_width=4)
        ac = Line(A, C, color=TEAL, stroke_width=4)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cities], lag_ratio=0.2), FadeIn(names))
        self.play(Create(ab), Create(bc), Create(ac))
        guesses = VGroup(*[safe_text(t, size=24, color=MUTED) for t in
                           ("¿nunca negativa?", "¿cero sólo consigo misma?",
                            "¿igual de ida que de vuelta?", "¿el rodeo nunca acorta?")])
        guesses.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(RIGHT, buff=0.45).shift(UP * 0.9)
        self.play(LaggedStart(*[FadeIn(g, shift=LEFT * 0.2) for g in guesses], lag_ratio=0.3))
        self.wait(0.3)

        # ── regla rota 1: sólo edad ────────────────────────────────────────
        self.beat("8–9 min. Regla rota 1: medir clientes SÓLO por su edad. Ana y Fernanda tienen "
                  "34 años pero ingresos muy distintos. La regla dice distancia 0 entre dos "
                  "personas diferentes: no puede distinguirlas. Note que esto es una proyección: "
                  "aplastamos el plano sobre un eje. Volveremos a esta idea.")
        head = self.swap_heading(head, "Regla rota 1", "Distancia cero entre casos distintos", RED)
        self.play(*[FadeOut(m) for m in (cities, names, ab, bc, ac, guesses)])
        ax = Axes(x_range=[20, 60, 10], y_range=[0, 80, 20], x_length=6.2, y_length=4.0,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 1.6,
                                           "include_numbers": True, "font_size": 22})
        ax.move_to([-2.6, -0.55, 0])
        xl = safe_text("edad", size=22, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = safe_text("ingreso", size=22, color=MUTED).rotate(PI / 2).next_to(ax.y_axis, LEFT,
                                                                              buff=0.5)
        ana = Dot(ax.c2p(34, 28), radius=0.11, color=YELLOW)
        fer = Dot(ax.c2p(34, 70), radius=0.11, color=BLUE)
        lana = safe_text("Ana", size=22, color=YELLOW).next_to(ana, RIGHT, buff=0.12)
        lfer = safe_text("Fernanda", size=22, color=BLUE).next_to(fer, RIGHT, buff=0.12)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(GrowFromCenter(ana), GrowFromCenter(fer), FadeIn(lana), FadeIn(lfer))
        pa = ana.copy().set_color(YELLOW)
        pf = fer.copy()
        self.play(pa.animate.move_to(ax.c2p(34, 0)), pf.animate.move_to(ax.c2p(34, 0)),
                  run_time=1.2)
        zero = tex(r"d_{\text{edad}}(\text{Ana},\text{Fer}) = |34-34| = 0", size=34, color=RED)
        zero.move_to([3.4, 1.1, 0])
        but = safe_text("pero Ana ≠ Fernanda", size=28, color=RED, weight=BOLD).next_to(zero, DOWN,
                                                                                       buff=0.35)
        lost = tex(r"\Rightarrow\ \text{pseudom\'etrica}", size=32, color=MUTED).next_to(but, DOWN,
                                                                                      buff=0.35)
        self.play(Write(zero))
        self.play(FadeIn(but), FadeIn(lost))

        # ── regla rota 2: tiempo de subida ─────────────────────────────────
        self.beat("9–10 min. Regla rota 2: medir 'distancia' con el tiempo de caminata en un "
                  "cerro. Subir del valle a la cima toma 30 minutos; bajar, 10. La regla depende "
                  "del sentido. Pregunte si conocen otra medida así: la divergencia de "
                  "Kullback-Leibler de t-SNE tampoco es simétrica.")
        head = self.swap_heading(head, "Regla rota 2", "Ida y vuelta no cuestan lo mismo", RED)
        self.play(*[FadeOut(m) for m in (ax, xl, yl, ana, fer, lana, lfer, pa, pf, zero, but, lost)])
        hill = FunctionGraph(lambda x: 2.6 * np.exp(-((x + 0.5) ** 2) / 2.2) - 2.0,
                             x_range=[-5.5, 4.5], color=MUTED, stroke_width=4)
        valley = Dot(hill.point_from_proportion(0.04), radius=0.12, color=YELLOW)
        top = Dot([-0.5, 0.6, 0], radius=0.12, color=BLUE)
        lv = safe_text("valle", size=24, color=YELLOW).next_to(valley, DOWN, buff=0.15)
        lt = safe_text("cima", size=24, color=BLUE).next_to(top, UP, buff=0.18)
        up = CurvedArrow(valley.get_center() + UP * 0.35 + RIGHT * 0.25, top.get_center() + LEFT * 0.35,
                         angle=-PI / 5, color=RED, stroke_width=4)
        down = CurvedArrow(top.get_center() + RIGHT * 0.4, [3.6, -1.55, 0], angle=-PI / 5,
                           color=GREEN, stroke_width=4)
        tu = safe_text("30 min", size=26, color=RED, weight=BOLD).next_to(up, UL, buff=0.05)
        td = safe_text("10 min", size=26, color=GREEN, weight=BOLD).next_to(down, UR, buff=0.05)
        self.play(Create(hill), FadeIn(valley), FadeIn(top), FadeIn(lv), FadeIn(lt))
        self.play(Create(up), FadeIn(tu))
        self.play(Create(down), FadeIn(td))
        asym = tex(r"d(\text{valle},\text{cima}) \neq d(\text{cima},\text{valle})", size=34,
                   color=RED).move_to([3.3, 2.1, 0])
        quasi = tex(r"\Rightarrow\ \text{cuasim\'etrica}", size=30, color=MUTED).next_to(asym, DOWN,
                                                                                        buff=0.25)
        self.play(Write(asym), FadeIn(quasi))

        # ── regla rota 3: euclidiana al cuadrado ───────────────────────────
        self.beat("10–11 min. Regla rota 3: elevar al cuadrado la distancia euclidiana, que es lo "
                  "que minimiza k-medias. Tres puntos en una recta: 0, 1 y 2. Ir directo cuesta 4; "
                  "dar el rodeo por el 1 cuesta 1 + 1 = 2. ¡El rodeo es más corto! Deje que el "
                  "grupo diga en voz alta por qué eso es absurdo para una regla de separación.")
        head = self.swap_heading(head, "Regla rota 3", "Un rodeo más corto que el camino directo",
                                 RED)
        self.play(*[FadeOut(m) for m in (hill, valley, top, lv, lt, up, down, tu, td, asym, quasi)])
        nl = NumberLine(x_range=[0, 2, 1], length=8, include_numbers=True, font_size=30,
                        color=MUTED).move_to([0, -1.6, 0])
        p0, p1, p2 = (Dot(nl.n2p(v), radius=0.13, color=c) for v, c in ((0, YELLOW), (1, BLUE),
                                                                        (2, BLUE)))
        self.play(Create(nl), FadeIn(p0), FadeIn(p1), FadeIn(p2))
        arc01 = ArcBetweenPoints(nl.n2p(0), nl.n2p(1), angle=-PI / 2, color=GREEN, stroke_width=4)
        arc12 = ArcBetweenPoints(nl.n2p(1), nl.n2p(2), angle=-PI / 2, color=GREEN, stroke_width=4)
        arc02 = ArcBetweenPoints(nl.n2p(0), nl.n2p(2), angle=-PI / 1.6, color=RED, stroke_width=4)
        l01 = tex(r"1^2=1", size=32, color=GREEN).next_to(arc01, UP, buff=0.08)
        l12 = tex(r"1^2=1", size=32, color=GREEN).next_to(arc12, UP, buff=0.08)
        l02 = tex(r"2^2=4", size=34, color=RED).next_to(arc02, UP, buff=0.1)
        self.play(Create(arc01), Write(l01), Create(arc12), Write(l12))
        self.play(Create(arc02), Write(l02))
        ineq = tex(r"d^2(0,2) = 4 \;>\; d^2(0,1)+d^2(1,2) = 2", size=38, color=RED)
        ineq.move_to([0, 2.2, 0])
        semi = tex(r"\Rightarrow\ \text{semim\'etrica (disimilitud)}", size=30, color=MUTED)
        semi.next_to(ineq, DOWN, buff=0.25)
        self.play(Write(ineq), FadeIn(semi))

        # ── formalización ───────────────────────────────────────────────────
        self.beat("11–14 min. Ahora sí, la definición. Cada axioma es exactamente lo que una de las "
                  "reglas rotas violaba. Recalque: una función que cumple los cuatro se llama "
                  "MÉTRICA. Las que fallan alguno no son inútiles (la euclidiana al cuadrado y el "
                  "coseno se usan todo el tiempo), pero no podemos pedirles lo que sólo garantiza "
                  "una métrica.")
        head = self.swap_heading(head, "Formalización", "Una métrica cumple cuatro axiomas", GREEN)
        self.play(*[FadeOut(m) for m in (nl, p0, p1, p2, arc01, arc12, arc02, l01, l12, l02, ineq,
                                         semi)])
        rows = [
            (r"d(\mathbf{x},\mathbf{y}) \ge 0", "no negatividad", None),
            (r"d(\mathbf{x},\mathbf{y}) = 0 \iff \mathbf{x}=\mathbf{y}", "identidad",
             "sólo edad → pseudométrica"),
            (r"d(\mathbf{x},\mathbf{y}) = d(\mathbf{y},\mathbf{x})", "simetría",
             "subir el cerro → cuasimétrica"),
            (r"d(\mathbf{x},\mathbf{z}) \le d(\mathbf{x},\mathbf{y}) + d(\mathbf{y},\mathbf{z})",
             "desigualdad triangular", "euclidiana² → semimétrica"),
        ]
        table = VGroup()
        for k, (formula, name, broken) in enumerate(rows):
            num = safe_text(f"{k + 1}", size=26, color=GREEN, weight=BOLD)
            f = tex(formula, size=36, color=INK, max_width=6.0)
            n = safe_text(name, size=22, color=GREEN)
            b = safe_text(broken or "", size=20, color=RED) if broken else VGroup()
            table.add(VGroup(num, f, n, b))
        y0 = 1.25
        for k, row in enumerate(table):
            y = y0 - 1.0 * k
            row[0].move_to([-6.1, y, 0])
            row[1].move_to([-5.7, y, 0], aligned_edge=LEFT)
            row[2].move_to([0.9, y + 0.17, 0], aligned_edge=LEFT)
            if len(row[3]):
                row[3].move_to([0.9, y - 0.2, 0], aligned_edge=LEFT)
        marks = VGroup(*[check().move_to([0.45, y0 - 1.0 * k, 0]) for k in range(4)])
        fails = VGroup(*[cross(size=0.22).move_to([5.9, y0 - 1.0 * k - 0.2, 0])
                         for k in range(1, 4)])
        for k, row in enumerate(table):
            self.play(FadeIn(row[0]), Write(row[1]), FadeIn(row[2]), Create(marks[k]),
                      run_time=0.7)
        self.play(LaggedStart(*[FadeIn(table[k][3]) for k in range(1, 4)], lag_ratio=0.25),
                  LaggedStart(*[Create(f) for f in fails], lag_ratio=0.25))
        concl = safe_text("Cada axioma nace de una regla rota", size=28, color=YELLOW, weight=BOLD)
        concl.move_to([0, -2.85, 0])
        self.play(Write(concl))
