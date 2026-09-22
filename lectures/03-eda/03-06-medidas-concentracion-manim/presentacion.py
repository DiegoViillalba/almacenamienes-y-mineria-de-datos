"""Clase de una hora: de cinco cafeterías a cuatro miradas de concentración.

Cada bloque parte de una pregunta sobre (5, 10, 15, 20, 50), construye una
representación visual y sólo entonces introduce su expresión matemática.
"""
from __future__ import annotations

from manim import *
from base_slide import BaseSlide
from utils import INK, MUTED, GRID, BLUE, YELLOW, GREEN, RED, PURPLE, ORANGE, safe_text, pill

SALES = [5, 10, 15, 20, 50]
NAMES = ["Norte", "Sur", "Centro", "Oriente", "Poniente"]
SHARES = [v / 100 for v in SALES]
LORENZ = [0, .05, .15, .30, .50, 1]


class MedidasConcentracion(BaseSlide):
    def construct(self) -> None:
        self.setup_slide()
        self.opening()
        self.cr_block()
        self.hhi_block()
        self.lorenz_block()
        self.gini_block()
        self.compare_and_practice()

    def start(self, section: str, question: str, color: str = BLUE) -> None:
        self.clear()
        self.play(FadeIn(self.heading(section, question, color)), run_time=.35)

    def line(self, content: str, *, y: float, size: int = 30, color: str = INK,
             width: float = 12.0) -> Text:
        mob = safe_text(content, size=size, color=color, max_width=width)
        mob.move_to([0, y, 0])
        return mob

    def math(self, tex: str, *, y: float, color: str = INK, size: int = 42,
             width: float = 11.8) -> MathTex:
        mob = MathTex(tex, font_size=size, color=color)
        if mob.width > width:
            mob.scale_to_fit_width(width)
        mob.move_to([0, y, 0])
        return mob

    def bars(self, values: list[float], names: list[str] | None = None,
             *, y: float = -1.30, color: str = BLUE, max_height: float = 2.55,
             scale_max: float = 50, labels: bool = True) -> VGroup:
        names = names or NAMES
        group = VGroup()
        for i, (value, name) in enumerate(zip(values, names)):
            x = (i - 2) * 2.22
            height = max(.035, value / scale_max * max_height)
            bar = Rectangle(width=1.12, height=height, stroke_width=0,
                            fill_color=ORANGE if i == 4 else color, fill_opacity=.92)
            bar.move_to([x, y - max_height / 2 + height / 2, 0])
            number = safe_text(f"{value:g}", size=27, color=INK, weight=BOLD)
            number.next_to(bar, UP, buff=.10)
            name_mob = safe_text(name if labels else str(i + 1), size=20, color=MUTED,
                                 max_width=1.95)
            name_mob.move_to([x, y - max_height / 2 - .31, 0])
            group.add(bar, number, name_mob)
        return group

    def lorenz_plot(self, *, center: tuple[float, float] = (0, -.65),
                    size: float = 4.45, values: list[float] = LORENZ,
                    compact: bool = False) -> VGroup:
        axes = Axes(x_range=[0, 1, .2], y_range=[0, 1, .2],
                    x_length=size, y_length=size,
                    axis_config={"color": MUTED, "include_numbers": False, "tip_length": 0})
        axes.move_to([center[0], center[1], 0])
        equality = Line(axes.c2p(0, 0), axes.c2p(1, 1), color=MUTED,
                        stroke_width=3)
        dots = VGroup(*[Dot(axes.c2p(i / 5, v), radius=.07, color=ORANGE)
                        for i, v in enumerate(values)])
        segments = VGroup(*[Line(dots[i].get_center(), dots[i + 1].get_center(),
                                 color=ORANGE, stroke_width=5) for i in range(5)])
        ticks = VGroup(
            safe_text("0", size=20, color=MUTED).next_to(axes.c2p(0, 0), DL, buff=.07),
            safe_text("100%" if compact else "100% sucursales", size=20, color=MUTED).next_to(axes.c2p(1, 0), DOWN, buff=.15),
            safe_text("100%" if compact else "100% ventas", size=20, color=MUTED).next_to(axes.c2p(0, 1), UP, buff=.10),
        )
        return VGroup(axes, equality, segments, dots, ticks)

    def opening(self) -> None:
        eyebrow = safe_text("ANÁLISIS EXPLORATORIO · CONCENTRACIÓN", size=24,
                            color=BLUE, weight=BOLD)
        title = safe_text("¿Quién acumula el total?", size=63, color=INK,
                          weight=BOLD, max_width=12.0)
        subtitle = safe_text("Cinco cafeterías · cuatro maneras de mirar 100 ventas",
                             size=30, color=MUTED)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE,
                           weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20,
                           color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo)
        cover.arrange(DOWN, buff=.25)
        self.play(FadeIn(cover, shift=UP * .16), run_time=.7)
        self.pause("0–2 min. Pregunte qué significa que las ventas estén concentradas. No nombre aún los índices. Anticipe que seguiremos las mismas cinco cafeterías durante la clase.")
        self.clear()
        self.add_course_branding()

        self.start("Ruta de la clase", "¿Qué sabremos hacer al terminar?", YELLOW)
        goals = VGroup(*[
            safe_text(s, size=27, color=INK, max_width=11.7)
            for s in [
                "1. Definir unidad, masa, total y participación.",
                "2. Calcular e interpretar CR₂, HHI, Lorenz y Gini.",
                "3. Predecir el efecto de transferir ventas.",
                "4. Elegir un indicador y reconocer sus límites.",
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=.30).move_to([0, -.35, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * .14) for g in goals],
                              lag_ratio=.15), run_time=.9)
        self.pause("2–3 min. Lea los objetivos como acciones. Indique que habrá dos cálculos en parejas y un cierre individual.")

        self.start("El problema", "Mismo total. ¿Misma concentración?", YELLOW)
        equal = self.bars([20] * 5, y=-.60, color=GREEN, scale_max=100)
        self.play(FadeIn(equal), run_time=.5)
        self.pause("3–4 min. Cinco sucursales, 100 ventas. Pida estimar la media y describir el reparto igualitario antes de cambiar las barras.")
        monopoly = self.bars([0, 0, 0, 0, 100], y=-.60, color=RED, scale_max=100)
        self.play(ReplacementTransform(equal, monopoly), run_time=.8)
        self.pause("4–5 min. Pregunte qué cambió y qué permaneció: total 100 y media 20 son iguales; distribución y dependencia son distintas. Estos son los extremos de concentración.")
        base = self.bars(SALES, y=-.60, scale_max=100)
        self.play(ReplacementTransform(monopoly, base), run_time=.8)
        self.pause("5–6 min. Presente el caso continuo: Norte 5, Sur 10, Centro 15, Oriente 20, Poniente 50. Pida identificar quién produce la mitad del total.")
        assumptions = self.line("Unidad = sucursal · masa = ventas · T = 100 > 0", y=-2.75,
                                size=26, color=YELLOW)
        self.play(FadeIn(assumptions))
        self.pause("6–7 min. Formalice la unidad y la masa. La masa debe ser no negativa y aditiva; un total nulo no permite participaciones. Multiplicar todas las ventas por 10 conserva el reparto.")

    def cr_block(self) -> None:
        self.start("1 · Razón de concentración", "¿Cuánto reúnen las líderes?", ORANGE)
        bars = self.bars(SALES)
        self.play(FadeIn(bars), run_time=.6)
        self.pause("7–8 min. Sin escribir fórmula, pida localizar las dos sucursales con más ventas y estimar qué parte de las 100 bebidas aportan.")
        shares = self.line("5%       10%       15%       20%       50%", y=1.25,
                           size=29, color=YELLOW)
        self.play(FadeIn(shares))
        self.pause("8–9 min. Divida cada venta entre T=100. Las participaciones suman 1. Señale que ahora la unidad de análisis sigue siendo sucursal, pero la escala es comparable.")
        self.play(Indicate(bars[9], color=YELLOW), Indicate(bars[12], color=YELLOW))
        cr = self.line("Poniente 50% + Oriente 20% = 70%", y=.58, size=30,
                       color=ORANGE)
        self.play(FadeIn(cr))
        self.pause("9–10 min. El grupo calcula CR₂=0.70. Pregunte si CR₂ habla de las otras tres sucursales; no revela su reparto interno.")
        formula = self.math(r"CR_k=\sum_{i=1}^{k}s_{[i]},\quad s_{[1]}\geq\cdots\geq s_{[K]}",
                            y=.58, color=ORANGE, size=39)
        self.play(ReplacementTransform(cr, formula))
        self.pause("10–11 min. Formalice el orden descendente y el subíndice entre corchetes. Para este caso CR₁=.50 y CR₂=.70. Siempre declare k y K.")

        self.start("Límite de CR₁", "¿Un mismo líder cuenta toda la historia?", ORANGE)
        a = self.line("A: 50, 20, 15, 10, 5", y=1.12, color=BLUE)
        b = self.line("B: 50, 12.5, 12.5, 12.5, 12.5", y=.30, color=GREEN)
        q = self.line("En ambos casos CR₁ = 0.50", y=-.75, color=ORANGE)
        self.play(FadeIn(a), FadeIn(b))
        self.pause("11–12 min. Antes de mostrar la conclusión, pida comparar cómo se reparte la otra mitad. ¿Captura CR₁ la diferencia?")
        self.play(FadeIn(q))
        self.pause("12–13 min. Ambos tienen líder de 50%, pero en B la cola es uniforme. Esto motiva una medida que use todas las participaciones.")

    def hhi_block(self) -> None:
        self.start("2 · HHI", "Dos ventas elegidas con reemplazo", PURPLE)
        urn = VGroup(*[pill(f"{name} {v}%", ORANGE if i == 4 else BLUE, width=2.35)
                       for i, (name, v) in enumerate(zip(NAMES, SALES))])
        urn.arrange(RIGHT, buff=.12).move_to([0, .36, 0])
        question = self.line("¿Probabilidad de que ambas vengan de la misma sucursal?",
                             y=-1.18, size=28, color=INK)
        self.play(FadeIn(urn), FadeIn(question))
        self.pause("13–15 min. Imagine 100 boletos de ventas. Sacar uno, devolverlo y sacar otro. Pida la probabilidad de Poniente dos veces: .5×.5=.25. Explique por qué hay reemplazo.")
        same = self.math(r"P(\text{misma})=\sum_{i=1}^{5}s_i^2", y=-2.35,
                         color=PURPLE, size=43)
        self.play(Write(same))
        self.pause("15–16 min. Los cinco sucesos 'ambas de la sucursal i' son excluyentes; sus probabilidades se suman. Formalice HHI como esa probabilidad de coincidencia.")

        self.start("HHI en el ejemplo", "El cuadrado da más peso a las grandes", PURPLE)
        terms = VGroup(*[
            safe_text(s, size=26, color=c)
            for s, c in [
                ("Norte: 0.05² = 0.0025", BLUE),
                ("Sur: 0.10² = 0.0100", BLUE),
                ("Centro: 0.15² = 0.0225", BLUE),
                ("Oriente: 0.20² = 0.0400", BLUE),
                ("Poniente: 0.50² = 0.2500", ORANGE),
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=.13).move_to([-1.7, -.30, 0])
        self.play(LaggedStart(*[FadeIn(t) for t in terms], lag_ratio=.10), run_time=.9)
        self.pause("16–18 min. Pida sumar primero las cuatro contribuciones pequeñas: .075. Poniente sola aporta .25; justifique por qué el cuadrado destaca a las grandes.")
        result = self.line("HHI = 0.325 · Diferencia = 1 − HHI = 0.675", y=-2.60,
                           size=29, color=PURPLE)
        self.play(FadeIn(result))
        self.pause("18–19 min. HHI=.325 es probabilidad de coincidencia; 1−HHI=.675 es probabilidad de dos ventas de sucursales distintas (Gini–Simpson). No lo lea como 32.5% del recorrido posible.")

        self.start("Escala del HHI", "¿Cómo interpretar 0.325 entre cinco unidades?", PURPLE)
        bound = self.math(r"\frac1K\leq HHI\leq 1,\qquad K=5\Rightarrow 0.20\leq HHI\leq1",
                          y=1.05, color=PURPLE, size=39)
        self.play(Write(bound))
        self.pause("19–20 min. Pida los extremos: cinco iguales dan 5(.2²)=.2; una sola con todo da 1. Cauchy–Schwarz justifica el mínimo 1/K.")
        norm = self.math(r"HHI^*=\frac{HHI-1/K}{1-1/K}=0.15625", y=-.20,
                         color=GREEN, size=40)
        effective = self.math(r"N_{\mathrm{efectivo}}=\frac1{HHI}\approx3.08",
                              y=-1.45, color=YELLOW, size=40)
        self.play(FadeIn(norm), FadeIn(effective))
        self.pause("20–22 min. Normalice respecto al mínimo .20 y al máximo 1. El número efectivo pregunta cuántas sucursales iguales generarían la misma probabilidad de coincidencia; son 3.08, aunque hay cinco sucursales nominales.")
        scale = self.line("Escala de mercados: 10 000 × 0.325 = 3250", y=-2.72,
                          size=25, color=MUTED)
        self.play(FadeIn(scale))
        self.pause("22–23 min. Aclare que 3250 es la misma medida en escala 0–10 000. Evite usar umbrales regulatorios sin mercado, jurisdicción y fecha definidos.")

        self.start("Experimento", "Mover 5 ventas de Norte a Poniente", RED)
        before = self.bars(SALES, y=-.45, scale_max=55)
        self.play(FadeIn(before))
        self.pause("23–24 min. Predicción: conservando las 100 ventas, ¿sube o baja CR₂? ¿Y HHI? Espere respuestas y justificación.")
        after = self.bars([0, 10, 15, 20, 55], y=-.45, scale_max=55)
        self.play(ReplacementTransform(before, after), run_time=.8)
        self.pause("24–25 min. Norte pasa a 0 y Poniente a 55. CR₂ pasa de .70 a .75; HHI pasa de .325 a .375. Señale que se conserva el total.")
        delta = self.math(r"\Delta HHI=2\delta(s_a-s_b)+2\delta^2=.050",
                          y=-2.79, color=RED, size=38)
        self.play(FadeIn(delta))
        self.pause("25–26 min. Con δ=.05, receptora sₐ=.50 y donante s_b=.05: 2(.05)(.45)+2(.05²)=.05. El cambio aumenta al transferir hacia la unidad mayor.")

    def lorenz_block(self) -> None:
        self.start("3 · Curva de Lorenz", "Acumular desde la sucursal menor", BLUE)
        bars = self.bars(SALES, y=-.7)
        self.play(FadeIn(bars))
        self.pause("26–27 min. Reordene mentalmente de menor a mayor (ya están así). En Lorenz el orden es ascendente, distinto del orden para CRₖ.")
        row = self.line("Ventas acumuladas: 0 → 5 → 15 → 30 → 50 → 100",
                        y=1.10, size=28, color=YELLOW)
        self.play(FadeIn(row))
        self.pause("27–28 min. Construya la suma paso a paso. Pida al grupo explicar por qué la cuarta sucursal acumulada deja 50 de 100 ventas.")
        fraction = self.line("% sucursales: 0 · 20 · 40 · 60 · 80 · 100",
                             y=.42, size=26, color=MUTED)
        masses = self.line("% ventas:       0 ·  5 · 15 · 30 · 50 · 100",
                           y=-.08, size=26, color=ORANGE)
        self.play(FadeIn(fraction), FadeIn(masses))
        self.pause("28–30 min. En parejas, formen los seis puntos (0,0),(.2,.05),(.4,.15),(.6,.30),(.8,.50),(1,1). Den 90 segundos antes de mostrar la curva.")

        self.start("Geometría de Lorenz", "¿Qué nos dice el punto (80%, 50%)?", BLUE)
        plot = self.lorenz_plot(center=(-1.7, -.69))
        self.play(Create(plot[0]), Create(plot[1]), run_time=.7)
        self.pause("30–31 min. La diagonal representa igualdad: el 80% de sucursales produciría 80% de ventas. Pida predecir si nuestra curva va arriba o abajo.")
        self.play(Create(plot[2]), FadeIn(plot[3]), FadeIn(plot[4]), run_time=.9)
        note = safe_text("80% de sucursales\nsolo acumula 50% de ventas",
                         size=29, color=ORANGE, max_width=5.0)
        note.move_to([3.55, -.52, 0])
        self.play(FadeIn(note))
        self.pause("31–33 min. Señale (.8,.5). El 20% restante, Poniente, genera la otra mitad. La curva completa usa todas las unidades, no sólo el grupo líder.")
        formula = self.math(r"L_i=\frac{\sum_{j=1}^{i}x_{(j)}}{T},\quad u_i=\frac{i}{K}",
                            y=-2.15, color=BLUE, size=37, width=5.3)
        formula.move_to([3.45, -2.15, 0])
        self.play(FadeIn(formula))
        self.pause("33–34 min. Formalice Lᵢ como masa acumulada y uᵢ como fracción acumulada de unidades; i=0,…,K y L₀=0. Trace segmentos entre puntos.")

        self.start("Transferencia y Lorenz", "¿Se arquea más al concentrar cinco ventas?", RED)
        changed = self.lorenz_plot(center=(-1.85, -.69), size=4.45)
        self.play(FadeIn(changed))
        self.pause("34–35 min. Pida ubicar el nuevo primer punto: Norte pasa de 5 a 0 ventas, así que L(.2)=0. La curva debe bajar o mantenerse en cada punto intermedio.")
        changed_axes = changed[0]
        new_values = [0, 0, .10, .25, .45, 1]
        new_dots = VGroup(*[Dot(changed_axes.c2p(i / 5, value), radius=.07, color=RED)
                            for i, value in enumerate(new_values)])
        new_lines = VGroup(*[Line(new_dots[i].get_center(), new_dots[i + 1].get_center(),
                                  color=RED, stroke_width=5) for i in range(5)])
        transfer_note = safe_text("Norte → Poniente: 5 ventas\nGini: .40 → .48",
                                  size=29, color=RED, max_width=5.0)
        transfer_note.move_to([3.35, -.52, 0])
        self.play(Create(new_lines), FadeIn(new_dots), FadeIn(transfer_note))
        self.pause("35–36 min. La nueva curva queda por debajo: (0,.10,.25,.45,1) tras el origen. El área bajo Lorenz baja de .30 a .26 y Gini sube de .40 a .48; es el mismo cambio descrito por HHI.")

        self.start("Comparar curvas", "Más arco, más desigualdad", BLUE)
        p = self.lorenz_plot(center=(-2.65, -.63), size=4.2, compact=True)
        q = self.lorenz_plot(center=(2.65, -.63), size=4.2,
                             values=[0, .2, .4, .6, .8, 1], compact=True)
        self.play(FadeIn(p), FadeIn(q))
        captions = VGroup(
            safe_text("Base: 5, 10, 15, 20, 50", size=23, color=ORANGE).move_to([-2.65, 2.08, 0]),
            safe_text("Igualdad: 20 cada una", size=23, color=GREEN).move_to([2.65, 2.08, 0]),
        )
        self.play(FadeIn(captions))
        self.pause("36–37 min. La curva igualitaria coincide con la diagonal. Si dos curvas se cruzan, Lorenz no ordena inequívocamente los casos; distintos índices pueden discrepar.")

    def gini_block(self) -> None:
        self.start("4 · Gini", "Convertir la separación en una cifra", GREEN)
        plot = self.lorenz_plot(center=(-2.65, -.67), size=4.2)
        self.play(FadeIn(plot))
        question = safe_text("¿Cómo resumir toda\nla distancia al reparto igual?",
                             size=30, color=INK, max_width=5.7)
        question.move_to([3.0, .35, 0])
        self.play(FadeIn(question))
        self.pause("37–38 min. Muestre que la línea y la curva encierran un área. Antes de escribir la fórmula, pregunte si igualdad perfecta debe dar 0 o 1.")
        axes = plot[0]
        points = [axes.c2p(i / 5, v) for i, v in enumerate(LORENZ)]
        area = Polygon(axes.c2p(0, 0), axes.c2p(1, 1), *list(reversed(points[1:-1])),
                       stroke_width=0, fill_color=GREEN, fill_opacity=.28)
        self.play(FadeIn(area))
        formula = self.math(r"G=\frac{\frac12-B}{\frac12}=1-2B",
                            y=-1.47, color=GREEN, size=43, width=5.5)
        formula.move_to([3.10, -1.47, 0])
        self.play(FadeIn(formula))
        self.pause("38–39 min. B es área bajo Lorenz. El triángulo bajo igualdad vale 1/2. G normaliza el área intermedia dividiendo entre 1/2.")

        self.start("Cálculo de Gini", "Cinco trapecios bajo la curva", GREEN)
        trap = self.math(r"B=\sum_{i=1}^{5}\frac{L_{i-1}+L_i}{2}\,\frac15",
                         y=1.07, color=BLUE, size=40)
        self.play(Write(trap))
        self.pause("39–40 min. Cada trapecio tiene ancho 1/5. Pida calcular en parejas los promedios de alturas antes de mostrar la suma.")
        parts = self.line("Alturas medias: .025 + .100 + .225 + .400 + .750 = 1.500",
                          y=-.28, size=28, color=YELLOW)
        self.play(FadeIn(parts))
        self.pause("40–42 min. Revise los cinco promedios de alturas. Multiplicar por .2 da B=.300.")
        answer = self.line("B = 0.300          G = 1 − 2(0.300) = 0.400",
                           y=-1.56, size=34, color=GREEN)
        self.play(FadeIn(answer))
        self.pause("42–43 min. Interprete: 0.4 resume la desigualdad del reparto, no significa que el líder posea 40%. Eso corresponde a CR₁=.50.")
        pair = self.math(r"G=\frac{\mathbb E|X_1-X_2|}{2\bar x}=\frac{16}{40}=.4",
                         y=-2.61, color=MUTED, size=35)
        self.play(FadeIn(pair))
        self.pause("43–44 min. Con dos sucursales elegidas al azar con reemplazo, la brecha absoluta esperada de ventas es 16; la media es 20. Esta es otra lectura del mismo Gini.")

        self.start("Escala de Gini", "¿Puede llegar a 1 con sólo cinco unidades?", GREEN)
        extremes = self.line("(20, 20, 20, 20, 20) → G = 0", y=.98,
                             size=30, color=GREEN)
        monopoly = self.line("(0, 0, 0, 0, 100) → G = 0.8", y=-.03,
                             size=30, color=RED)
        self.play(FadeIn(extremes), FadeIn(monopoly))
        self.pause("44–45 min. Pida predecir el máximo muestral. Una sola sucursal con todo da .8, no 1, porque K=5.")
        max_formula = self.math(r"G_{\max}=\frac{K-1}{K},\qquad G^*=\frac{K}{K-1}G",
                                y=-1.28, color=GREEN, size=39)
        base = self.line("Base: G* = (5/4)(0.4) = 0.5", y=-2.42,
                         size=29, color=YELLOW)
        self.play(FadeIn(max_formula), FadeIn(base))
        self.pause("45–46 min. Si se necesita máximo exactamente 1 para K fijo, use G*. Declare siempre si se normalizó. Omitir unidades con cero altera K, la curva y Gini.")

    def compare_and_practice(self) -> None:
        self.start("Elegir la herramienta", "Una pregunta, una lectura", YELLOW)
        rows = [
            ("CR₂ = .70", "¿Qué parte reúnen las dos líderes?", ORANGE),
            ("HHI = .325", "¿Coinciden dos ventas al azar?", PURPLE),
            ("Lorenz", "¿Cómo se acumula desde abajo?", BLUE),
            ("Gini = .40", "¿Qué tan grande es la brecha global?", GREEN),
        ]
        group = VGroup()
        for label, question, color in rows:
            badge = pill(label, color, width=3.0)
            description = safe_text(question, size=27, color=INK, max_width=8.5)
            group.add(VGroup(badge, description).arrange(RIGHT, buff=.4))
        group.arrange(DOWN, aligned_edge=LEFT, buff=.32).move_to([0, -.47, 0])
        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * .10) for row in group],
                              lag_ratio=.14), run_time=.8)
        self.pause("46–48 min. Pida que cada estudiante proponga una pregunta de negocio y seleccione el indicador que contesta exactamente esa pregunta. Los índices describen; no prueban causalidad ni daño.")

        self.start("Comparación", "El mismo CR₁ puede ocultar otra estructura", YELLOW)
        a = self.line("A = (50, 20, 15, 10, 5):    CR₁=.50 · HHI=.3250 · G=.40",
                      y=.95, size=27, color=BLUE)
        b = self.line("B = (50, 12.5, 12.5, 12.5, 12.5): CR₁=.50 · HHI=.3125 · G=.30",
                      y=-.10, size=26, color=GREEN)
        self.play(FadeIn(a))
        self.pause("48–49 min. Pregunte qué anticipan HHI y Gini en B antes de mostrar la fila. CR₁ empata porque sólo mira al líder.")
        self.play(FadeIn(b))
        self.pause("48–49 min. La cola igualitaria en B reduce HHI y Gini. Las medidas pueden distinguir repartos que CR₁ considera idénticos.")

        self.start("Práctica en parejas", "Ahora ustedes: 10, 10, 20, 20, 40", YELLOW)
        prompts = VGroup(*[safe_text(s, size=29, color=INK, max_width=11.6) for s in [
            "1. Total y participaciones; CR₂.",
            "2. HHI y número efectivo de sucursales.",
            "3. Punto de Lorenz para el 80% inferior y Gini.",
            "4. Una frase: ¿qué no permite concluir un índice?",
        ]]).arrange(DOWN, aligned_edge=LEFT, buff=.35).move_to([0, -.4, 0])
        self.play(FadeIn(prompts))
        self.pause("49–54 min. Dé cinco minutos en parejas. No avance a la solución. Circule: verifique orden descendente para CR₂ y ascendente para Lorenz; total 100; fórmula de Gini por trapecios o rangos.")

        self.start("Puesta en común", "Comprobar el nuevo reparto", GREEN)
        answers = VGroup(*[safe_text(s, size=29, color=c, max_width=11.6) for s, c in [
            ("CR₂ = (40+20)/100 = 0.60", ORANGE),
            ("HHI = .10²+.10²+.20²+.20²+.40² = .260", PURPLE),
            ("N efectivo = 1/.260 ≈ 3.85", YELLOW),
            ("Lorenz (80%, 60%); B=.360; G=.280", GREEN),
        ]]).arrange(DOWN, aligned_edge=LEFT, buff=.26).move_to([0, -.55, 0])
        self.play(LaggedStart(*[FadeIn(a) for a in answers], lag_ratio=.20), run_time=.8)
        self.pause("54–57 min. Solicite resultados antes de mostrar cada fila. Puntos Lorenz: 0,.1,.2,.4,.6,1; el área por trapecios es .36. Corrija la confusión entre G=.28 y el 40% del líder.")

        self.start("Antes de interpretar", "¿Definimos bien lo que contamos?", RED)
        cautions = VGroup(*[safe_text(s, size=28, color=INK, max_width=11.6) for s in [
            "Unidades con cero: no borrarlas sin justificarlo.",
            "Agrupar o dividir sucursales cambia K y los índices.",
            "No comparar diferentes unidades, periodos o escalas del HHI.",
            "Concentración descriptiva ≠ causa ni juicio normativo.",
        ]]).arrange(DOWN, aligned_edge=LEFT, buff=.34).move_to([0, -.45, 0])
        self.play(FadeIn(cautions))
        self.pause("57–59 min. Conecte con BigMart: ¿unidad producto, categoría u outlet? ¿Masa ventas o unidades vendidas? Pida un ejemplo donde dividir una marca en dos cambia HHI sin cambiar el negocio real.")

        self.start("Salida", "Una frase y una decisión", BLUE)
        exit_task = self.line("Si una sucursal concentra 50% de ventas, ¿qué reportarías",
                              y=.42, size=29)
        exit_task2 = self.line("para estudiar también el reparto del otro 50%? ¿Por qué?",
                               y=-.20, size=29, color=YELLOW)
        self.play(FadeIn(exit_task), FadeIn(exit_task2))
        self.pause("59–60 min. Ticket individual: CR₁ por sí solo no distingue el reparto de la cola; añadir HHI y/o Lorenz–Gini. Pida justificar la elección. Enlace con el caso BigMart del capítulo siguiente.")
        bridge = self.line("Siguiente caso: concentración de productos y outlets en BigMart",
                           y=-1.68, size=25, color=MUTED)
        self.play(FadeIn(bridge))
