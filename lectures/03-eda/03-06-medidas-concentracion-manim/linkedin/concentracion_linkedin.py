"""Video corto (≈45 s, 4:5) para LinkedIn: CRₖ → HHI → Lorenz → Gini.

Un solo hilo geométrico sobre las cinco cafeterías (5, 10, 15, 20, 50):
las barras se apilan en una columna (CRₖ), la columna genera el cuadrado de
coincidencias (HHI), las barras ordenadas forman la escalera de Lorenz, la
curva se deforma entre igualdad y monopolio y su arco se convierte en Gini.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import BG, INK, MUTED, GRID, BLUE, YELLOW, GREEN, PURPLE, ORANGE, LOGO_PATH, safe_text  # noqa: E402

SALES = [5, 10, 15, 20, 50]            # orden ascendente: Norte … Poniente
NAMES = ["Norte", "Sur", "Centro", "Oriente", "Poniente"]
S = 5.0                                # lado del cuadro principal
C = np.array([0.25, -0.15, 0.0])       # centro del cuadro
STEPS = [("CR<sub>k</sub>", YELLOW), ("HHI", GREEN), ("Lorenz", BLUE), ("Gini", PURPLE)]


def sq(u: float, v: float) -> np.ndarray:
    """Coordenadas unitarias del cuadro → escena."""
    return C + np.array([(u - .5) * S, (v - .5) * S, 0])


def lorenz_points(values) -> list[np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    L = np.r_[0, np.cumsum(x) / x.sum()]
    return [sq(i / len(x), l) for i, l in enumerate(L)]


def label(markup: str, *, size: float, color: str, weight: str = NORMAL) -> MarkupText:
    """Texto corto con subíndices de Pango (la fuente no trae «ₖ»)."""
    return MarkupText(markup, font="Sans", font_size=size, color=color, weight=weight)


def unit_color(i_asc: int) -> str:
    return ORANGE if i_asc == 4 else BLUE


class ConcentracionLinkedIn(Scene):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.add(self.footer())
        self.hook()
        self.cr_act()
        self.hhi_act()
        self.lorenz_act()
        self.morph_act()
        self.gini_act()
        self.closing()

    # ── piezas reutilizables ──────────────────────────────────────────
    def footer(self) -> Group:
        sig = safe_text("Diego Villalba · Almacenes y Minería de Datos", size=17, color=MUTED)
        logo = ImageMobject(str(LOGO_PATH)).scale_to_fit_height(.36)
        sig.move_to([-3.75 + sig.width / 2, -4.62, 0])
        logo.move_to([3.75 - logo.width / 2, -4.62, 0])
        rule = Line([-4, -4.30, 0], [4, -4.30, 0], color=GRID, stroke_width=1.2)
        return Group(rule, sig, logo)

    def nav(self, active: int) -> VGroup:
        items = VGroup(*[
            label(name, size=26, weight=BOLD if i == active else NORMAL,
                  color=col if i == active else GRID)
            for i, (name, col) in enumerate(STEPS)
        ]).arrange(RIGHT, buff=.62).move_to([0, 4.40, 0])
        bar = Line(LEFT * .5, RIGHT * .5, color=STEPS[active][1], stroke_width=5)
        bar.match_width(items[active]).next_to(items[active], DOWN, buff=.10)
        return VGroup(items, bar)

    def title(self, text: str, color: str = INK) -> Text:
        return safe_text(text, size=44, weight=BOLD, color=color, max_width=7.4).move_to([0, 3.50, 0])

    def caption(self, text: str, *, y: float = -3.55, color: str = INK, size: int = 30) -> Text:
        return safe_text(text, size=size, color=color, max_width=7.5).move_to([0, y, 0])

    def swap(self, old: Mobject, new: Mobject, rt: float = .35):
        return FadeTransform(old, new, run_time=rt)

    # ── 0 · gancho ────────────────────────────────────────────────────
    def hook(self) -> None:
        kicker = safe_text("MEDIDAS DE CONCENTRACIÓN", size=24, weight=BOLD, color=BLUE).move_to([0, 4.40, 0])
        self.head = self.title("¿Quién acumula el total?")
        sub = self.caption("5 sucursales · 100 ventas", y=2.85, color=MUTED, size=28)

        base = C[1] - S / 2
        self.bars = VGroup()
        self.bar_nums = VGroup()
        names = VGroup()
        for i, v in enumerate(SALES):
            x = sq((i + .5) / 5, 0)[0]
            h = v / 50 * 4.0
            bar = Rectangle(width=.78, height=h, stroke_width=0, fill_color=unit_color(i), fill_opacity=.92)
            bar.move_to([x, base + h / 2, 0])
            self.bars.add(bar)
            self.bar_nums.add(safe_text(str(v), size=30, weight=BOLD).next_to(bar, UP, buff=.10))
            names.add(safe_text(NAMES[i], size=18, color=MUTED).move_to([x, base - .25, 0]))

        self.play(FadeIn(kicker, shift=DOWN * .2), Write(self.head), run_time=.7)
        self.play(FadeIn(sub),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in self.bars], lag_ratio=.12),
                  LaggedStart(*[FadeIn(n) for n in self.bar_nums], lag_ratio=.12),
                  FadeIn(names), run_time=1.1)
        mean_y = base + 20 / 50 * 4.0
        mean = DashedLine([sq(0, 0)[0] - .1, mean_y, 0], [sq(1, 0)[0] + .1, mean_y, 0], color=YELLOW, stroke_width=3)
        self.cap = self.caption("La media es 20… pero no dice cómo se reparte")
        self.play(Create(mean), FadeIn(self.cap), run_time=.6)
        self.wait(1.6)

        self.nav_mob = self.nav(0)
        self.play(FadeTransform(kicker, self.nav_mob), FadeOut(sub), FadeOut(mean), FadeOut(names), run_time=.4)

    # ── 1 · CRₖ: apilar desde la más grande ──────────────────────────
    def cr_act(self) -> None:
        top, x0, w = C[1] + S / 2, -0.55, 1.25
        self.col = VGroup()                 # segmentos en orden descendente
        labels = VGroup()
        cum = 0.0
        for i in reversed(range(5)):
            s = SALES[i] / 100
            seg = Rectangle(width=w, height=s * S, stroke_color=BG, stroke_width=2,
                            fill_color=unit_color(i), fill_opacity=.92)
            seg.move_to([x0, top - (cum + s / 2) * S, 0])
            cum += s
            self.col.add(seg)
            labels.add(safe_text(f"{SALES[i]}%", size=22 if s < .12 else 26, weight=BOLD)
                       .move_to([x0 - w / 2 - .45, seg.get_center()[1], 0]))

        self.play(self.swap(self.head, t := self.title("¿Cuánto reúne la cima?")),
                  *[ReplacementTransform(self.bars[i], self.col[4 - i]) for i in range(5)],
                  *[ReplacementTransform(self.bar_nums[i], labels[4 - i]) for i in range(5)],
                  run_time=1.0)
        self.head = t

        brace = Brace(self.col[0], RIGHT, buff=.12, color=YELLOW)
        cr = MathTex(r"CR_1 = 50\%", font_size=46, color=YELLOW).next_to(brace, RIGHT, buff=.12)
        self.play(GrowFromCenter(brace), Write(cr),
                  self.swap(self.cap, c := self.caption("Suma las k unidades más grandes")), run_time=.6)
        self.cap = c
        self.wait(.9)
        brace2 = Brace(VGroup(self.col[0], self.col[1]), RIGHT, buff=.12, color=YELLOW)
        cr2 = MathTex(r"CR_2 = 70\%", font_size=46, color=YELLOW).next_to(brace2, RIGHT, buff=.12)
        self.play(Transform(brace, brace2), Transform(cr, cr2), run_time=.6)
        self.wait(.8)
        self.play(self.swap(self.cap, c := self.caption("Rápido de leer… pero ignora el resto", color=MUTED)))
        self.cap = c
        self.wait(1.1)
        self.play(FadeOut(brace), FadeOut(cr), FadeOut(labels), run_time=.35)

    # ── 2 · HHI: cuadrado de parejas ─────────────────────────────────
    def hhi_act(self) -> None:
        cum = np.cumsum([SALES[i] / 100 for i in reversed(range(5))])
        edges = np.r_[0, cum]
        left, top = sq(0, 1)[0], sq(0, 1)[1]

        # columna → franja izquierda; copia girada → franja superior
        strip_w = .32
        row = VGroup()
        for k, i in enumerate(reversed(range(5))):
            s = SALES[i] / 100
            seg = Rectangle(width=s * S, height=strip_w, stroke_color=BG, stroke_width=2,
                            fill_color=unit_color(i), fill_opacity=.92)
            seg.move_to([left + (edges[k] + s / 2) * S, top + .12 + strip_w / 2, 0])
            row.add(seg)
        frame = Square(S, color=MUTED, stroke_width=2).move_to(C)
        grid = VGroup(
            *[Line([left + e * S, top, 0], [left + e * S, top - S, 0], color=GRID, stroke_width=1.5) for e in cum[:-1]],
            *[Line([left, top - e * S, 0], [left + S, top - e * S, 0], color=GRID, stroke_width=1.5) for e in cum[:-1]],
        )

        self.nav_mob2 = self.nav(1)
        self.play(Transform(self.nav_mob, self.nav_mob2),
                  self.swap(self.head, t := self.title("¿2 ventas, misma sucursal?")),
                  self.col.animate.stretch_to_fit_width(strip_w).move_to([left - .12 - strip_w / 2, C[1], 0]),
                  run_time=.8)
        self.head = t
        self.play(TransformFromCopy(self.col, row, path_arc=-PI / 2),
                  Create(frame), Create(grid, lag_ratio=.1),
                  self.swap(self.cap, c := self.caption("Cada celda: una pareja de sucursales")), run_time=1.0)
        self.cap = c
        self.wait(.8)

        blocks = VGroup()
        for k, i in enumerate(reversed(range(5))):
            s = SALES[i] / 100
            b = Square(s * S, stroke_width=0, fill_color=unit_color(i), fill_opacity=.85)
            b.move_to([left + (edges[k] + s / 2) * S, top - (edges[k] + s / 2) * S, 0])
            blocks.add(b)
        big = MathTex(r"0.5^2 = 0.25", font_size=40, color=BG).move_to(blocks[0])
        self.play(LaggedStart(*[GrowFromCenter(b) for b in blocks], lag_ratio=.15),
                  self.swap(self.cap, c := self.caption("Coincidir = cuadros de la diagonal", color=GREEN)),
                  run_time=1.0)
        self.play(Write(big), run_time=.4)
        self.cap = c
        self.wait(.9)
        hhi = MathTex(r"\mathrm{HHI}=\sum s_i^2=0.325", font_size=46, color=GREEN).move_to([0, -3.40, 0])
        eff = safe_text("≈ 3 sucursales “efectivas” (1/HHI)", size=24, color=MUTED).move_to([0, -3.95, 0])
        self.play(self.swap(self.cap, hhi), run_time=.5)
        self.play(FadeIn(eff, shift=UP * .1), run_time=.4)
        self.cap = VGroup(hhi, eff)
        self.wait(1.8)
        self.play(FadeOut(VGroup(self.col, row, grid, blocks, big)), run_time=.4)
        self.frame = frame

    # ── 3 · Lorenz: escalera de acumulados ──────────────────────────
    def lorenz_act(self) -> None:
        shares = [v / 100 for v in SALES]
        L = np.r_[0, np.cumsum(shares)]
        bars = VGroup()
        for i, s in enumerate(shares):
            b = Rectangle(width=S / 5, height=s * S, stroke_color=BG, stroke_width=2,
                          fill_color=unit_color(i), fill_opacity=.85)
            b.move_to(sq((i + .5) / 5, s / 2))
            bars.add(b)
        xlab = safe_text("% acumulado de sucursales →", size=20, color=MUTED).next_to(self.frame, DOWN, buff=.14)
        ylab = safe_text("% acumulado de ventas →", size=20, color=MUTED).rotate(PI / 2).next_to(self.frame, LEFT, buff=.14)

        self.play(Transform(self.nav_mob, self.nav(2)),
                  self.swap(self.head, t := self.title("Ordena de menor a mayor…")),
                  FadeOut(self.cap), FadeIn(xlab), FadeIn(ylab),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.1), run_time=.9)
        self.head = t
        self.play(LaggedStart(*[b.animate.shift(UP * L[i] * S) for i, b in enumerate(bars)], lag_ratio=.15),
                  self.swap(self.head, t := self.title("…y acumula las ventas")), run_time=1.3)
        self.head = t
        self.wait(.4)

        pts = lorenz_points(SALES)
        dots = VGroup(*[Dot(p, radius=.075, color=INK) for p in pts])
        curve = VMobject(color=INK, stroke_width=6).set_points_as_corners(pts)
        diag = DashedLine(sq(0, 0), sq(1, 1), color=MUTED, stroke_width=3)
        eq = safe_text("igualdad", size=22, color=MUTED).rotate(PI / 4).move_to(sq(.40, .52))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=.1), Create(curve), run_time=1.0)
        self.play(Create(diag), FadeIn(eq), bars.animate.set_fill(opacity=.25),
                  self.swap(self.head, t := self.title("Curva de Lorenz", color=BLUE)), run_time=.7)
        self.head = t

        p = sq(.8, .5)
        guides = VGroup(DashedLine(sq(.8, 0), p, color=YELLOW, stroke_width=3),
                        DashedLine(sq(0, .5), p, color=YELLOW, stroke_width=3))
        self.cap = self.caption("80% de sucursales → solo 50% de ventas", color=YELLOW)
        self.play(Create(guides), Flash(p, color=YELLOW, flash_radius=.3), FadeIn(self.cap), run_time=.7)
        self.wait(2.0)
        self.play(FadeOut(guides), FadeOut(bars), FadeOut(dots), FadeOut(curve), run_time=.35)
        self.diag, self.eq, self.axes_labels = diag, eq, VGroup(xlab, ylab)

    # ── 4 · la curva se deforma ─────────────────────────────────────
    def morph_act(self) -> None:
        t = ValueTracker(0)
        self.src, self.dst = list(map(float, SALES)), list(map(float, SALES))

        def current():
            a = t.get_value()
            return [(1 - a) * x + a * y for x, y in zip(self.src, self.dst)]

        gap = always_redraw(lambda: Polygon(*lorenz_points(current()), stroke_width=0,
                                            fill_color=PURPLE, fill_opacity=.40))
        curve = always_redraw(lambda: VMobject(color=INK, stroke_width=6)
                              .set_points_as_corners(lorenz_points(current())))
        dots = always_redraw(lambda: VGroup(*[Dot(p, radius=.075, color=INK) for p in lorenz_points(current())]))
        self.add(gap, curve, dots)
        self.play(self.swap(self.head, h := self.title("La forma del reparto", color=BLUE)),
                  FadeIn(gap), run_time=.4)
        self.head = h

        for target, text, col in [
            ([20] * 5, "Reparto igual → la curva es la diagonal", GREEN),
            ([0, 0, 0, 0, 100], "Una sola acapara → la curva se hunde", ORANGE),
            (SALES, "Más arco = más concentración", PURPLE),
        ]:
            self.src, self.dst = current(), list(map(float, target))
            t.set_value(0)
            self.play(t.animate.set_value(1), self.swap(self.cap, c := self.caption(text, color=col)),
                      run_time=1.6, rate_func=smooth)
            self.cap = c
            self.wait(.9)

        # congelar el estado base para el acto final
        self.src = self.dst = list(map(float, SALES))
        self.gap = Polygon(*lorenz_points(SALES), stroke_width=0, fill_color=PURPLE, fill_opacity=.40)
        self.curve = VMobject(color=INK, stroke_width=6).set_points_as_corners(lorenz_points(SALES))
        self.dots = VGroup(*[Dot(p, radius=.075, color=INK) for p in lorenz_points(SALES)])
        self.remove(gap, curve, dots)
        self.add(self.gap, self.curve, self.dots)

    # ── 5 · Gini: el arco en un número ──────────────────────────────
    def gini_act(self) -> None:
        tri = Polygon(sq(0, 0), sq(1, 0), sq(1, 1), stroke_color=YELLOW, stroke_width=4, fill_opacity=0)
        self.play(Transform(self.nav_mob, self.nav(3)),
                  self.swap(self.head, h := self.title("Gini: el arco en un número", color=PURPLE)),
                  FadeOut(self.eq), FadeOut(self.axes_labels), run_time=.5)
        self.head = h
        self.play(Create(tri), self.swap(self.cap, c := self.caption("Brecha ÷ triángulo bajo la igualdad")), run_time=.8)
        self.cap = c
        self.wait(.6)
        g = MathTex(r"G=\frac{\text{brecha}}{\text{triángulo}}=\frac{0.20}{0.50}=0.40",
                    font_size=42, color=INK)
        g.move_to([0, -3.50, 0])
        self.play(self.swap(self.cap, g), Indicate(self.gap, color=PURPLE, scale_factor=1.0), run_time=.7)
        self.cap = g
        self.wait(1.0)
        scale = safe_text("0 = igualdad   ·   → 1 = una sola lo tiene todo", size=22, color=MUTED).move_to([0, 2.75, 0])
        self.play(FadeIn(scale, shift=DOWN * .1), run_time=.4)
        self.wait(1.9)
        self.play(*[FadeOut(m) for m in [tri, scale, self.gap, self.curve, self.dots, self.diag,
                                        self.frame, self.cap, self.head, self.nav_mob]],
                  run_time=.45)

    # ── cierre ──────────────────────────────────────────────────────
    def closing(self) -> None:
        head = self.title("4 miradas, 1 reparto")
        head.move_to([0, 3.55, 0])
        rows = VGroup()
        for (name, col), text in zip(STEPS, ["cuánto reúne la cima",
                                             "prob. de coincidir",
                                             "la forma de acumular",
                                             "el arco en un número"]):
            chip = RoundedRectangle(corner_radius=.14, width=1.85, height=.72, stroke_color=col,
                                    fill_color=col, fill_opacity=.14, stroke_width=2.5)
            tag = label(name, size=30, weight=BOLD, color=col).move_to(chip)
            desc = safe_text(text, size=30, color=INK)
            rows.add(VGroup(VGroup(chip, tag), desc).arrange(RIGHT, buff=.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=.42).move_to([0, .55, 0])
        note = safe_text("Ningún índice lo dice todo: combínalos.", size=26, color=YELLOW).move_to([0, -2.05, 0])
        course = safe_text("Almacenes y Minería de Datos", size=28, weight=BOLD, color=BLUE).move_to([0, -3.05, 0])
        who = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=22, color=MUTED).move_to([0, -3.55, 0])
        self.play(FadeIn(head, shift=DOWN * .1),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.18), run_time=1.1)
        self.play(FadeIn(note), FadeIn(course), FadeIn(who), run_time=.5)
        self.wait(3.2)
