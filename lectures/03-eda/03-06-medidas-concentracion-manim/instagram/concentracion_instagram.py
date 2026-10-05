"""Reel (9:16, ≈90 s) para Instagram: CRₖ → HHI → Lorenz → Gini.

Mismo hilo geométrico que el clip de LinkedIn (cinco cafeterías 5, 10, 15,
20, 50), reescrito para verse sin audio en Reels: cada paso tiene una
explicación de 1–2 líneas que permanece el tiempo de lectura.

El acto de Gini explica *por qué* funciona el índice:
1. la diagonal es el reparto igualitario;
2. la brecha entre curva y diagonal es la desigualdad;
3. el triángulo bajo la diagonal es la brecha máxima;
4. G = brecha ÷ triángulo, y un medidor en vivo muestra G = 0 con igualdad,
   G = 0.80 cuando una sola sucursal vende todo (máx. con 5 = 1 − 1/5) y
   G = 0.40 en el caso base.

Sin marca de curso ni de universidad: solo nombre y usuario de Instagram.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import BG, INK, MUTED, GRID, BLUE, YELLOW, GREEN, PURPLE, ORANGE, FONT  # noqa: E402

NAME, HANDLE = "Diego Villalba", "@diego_villalba06"
SALES = [5, 10, 15, 20, 50]            # orden ascendente: Norte … Poniente
NAMES = ["Norte", "Sur", "Centro", "Oriente", "Poniente"]
STEPS = [("CR<sub>k</sub>", YELLOW), ("HHI", GREEN), ("Lorenz", BLUE), ("Gini", PURPLE)]

# ── maquetación 9:16 (marco 8 × 14.22) ─────────────────────────────────
# La interfaz de Reels tapa ≈220 px arriba y ≈380 px abajo: el contenido
# vive en y ∈ [-4.1, 5.6]; la firma queda justo encima de esa franja.
NAV_Y, TITLE_Y = 5.50, 4.55
S = 5.0                                # lado del cuadro principal
C = np.array([0.15, 0.95, 0.0])        # centro del cuadro
TEXT_Y, TEXT_Y_UNDER = -2.20, -2.95    # arriba del bloque de texto (sin / con fórmula)
FORMULA_Y = -2.38


def sq(u: float, v: float) -> np.ndarray:
    """Coordenadas unitarias del cuadro → escena."""
    return C + np.array([(u - .5) * S, (v - .5) * S, 0])


def lorenz_points(values) -> list[np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    L = np.r_[0, np.cumsum(x) / x.sum()]
    return [sq(i / len(x), l) for i, l in enumerate(L)]


def gini(values) -> float:
    """G = 1 − 2B con B = área (trapecios) bajo la curva de Lorenz."""
    x = np.sort(np.asarray(values, dtype=float))
    L = np.r_[0, np.cumsum(x) / x.sum()]
    return 1 - 2 * float(np.sum((L[:-1] + L[1:]) / 2) / len(x))


def unit_color(i_asc: int) -> str:
    return ORANGE if i_asc == 4 else BLUE


def hl(text: str, color: str, bold: bool = True) -> str:
    """Resalta un fragmento dentro de MarkupText."""
    w = ' font_weight="bold"' if bold else ""
    return f'<span foreground="{color}"{w}>{text}</span>'


def mk(markup: str, *, size: float = 30, color: str = INK, weight: str = NORMAL) -> MarkupText:
    return MarkupText(markup, font=FONT, font_size=size, color=color, weight=weight)


def read_time(lines) -> float:
    """Segundos para leer en móvil (~4 palabras/s) más una pausa para mirar."""
    words = len(re.sub(r"<[^>]+>", "", " ".join(lines)).split())
    return max(1.6, .6 + .25 * words)


class ConcentracionInstagram(Scene):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.txt: Mobject | None = None
        self.fml: Mobject | None = None
        self.add(self.footer())
        self.hook()
        self.cr_act()
        self.hhi_act()
        self.lorenz_act()
        self.gini_act()
        self.closing()

    # ── piezas reutilizables ──────────────────────────────────────────
    def footer(self) -> VGroup:
        rule = Line([-3.6, -4.30, 0], [3.6, -4.30, 0], color=GRID, stroke_width=1.2)
        sig = mk(f"{NAME}  ·  {hl(HANDLE, BLUE, bold=False)}", size=22, color=MUTED).move_to([0, -4.64, 0])
        return VGroup(rule, sig)

    def nav(self, active: int) -> VGroup:
        items = VGroup(*[
            mk(name, size=28, weight=BOLD if i == active else NORMAL, color=col if i == active else GRID)
            for i, (name, col) in enumerate(STEPS)
        ]).arrange(RIGHT, buff=.62).move_to([0, NAV_Y, 0])
        bar = Line(LEFT * .5, RIGHT * .5, color=STEPS[active][1], stroke_width=5)
        bar.match_width(items[active]).next_to(items[active], DOWN, buff=.10)
        return VGroup(items, bar)

    def title(self, *lines: str, color: str = INK) -> VGroup:
        g = VGroup(*[mk(l, size=42, weight=BOLD, color=color) for l in lines]).arrange(DOWN, buff=.16)
        if g.width > 7.1:
            g.scale_to_fit_width(7.1)
        return g.move_to([0, TITLE_Y, 0])

    def set_title(self, *lines: str, color: str = INK, extra=(), rt: float = .6) -> None:
        new = self.title(*lines, color=color)
        self.play(FadeTransform(self.head, new), *extra, run_time=rt)
        self.head = new

    def say(self, *lines: str, color: str = INK, extra=(), rt: float = .5, hold: float | None = None) -> None:
        """Cambia la explicación inferior y espera el tiempo de lectura."""
        new = VGroup(*[mk(l, size=33, color=color) for l in lines]).arrange(DOWN, buff=.20)
        if new.width > 7.3:
            new.scale_to_fit_width(7.3)
        top = TEXT_Y_UNDER if self.fml is not None else TEXT_Y
        new.move_to([0, top - new.height / 2, 0])
        swap = [AnimationGroup(FadeOut(self.txt), FadeIn(new, shift=UP * .08), lag_ratio=.5)] if self.txt \
            else [FadeIn(new, shift=UP * .08)]
        self.play(*swap, *extra, run_time=rt)
        self.txt = new
        self.wait(read_time(lines) if hold is None else hold)

    def set_formula(self, mob: Mobject | None) -> list:
        """Anima el cambio de fórmula (o su retiro) y devuelve las animaciones."""
        anims = [FadeOut(self.fml)] if self.fml is not None else []
        self.fml = mob
        if mob is not None:
            mob.move_to([0, FORMULA_Y, 0])
            anims.append(FadeIn(mob, shift=UP * .08))
        return anims

    def clear_text(self) -> list:
        anims = [FadeOut(m) for m in (self.txt, self.fml) if m is not None]
        self.txt = self.fml = None
        return anims

    # ── 0 · gancho ────────────────────────────────────────────────────
    def hook(self) -> None:
        kicker = mk("CIENCIA DE DATOS", size=24, weight=BOLD, color=BLUE).move_to([0, NAV_Y, 0])
        self.head = self.title("¿Quién acumula", "el total?")

        base = sq(0, 0)[1]
        self.bars, self.bar_nums, names = VGroup(), VGroup(), VGroup()
        for i, v in enumerate(SALES):
            x = sq((i + .5) / 5, 0)[0]
            h = v / 50 * 4.1
            bar = Rectangle(width=.8, height=h, stroke_width=0, fill_color=unit_color(i), fill_opacity=.92)
            bar.move_to([x, base + h / 2, 0])
            self.bars.add(bar)
            self.bar_nums.add(mk(str(v), size=32, weight=BOLD).next_to(bar, UP, buff=.10))
            names.add(mk(NAMES[i], size=18, color=MUTED).move_to([x, base - .27, 0]))

        self.play(FadeIn(kicker, shift=DOWN * .2), FadeIn(self.head, shift=DOWN * .15),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in self.bars], lag_ratio=.12),
                  LaggedStart(*[FadeIn(n) for n in self.bar_nums], lag_ratio=.12),
                  FadeIn(names), run_time=1.2)
        self.say("5 cafeterías venden 100 cafés.")
        mean_y = base + 20 / 50 * 4.1
        mean = DashedLine([sq(0, 0)[0] - .1, mean_y, 0], [sq(1, 0)[0] + .1, mean_y, 0],
                          color=YELLOW, stroke_width=3)
        mean_lab = mk("media = 20", size=22, color=YELLOW).next_to(mean, UP, buff=.08).align_to(mean, LEFT)
        self.say(f"La {hl('media', YELLOW)} es 20…", f"pero {hl('Poniente', ORANGE)} sola vende 50.",
                 extra=[Create(mean), FadeIn(mean_lab)], rt=.7)
        self.say("El promedio no dice cómo", "se reparte el total. Cuatro medidas sí.", color=MUTED)

        self.nav_mob = self.nav(0)
        self.play(FadeTransform(kicker, self.nav_mob), FadeOut(mean), FadeOut(mean_lab), FadeOut(names),
                  run_time=.5)

    # ── 1 · CRₖ: apilar desde la más grande ──────────────────────────
    def cr_act(self) -> None:
        top, x0, w = sq(0, 1)[1], -0.45, 1.3
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
            labels.add(mk(f"{SALES[i]}%", size=22 if s < .12 else 28, weight=BOLD)
                       .move_to([x0 - w / 2 - .5, seg.get_center()[1], 0]))

        self.set_title(f"CR<sub>k</sub>: ¿cuánto", "reúne la cima?",
                       extra=[*[ReplacementTransform(self.bars[i], self.col[4 - i]) for i in range(5)],
                              *[ReplacementTransform(self.bar_nums[i], labels[4 - i]) for i in range(5)]],
                       rt=1.1)
        self.say("Apila las ventas de la más", "grande a la más chica (100%).")

        brace = Brace(self.col[0], RIGHT, buff=.12, color=YELLOW)
        cr = MathTex(r"CR_1 = 50\%", font_size=46, color=YELLOW).next_to(brace, RIGHT, buff=.12)
        self.say(f"{hl('CR<sub>1</sub> = 50%', YELLOW)}: la más grande", "sola tiene la mitad.",
                 extra=[GrowFromCenter(brace), Write(cr)], rt=.7)
        brace2 = Brace(VGroup(self.col[0], self.col[1]), RIGHT, buff=.12, color=YELLOW)
        cr2 = MathTex(r"CR_2 = 70\%", font_size=46, color=YELLOW).next_to(brace2, RIGHT, buff=.12)
        self.say(f"{hl('CR<sub>2</sub> = 70%', YELLOW)}: las dos más grandes.",
                 extra=[Transform(brace, brace2), Transform(cr, cr2)], rt=.7)
        self.say("Fácil de leer… pero no mira", "cómo se reparte el resto.", color=MUTED)
        self.play(FadeOut(brace), FadeOut(cr), FadeOut(labels), run_time=.4)

    # ── 2 · HHI: cuadrado de parejas ─────────────────────────────────
    def hhi_act(self) -> None:
        cum = np.cumsum([SALES[i] / 100 for i in reversed(range(5))])
        edges = np.r_[0, cum]
        left, top = sq(0, 1)[0], sq(0, 1)[1]

        strip_w = .30
        row = VGroup()
        for k, i in enumerate(reversed(range(5))):
            s = SALES[i] / 100
            seg = Rectangle(width=s * S, height=strip_w, stroke_color=BG, stroke_width=2,
                            fill_color=unit_color(i), fill_opacity=.92)
            seg.move_to([left + (edges[k] + s / 2) * S, top + .08 + strip_w / 2, 0])
            row.add(seg)
        frame = Square(S, color=MUTED, stroke_width=2).move_to(C)
        grid = VGroup(
            *[Line([left + e * S, top, 0], [left + e * S, top - S, 0], color=GRID, stroke_width=1.5) for e in cum[:-1]],
            *[Line([left, top - e * S, 0], [left + S, top - e * S, 0], color=GRID, stroke_width=1.5) for e in cum[:-1]],
        )
        lab1 = mk("1ª venta", size=20, color=MUTED).next_to(row, RIGHT, buff=.12)
        lab2 = mk("2ª venta", size=20, color=MUTED).rotate(PI / 2) \
            .move_to([left - .08 - strip_w - .27, C[1], 0])

        self.nav_mob2 = self.nav(1)
        self.set_title("HHI: ¿2 ventas al azar,", "misma cafetería?",
                       extra=[Transform(self.nav_mob, self.nav_mob2),
                              self.col.animate.stretch_to_fit_width(strip_w)
                              .move_to([left - .08 - strip_w / 2, C[1], 0])],
                       rt=.9)
        self.say("Elige 2 ventas al azar.", "Cada celda es una pareja;", f"su {hl('área', YELLOW)} es su {hl('probabilidad', YELLOW)}.",
                 extra=[TransformFromCopy(self.col, row, path_arc=-PI / 2), Create(frame),
                        Create(grid, lag_ratio=.1), FadeIn(lab1), FadeIn(lab2)], rt=1.1)

        blocks = VGroup()
        for k, i in enumerate(reversed(range(5))):
            s = SALES[i] / 100
            b = Square(s * S, stroke_width=0, fill_color=unit_color(i), fill_opacity=.88)
            b.move_to([left + (edges[k] + s / 2) * S, top - (edges[k] + s / 2) * S, 0])
            blocks.add(b)
        big = MathTex(r"0.5^2 = 0.25", font_size=42, color=BG).move_to(blocks[0])
        self.say(f"{hl('Diagonal', GREEN)} = misma cafetería.",
                 f"{hl('Poniente', ORANGE)} sola: 0.5 × 0.5 = 0.25",
                 extra=[LaggedStart(*[GrowFromCenter(b) for b in blocks], lag_ratio=.15), Write(big)], rt=1.1)

        hhi = MathTex(r"\mathrm{HHI}=\sum s_i^2=0.325", font_size=48, color=GREEN)
        self.play(*self.set_formula(hhi), FadeOut(self.txt), run_time=.7)
        self.txt = None
        self.say(f"{hl('32.5%', GREEN)} de las parejas coinciden.",
                 "Equivale a ≈ 3 cafeterías iguales (1/HHI).")
        self.play(FadeOut(VGroup(self.col, row, grid, blocks, big, lab1, lab2)), *self.clear_text(), run_time=.5)
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
        xlab = mk("% acumulado de cafeterías →", size=21, color=MUTED).next_to(self.frame, DOWN, buff=.14)
        ylab = mk("% acumulado de ventas →", size=21, color=MUTED).rotate(PI / 2) \
            .next_to(self.frame, LEFT, buff=.14)

        self.set_title("Lorenz: ordena", "de menor a mayor…",
                       extra=[Transform(self.nav_mob, self.nav(2)), FadeIn(xlab), FadeIn(ylab),
                              LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.1)], rt=1.0)
        self.set_title("Lorenz: …y acumula", "las ventas",
                       extra=[LaggedStart(*[b.animate.shift(UP * L[i] * S) for i, b in enumerate(bars)],
                                          lag_ratio=.15)], rt=1.4)

        pts = lorenz_points(SALES)
        dots = VGroup(*[Dot(p, radius=.08, color=INK) for p in pts])
        curve = VMobject(color=INK, stroke_width=6).set_points_as_corners(pts)
        self.set_title("Curva de Lorenz", color=BLUE,
                       extra=[LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=.1), Create(curve),
                              bars.animate.set_fill(opacity=.22)], rt=1.1)

        p = sq(.8, .5)
        guides = VGroup(DashedLine(sq(.8, 0), p, color=YELLOW, stroke_width=3),
                        DashedLine(sq(0, .5), p, color=YELLOW, stroke_width=3))
        self.say(f"{hl('80%', YELLOW)} de las cafeterías reúne", f"solo el {hl('50%', YELLOW)} de las ventas.",
                 extra=[Create(guides), Flash(p, color=YELLOW, flash_radius=.3)], rt=.8)
        self.say(f"El otro 50% es de {hl('Poniente', ORANGE)} sola:", "ese salto final es la concentración.")
        self.play(FadeOut(guides), FadeOut(bars), run_time=.4)
        self.curve, self.dots, self.axes_labels = curve, dots, VGroup(xlab, ylab)

    # ── 4 · Gini: por qué el área mide la desigualdad ───────────────
    def gini_act(self) -> None:
        diag = DashedLine(sq(0, 0), sq(1, 1), color=GREEN, stroke_width=4)
        eq = mk("igualdad", size=24, color=GREEN).rotate(PI / 4).move_to(sq(.36, .50))
        self.set_title("Gini: ¿qué tan lejos", "de la igualdad?",
                       extra=[Transform(self.nav_mob, self.nav(3))], rt=.6)
        self.say(f"Si las 5 vendieran 20, la curva", f"sería la {hl('diagonal', GREEN)}.",
                 extra=[Create(diag), FadeIn(eq)], rt=.8)

        gap = Polygon(*lorenz_points(SALES), stroke_width=0, fill_color=PURPLE, fill_opacity=.55)
        self.bring_to_back(gap)
        self.say(f"La {hl('brecha', PURPLE)} entre las dos", "mide la desigualdad.",
                 extra=[FadeIn(gap)], rt=.7)

        tri = Polygon(sq(0, 0), sq(1, 0), sq(1, 1), stroke_color=YELLOW, stroke_width=5, fill_opacity=0)
        self.say(f"Lo más desigual posible: la brecha", f"llena el {hl('triángulo', YELLOW)} (área 0.5).",
                 extra=[Create(tri)], rt=.8)

        g_tex = MathTex(r"G=\frac{\text{brecha}}{\text{triángulo}}=\frac{0.20}{0.50}=0.40", font_size=44)
        for sl, col in [(slice(2, 8), PURPLE), (slice(9, 19), YELLOW),      # brecha / triángulo
                        (slice(20, 24), PURPLE), (slice(25, 29), YELLOW)]:  # 0.20 / 0.50
            g_tex[0][sl].set_color(col)
        self.play(*self.set_formula(g_tex), FadeOut(self.txt), FadeOut(eq), FadeOut(self.axes_labels),
                  Indicate(gap, color=PURPLE, scale_factor=1.0), run_time=.9)
        self.txt = None
        self.say(f"{hl('Gini', PURPLE)} = qué fracción del triángulo", f"ocupa la brecha: {hl('40%', PURPLE)}.")

        # ── medidor en vivo: la curva se deforma y G la sigue ──
        bar_l, bar_r, bar_y = sq(0, 0)[0], sq(1, 0)[0], sq(0, 0)[1] - .42
        track = Line([bar_l, bar_y, 0], [bar_r, bar_y, 0], color=GRID, stroke_width=12)
        ends = VGroup(mk("0", size=22, color=MUTED).next_to(track, LEFT, buff=.14),
                      mk("1", size=22, color=MUTED).next_to(track, RIGHT, buff=.14))
        t = ValueTracker(0)
        self.src, self.dst = list(map(float, SALES)), list(map(float, SALES))

        def current():
            a = t.get_value()
            return [(1 - a) * x + a * y for x, y in zip(self.src, self.dst)]

        fill = always_redraw(lambda: Line([bar_l, bar_y, 0], [bar_l + max(gini(current()), .003) * (bar_r - bar_l),
                                                                bar_y, 0], color=PURPLE, stroke_width=12))
        live_gap = always_redraw(lambda: Polygon(*lorenz_points(current()), stroke_width=0,
                                                 fill_color=PURPLE, fill_opacity=.55))
        live_curve = always_redraw(lambda: VMobject(color=INK, stroke_width=6)
                                   .set_points_as_corners(lorenz_points(current())))
        live_dots = always_redraw(lambda: VGroup(*[Dot(p, radius=.08, color=INK)
                                                   for p in lorenz_points(current())]))
        g_val = DecimalNumber(gini(SALES), num_decimal_places=2, font_size=48, color=PURPLE)
        g_val.add_updater(lambda m: m.set_value(gini(current())))
        g_live = VGroup(MathTex("G =", font_size=48, color=PURPLE), g_val).arrange(RIGHT, buff=.18)
        g_val.add_updater(lambda m: m.next_to(g_live[0], RIGHT, buff=.18))

        self.remove(gap, self.curve, self.dots, *self.dots)
        self.add(live_gap, tri, diag, live_curve, live_dots)
        self.play(FadeIn(track), FadeIn(ends), FadeIn(fill), *self.set_formula(g_live), FadeOut(self.txt),
                  run_time=.6)
        self.txt = None
        for target, lines in [
            ([20] * 5, ["Si todas venden igual:", f"sin brecha → {hl('G = 0', GREEN)}."]),
            ([0, 0, 0, 0, 100], [f"Si una sola vende todo: {hl('G = 0.80', ORANGE)}",
                                 "(tope con 5; con muchas, casi 1)."]),
            (SALES, [f"Nuestro caso: {hl('G = 0.40', PURPLE)}.", "Más arco = más concentración."]),
        ]:
            self.src, self.dst = current(), list(map(float, target))
            t.set_value(0)
            self.say(*lines, extra=[t.animate(rate_func=smooth).set_value(1)], rt=1.8)

        g_val.clear_updaters()
        self.play(*[FadeOut(m) for m in (live_gap, live_curve, live_dots, tri, diag, self.frame, track, ends,
                                         fill, self.head, self.nav_mob)],
                  *self.clear_text(), run_time=.6)

    # ── cierre ──────────────────────────────────────────────────────
    def closing(self) -> None:
        head = self.title("4 miradas, 1 reparto")
        rows = VGroup()
        for (name, col), text in zip(STEPS, ["cuánto reúne la cima",
                                             "prob. de coincidir",
                                             "la forma de acumular",
                                             "el arco en un número"]):
            chip = RoundedRectangle(corner_radius=.14, width=1.9, height=.78, stroke_color=col,
                                    fill_color=col, fill_opacity=.14, stroke_width=2.5)
            tag = mk(name, size=32, weight=BOLD, color=col).move_to(chip)
            desc = mk(text, size=32)
            rows.add(VGroup(VGroup(chip, tag), desc).arrange(RIGHT, buff=.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=.32).move_to([0, 1.75, 0])
        vals = mk(f"{hl('CR<sub>2</sub> 70%', YELLOW)} · {hl('HHI 0.325', GREEN)} · {hl('G 0.40', PURPLE)}",
                  size=30).move_to([0, -.85, 0])
        note = mk("Ningún índice lo dice todo: combínalos.", size=32, color=YELLOW).move_to([0, -1.6, 0])
        who = VGroup(mk(NAME, size=38, weight=BOLD), mk(HANDLE, size=36, color=BLUE)) \
            .arrange(DOWN, buff=.16).move_to([0, -3.0, 0])
        self.play(FadeIn(head, shift=DOWN * .1),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.2), run_time=1.3)
        self.wait(2.6)
        self.play(FadeIn(vals), FadeIn(note, shift=UP * .1), run_time=.6)
        self.wait(2.4)
        self.play(FadeIn(who, shift=UP * .1), run_time=.6)
        self.wait(3.0)
