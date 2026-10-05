"""Reel (9:16, ≈75 s) para Instagram: Gini–Simpson → Shannon.

Un solo hilo geométrico sobre la bolsa de 6 canicas (3 azules, 1 morada,
2 verdes; p=(1/2, 1/6, 1/3)): las canicas se vuelven una franja de
proporciones; la franja arma el cuadrado de parejas «1ª canica × 2ª canica»,
cuya zona fuera de la diagonal es Gini–Simpson; los tres bloques diagonales
se transforman en las barras de sorpresa cuya área total es Shannon.

Pensado para verse sin audio: cada paso tiene una explicación de 1–2 líneas
que permanece en pantalla el tiempo de lectura (ver ``read_time``). Sin marca
de curso ni de universidad: solo nombre y usuario de Instagram.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import BG, INK, MUTED, GRID, BLUE, PURPLE, GREEN, ORANGE, YELLOW, FONT  # noqa: E402

NAME, HANDLE = "Diego Villalba", "@diego_villalba06"
COUNTS = [3, 1, 2]
PROBS = [c / 6 for c in COUNTS]
FRACS = [r"\tfrac{1}{2}", r"\tfrac{1}{6}", r"\tfrac{1}{3}"]
CAT_COLORS = [BLUE, PURPLE, GREEN]
STEPS = [("Gini–Simpson", ORANGE), ("Shannon", PURPLE)]

SUM_SQ = sum(p * p for p in PROBS)          # 7/18 ≈ 0.3889
D_VALUE = 1 - SUM_SQ                         # 11/18 ≈ 0.6111
SURPRISES = [-np.log2(p) for p in PROBS]     # 1, 2.585, 1.585 bits
CONTRIB = [p * s for p, s in zip(PROBS, SURPRISES)]
H_VALUE = sum(CONTRIB)                       # ≈ 1.4591 bits
H_MAX = np.log2(3)                           # ≈ 1.585 bits (K=3)

# ── maquetación 9:16 (marco 8 × 14.22) ─────────────────────────────────
# La interfaz de Reels tapa ≈220 px arriba y ≈380 px abajo: todo el
# contenido vive en y ∈ [-4.1, 5.6]; la firma queda justo encima.
NAV_Y, TITLE_Y = 5.50, 4.55
C = np.array([0.10, 0.95, 0.0])              # centro del visual
S = 5.0                                      # lado del cuadrado
LEFT_X, RIGHT_X = C[0] - S / 2, C[0] + S / 2
TOP_Y, BOT_Y = C[1] + S / 2, C[1] - S / 2
TEXT_Y, TEXT_Y_UNDER = -2.20, -2.95          # arriba del bloque de texto (sin / con fórmula)
FORMULA_Y = -2.38


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


class HeterogeneidadInstagram(Scene):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.txt: Mobject | None = None
        self.fml: Mobject | None = None
        self.add(self.footer())
        self.hook()
        self.proportions()
        self.gini_act()
        self.shannon_act()
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
        ]).arrange(RIGHT, buff=1.0).move_to([0, NAV_Y, 0])
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

    def set_formula(self, tex: str | None, color: str = INK, size: float = 44) -> list:
        """Anima el cambio de fórmula (o su retiro) y devuelve las animaciones."""
        anims = []
        if self.fml is not None:
            anims.append(FadeOut(self.fml))
        self.fml = None
        if tex is not None:
            self.fml = MathTex(tex, font_size=size, color=color).move_to([0, FORMULA_Y, 0])
            anims.append(Write(self.fml))
        return anims

    def clear_text(self) -> list:
        anims = [FadeOut(m) for m in (self.txt, self.fml) if m is not None]
        self.txt = self.fml = None
        return anims

    # ── 0 · gancho ────────────────────────────────────────────────────
    def hook(self) -> None:
        kicker = mk("CIENCIA DE DATOS", size=24, weight=BOLD, color=BLUE).move_to([0, NAV_Y, 0])
        self.head = self.title("¿Qué tan mezcladas", "están las clases?")

        seq = [c for c, n in zip(CAT_COLORS, COUNTS) for _ in range(n)]
        self.marbles = VGroup(*[
            Circle(radius=.36, stroke_color=BG, stroke_width=3, fill_color=col, fill_opacity=.95)
            .move_to([x, C[1], 0])
            for x, col in zip(np.linspace(-2.6, 2.6, 6), seq)
        ])
        self.play(FadeIn(kicker, shift=DOWN * .2), FadeIn(self.head, shift=DOWN * .15),
                  LaggedStart(*[GrowFromCenter(m) for m in self.marbles], lag_ratio=.12), run_time=1.2)
        self.say(f"Una bolsa con 6 canicas:",
                 f"{hl('3 azules', BLUE)}, {hl('1 morada', PURPLE)} y {hl('2 verdes', GREEN)}.")
        self.say("Dos índices lo miden,", "cada uno con su propia geometría.", color=MUTED)
        self.kicker = kicker

    # ── 1 · de canicas a proporciones ─────────────────────────────────
    def proportions(self) -> None:
        self.strip = VGroup()
        cum = 0.0
        for p, col in zip(PROBS, CAT_COLORS):
            seg = Rectangle(width=S * p, height=.95, stroke_color=BG, stroke_width=3,
                            fill_color=col, fill_opacity=.95)
            seg.move_to([LEFT_X + (cum + p / 2) * S, C[1], 0])
            self.strip.add(seg)
            cum += p
        fracs = VGroup(*[MathTex(f, font_size=46, color=BG).move_to(seg) for f, seg in zip(FRACS, self.strip)])

        groups, i = [], 0
        for n in COUNTS:
            groups.append(VGroup(*self.marbles[i:i + n]))
            i += n
        self.play(*[ReplacementTransform(g, s) for g, s in zip(groups, self.strip)], run_time=1.2)
        self.play(FadeIn(fracs), run_time=.4)
        self.say("Lo que importa es la fracción", "de la bolsa que ocupa cada color.")
        self.fracs = fracs

    # ── 2 · Gini–Simpson: cuadrado de parejas ─────────────────────────
    def gini_act(self) -> None:
        self.nav_mob = self.nav(0)
        self.set_title("Gini–Simpson: ¿dos canicas", "de colores distintos?",
                       extra=[FadeTransform(self.kicker, self.nav_mob), FadeOut(self.fracs)], rt=.7)

        thick = .30
        top_strip = self.strip.copy()
        for k, seg in enumerate(top_strip):
            seg.stretch_to_fit_height(thick).move_to([self.strip[k].get_x(), TOP_Y + .08 + thick / 2, 0])
        left_strip = VGroup()
        cum = 0.0
        for p, col in zip(PROBS, CAT_COLORS):
            seg = Rectangle(width=thick, height=S * p, stroke_color=BG, stroke_width=2,
                            fill_color=col, fill_opacity=.95)
            seg.move_to([LEFT_X - .08 - thick / 2, TOP_Y - (cum + p / 2) * S, 0])
            left_strip.add(seg)
            cum += p
        frame = Square(S, color=MUTED, stroke_width=2).move_to(C)
        cuts = np.cumsum(PROBS)[:-1]
        grid = VGroup(
            *[Line([LEFT_X + e * S, TOP_Y, 0], [LEFT_X + e * S, BOT_Y, 0], color=GRID, stroke_width=2) for e in cuts],
            *[Line([LEFT_X, TOP_Y - e * S, 0], [RIGHT_X, TOP_Y - e * S, 0], color=GRID, stroke_width=2) for e in cuts],
        )
        lab1 = mk("1ª canica", size=20, color=MUTED).next_to(top_strip, RIGHT, buff=.15)
        lab2 = mk("2ª canica", size=20, color=MUTED).rotate(PI / 2).next_to(left_strip, LEFT, buff=.12)

        self.play(Transform(self.strip, top_strip), FadeIn(lab1), run_time=.9)
        self.play(TransformFromCopy(self.strip, left_strip, path_arc=PI / 2), FadeIn(lab2),
                  Create(frame), Create(grid, lag_ratio=.1), run_time=1.1)
        self.say("Saca una canica, devuélvela", "y saca otra.")
        self.say("Cada celda es una pareja posible;", f"su {hl('área', YELLOW)} es su {hl('probabilidad', YELLOW)}.")

        diagonal, off_diagonal = self.build_cells()
        quarter = MathTex(r"\tfrac14", font_size=56, color=BG).move_to(diagonal[0])
        self.say(f"{hl('Azul y azul', BLUE)}: ½ × ½ = ¼",
                 extra=[GrowFromCenter(diagonal[0]), FadeIn(quarter, scale=.6)], rt=.8)
        self.say(f"La {hl('diagonal', YELLOW)} = mismo color:",
                 "¼ + 1/36 + 1/9 = 0.389",
                 extra=[LaggedStart(*[GrowFromCenter(c) for c in diagonal[1:]], lag_ratio=.3)], rt=.9)
        self.say(f"Lo {hl('naranja', ORANGE)} = colores distintos:",
                 f"1 − 0.389 = {hl('0.611', ORANGE)}",
                 extra=[LaggedStart(*[GrowFromCenter(c) for c in off_diagonal], lag_ratio=.06)], rt=1.1)

        self.play(*self.set_formula(rf"D = 1-\sum p_k^2 = {D_VALUE:.4f}", ORANGE), FadeOut(self.txt), run_time=.9)
        self.txt = None
        self.say(f"{hl('61%', ORANGE)} de las parejas mezclan colores.")
        self.say("Tope con 3 colores: 2/3 ≈ 0.667", "(si los 3 salieran por igual).", color=MUTED)

        self.diagonal = diagonal
        self.play(FadeOut(VGroup(off_diagonal, quarter, grid, frame, self.strip, left_strip, lab1, lab2)),
                  *self.clear_text(), run_time=.6)

    def build_cells(self):
        """Celdas del cuadrado: fila = 2ª canica, columna = 1ª canica."""
        diagonal, off_diagonal = VGroup(), VGroup()
        cum_i = 0.0
        for i, pi in enumerate(PROBS):
            cum_j = 0.0
            for j, pj in enumerate(PROBS):
                same = i == j
                cell = Rectangle(width=S * pj, height=S * pi, stroke_color=BG, stroke_width=2,
                                 fill_color=CAT_COLORS[i] if same else ORANGE,
                                 fill_opacity=.92 if same else .78)
                cell.move_to([LEFT_X + (cum_j + pj / 2) * S, TOP_Y - (cum_i + pi / 2) * S, 0])
                (diagonal if same else off_diagonal).add(cell)
                cum_j += pj
            cum_i += pi
        return diagonal, off_diagonal

    # ── 3 · Shannon: barras de sorpresa ───────────────────────────────
    def shannon_act(self) -> None:
        self.set_title("Shannon: ¿cuánta sorpresa", "trae una canica?",
                       extra=[Transform(self.nav_mob, self.nav(1))], rt=.7)
        base_y, y_scale = BOT_Y + .40, 1.55
        bars, cum = VGroup(), 0.0
        for p, s, col in zip(PROBS, SURPRISES, CAT_COLORS):
            w, h = S * p, s * y_scale
            bar = Rectangle(width=w, height=h, stroke_color=BG, stroke_width=2, fill_color=col, fill_opacity=.92)
            bar.move_to([LEFT_X + (cum + p / 2) * S, base_y + h / 2, 0])
            bars.add(bar)
            cum += p
        baseline = Line([LEFT_X, base_y, 0], [RIGHT_X, base_y, 0], color=MUTED, stroke_width=3)
        xlab = mk("ancho = probabilidad →", size=21, color=MUTED).next_to(baseline, DOWN, buff=.14)
        ylab = mk("alto = sorpresa →", size=21, color=MUTED).rotate(PI / 2) \
            .move_to([LEFT_X - .32, base_y + 1.5, 0])
        s_labels = VGroup(*[
            mk(f"{s:.2f}".rstrip("0").rstrip("."), size=26, weight=BOLD).next_to(b, UP, buff=.10)
            for s, b in zip(SURPRISES, bars)
        ])

        self.say("Lo raro sorprende más.",
                 f"Sorpresa = −log<sub>2</sub> p  ({hl('bits', PURPLE)})",
                 extra=[Create(baseline), FadeIn(xlab), FadeIn(ylab),
                        *[ReplacementTransform(c, b) for c, b in zip(self.diagonal, bars)]], rt=1.3)
        self.say(f"{hl('Azul', BLUE)} (½): 1 bit.  {hl('Verde', GREEN)} (⅓): 1.58.",
                 f"{hl('Morada', PURPLE)} (1/6): 2.58 bits, la más rara.",
                 extra=[LaggedStart(*[FadeIn(l, shift=UP * .1) for l in s_labels], lag_ratio=.25)], rt=.9)

        c_labels = VGroup(*[
            mk(f"{c:.3f}", size=24, weight=BOLD, color=YELLOW).next_to(b, UP, buff=.10)
            for c, b in zip(CONTRIB, bars)
        ])
        self.say(f"{hl('Área', YELLOW)} de cada barra =",
                 "probabilidad × sorpresa.",
                 extra=[ReplacementTransform(s_labels, c_labels)], rt=.8)
        self.say(f"La {hl('morada', PURPLE)} sorprende más,", "pero sale poco:",
                 f"aporta menos que la {hl('verde', GREEN)}.",
                 extra=[Indicate(bars[1], color=PURPLE, scale_factor=1.06),
                        Indicate(c_labels[1], color=PURPLE)], rt=1.0)

        self.play(*self.set_formula(rf"H = -\sum p_k \log_2 p_k = {H_VALUE:.4f}\ \text{{bits}}", PURPLE, size=40),
                  FadeOut(self.txt), run_time=.9)
        self.txt = None
        self.say("En promedio, cada canica trae", f"{hl('1.46 bits', PURPLE)} de sorpresa.")
        self.say(f"Tope con 3 colores: log<sub>2</sub>3 ≈ {H_MAX:.3f} bits", "(de nuevo, si salieran por igual).",
                 color=MUTED)

        self.play(*[FadeOut(m) for m in (bars, baseline, xlab, ylab, c_labels, self.head, self.nav_mob)],
                  *self.clear_text(), run_time=.6)

    # ── cierre ──────────────────────────────────────────────────────
    def closing(self) -> None:
        head = self.title("Misma bolsa, dos preguntas")
        cards = VGroup()
        for (name, col), q, val in zip(STEPS, ["¿dos al azar difieren?", "¿sorpresa promedio?"],
                                       [f"{D_VALUE:.3f}", f"{H_VALUE:.2f} bits"]):
            body = VGroup(mk(name, size=34, weight=BOLD, color=col), mk(q, size=25),
                          mk(val, size=40, weight=BOLD, color=col)).arrange(DOWN, buff=.22)
            box = RoundedRectangle(corner_radius=.2, width=body.width + .4, height=body.height + .6, stroke_color=col,
                                   fill_color=col, fill_opacity=.10, stroke_width=2.5).move_to(body)
            cards.add(VGroup(box, body))
        cards.arrange(RIGHT, buff=.3).move_to([0, 2.2, 0])
        if cards.width > 7.6:
            cards.scale_to_fit_width(7.6)
        note = VGroup(mk("Son los criterios con que los", size=32),
                      mk("árboles de decisión eligen sus cortes:", size=32),
                      mk(f"{hl('CART', ORANGE)} usa Gini · {hl('ID3/C4.5', PURPLE)}, Shannon", size=32, color=MUTED)) \
            .arrange(DOWN, buff=.2).move_to([0, -.2, 0])
        who = VGroup(mk(NAME, size=38, weight=BOLD), mk(HANDLE, size=36, color=BLUE)) \
            .arrange(DOWN, buff=.16).move_to([0, -2.75, 0])
        rows = cards

        self.play(FadeIn(head, shift=DOWN * .1),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.3), run_time=1.2)
        self.wait(2.2)
        self.play(FadeIn(note, shift=UP * .1), run_time=.6)
        self.wait(3.6)
        self.play(FadeIn(who, shift=UP * .1), run_time=.6)
        self.wait(3.0)
