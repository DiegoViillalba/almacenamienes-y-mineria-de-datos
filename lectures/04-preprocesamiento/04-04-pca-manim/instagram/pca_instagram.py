"""Reel (9:16, ≈90 s) para Instagram: Centrar → Barrer → Autovector → Reducir.

Un solo ejemplo de principio a fin: la nube de 26 lecturas de dos sensores
del acto 4 de la clase (``sections.data.CLOUD``), con PC1 a 32°,
λ = 2.10 y 0.23, varianza total 2.33 y PVE₁ = 90.1 %.

Hilo geométrico continuo:
1. la nube se traslada para que su media quede en el origen;
2. una recta gira; cada punto se proyecta (naranja) y deja un residuo (rojo);
   la barra conservada + perdida suma siempre 2.33 (Pitágoras);
3. la covarianza S aplicada a un círculo de flechas: solo dos direcciones no
   giran y su estiramiento es la varianza del barrido;
4. la nube gira hasta que PC1 queda horizontal y se aplana sobre él.

Sin marca de curso ni de universidad: solo nombre y usuario de Instagram.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sections.data import CLOUD, CLOUD_ANGLE, CLOUD_S, CLOUD_VALS, CLOUD_VECS, cloud_split  # noqa: E402
from sections.theme import BG, INK, MUTED, GRID, BLUE, YELLOW, ORANGE, GREEN, PURPLE, RED, FONT  # noqa: E402

NAME, HANDLE = "Diego Villalba", "@diego_villalba06"
STEPS = [("Centrar", YELLOW), ("Barrer", ORANGE), ("Autovector", PURPLE), ("Reducir", BLUE)]
TOTAL = float(np.trace(CLOUD_S))                    # 2.33
PVE1 = CLOUD_VALS[0] / CLOUD_VALS.sum()             # 0.901
BEST_DEG = np.rad2deg(CLOUD_ANGLE)                  # 32°

# ── maquetación 9:16 (marco 8 × 14.22) ─────────────────────────────────
# La interfaz de Reels tapa ≈220 px arriba y ≈380 px abajo: el contenido
# vive en y ∈ [-4.1, 5.6]; la firma queda justo encima de esa franja.
NAV_Y, TITLE_Y = 5.50, 4.55
S = 5.0                                # lado del cuadro principal
C = np.array([0.0, 1.1, 0.0])          # centro del cuadro (y origen tras centrar)
UNIT = 0.85                            # escala de la nube
TEXT_Y, TEXT_Y_UNDER = -2.05, -3.0    # arriba del bloque de texto (sin / con fórmula)
FORMULA_Y = -2.15
RAW_MEAN = np.array([3.1, 2.25])       # posición de la nube antes de centrar
O0 = C + np.array([-S / 2 + .05, -S / 2 + .05, 0])  # origen de los ejes sin centrar


def sq(u: float, v: float) -> np.ndarray:
    """Coordenadas unitarias del cuadro → escena."""
    return C + np.array([(u - .5) * S, (v - .5) * S, 0])


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


def unit(a: float) -> np.ndarray:
    return np.array([np.cos(a), np.sin(a), 0])


class PCAInstagram(Scene):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.txt: Mobject | None = None
        self.fml: Mobject | None = None
        self.add(self.footer())
        self.hook()
        self.center_act()
        self.sweep_act()
        self.eigen_act()
        self.reduce_act()
        self.closing()

    # ── piezas reutilizables ──────────────────────────────────────────
    def footer(self) -> VGroup:
        rule = Line([-3.6, -4.30, 0], [3.6, -4.30, 0], color=GRID, stroke_width=1.2)
        sig = mk(f"{NAME}  ·  {hl(HANDLE, BLUE, bold=False)}", size=22, color=MUTED).move_to([0, -4.64, 0])
        return VGroup(rule, sig)

    def nav(self, active: int) -> VGroup:
        items = VGroup(*[
            mk(name, size=27, weight=BOLD if i == active else NORMAL, color=col if i == active else GRID)
            for i, (name, col) in enumerate(STEPS)
        ]).arrange(RIGHT, buff=.48)
        if items.width > 7.3:
            items.scale_to_fit_width(7.3)
        items.move_to([0, NAV_Y, 0])
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
        """Cambia la explicación inferior y espera el tiempo de lectura.

        El cambio de texto dura ≈0.5 s aunque ``extra`` (la animación que el
        texto describe) dure más: así se lee mientras la figura se mueve.
        """
        new = VGroup(*[mk(l, size=33, color=color) for l in lines]).arrange(DOWN, buff=.20)
        if new.width > 7.3:
            new.scale_to_fit_width(7.3)
        top = TEXT_Y_UNDER if self.fml is not None else TEXT_Y
        new.move_to([0, top - new.height / 2, 0])
        f = min(1.0, .5 / rt)
        swap = [FadeIn(new, shift=UP * .08, rate_func=squish_rate_func(smooth, f / 2 if self.txt else 0, f))]
        if self.txt is not None:
            swap.append(FadeOut(self.txt, rate_func=squish_rate_func(smooth, 0, f / 2)))
        self.play(*swap, *extra, run_time=rt)
        self.txt = new
        self.wait(read_time(lines) if hold is None else hold)

    def set_formula(self, mob: Mobject | None, y: float = FORMULA_Y) -> list:
        """Anima el cambio de fórmula (o su retiro) y devuelve las animaciones."""
        anims = [FadeOut(self.fml)] if self.fml is not None else []
        self.fml = mob
        if mob is not None:
            mob.move_to([0, y, 0])
            anims.append(FadeIn(mob, shift=UP * .08))
        return anims

    def clear_text(self) -> list:
        anims = [FadeOut(m) for m in (self.txt, self.fml) if m is not None]
        self.txt = self.fml = None
        return anims

    # ── 0 · gancho ────────────────────────────────────────────────────
    def hook(self) -> None:
        kicker = mk("CIENCIA DE DATOS", size=24, weight=BOLD, color=BLUE).move_to([0, NAV_Y, 0])
        self.head = self.title("¿Cuántas columnas", "necesita tu tabla?")

        right, top = sq(1, 0)[0] - .05, sq(0, 1)[1] - .05
        self.axes = VGroup(
            Arrow(O0, [right, O0[1], 0], buff=0, color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=.04),
            Arrow(O0, [O0[0], top, 0], buff=0, color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=.04),
        )
        self.ax_labels = VGroup(
            mk("sensor 1", size=20, color=MUTED).next_to([right, O0[1], 0], UP, buff=.12).align_to([right, 0, 0], RIGHT),
            mk("sensor 2", size=20, color=MUTED).rotate(PI / 2).next_to([O0[0], top, 0], RIGHT, buff=.12)
            .align_to([0, top, 0], UP),
        )
        self.raw_pts = np.array([O0 + UNIT * np.array([x + RAW_MEAN[0], y + RAW_MEAN[1], 0]) for x, y in CLOUD])
        self.dots = VGroup(*[Dot(p, radius=.075, color=BLUE) for p in self.raw_pts])

        self.play(FadeIn(kicker, shift=DOWN * .2), FadeIn(self.head, shift=DOWN * .15),
                  GrowArrow(self.axes[0]), GrowArrow(self.axes[1]), FadeIn(self.ax_labels),
                  LaggedStart(*[FadeIn(d, scale=.4) for d in self.dots], lag_ratio=.04), run_time=1.4)
        self.say(f"26 lecturas de 2 sensores:", f"{hl('2 columnas', BLUE)} por lectura.")
        self.say("Pero la nube es casi una línea…", extra=[Indicate(self.dots, color=INK, scale_factor=1.15)],
                 rt=1.0)
        self.say(f"{hl('PCA', ORANGE)} encuentra esa línea", "y te deja quedarte solo con ella.", color=MUTED)

        self.nav_mob = self.nav(0)
        self.play(FadeTransform(kicker, self.nav_mob), run_time=.5)

    # ── 1 · centrar ───────────────────────────────────────────────────
    def center_act(self) -> None:
        mean = O0 + UNIT * np.array([*RAW_MEAN, 0])
        mdot = Dot(mean, radius=.13, color=YELLOW)
        mring = Circle(radius=.24, color=YELLOW, stroke_width=3).move_to(mean)
        self.set_title("Paso 1: resta", "la media")
        self.say(f"Marca el {hl('centro', YELLOW)} de la nube:", "la media de cada sensor.",
                 extra=[GrowFromCenter(mdot), Create(mring)], rt=.7)

        shift = C - mean
        cross = VGroup(Line(sq(0, .5), sq(1, .5), color=GRID, stroke_width=2),
                       Line(sq(.5, 0), sq(.5, 1), color=GRID, stroke_width=2))
        self.say("Muévelo al origen (0, 0):", "ahora cada punto es una desviación.",
                 extra=[self.dots.animate.shift(shift), mdot.animate.shift(shift), mring.animate.shift(shift),
                        FadeOut(self.ax_labels), ReplacementTransform(self.axes, cross)], rt=1.4)
        self.cross = cross
        self.pts = self.raw_pts + shift
        self.play(FadeOut(mring), mdot.animate.scale(.6), run_time=.4)
        self.origin_dot = mdot

    # ── 2 · barrer direcciones ───────────────────────────────────────
    def sweep_act(self) -> None:
        theta = ValueTracker(100 * DEGREES)
        u = lambda: unit(theta.get_value())  # noqa: E731
        line = always_redraw(lambda: Line(C - 2.45 * u(), C + 2.45 * u(), color=ORANGE, stroke_width=5))

        def residuals() -> VGroup:
            g = VGroup()
            for p in self.pts:
                foot = C + np.dot(p - C, u()) * u()
                g.add(Line(p, foot, color=RED, stroke_width=2.2, stroke_opacity=.85),
                      Dot(foot, radius=.05, color=ORANGE))
            return g

        res = always_redraw(residuals)

        # Barra apilada: conservada (naranja) + perdida (roja) = 2.33 siempre.
        W, BAR_Y = 6.2, FORMULA_Y - .35
        left = -W / 2
        split = lambda: cloud_split(theta.get_value())  # noqa: E731

        def bars() -> VGroup:
            p, _, t = split()
            wp = W * p / t
            return VGroup(
                Rectangle(width=max(wp, .01), height=.42, stroke_width=0, fill_color=ORANGE, fill_opacity=.95)
                .move_to([left + wp / 2, BAR_Y, 0]),
                Rectangle(width=max(W - wp, .01), height=.42, stroke_width=0, fill_color=RED, fill_opacity=.85)
                .move_to([left + wp + (W - wp) / 2, BAR_Y, 0]),
            )

        bar = always_redraw(bars)
        outline = Rectangle(width=W, height=.42, stroke_color=INK, stroke_width=2).move_to([0, BAR_Y, 0])
        num_p = DecimalNumber(split()[0], num_decimal_places=2, font_size=34, color=ORANGE)
        num_r = DecimalNumber(split()[1], num_decimal_places=2, font_size=34, color=RED)
        num_a = DecimalNumber(100, num_decimal_places=0, font_size=34, color=INK, unit=r"^\circ")
        row_y = BAR_Y + .52
        row_p = VGroup(mk("conservada", size=22, color=ORANGE), num_p).arrange(RIGHT, buff=.16)
        row_r = VGroup(mk("perdida", size=22, color=RED), num_r).arrange(RIGHT, buff=.16)
        row_p.move_to([0, row_y, 0]).align_to(outline, LEFT)
        row_r.move_to([0, row_y, 0]).align_to(outline, RIGHT)
        num_a.move_to([.1, row_y, 0])
        num_p.add_updater(lambda m: m.set_value(split()[0]))
        num_r.add_updater(lambda m: m.set_value(split()[1]))
        num_a.add_updater(lambda m: m.set_value(np.rad2deg(theta.get_value())))
        meter = VGroup(outline, bar, row_p, row_r, num_a)

        self.set_title("Paso 2: busca la recta", "con más variación",
                       extra=[Transform(self.nav_mob, self.nav(1))], rt=.6)
        self.say("Traza una recta por el origen", f"y {hl('proyecta', ORANGE)} cada punto sobre ella.",
                 extra=[Create(line), FadeIn(res)], rt=1.0)

        self.fml = meter
        self.say(f"{hl('Naranja', ORANGE)}: la varianza que conserva.", f"{hl('Rojo', RED)}: la que se pierde.",
                 extra=[FadeIn(meter, shift=UP * .08)], rt=.6)
        self.say("Al girar, lo que gana el naranja", "lo pierde el rojo.",
                 extra=[theta.animate(rate_func=smooth).set_value(-40 * DEGREES)], rt=3.6, hold=.6)
        self.say(f"El máximo está en {hl(f'{BEST_DEG:.0f}°', ORANGE)}:", f"conserva 2.10 de {TOTAL:.2f}.",
                 extra=[theta.animate(rate_func=smooth).set_value(CLOUD_ANGLE)], rt=2.2)
        self.say("Esa recta es la", f"{hl('componente principal 1', ORANGE)}: {hl('PC1', ORANGE)}.",
                 extra=[Indicate(line, color=ORANGE, scale_factor=1.02)], rt=.8)

        # Pitágoras sobre un punto: por qué la suma no cambia.
        for m in (line, res, num_p, num_r, num_a):
            m.clear_updaters()
        bar.clear_updaters()
        uu = unit(CLOUD_ANGLE)
        rel = (self.pts - C)[:, :2]
        along, across = np.abs(rel @ uu[:2]), np.abs(rel @ np.array([-uu[1], uu[0]]))
        k = int(np.argmax(np.minimum(along, 2.2 * across)))
        p = self.pts[k]
        foot = C + np.dot(p - C, uu) * uu
        tri = VGroup(Line(C, p, color=INK, stroke_width=5),
                     Line(C, foot, color=ORANGE, stroke_width=8),
                     Line(foot, p, color=RED, stroke_width=8),
                     RightAngle(Line(foot, C), Line(foot, p), length=.2, color=INK, quadrant=(1, 1)))
        self.say("¿Por qué la suma es fija? Pitágoras:", f"{hl('proyección', ORANGE)}² + {hl('error', RED)}² = distancia².",
                 extra=[res.animate.set_opacity(.12), self.dots.animate.set_opacity(.3), Create(tri),
                        self.dots[k].animate.set_opacity(1).scale(1.4)], rt=1.2, hold=3.4)
        self.say("Maximizar lo conservado =", "minimizar el error.", color=MUTED)

        self.play(FadeOut(VGroup(line, res, tri, self.dots, self.origin_dot)), *self.clear_text(), run_time=.5)

    # ── 3 · autovectores ──────────────────────────────────────────────
    def eigen_act(self) -> None:
        R = 1.1
        angles = np.linspace(0, TAU, 16, endpoint=False)
        arrow = lambda v, col, w: Arrow(C, C + R * v, buff=0, color=col, stroke_width=w,  # noqa: E731
                                        max_tip_length_to_length_ratio=.2)
        arrows = VGroup(*[arrow(unit(a), BLUE, 3.5) for a in angles])
        circle = Circle(radius=R, color=MUTED, stroke_width=2).move_to(C)
        S_tex = MathTex(rf"S=\begin{{pmatrix}}{CLOUD_S[0, 0]:.2f}&{CLOUD_S[0, 1]:.2f}\\"
                        rf"{CLOUD_S[1, 0]:.2f}&{CLOUD_S[1, 1]:.2f}\end{{pmatrix}}", font_size=34, color=YELLOW)
        S_tex.move_to(sq(.22, .87))

        self.set_title("Paso 3: ¿por qué 32°?", "La covarianza lo sabe",
                       extra=[Transform(self.nav_mob, self.nav(2))], rt=.6)
        self.say("Toma flechas en todas", "las direcciones…",
                 extra=[Create(circle), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=.04)], rt=1.2)
        self.say(f"…y multiplícalas por la", f"{hl('matriz de covarianza S', YELLOW)} de la nube.",
                 extra=[Write(S_tex)], rt=.8)

        M = np.eye(3)
        M[:2, :2] = CLOUD_S
        image = lambda a, col, w: arrow(np.r_[CLOUD_S @ unit(a)[:2], 0], col, w)  # noqa: E731
        self.say("Casi todas cambian", "de dirección.",
                 extra=[ApplyMatrix(M, circle, about_point=C),
                        *[Transform(a, image(t, BLUE, 3.5)) for a, t in zip(arrows, angles)]], rt=2.2)

        eig_angles = [np.arctan2(CLOUD_VECS[1, j], CLOUD_VECS[0, j]) for j in range(2)]
        eig = VGroup(*[arrow(unit(a), c, 7) for a, c in zip(eig_angles, [ORANGE, GREEN])])
        self.say("Dos solo se estiran sobre su recta:", f"son los {hl('autovectores', PURPLE)} de S.",
                 extra=[arrows.animate.set_opacity(.25), circle.animate.set_stroke(opacity=.4),
                        Succession(FadeIn(eig, run_time=.6),
                                   AnimationGroup(*[Transform(e, image(a, c, 7))
                                                    for e, a, c in zip(eig, eig_angles, [ORANGE, GREEN])],
                                                  run_time=1.4))], rt=2.0)

        eq = MathTex(rf"S\mathbf v_1={CLOUD_VALS[0]:.2f}\,\mathbf v_1", rf"\qquad S\mathbf v_2={CLOUD_VALS[1]:.2f}\,\mathbf v_2",
                     font_size=46)
        eq[0].set_color(ORANGE)
        eq[1].set_color(GREEN)
        self.play(*self.set_formula(eq), FadeOut(self.txt), run_time=.7)
        self.txt = None
        self.say(f"{hl('λ<sub>1</sub> = 2.10', ORANGE)}: la varianza que PC1 conserva,", f"la misma del barrido.")
        self.say(f"{hl('λ<sub>2</sub> = 0.23', GREEN)}: lo que se pierde (PC2).", f"Suman {TOTAL:.2f}, el total.")
        self.play(FadeOut(VGroup(arrows, circle, eig, S_tex)), *self.clear_text(), run_time=.5)

    # ── 4 · reducir ───────────────────────────────────────────────────
    def reduce_act(self) -> None:
        dots = VGroup(*[Dot(p, radius=.075, color=BLUE) for p in self.pts])
        u1, u2 = unit(CLOUD_ANGLE), unit(CLOUD_ANGLE + PI / 2)
        pc1 = Line(C - 2.45 * u1, C + 2.45 * u1, color=ORANGE, stroke_width=5)
        pc2 = Line(C - 1.6 * u2, C + 1.6 * u2, color=GREEN, stroke_width=4)
        lab1 = mk("PC1", size=26, weight=BOLD, color=ORANGE).move_to(C + 2.2 * u1 + .38 * u2)
        lab2 = mk("PC2", size=26, weight=BOLD, color=GREEN).move_to(C + 1.6 * u2 + .45 * u1)
        group = VGroup(pc1, pc2, dots)

        self.set_title("Paso 4: gira los ejes", "y quédate con PC1",
                       extra=[Transform(self.nav_mob, self.nav(3))], rt=.6)
        self.say(f"Los nuevos ejes: {hl('PC1', ORANGE)} sigue la nube,", f"{hl('PC2', GREEN)} es perpendicular.",
                 extra=[FadeIn(dots), Create(pc1), Create(pc2), FadeIn(lab1), FadeIn(lab2)], rt=1.0)

        lab1_end = mk("PC1", size=26, weight=BOLD, color=ORANGE).move_to(C + np.array([2.2, .38, 0]))
        lab2_end = mk("PC2", size=26, weight=BOLD, color=GREEN).move_to(C + np.array([.45, 1.6, 0]))
        self.say(f"Gira todo {BEST_DEG:.0f}°: {hl('PC1', ORANGE)} queda horizontal.",
                 extra=[Rotate(group, -CLOUD_ANGLE, about_point=C), FadeOut(self.cross),
                        Transform(lab1, lab1_end), Transform(lab2, lab2_end)], rt=1.6)

        drops = VGroup(*[Line(d.get_center(), [d.get_x(), C[1], 0], color=RED, stroke_width=2.2) for d in dots])
        self.say("Borra la coordenada de PC2:", f"cada punto cae sobre {hl('PC1', ORANGE)}.",
                 extra=[Create(drops)], rt=.8, hold=.4)
        self.play(*[d.animate.set_y(C[1]).set_color(ORANGE) for d in dots], FadeOut(drops), FadeOut(pc2),
                  FadeOut(lab2), run_time=1.4)

        Wb = 6.2
        w1 = Wb * PVE1
        pve = VGroup(
            Rectangle(width=w1, height=.42, stroke_width=0, fill_color=ORANGE, fill_opacity=.95)
            .move_to([-Wb / 2 + w1 / 2, 0, 0]),
            Rectangle(width=Wb - w1, height=.42, stroke_width=0, fill_color=GREEN, fill_opacity=.9)
            .move_to([-Wb / 2 + w1 + (Wb - w1) / 2, 0, 0]),
        )
        pl = VGroup(mk(f"PC1 {100 * PVE1:.1f}%", size=24, weight=BOLD, color=ORANGE)
                    .next_to(pve, UP, buff=.12).align_to(pve, LEFT),
                    mk(f"PC2 {100 * (1 - PVE1):.1f}%", size=24, weight=BOLD, color=GREEN)
                    .next_to(pve, UP, buff=.12).align_to(pve, RIGHT))
        meter = VGroup(pl, pve)
        self.play(*self.set_formula(meter, y=FORMULA_Y - .1), FadeOut(self.txt), run_time=.6)
        self.txt = None
        self.say(f"De 2 columnas a 1, y aún guardas", f"el {hl(f'{100 * PVE1:.1f}%', ORANGE)} de la varianza.")
        self.say("Con 6 columnas o 600,", "la receta es la misma.", color=MUTED)
        self.play(*[FadeOut(m) for m in (dots, pc1, lab1, self.head, self.nav_mob)], *self.clear_text(),
                  run_time=.6)

    # ── cierre ──────────────────────────────────────────────────────
    def closing(self) -> None:
        head = self.title("PCA en 4 pasos")
        rows = VGroup()
        chip_w = 2.75
        for (name, col), text in zip(STEPS, ["resta la media",
                                             "la recta de más varianza",
                                             "esa recta: Su = λu",
                                             "quédate con las primeras"]):
            chip = RoundedRectangle(corner_radius=.14, width=chip_w, height=.74, stroke_color=col,
                                    fill_color=col, fill_opacity=.14, stroke_width=2.5)
            tag = mk(name, size=28, weight=BOLD, color=col).move_to(chip)
            desc = mk(text, size=28)
            rows.add(VGroup(VGroup(chip, tag), desc).arrange(RIGHT, buff=.3))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=.3)
        if rows.width > 7.3:
            rows.scale_to_fit_width(7.3)
        rows.move_to([0, 1.85, 0])
        vals = mk(f"{hl(f'PC1 {BEST_DEG:.0f}°', ORANGE)} · {hl('λ<sub>1</sub> 2.10', ORANGE)} · "
                  f"{hl(f'{100 * PVE1:.1f}% con 1 columna', BLUE)}", size=28).move_to([0, -.75, 0])
        note = mk("Muchas columnas, pocas direcciones.", size=32, color=YELLOW).move_to([0, -1.5, 0])
        who = VGroup(mk(NAME, size=38, weight=BOLD), mk(HANDLE, size=36, color=BLUE)) \
            .arrange(DOWN, buff=.16).move_to([0, -3.0, 0])
        self.play(FadeIn(head, shift=DOWN * .1),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.2), run_time=1.3)
        self.wait(2.8)
        self.play(FadeIn(vals), FadeIn(note, shift=UP * .1), run_time=.6)
        self.wait(2.4)
        self.play(FadeIn(who, shift=UP * .1), run_time=.6)
        self.wait(3.0)
