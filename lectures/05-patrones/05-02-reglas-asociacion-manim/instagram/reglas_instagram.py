"""Reel (9:16, ≈90 s) para Instagram: Regla → Soporte → Confianza → Lift.

Un solo ejemplo de principio a fin: 20 pacientes (datos ilustrativos) con
sus diagnósticos de diabetes, hipertensión y asma. La regla
Diabetes ⇒ Hipertensión tiene soporte 5/20 = 25 %, confianza 5/6 = 83 % y
lift 0.83 / 0.40 = 2.1; el contraste Asma ⇒ Hipertensión da lift 1.0.

Hilo visual continuo:
1. la cuadrícula de pacientes es la tabla de transacciones;
2. soporte: se marcan los pacientes con ambas enfermedades (fracción del total);
3. confianza: se apagan quienes no tienen diabetes (fracción dentro de A);
4. lift: cada paciente se vuelve una celda de una barra; la línea amarilla
   del 40 % es lo esperado al azar y las barras de diabetes y asma se comparan
   contra ella.

Las métricas de evaluación (leverage, conviction, …) quedan para otro video.
Sin marca de curso ni de universidad: solo nombre y usuario de Instagram.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sections.theme import BG, PANEL, INK, MUTED, GRID, BLUE, ORANGE, GREEN, YELLOW, PURPLE, FONT  # noqa: E402

NAME, HANDLE = "Diego Villalba", "@diego_villalba06"
PINK = "#F28CB1"                       # asma: solo en este clip

# ── datos ilustrativos: 20 pacientes ──────────────────────────────────
N = 20
DIAB = {0, 3, 6, 8, 11, 15}
HTA = {0, 1, 3, 5, 8, 11, 15, 17}
ASMA = {2, 5, 9, 13, 17}
CONDS = [(DIAB, BLUE), (HTA, ORANGE), (ASMA, PINK)]   # posición fija del punto en la ficha

BOTH = DIAB & HTA
SUP = len(BOTH) / N                    # 0.25
CONF = len(BOTH) / len(DIAB)           # 0.83
BASE = len(HTA) / N                    # 0.40
LIFT = CONF / BASE                     # 2.08
CONF_ASMA = len(ASMA & HTA) / len(ASMA)
LIFT_ASMA = CONF_ASMA / BASE           # 1.00
assert (len(DIAB), len(HTA), len(BOTH), len(ASMA & HTA)) == (6, 8, 5, 2)
assert abs(LIFT_ASMA - 1) < 1e-9


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


STEPS = [("Regla", INK), ("Soporte", GREEN), ("Confianza", ORANGE), ("Lift", PURPLE)]

# ── maquetación 9:16 (marco 8 × 14.22) ─────────────────────────────────
# La interfaz de Reels tapa ≈220 px arriba y ≈380 px abajo: el contenido
# vive en y ∈ [-4.1, 5.6]; la firma queda justo encima de esa franja.
NAV_Y, TITLE_Y = 5.50, 4.55
COLS, TILE, PITCH = 5, 1.12, 1.26
GRID_Y = 1.25                          # centro de la cuadrícula de pacientes
LEGEND_Y = -1.62
TEXT_Y, TEXT_Y_UNDER = -2.05, -3.10    # arriba del bloque de texto (sin / con fórmula)
FORMULA_Y = -2.42
BAR_W, BAR_H, BAR_X0 = 6.4, .62, -3.2  # barras del acto de lift


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


def tile_center(i: int) -> np.ndarray:
    col, row = i % COLS, i // COLS
    return np.array([(col - 2) * PITCH, GRID_Y + (1.5 - row) * PITCH, 0])


def patient(i: int) -> VGroup:
    """Ficha: silueta + tres puntos (diabetes, hipertensión, asma); aro vacío = no."""
    c = tile_center(i)
    box = RoundedRectangle(corner_radius=.14, width=TILE, height=TILE, stroke_color=GRID,
                           stroke_width=1.6, fill_color=PANEL, fill_opacity=1).move_to(c)
    head = Circle(radius=.13, stroke_width=0, fill_color=MUTED, fill_opacity=1).move_to(c + [0, .27, 0])
    body = AnnularSector(inner_radius=0, outer_radius=.25, angle=PI, start_angle=0, stroke_width=0,
                         fill_color=MUTED, fill_opacity=1).move_to(c + [0, .02, 0])
    chips = VGroup()
    for k, (members, col) in enumerate(CONDS):
        pos = c + [(k - 1) * .31, -.33, 0]
        if i in members:
            chips.add(Circle(radius=.115, stroke_width=0, fill_color=col, fill_opacity=1).move_to(pos))
        else:
            chips.add(Circle(radius=.09, stroke_color=GRID, stroke_width=1.6, fill_opacity=0).move_to(pos))
    return VGroup(box, VGroup(head, body), chips)


def bar_row(members: list[int], y: float) -> VGroup:
    """Una celda por paciente; las de hipertensión primero, para leerla como barra."""
    order = sorted(members, key=lambda i: i not in HTA)
    w = BAR_W / len(order)
    return VGroup(*[
        Rectangle(width=w, height=BAR_H, stroke_color=BG, stroke_width=1.5,
                  fill_color=ORANGE if i in HTA else PANEL, fill_opacity=1)
        .move_to([BAR_X0 + w * (k + .5), y, 0])
        for k, i in enumerate(order)
    ]), order


class ReglasInstagram(Scene):
    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        self.txt: Mobject | None = None
        self.fml: Mobject | None = None
        self.add(self.footer())
        self.hook()
        self.rule_act()
        self.support_act()
        self.confidence_act()
        self.lift_act()
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

    def show_formula(self, mob: Mobject, y: float = FORMULA_Y) -> None:
        """Pone una fórmula y retira el texto: el siguiente ``say`` va debajo."""
        anims = self.set_formula(mob, y)
        if self.txt is not None:
            anims.append(FadeOut(self.txt))
        self.play(*anims, run_time=.7)
        self.txt = None

    def clear_text(self) -> list:
        anims = [FadeOut(m) for m in (self.txt, self.fml) if m is not None]
        self.txt = self.fml = None
        return anims

    def focus(self, keep: set[int], color: str) -> list:
        """Apaga las fichas fuera de ``keep`` y bordea las de dentro con ``color``."""
        anims = []
        for i, t in enumerate(self.tiles):
            target = patient(i)
            if i in keep:
                target[0].set_stroke(color=color, width=4)
            else:
                for m in target.family_members_with_points():
                    m.set_fill(opacity=m.get_fill_opacity() * .18)
                    m.set_stroke(opacity=m.get_stroke_opacity() * .18)
            anims.append(Transform(t, target))
        return anims

    def reset_tiles(self) -> None:
        """Devuelve todas las fichas a su aspecto original."""
        fresh = [patient(i) for i in range(N)]
        self.play(*[Transform(t, f) for t, f in zip(self.tiles, fresh)], run_time=.6)

    # ── 0 · gancho ────────────────────────────────────────────────────
    def hook(self) -> None:
        kicker = mk("MINERÍA DE DATOS", size=24, weight=BOLD, color=BLUE).move_to([0, NAV_Y, 0])
        self.head = self.title("¿Qué enfermedades", "aparecen juntas?")
        self.tiles = VGroup(*[patient(i) for i in range(N)])
        dot = lambda col: Circle(radius=.1, stroke_width=0, fill_color=col, fill_opacity=1)  # noqa: E731
        items = [(dot(col), mk(name, size=24, color=col)) for name, col in
                 [("diabetes", BLUE), ("hipertensión", ORANGE), ("asma", PINK)]]
        self.legend = VGroup(*[VGroup(d, t).arrange(RIGHT, buff=.12) for d, t in items]).arrange(RIGHT, buff=.42)
        self.legend.move_to([0, LEGEND_Y, 0])

        self.play(FadeIn(kicker, shift=DOWN * .2), FadeIn(self.head, shift=DOWN * .15),
                  LaggedStart(*[FadeIn(t, scale=.7) for t in self.tiles], lag_ratio=.05),
                  FadeIn(self.legend), run_time=1.6)
        self.say("20 pacientes y sus diagnósticos", hl("(datos ilustrativos)", MUTED, bold=False))
        self.say("Es el mismo problema del súper:", "¿qué productos se compran juntos?",
                 extra=[LaggedStart(*[Indicate(t.submobjects[2], color=INK, scale_factor=1.3)
                                      for t in self.tiles], lag_ratio=.03)], rt=1.2)
        self.say(f"Cada paciente es una {hl('canasta', YELLOW)}", "de diagnósticos.", color=MUTED)

        self.nav_mob = self.nav(0)
        self.play(FadeTransform(kicker, self.nav_mob), run_time=.5)

    # ── 1 · la regla ──────────────────────────────────────────────────
    def rule_act(self) -> None:
        a = mk("Diabetes", size=40, weight=BOLD, color=BLUE)
        arrow = MathTex(r"\Rightarrow", font_size=56, color=INK)
        b = mk("Hipertensión", size=40, weight=BOLD, color=ORANGE)
        rule = VGroup(a, arrow, b).arrange(RIGHT, buff=.3)
        la = mk("antecedente", size=21, color=MUTED).next_to(a, DOWN, buff=.12)
        lb = mk("consecuente", size=21, color=MUTED).next_to(b, DOWN, buff=.12)
        self.rule = VGroup(rule, la, lb)

        self.set_title("Una regla: si A,", "entonces B")
        self.show_formula(self.rule)
        self.say(f"Si un paciente tiene {hl('diabetes', BLUE)},", f"¿suele tener {hl('hipertensión', ORANGE)}?",
                 extra=self.focus(DIAB, BLUE), rt=1.0)
        self.say("Ojo: no dice que una cause la otra,", "solo que aparecen juntas.", color=MUTED)
        self.say("Para creerle, la medimos", f"con {hl('soporte', GREEN)}, {hl('confianza', ORANGE)} y {hl('lift', PURPLE)}.")
        self.play(*self.clear_text(), run_time=.4)
        self.reset_tiles()

    # ── 2 · soporte ───────────────────────────────────────────────────
    def support_act(self) -> None:
        self.set_title("Soporte: ¿qué tan", "común es la pareja?",
                       extra=[Transform(self.nav_mob, self.nav(1))])
        self.say(f"Cuenta a quienes tienen {hl('diabetes', BLUE)}", f"{hl('y', INK)} {hl('hipertensión', ORANGE)} a la vez.",
                 extra=self.focus(BOTH, GREEN), rt=1.0, hold=.8)
        badges = VGroup()
        for k, i in enumerate(sorted(BOTH)):
            c = tile_center(i) + [TILE / 2 - .05, TILE / 2 - .05, 0]
            badges.add(VGroup(Circle(radius=.2, stroke_width=0, fill_color=GREEN, fill_opacity=1),
                              mk(str(k + 1), size=22, weight=BOLD, color=BG)).move_to(c))
        self.play(LaggedStart(*[FadeIn(bd, scale=.4) for bd in badges], lag_ratio=.35), run_time=1.6)
        self.wait(.5)

        sup = MathTex(r"\text{soporte}", r"=\frac{5}{20}", rf"={100 * SUP:.0f}\%", font_size=50, color=GREEN)
        self.show_formula(sup)
        self.say(f"{hl('1 de cada 4', GREEN)} pacientes tiene ambas.")
        self.say("Con soporte muy bajo, el patrón", "puede ser pura casualidad.", color=MUTED)
        self.play(FadeOut(badges), *self.clear_text(), run_time=.5)
        self.reset_tiles()

    # ── 3 · confianza ─────────────────────────────────────────────────
    def confidence_act(self) -> None:
        self.set_title("Confianza: ¿qué tan", "fiable es la regla?",
                       extra=[Transform(self.nav_mob, self.nav(2))])
        self.say(f"Mira solo a quienes tienen", f"{hl('diabetes', BLUE)}: son {hl(str(len(DIAB)), BLUE)}.",
                 extra=self.focus(DIAB, BLUE), rt=1.0)
        hta_chips = [self.tiles[i].submobjects[2][1] for i in sorted(BOTH)]
        self.say(f"De esos {len(DIAB)}, {hl(str(len(BOTH)), ORANGE)} tienen {hl('hipertensión', ORANGE)}.",
                 extra=[LaggedStart(*[Indicate(ch, color=ORANGE, scale_factor=2) for ch in hta_chips],
                                    lag_ratio=.25)], rt=1.4)
        conf = MathTex(r"\text{confianza}", r"=\frac{5}{6}", rf"={100 * CONF:.0f}\%", font_size=50, color=ORANGE)
        self.show_formula(conf)
        self.say(f"Con diabetes, hay hipertensión", f"en {hl('83 de cada 100', ORANGE)} casos.")
        self.say("¿83% es mucho? Depende de qué", "tan común sea la hipertensión…", color=MUTED)
        self.play(*self.clear_text(), run_time=.4)

    # ── 4 · lift ──────────────────────────────────────────────────────
    def lift_act(self) -> None:
        y_all, y_d, y_a = 2.75, 1.05, -.65
        row_all, order_all = bar_row(list(range(N)), y_all)
        row_d, order_d = bar_row(sorted(DIAB), y_d)
        row_a, order_a = bar_row(sorted(ASMA), y_a)

        def label(text: str, value: float, col: str, y: float) -> VGroup:
            l = mk(text, size=24, color=MUTED).move_to([0, y + BAR_H / 2 + .26, 0]).align_to([BAR_X0, 0, 0], LEFT)
            v = mk(pct(value), size=28, weight=BOLD, color=col)
            v.move_to([0, y + BAR_H / 2 + .26, 0]).align_to([BAR_X0 + BAR_W, 0, 0], RIGHT)
            l.add_background_rectangle(color=BG, opacity=1, buff=.06)
            return VGroup(l, v).set_z_index(2)

        lab_all = label(f"Todos ({N})", BASE, YELLOW, y_all)
        lab_d = label(f"Con {hl('diabetes', BLUE)} ({len(DIAB)})", CONF, ORANGE, y_d)
        lab_a = label(f"Con {hl('asma', PINK)} ({len(ASMA)})", CONF_ASMA, ORANGE, y_a)

        xb = BAR_X0 + BASE * BAR_W
        base_line = DashedLine([xb, y_all + BAR_H / 2 + .12, 0], [xb, y_a - BAR_H / 2 - .5, 0],
                               color=YELLOW, stroke_width=4, dash_length=.12).set_z_index(1)
        base_tag = mk("lo esperado", size=22, weight=BOLD, color=YELLOW).next_to(base_line, DOWN, buff=.1)

        self.set_title("Lift: ¿más de lo", "esperado?", extra=[Transform(self.nav_mob, self.nav(3))])
        cells = {i: row_all[k] for k, i in enumerate(order_all)}
        self.say("Cada paciente se vuelve una celda;", f"las {hl('naranjas', ORANGE)} tienen hipertensión.",
                 extra=[FadeOut(self.legend),
                        *[ReplacementTransform(self.tiles[i], cells[i]) for i in range(N)],
                        FadeIn(lab_all[0])], rt=1.6)
        self.say(f"En todos: {hl('8 de 20', ORANGE)} = {hl('40%', YELLOW)}.", "Eso esperaríamos al azar.",
                 extra=[FadeIn(lab_all[1]), Create(base_line), FadeIn(base_tag)], rt=1.0)

        self.say(f"Entre quienes tienen {hl('diabetes', BLUE)}:", f"{hl('83%', ORANGE)}, muy arriba de la línea.",
                 extra=[*[TransformFromCopy(cells[i], row_d[k]) for k, i in enumerate(order_d)],
                        FadeIn(lab_d)], rt=1.4)
        lift = MathTex(r"\text{lift}", r"=\frac{83\%}{40\%}", rf"={LIFT:.1f}", font_size=50, color=PURPLE)
        self.show_formula(lift)
        self.say("Con diabetes, la hipertensión es", f"{hl('2.1 veces', PURPLE)} más frecuente.")

        lift_a = MathTex(r"\text{lift}", r"=\frac{40\%}{40\%}", rf"={LIFT_ASMA:.1f}", font_size=50, color=YELLOW)
        self.play(*[TransformFromCopy(cells[i], row_a[k]) for k, i in enumerate(order_a)], FadeIn(lab_a),
                  *self.set_formula(lift_a), FadeOut(self.txt), run_time=1.4)
        self.txt = None
        self.say(f"Contraste: con {hl('asma', PINK)}, 2 de 5 = 40%.", "Justo en la línea: no cambia nada.")
        self.play(*self.set_formula(None), FadeOut(self.txt), run_time=.5)
        self.txt = None
        self.say(f"{hl('Lift = 1', YELLOW)}: no hay asociación.", f"{hl('Lift &gt; 1', PURPLE)}: aparecen juntas más",
                 "de lo que el azar explicaría.")
        self.play(*[FadeOut(m) for m in (row_all, row_d, row_a, lab_all, lab_d, lab_a, base_line, base_tag,
                                          self.head, self.nav_mob)], *self.clear_text(), run_time=.6)

    # ── cierre ──────────────────────────────────────────────────────
    def closing(self) -> None:
        head = self.title("Reglas de asociación")
        rows = VGroup()
        chip_w = 2.75
        for (name, col), text in zip(STEPS, ["si A, entonces B",
                                             f"¿qué tan común? {pct(SUP)}",
                                             f"¿qué tan fiable? {pct(CONF)}",
                                             f"¿más que el azar? {LIFT:.1f}×"]):
            chip = RoundedRectangle(corner_radius=.14, width=chip_w, height=.74, stroke_color=col,
                                    fill_color=col, fill_opacity=.14, stroke_width=2.5)
            tag = mk(name, size=28, weight=BOLD, color=col).move_to(chip)
            desc = mk(text, size=28)
            rows.add(VGroup(VGroup(chip, tag), desc).arrange(RIGHT, buff=.3))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=.3)
        if rows.width > 7.3:
            rows.scale_to_fit_width(7.3)
        rows.move_to([0, 1.85, 0])
        rule = mk(f"{hl('Diabetes', BLUE)} ⇒ {hl('Hipertensión', ORANGE)}", size=30, weight=BOLD).move_to([0, -.6, 0])
        note = mk("Juntas no significa causa.", size=32, color=YELLOW).move_to([0, -1.35, 0])
        nxt = mk("Próximo video: cómo evaluar reglas", size=26, color=MUTED).move_to([0, -2.0, 0])
        who = VGroup(mk(NAME, size=38, weight=BOLD), mk(HANDLE, size=36, color=BLUE)) \
            .arrange(DOWN, buff=.16).move_to([0, -3.2, 0])
        self.play(FadeIn(head, shift=DOWN * .1),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.2), run_time=1.3)
        self.wait(2.8)
        self.play(FadeIn(rule), FadeIn(note, shift=UP * .1), run_time=.6)
        self.wait(2.2)
        self.play(FadeIn(nxt), FadeIn(who, shift=UP * .1), run_time=.6)
        self.wait(3.0)
