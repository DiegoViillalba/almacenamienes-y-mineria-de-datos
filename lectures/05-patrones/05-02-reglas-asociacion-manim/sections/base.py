"""Clase base: pausas con notas, limpieza, cintillo, encabezados y el cuadrado."""
from __future__ import annotations

from manim import *
from manim_slides import Slide

from sections.theme import (BG, BLUE, GRID, INK, LOGO_PATH, MUTED, ORANGE, YELLOW,
                            safe_text)


class BaseSlide:
    """Mezcla con utilidades de composición para ``Slide``.

    Convención de notas: ``beat(notas)`` se llama *antes* de las animaciones de
    un estado; Manim Slides asocia esas notas con el estado que empieza.
    """

    branding: Group | None = None

    def beat(self: Slide, notes: str, *, loop: bool = False) -> None:
        self.next_slide(notes=notes, loop=loop)

    def clear(self: Slide, run_time: float = 0.45) -> None:
        removable = [m for m in self.mobjects if m is not self.branding]
        if removable:
            self.play(*[FadeOut(m) for m in removable], run_time=run_time)
        for m in removable:
            for sub in m.get_family():
                sub.clear_updaters()
        self.remove(*removable)

    def logo_lockup(self: Slide, *, width: float, height: float) -> Group:
        box = RoundedRectangle(corner_radius=0.10, width=width, height=height,
                               stroke_color=GRID, stroke_width=1.2, stroke_opacity=0.65,
                               fill_color=BG, fill_opacity=0.35)
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width - 0.26)
        if logo.height > height - 0.18:
            logo.scale_to_fit_height(height - 0.18)
        logo.move_to(box)
        return Group(box, logo)

    def add_branding(self: Slide) -> None:
        bar = Rectangle(width=config.frame_width, height=0.40, stroke_width=0,
                        fill_color="#050D18", fill_opacity=0.98).to_edge(DOWN, buff=0)
        rule = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2,
                    color=GRID, stroke_width=1.5).move_to([0, -3.59, 0])
        signature = safe_text("Diego Villalba  ·  Almacenes y Minería de Datos",
                              size=13, color=MUTED).move_to([-4.75, -3.80, 0])
        logo = self.logo_lockup(width=1.42, height=0.34).move_to([6.12, -3.80, 0])
        self.branding = Group(bar, rule, signature, logo).set_z_index(100)
        self.add(self.branding)

    def heading(self: Slide, kicker: str, title: str, color: str = BLUE) -> VGroup:
        kicker_mob = safe_text(kicker.upper(), size=20, color=color, weight=BOLD)
        title_mob = safe_text(title, size=40, color=INK, weight=BOLD, max_width=12.4)
        block = VGroup(kicker_mob, title_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        block.to_corner(UL, buff=0.45)
        rule = Line(LEFT * 0.55, RIGHT * 0.55, color=color, stroke_width=4)
        rule.next_to(block, DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(block, rule)

    def start(self: Slide, kicker: str, title: str, color: str = BLUE) -> VGroup:
        """Limpia la pantalla y muestra el encabezado de un nuevo estado."""
        self.clear()
        head = self.heading(kicker, title, color)
        self.play(FadeIn(head), run_time=0.35)
        return head

    def swap_heading(self: Slide, old: VGroup, kicker: str, title: str,
                     color: str = BLUE) -> VGroup:
        new = self.heading(kicker, title, color)
        self.play(FadeOut(old), FadeIn(new), run_time=0.4)
        return new


class UnitSquare(VGroup):
    """Cuadrado unitario de probabilidad: área total = 1 = todos los tickets.

    La columna izquierda (ancho ``pa``) son los tickets con A. El consecuente B
    se dibuja como "agua": altura ``h_a`` dentro de la columna A y ``h_na``
    fuera de ella. Bajo independencia ambas alturas valen P(B).
    """

    def __init__(self, side: float, pa: float, h_a: float, h_na: float,
                 *, show_a: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.side = side
        self.frame = Square(side_length=side, stroke_color=INK, stroke_width=2.5,
                            fill_opacity=0)
        self.a_col = Rectangle(width=pa * side, height=side, stroke_width=0,
                               fill_color=BLUE, fill_opacity=0.16 if show_a else 0)
        self.a_col.align_to(self.frame, LEFT).align_to(self.frame, DOWN)
        self.b_in = self._water(0, pa, h_a)
        self.b_out = self._water(pa, 1, h_na)
        self.divider = Line(self.p(pa, 0), self.p(pa, 1), color=BLUE,
                            stroke_width=3 if show_a else 0)
        self.add(self.a_col, self.b_in, self.b_out, self.frame, self.divider)

    def p(self, x: float, y: float) -> np.ndarray:
        """Coordenada de pantalla de un punto (x, y) del cuadrado unitario."""
        o = self.frame.get_corner(DL)
        return o + RIGHT * x * self.side + UP * y * self.side

    def _water(self, x0: float, x1: float, h: float) -> Rectangle:
        rect = Rectangle(width=max(1e-3, (x1 - x0) * self.side),
                         height=max(1e-3, h * self.side), stroke_width=0,
                         fill_color=ORANGE, fill_opacity=0.78)
        rect.move_to(self.p((x0 + x1) / 2, h / 2))
        return rect

    def level_line(self, y: float, color: str = YELLOW, x0: float = 0,
                   x1: float = 1) -> DashedLine:
        return DashedLine(self.p(x0, y), self.p(x1, y), color=color, stroke_width=3,
                          dash_length=0.12)

    def level_label(self, content: str, y: float, color: str = YELLOW,
                    size: float = 26) -> MathTex:
        """Etiqueta del nivel dentro del cuadrado, sobre la línea, junto al borde derecho."""
        lab = MathTex(content, font_size=size, color=color)
        lab.next_to(self.p(1, y), UP + LEFT, buff=0.08)
        lab.add_background_rectangle(color=BG, opacity=0.75, buff=0.04)
        return lab

    def region(self, x0: float, x1: float, y0: float, y1: float, color: str,
               opacity: float = 0.55) -> Rectangle:
        rect = Rectangle(width=max(1e-3, (x1 - x0) * self.side),
                         height=max(1e-3, (y1 - y0) * self.side), stroke_color=color,
                         stroke_width=2, fill_color=color, fill_opacity=opacity)
        rect.move_to(self.p((x0 + x1) / 2, (y0 + y1) / 2))
        return rect
