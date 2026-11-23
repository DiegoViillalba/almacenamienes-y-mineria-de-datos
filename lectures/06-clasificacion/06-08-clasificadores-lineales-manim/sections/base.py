"""Clase base: pausas con notas, limpieza, cintillo y encabezados."""
from __future__ import annotations

import json
from pathlib import Path

from manim import *
from manim_slides import Slide

from sections.theme import BG, BLUE, GRID, INK, LOGO_PATH, MUTED, safe_text


NARRATION = json.loads((Path(__file__).resolve().parents[1] / "narracion.json").read_text())


class BaseSlide:
    """Mezcla con utilidades de composición para ``Slide``.

    Convención de notas: ``beat(id)`` se llama *antes* de las animaciones de
    un estado; el ID consulta narración y pauta en ``narracion.json``.
    Manim Slides asocia esas notas con el estado que empieza.
    """

    branding: Group | None = None

    def beat(self: Slide, state: str, *, loop: bool = False) -> None:
        # Hold a real stable frame before every cut (also safe at 15 fps).
        if self.mobjects:
            self.wait(0.4)
        entry = NARRATION[state]
        notes = (f"E{state} · {entry['title']}\n\nNarración: {entry['narration']}"
                 f"\n\nPresentador: {entry['presenter']}")
        self.next_slide(notes=notes, loop=loop)

    def clear(self: Slide, run_time: float = 0.45) -> None:
        removable = [m for m in self.mobjects if m is not self.branding]
        if removable:
            self.play(*[FadeOut(m) for m in removable], run_time=run_time)
        for m in removable:
            for sub in m.get_family():
                sub.clear_updaters()
        self.remove(*removable)

    def logo_lockup(self: Slide, *, width: float, height: float | None = None) -> ImageMobject:
        """Logo blanco de la Facultad, sin recuadro: se apoya directo sobre el fondo."""
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width)
        if height is not None and logo.height > height:
            logo.scale_to_fit_height(height)
        return logo

    def add_branding(self: Slide) -> None:
        bar = Rectangle(width=config.frame_width, height=0.40, stroke_width=0,
                        fill_color=BG, fill_opacity=0.98).to_edge(DOWN, buff=0)
        rule = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2,
                    color=GRID, stroke_width=1.5).move_to([0, -3.59, 0])
        signature = safe_text("Diego Villalba  ·  Almacenes y Minería de Datos",
                              size=13, color=MUTED).move_to([-4.75, -3.80, 0])
        logo = self.logo_lockup(width=1.25, height=0.30).move_to([6.20, -3.80, 0])
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
