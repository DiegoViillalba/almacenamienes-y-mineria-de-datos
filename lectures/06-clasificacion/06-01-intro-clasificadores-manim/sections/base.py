"""Clase base: pausas con notas, limpieza, cintillo y encabezados."""
from __future__ import annotations

import re

from manim import *
from manim_slides import Slide

from sections.theme import BG, BLUE, GRID, INK, LOGO_PATH, MUTED, PANEL, safe_text


class BaseSlide:
    """Mezcla con utilidades de composición para ``Slide``.

    Convención de notas: ``beat(notas)`` se llama *antes* de las animaciones de
    un estado; Manim Slides asocia esas notas con el estado que empieza.
    """

    branding: Group | None = None
    video_mode = False

    def beat(self: Slide, notes: str, *, loop: bool = False) -> None:
        if self.renderer.time > 0:
            # Registrar el estado ya asentado, incluidos los cambios de orden
            # que ocurren fuera de play(). El HTML pausa sobre este fotograma.
            self.wait(0.25)
        if self.video_mode:
            # Los tiempos del aula no son marcas de edición del video.
            notes = re.sub(r"\b\d+(?:[–−-]\d+)?\s*min\.?\s*", "", notes)
            notes = notes.replace("Pida a la clase", "Invite a quien mira")
            notes = notes.replace("con la clase", "con quien mira")
            notes = notes.replace("en parejas", "individual, pausando el video")
            notes = notes.replace("Pida que griten «¡alto!»", "Invite a predecir el mejor corte")
            notes = notes.replace("No revele el círculo hasta que la mayoría tenga respuesta",
                                  "Invitar a pausar antes de revelar el círculo")
            notes = notes.replace("en papel", "al pausar el video")
            notes = "Grabación · " + notes
        self.next_slide(notes=notes, loop=loop)

    def clear(self: Slide, run_time: float = 0.45) -> None:
        removable = [m for m in self.mobjects if m is not self.branding]
        for m in removable:
            m.suspend_updating()
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
                        fill_color=PANEL, fill_opacity=0.98).to_edge(DOWN, buff=0)
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
