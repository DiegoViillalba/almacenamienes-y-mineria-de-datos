"""Clase base con utilidades de composición, pausas y branding institucional."""

from __future__ import annotations

from pathlib import Path
from manim import *
from manim_slides import Slide

from sections.theme import (
    BG, INK, MUTED, GRID, BLUE, ORANGE, PURPLE, YELLOW, GREEN, RED,
    CAT_COLORS, FONT, LOGO_PATH, safe_text, pill, metric_card
)


class BaseSlide:
    """Composición compartida para la presentación."""

    branding: Group | None = None

    def pause(self: Slide, notes: str) -> None:
        self.next_slide(notes=notes)

    def clear_content(self: Slide, run_time: float = 0.42) -> None:
        removable = [mob for mob in self.mobjects if mob is not getattr(self, "branding", None)]
        if removable:
            self.play(*[FadeOut(mob) for mob in removable], run_time=run_time)

    def logo_lockup(self: Slide, width: float, height: float) -> Group:
        card = RoundedRectangle(
            corner_radius=0.10,
            width=width,
            height=height,
            stroke_color=GRID,
            stroke_width=1.2,
            fill_color=BG,
            fill_opacity=0.30,
        )
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width - 0.25)
        if logo.height > height - 0.16:
            logo.scale_to_fit_height(height - 0.16)
        logo.move_to(card)
        return Group(card, logo)

    def add_branding(self: Slide) -> None:
        bar = Rectangle(
            width=config.frame_width,
            height=0.40,
            stroke_width=0,
            fill_color="#050D18",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0)
        rule = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2, color=GRID)
        rule.move_to([0, -3.58, 0])
        signature = safe_text(
            "Diego Villalba  ·  Almacenes y Minería de Datos",
            size=13,
            color=MUTED,
        ).move_to([-4.72, -3.80, 0])
        logo = self.logo_lockup(1.42, 0.34).move_to([6.12, -3.80, 0])
        self.branding = Group(bar, rule, signature, logo).set_z_index(100)
        self.add(self.branding)

    def heading(self: Slide, kicker: str, title: str, color: str) -> VGroup:
        kicker_mob = safe_text(kicker.upper(), size=19, color=color, weight=BOLD)
        title_mob = safe_text(title, size=42, color=INK, weight=BOLD, max_width=12.5)
        text = VGroup(kicker_mob, title_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        text.to_corner(UL, buff=0.43)
        rule = Line(LEFT * 0.55, RIGHT * 0.55, color=color, stroke_width=4)
        rule.next_to(text, DOWN, aligned_edge=LEFT, buff=0.10)
        return VGroup(text, rule)
