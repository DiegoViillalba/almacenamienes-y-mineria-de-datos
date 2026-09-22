from manim import *
from manim_slides import Slide
from utils import BG, GRID, MUTED, INK, BLUE, LOGO_PATH, safe_text

class BaseSlide(Slide):
    def setup_slide(self):
        self.camera.background_color = ManimColor(BG)
        self.branding: Group | None = None

    def pause(self, notes: str) -> None:
        self.next_slide(notes=notes)

    def clear(self, run_time: float = 0.55) -> None:
        removable = [m for m in self.mobjects if m is not self.branding]
        if removable:
            self.play(*[FadeOut(m) for m in removable], run_time=run_time)

    def logo_lockup(self, *, width: float, height: float) -> Group:
        card = RoundedRectangle(
            corner_radius=0.10, width=width, height=height,
            stroke_color=GRID, stroke_width=1.2, stroke_opacity=0.65,
            fill_color=BG, fill_opacity=0.35,
        )
        logo = ImageMobject(str(LOGO_PATH))
        logo.scale_to_fit_width(width - 0.26)
        if logo.height > height - 0.18:
            logo.scale_to_fit_height(height - 0.18)
        logo.move_to(card)
        return Group(card, logo)

    def add_course_branding(self) -> None:
        bar = Rectangle(width=config.frame_width, height=0.40, stroke_width=0, fill_color="#050D18", fill_opacity=0.98).to_edge(DOWN, buff=0)
        rule = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2, color=GRID, stroke_width=1.5).move_to([0, -3.59, 0])
        signature = safe_text("Diego Villalba  ·  Almacenes y Minería de Datos", size=13, color=MUTED).move_to([-4.75, -3.80, 0])
        logo = self.logo_lockup(width=1.42, height=0.34).move_to([6.12, -3.80, 0])
        self.branding = Group(bar, rule, signature, logo).set_z_index(100)
        self.add(self.branding)

    def heading(self, kicker: str, title: str, color: str = BLUE) -> VGroup:
        kicker_mob = safe_text(kicker.upper(), size=20, color=color, weight=BOLD)
        title_mob = safe_text(title, size=43, color=INK, weight=BOLD, max_width=12.4)
        block = VGroup(kicker_mob, title_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        block.to_corner(UL, buff=0.45)
        rule = Line(LEFT * 0.55, RIGHT * 0.55, color=color, stroke_width=4)
        rule.next_to(block, DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(block, rule)
