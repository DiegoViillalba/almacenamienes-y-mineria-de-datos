from pathlib import Path
from manim import *

BG = "#07111F"
INK = "#F7F7F2"
MUTED = "#9FB3C8"
GRID = "#27435D"
BLUE = "#58C4DD"
YELLOW = "#F7C948"
GREEN = "#6FCF97"
RED = "#FF6B6B"
PURPLE = "#B28DFF"
ORANGE = "#FFB86B"

FONT = "Sans"
LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "Logo_FC_Blanco.png"

def safe_text(content: str, *, size: float = 34, color: str = INK, weight: str = NORMAL, max_width: float | None = None) -> Text:
    mob = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob

def pill(content: str, color: str, width: float = 2.25) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.20, width=width, height=0.58,
        stroke_color=color, fill_color=color, fill_opacity=0.13, stroke_width=2
    )
    label = safe_text(content, size=25, color=color, weight=BOLD, max_width=width - 0.25)
    label.move_to(box)
    return VGroup(box, label)
