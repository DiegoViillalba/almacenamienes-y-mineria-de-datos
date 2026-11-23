"""Paleta, tipografía y piezas visuales reutilizables.

Colores con función estable en toda la clase (cambiar aquí cambia toda la clase):
- azul + círculo = clase −1 (y = 0); naranja + cuadrado = clase +1 (y = 1);
- amarillo = frontera de decisión (g = 0) y foco;
- morado = vector de pesos w y puntaje z; verde = probabilidad / sigmoide;
- rojo = error y pérdida; turquesa = proyecciones (sombras de Fisher);
- rosa = espacio de características φ(x) y kernels; gris = estructura.
La forma (círculo/cuadrado) acompaña siempre al color.
"""
from __future__ import annotations

import re
from pathlib import Path

from manim import *

BG = "#14191B"
PANEL = "#1E2729"
INK = "#F2EDE3"
MUTED = "#A9B3AF"
GRID = "#5F7C78"
BLUE = "#8FA6E0"     # clase −1
ORANGE = "#E3A857"   # clase +1
PURPLE = "#BE8CE8"   # w y puntaje z
TEAL = "#5FB3A8"     # sombras / proyecciones
PINK = "#D98FA6"     # espacio de características
GREEN = "#9BC4A0"    # probabilidad, acierto
YELLOW = "#F2EDE3"   # frontera, foco
RED = "#E0654C"      # error

CLASS_COLOR = {0: BLUE, 1: ORANGE, -1: BLUE}
CELL_COLOR = {"TP": GREEN, "TN": GREEN, "FP": RED, "FN": RED}

FONT = "DejaVu Sans"
MONO = "Menlo"
LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "Logo_FC_Blanco.png"


# Pango coloca los glifos con hinting a tamaños pequeños: algunas letras quedan
# pegadas y otras separadas. Se compone a SUPERSAMPLE veces el tamaño y se
# reduce, para que el espaciado salga de los métricos reales de la fuente.
SUPERSAMPLE = 4
_MATH_K: list = []  # factor para igualar la altura de los dígitos CM con los de DejaVu

_NUM = re.compile(r"(?<![\w²³.])([-−+±]?\d+(?:\.\d+)?)(?![\w²³])")
_GLYPHS: dict = {}
_METRICS: dict = {}


def _text(content: str, size: float, color: str, weight: str, font: str) -> Text:
    mob = Text(content, font=font, font_size=size * SUPERSAMPLE, color=color,
               weight=weight, disable_ligatures=True)
    return mob.scale(1 / SUPERSAMPLE)


def _math_scale() -> float:
    if not _MATH_K:
        dj = _text("0", 24, INK, NORMAL, FONT).height
        cm = MathTex("0", font_size=24).height
        _MATH_K.append(dj / cm * 1.02)
    return _MATH_K[0]


def _glyph(ch: str, size: float, bold: bool) -> MathTex:
    """Glifo LaTeX de un dígito o signo; se compila una vez y se copia."""
    key = (ch, round(size, 2), bold)
    if key not in _GLYPHS:
        src = {"−": "-", "±": r"\pm"}.get(ch, ch)
        if bold and ch.isdigit():
            src = r"\mathbf{" + src + "}"
        _GLYPHS[key] = MathTex(src, font_size=size * _math_scale())
    return _GLYPHS[key].copy()


def numeral(content: str, *, size: float = 24, color: str = INK, bold: bool = False) -> VGroup:
    """Número en LaTeX (dígitos de Computer Modern) armado con glifos cacheados.

    Sirve igual para números fijos y para valores que cambian cada cuadro
    (``always_redraw``), porque no compila LaTeX nuevo en cada llamada. La
    línea base queda en y = 0 antes de reubicar el grupo.
    """
    h0 = _glyph("0", size, bold).height
    out = VGroup()
    x = 0.0
    for ch in content:
        g = _glyph(ch, size, bold).set_color(color)
        sign = ch in "-−+±"
        cy = 0.40 * h0 if sign else g.height / 2
        g.move_to([x + g.width / 2, cy, 0])
        x += g.width + (0.16 if sign else 0.09) * h0
        out.add(g)
    return out


def _space(size: float, weight: str, font: str) -> float:
    """Ancho de un espacio a este tamaño."""
    key = (round(size, 2), weight, font)
    if key not in _METRICS:
        _METRICS[key] = (_text("a a", size, INK, weight, font).width
                         - _text("aa", size, INK, weight, font).width)
    return _METRICS[key]


def _math_piece(src: str, size: float, color: str):
    """Fragmento LaTeX con su línea base: «x» de referencia + el fragmento."""
    mob = MathTex("x", src, font_size=size * _math_scale() * 0.92, color=color)
    return VGroup(*mob.submobjects[1:]), mob.submobjects[0].get_bottom()[1]


def safe_text(content: str, *, size: float = 32, color: str = INK,
              weight: str = NORMAL, max_width: float | None = None,
              font: str = FONT) -> VMobject:
    """Texto DejaVu con matemáticas en LaTeX.

    - Lo que va entre ``$…$`` es LaTeX (variables, ejes, expresiones).
    - Los números sueltos (``3.5``, ``−0.41``, ``1.04``…) también se componen
      en LaTeX con glifos cacheados, de modo que valen para valores que cambian
      cada cuadro. Los títulos y conceptos quedan en la fuente normal.
    """
    if "$" not in content and _NUM.search(content) is None:
        mob = _text(content, size, color, weight, font)
    else:
        tokens = []
        for k, seg in enumerate(content.split("$")):
            if k % 2 == 1:
                if seg.strip():
                    tokens.append(("math", seg))
            else:
                for j, part in enumerate(_NUM.split(seg)):
                    if j % 2 == 1:
                        tokens.append(("num", part))
                    elif part:
                        tokens.append(("txt", part))
        space = _space(size, weight, font)
        pad = 0.0016 * size    # aire entre fragmentos de distinto tipo pegados
        bold = weight == BOLD
        x = 0.0
        row = VGroup()
        prev = None
        gap = True
        for kind, val in tokens:
            trail = 0
            if kind == "txt":
                lead = len(val) - len(val.lstrip(" "))
                trail = len(val) - len(val.rstrip(" "))
                chunk = val.strip(" ")
                x += lead * space
                gap = gap or lead > 0
                if not chunk:
                    gap = True
                    continue
                ref = _text("x" + chunk, size, color, weight, font)
                base = ref.submobjects[0].get_bottom()[1]
                mob = VGroup(*ref.submobjects[1:])
            elif kind == "num":
                mob = numeral(val, size=size, color=color, bold=bold)
                base = mob.get_bottom()[1]
            else:
                mob, base = _math_piece(val, size, color)
            if prev is not None and prev != kind and not gap:
                x += pad
            mob.shift(RIGHT * (x - mob.get_left()[0]) + UP * (0.0 - base))
            x += mob.width + trail * space
            gap = trail > 0
            prev = kind
            row.add(mob)
        mob = row.move_to(ORIGIN)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def tex(content: str, *, size: float = 40, color: str = INK,
        max_width: float | None = 12.0) -> MathTex:
    mob = MathTex(content, font_size=size, color=color)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def pill(content: str, color: str, *, width: float | None = None,
         size: float = 24) -> VGroup:
    label = safe_text(content, size=size, color=color, weight=BOLD)
    w = width if width is not None else label.width + 0.55
    if label.width > w - 0.3:
        label.scale_to_fit_width(w - 0.3)
    box = RoundedRectangle(corner_radius=0.2, width=w, height=label.height + 0.34,
                           stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.12)
    label.move_to(box)
    return VGroup(box, label)


def card(width: float, height: float, color: str = GRID) -> RoundedRectangle:
    return RoundedRectangle(corner_radius=0.14, width=width, height=height,
                            stroke_color=color, stroke_width=1.6,
                            fill_color=PANEL, fill_opacity=0.92)


def bullet_list(items: list[str], *, size: float = 28, color: str = INK,
                width: float = 11.5, buff: float = 0.28) -> VGroup:
    rows = VGroup(*[safe_text(t, size=size, color=color, max_width=width) for t in items])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return rows


def table(rows: list[list[str]], *, col_widths: list[float], size: float = 24,
          row_h: float = 0.52, header_color: str = MUTED,
          colors: list[list[str]] | None = None) -> VGroup:
    """Tabla simple de texto. Devuelve VGroup(filas…, regla)."""
    out = VGroup()
    for i, r in enumerate(rows):
        row = VGroup()
        x = 0.0
        for j, (content, w) in enumerate(zip(r, col_widths)):
            if i == 0:
                c = header_color
            elif colors is not None:
                c = colors[i - 1][j]
            else:
                c = INK
            cell = safe_text(content, size=size, color=c,
                             weight=BOLD if i == 0 else NORMAL, max_width=w - 0.15)
            cell.move_to([x + w / 2, -i * row_h, 0])
            row.add(cell)
            x += w
        out.add(row)
    rule = Line([0, -row_h / 2, 0], [sum(col_widths), -row_h / 2, 0], color=GRID,
                stroke_width=2)
    out.add(rule)
    return out


def marker(cls: int | None, *, r: float = 0.13, filled: bool = True) -> VMobject:
    """Círculo azul (clase −1 ó 0), cuadrado naranja (+1) o rombo amarillo (desconocido)."""
    if cls is None:
        m = Square(side_length=2 * r * 0.95).rotate(PI / 4)
        m.set_stroke(YELLOW, 3).set_fill(YELLOW, 0.9 if filled else 0.0)
        return m.set_z_index(20)
    col = CLASS_COLOR[cls]
    m = Circle(radius=r) if cls <= 0 else Square(side_length=2 * r * 0.88)
    m.set_stroke(col, 3).set_fill(col, 0.9 if filled else 0.0)
    return m.set_z_index(20)


def legend_item(cls: int, text: str, size: float = 20) -> VGroup:
    return VGroup(marker(cls, r=0.11), safe_text(text, size=size, color=MUTED)
                  ).arrange(RIGHT, buff=0.15)


def fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"
