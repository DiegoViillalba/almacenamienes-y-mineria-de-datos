"""Vista previa rápida de laboratorios interactivos sin renderizar el libro.

Extrae el kit (`unidades/preprocesamiento/_labkit.qmd`) y los bloques ```{=html}
que contienen `class="plab"` de uno o varios .qmd, y escribe una página HTML
aislada por capítulo y por tema (claro/oscuro).

Uso:
    python tools/lab_preview.py unidades/preprocesamiento/4-4-PCA.qmd [otros.qmd ...]
    python -m http.server 8766 --directory _lab_preview
    # abrir http://localhost:8766/4-4-PCA-light.html y ...-dark.html

La salida va a `_lab_preview/` (ignorada por Git). Ver LABORATORIOS.md.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIT = ROOT / "unidades" / "preprocesamiento" / "_labkit.qmd"
OUT = ROOT / "_lab_preview"
HTML_BLOCK = re.compile(r"```\{=html\}\n(.*?)\n```", re.S)


def main(paths):
    if not paths:
        sys.exit(__doc__)
    OUT.mkdir(exist_ok=True)
    kit = HTML_BLOCK.findall(KIT.read_text())[0]
    for path in paths:
        qmd = pathlib.Path(path)
        labs = [b for b in HTML_BLOCK.findall(qmd.read_text()) if 'class="plab"' in b]
        for theme, body_class, bg in (("light", "quarto-light", "#fff"), ("dark", "quarto-dark", "#222")):
            page = (
                '<!doctype html><html lang="es"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f"<title>{qmd.stem} ({theme})</title>"
                f"<style>body{{background:{bg};margin:0;padding:16px}}main{{max-width:760px;margin:auto}}</style>"
                f'</head><body class="{body_class}"><main>{kit}{"".join(labs)}</main></body></html>'
            )
            (OUT / f"{qmd.stem}-{theme}.html").write_text(page)
        print(f"{qmd}: {len(labs)} bloque(s) de laboratorio -> {OUT.name}/{qmd.stem}-{{light,dark}}.html")


if __name__ == "__main__":
    main(sys.argv[1:])
