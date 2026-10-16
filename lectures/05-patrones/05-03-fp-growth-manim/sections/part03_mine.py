"""Acto 3 · Minar el árbol (24–42 min).

Idea central: cada patrón tiene un «último» ítem en el orden global, así que
el espacio de búsqueda se parte en subproblemas disjuntos (8 + 4 + 2 + 1).
Para el subproblema de un ítem basta su base condicional: los caminos prefijo
de sus nodos, cada uno con el contador del nodo del ítem (no el del prefijo).
Cerveza se resuelve guiado, Pañales es el «tu turno», y al final se muestra la
recursión cuando el árbol condicional se ramifica.
"""
from __future__ import annotations

from itertools import combinations

from manim import *

from sections.data import (BASE_CERVEZA, BASE_PAN, BASE_PANALES, COUNTS, FREQUENT, MIN_SUP,
                           ORDER, REC_MIN, REC_PATTERNS, REC_TREE, TREE, base_counts)
from sections.fptree import TreeView
from sections.part02_build import main_tree
from sections.theme import (ABBR, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK,
                            PURPLE, RED, YELLOW, bullet_list, card, chip, chip_row, code_block, color_of,
                            pill, safe_text, slot_row, tex)

FREQ = {frozenset(p): c for p, c in FREQUENT}
COL_X = [-5.0, -1.65, 1.7, 5.05]
BASE_X0 = -6.7          # borde izquierdo de las filas de la base condicional
BASE_Y = [1.15, 0.45]
COND_ROOT = [2.6, 2.0, 0]


def fmt_pattern(items) -> str:
    return "{" + ", ".join(i for i in ORDER if i in items) + "}"


def last_item(subset) -> str:
    return max(subset, key=ORDER.index)


def build_partition() -> tuple[VGroup, dict[frozenset, VGroup]]:
    """Los 15 itemsets agrupados por su último ítem (columnas Cerveza … Leche)."""
    groups: dict[str, list[frozenset]] = {i: [] for i in ORDER}
    for k in range(1, len(ORDER) + 1):
        for combo in combinations(ORDER, k):
            groups[last_item(combo)].append(frozenset(combo))
    whole, rows = VGroup(), {}
    for x, item in zip(COL_X, reversed(ORDER)):
        head = VGroup(safe_text("terminan en", size=16, color=MUTED),
                      VGroup(chip(item, size=20), safe_text(f"({len(groups[item])})", size=19,
                                                            color=MUTED)).arrange(RIGHT, buff=0.12)
                      ).arrange(DOWN, buff=0.08).move_to([x, 2.05, 0])
        col = VGroup(head)
        for r, s in enumerate(groups[item]):
            row = slot_row(s, ORDER).move_to([x, 1.3 - r * 0.47, 0])
            rows[s] = row
            col.add(row)
        whole.add(col)
    return whole, rows


def base_rows(base, keep: set[str] | None = None) -> VGroup:
    """Filas «camino prefijo × conteo» alineadas a la izquierda."""
    out = VGroup()
    for (path, w), y in zip(base, [1.15, 0.45, -0.25]):
        chips = chip_row(path, size=20).move_to([0, y, 0])
        chips.shift(RIGHT * (BASE_X0 - chips.get_left()[0]))
        wt = safe_text(f"× {w}", size=26, color=YELLOW, weight=BOLD).move_to([-2.45, y, 0])
        out.add(VGroup(chips, wt))
    return out


def count_row(counts: dict[str, int], y: float = -0.75) -> VGroup:
    row = VGroup()
    for i in ORDER:
        if i in counts:
            ok = counts[i] >= MIN_SUP
            row.add(VGroup(chip(i, size=19), safe_text(f"{counts[i]}", size=26,
                                                       color=GREEN if ok else RED, weight=BOLD)
                           ).arrange(RIGHT, buff=0.12))
    return row.arrange(RIGHT, buff=0.35).move_to([0, y, 0]).align_to([BASE_X0, 0, 0], LEFT)


def pattern_grid(patterns: list[tuple[tuple, int]]) -> VGroup:
    cells = VGroup(*[safe_text(f"{fmt_pattern(p)}: {c}", size=24, color=GREEN, weight=BOLD)
                     for p, c in patterns])
    cells.arrange_in_grid(rows=2, cols=2, buff=(0.9, 0.28), col_alignments="ll")
    return cells.move_to([-0.6, -2.45, 0])


class MineSection:
    def mine_tree(self) -> None:
        # ── Partición del espacio ──────────────────────────────────────────
        self.beat("24–26 min. Cambio de pregunta: en vez de fabricar candidatos, partimos el "
                  "problema. Ordene cada itemset según el orden global: su ÚLTIMO ítem lo manda "
                  "a una sola columna. 8 + 4 + 2 + 1 = 15, sin repetir. Pregunte: ¿por cuál "
                  "columna conviene empezar? (Por el ítem menos frecuente: Cerveza.)")
        self.start("2 · Minar", "Dividir: cada patrón tiene un último ítem", PINK)
        grid, _ = build_partition()
        for col in grid:
            self.play(FadeIn(col[0]), LaggedStart(*[FadeIn(r, shift=DOWN * 0.08) for r in col[1:]],
                                                  lag_ratio=0.08), run_time=0.7)
        total = safe_text("8 + 4 + 2 + 1 = 15 itemsets posibles, cada uno en una sola columna",
                          size=22, color=MUTED).move_to([0, -2.75, 0])
        self.play(FadeIn(total))

        # ── El subproblema de Cerveza ──────────────────────────────────────
        self.beat("26–28 min. Subproblema «terminan en Cerveza». Seguimos los enlaces de Cerveza "
                  "desde la tabla de encabezados: dos nodos, 2 + 1 = 3 tickets ⇒ {Cerveza} es "
                  "frecuente. Lo que va ANTES de Cerveza en esos caminos es lo único que puede "
                  "acompañarla.")
        self.start("2 · Minar", "Los caminos que terminan en Cerveza", PINK)
        tree = main_tree().build_all()
        self.play(FadeIn(tree))
        ce = tree.nodes_of("Cerveza")
        links = tree.links("Cerveza", PURPLE)
        rings = VGroup(*[SurroundingRectangle(tree.nodes[p], color=YELLOW, buff=0.06) for p in ce])
        self.play(Create(links), Create(rings))
        paths = VGroup(*[tree.path_group(p) for p in ce])
        self.play(Indicate(paths, color=YELLOW, scale_factor=1.03), run_time=1.0)
        sup = VGroup(chip("Cerveza", size=22), tex(r"2 + 1 = 3 \geq 3", size=36, color=GREEN)
                     ).arrange(RIGHT, buff=0.3).move_to([-4.3, 1.6, 0])
        self.play(FadeIn(sup))

        self.beat("28–29 min. PREDICCIÓN antes de revelar. En el camino Leche:5 → Pan:4 → "
                  "Pañales:3 → Cerveza:2, ¿con qué conteo entra Leche a la base de Cerveza? "
                  "Pida votar: 5, 4, 3 o 2. No responda todavía.")
        q = VGroup(safe_text("¿Con qué conteo entra Leche?", size=26, color=YELLOW, weight=BOLD),
                   safe_text("5 · 4 · 3 · 2", size=30, weight=BOLD)).arrange(DOWN, buff=0.2)
        q.move_to([-4.3, 0.2, 0])
        self.play(FadeIn(q, shift=UP * 0.1))

        self.beat("29–31 min. Respuesta: 2. De los 5 tickets con Leche, sólo 2 siguen por ese "
                  "camino hasta Cerveza. Cada camino prefijo hereda el contador del nodo "
                  "Cerveza. Es la probabilidad condicional de la clase pasada: el universo se "
                  "encoge a los tickets con Cerveza. Error típico: copiar 5, 4, 3.")
        self.play(FadeOut(q), FadeOut(sup))
        title = safe_text("base condicional de Cerveza", size=22, color=PURPLE, weight=BOLD)
        title.move_to([0, 1.85, 0]).align_to([BASE_X0, 0, 0], LEFT)
        self.play(FadeIn(title))
        rows = base_rows(BASE_CERVEZA)
        for (path, w), row, p in zip(BASE_CERVEZA, rows, ce):
            src = VGroup(*[tree.nodes[path[:d]].copy() for d in range(1, len(path) + 1)])
            wt_src = tree.nodes[p][2].copy()
            self.play(ReplacementTransform(src, row[0]), ReplacementTransform(wt_src, row[1]),
                      run_time=1.1)
        trap = pill("Leche entra × 2, no × 5", RED, size=21).move_to([-4.6, -0.3, 0])
        self.play(FadeIn(trap))

        self.beat("31–32 min. Conteos condicionales: sumamos por ítem con esos pesos. Leche "
                  "2 + 1 = 3, Pan 2, Pañales 2 + 1 = 3. Pan no llega a 3: sale de la base. "
                  "Pregunte qué significa: Pan y Cerveza juntos sólo en 2 tickets.")
        bc = base_counts(BASE_CERVEZA)
        counts = count_row(bc)
        self.play(FadeOut(trap), FadeIn(counts, shift=UP * 0.1))
        pan_chips = VGroup(*[c for row in rows for c in row[0] if c.item == "Pan"])
        pan_count = [g for g in counts if g[0].item == "Pan"][0]
        self.play(Create(Cross(pan_count, stroke_color=RED, stroke_width=4)),
                  pan_chips.animate.set_opacity(0.2))

        self.beat("32–34 min. Con lo que queda construimos un FP-tree pequeño: el árbol "
                  "condicional de Cerveza, Leche:3 → Pañales:3. Es un solo camino, así que "
                  "cualquier combinación de sus nodos, junto con Cerveza, es frecuente, con el "
                  "menor contador elegido. Salen 4 patrones.")
        self.play(FadeOut(VGroup(tree, links, rings)), run_time=0.4)
        cond_counts = {("Leche",): bc["Leche"], ("Leche", "Pañales"): bc["Pañales"]}
        cond = TreeView(cond_counts, root_pos=COND_ROOT, dx=2.3, dy=0.95, node_w=2.1, size=22)
        cond.build_all()
        cond_lab = safe_text("árbol condicional de Cerveza", size=20, color=PURPLE)
        cond_lab.next_to(cond.root, RIGHT, buff=0.4)
        self.play(FadeIn(cond), FadeIn(cond_lab))
        pats = [(p, c) for p, c in FREQUENT if p[-1] == "Cerveza"]
        grid = pattern_grid(pats)
        rule = VGroup(safe_text("un solo camino ⇒", size=20, color=MUTED),
                      safe_text("todas sus combinaciones + Cerveza", size=20, color=MUTED)
                      ).arrange(DOWN, buff=0.08).move_to([4.6, -0.55, 0])
        self.play(FadeIn(rule))
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in grid], lag_ratio=0.2))

        # ── Tu turno: Pañales ──────────────────────────────────────────────
        self.beat("34–37 min. TU TURNO, en parejas, 3 minutos: el subproblema «terminan en "
                  "Pañales» (sin Cerveza). Recorra el salón. Pista si se atoran: los nodos "
                  "Cerveza que cuelgan debajo de Pañales NO entran; esos patrones ya se contaron "
                  "en la columna anterior.")
        self.start("Tu turno", "Mina Pañales con el mismo procedimiento", YELLOW)
        tree = main_tree().build_all()
        pn = tree.nodes_of("Pañales")
        self.play(FadeIn(tree))
        self.play(Create(tree.links("Pañales", PINK)),
                  Create(VGroup(*[SurroundingRectangle(tree.nodes[p], color=YELLOW, buff=0.06)
                                  for p in pn])))
        tasks = bullet_list([
            "1. Sigue los enlaces de Pañales.",
            "2. Escribe su base condicional",
            "    (¿con qué conteos?).",
            "3. Suma por ítem y poda (< 3).",
            "4. Árbol condicional y patrones.",
        ], size=24, width=6.3, buff=0.25).move_to([-4.0, 0.4, 0])
        timer = pill("3 min · ¿qué hacemos con Cerveza?", MUTED, size=19).move_to([-4.0, -2.3, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tasks], lag_ratio=0.15))
        self.play(FadeIn(timer))

        self.beat("37–38 min. Solución. Dos nodos Pañales: el de contador 3 (camino Leche → Pan) "
                  "y el de contador 1 (camino Leche). Base: [Leche, Pan] × 3 y [Leche] × 1. "
                  "Conteos: Leche 4, Pan 3: nadie sale. Árbol Leche:4 → Pan:3, un solo camino "
                  "⇒ 4 patrones.")
        self.clear()
        self.play(FadeIn(self.heading("2 · Minar", "Solución: terminan en Pañales", PINK)),
                  run_time=0.3)
        title = safe_text("base condicional de Pañales", size=22, color=PINK, weight=BOLD)
        title.move_to([0, 1.85, 0]).align_to([BASE_X0, 0, 0], LEFT)
        rows = base_rows(BASE_PANALES)
        bc = base_counts(BASE_PANALES)
        counts = count_row(bc)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows],
                                             lag_ratio=0.3))
        self.play(FadeIn(counts, shift=UP * 0.1))
        cond = TreeView({("Leche",): bc["Leche"], ("Leche", "Pan"): bc["Pan"]},
                        root_pos=COND_ROOT, dx=2.3, dy=0.95, node_w=2.1, size=22).build_all()
        cond_lab = safe_text("árbol condicional de Pañales", size=20, color=PINK)
        cond_lab.next_to(cond.root, RIGHT, buff=0.4)
        self.play(FadeIn(cond), FadeIn(cond_lab))
        grid = pattern_grid([(p, c) for p, c in FREQUENT if p[-1] == "Pañales"])
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in grid], lag_ratio=0.2))

        self.beat("38–39 min. Las dos columnas restantes son inmediatas. Pan: un nodo, camino "
                  "[Leche] × 4 ⇒ {Pan}: 4 y {Leche, Pan}: 4. Leche cuelga de la raíz: su base "
                  "está vacía ⇒ sólo {Leche}: 5.")
        self.clear()
        self.play(FadeIn(self.heading("2 · Minar", "Pan y Leche: las columnas cortas", PINK)),
                  run_time=0.3)
        (pan_path, pan_w), = BASE_PAN
        lines = VGroup(
            VGroup(chip("Pan", size=24), safe_text("base:", size=26, color=MUTED),
                   chip_row(pan_path, size=22), safe_text(f"× {pan_w}", size=28, color=YELLOW,
                                                          weight=BOLD)).arrange(RIGHT, buff=0.3),
            safe_text(f"{{Pan}}: {FREQ[frozenset(['Pan'])]}    "
                      f"{{Leche, Pan}}: {FREQ[frozenset(['Leche', 'Pan'])]}", size=28,
                      color=GREEN, weight=BOLD),
            VGroup(chip("Leche", size=24), safe_text("base: vacía (cuelga de ∅)", size=26,
                                                     color=MUTED)).arrange(RIGHT, buff=0.3),
            safe_text(f"{{Leche}}: {COUNTS['Leche']}", size=28, color=GREEN, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([0, 0.1, 0])
        for k in range(0, 4, 2):
            self.play(FadeIn(lines[k], shift=RIGHT * 0.1))
            self.play(FadeIn(lines[k + 1], shift=UP * 0.1))

        # ── Balance sobre la partición ─────────────────────────────────────
        self.beat("39–40 min. Volvemos a las 15 casillas: 11 frecuentes (verde). Las 4 que "
                  "faltan contienen Pan y Cerveza: desaparecieron cuando Pan salió de la base de "
                  "Cerveza. FP-Growth nunca las fabricó ni las contó: eso es «sin candidatos».")
        self.start("2 · Minar", "Once patrones frecuentes, ningún candidato", PINK)
        grid, rows = build_partition()
        self.play(FadeIn(grid), run_time=0.6)
        marks = []
        for s, row in rows.items():
            if s in FREQ:
                marks.append(VGroup(SurroundingRectangle(row, color=GREEN, buff=0.05),
                                    safe_text(f"{FREQ[s]}", size=20, color=GREEN, weight=BOLD)
                                    .next_to(row, RIGHT, buff=0.12)))
            else:
                marks.append(Cross(row, stroke_color=RED, stroke_width=3, scale_factor=0.9))
                row.set_opacity(0.35)
        self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.05), run_time=1.5)
        verdict = pill(f"{len(FREQUENT)} frecuentes · las 4 que caen contienen Pan y Cerveza",
                       GREEN, size=20).move_to([0, -2.75, 0])
        self.play(FadeIn(verdict, shift=UP * 0.1))

        # ── Recursión ──────────────────────────────────────────────────────
        self.beat("40–41 min. ¿Y si el árbol condicional NO es un solo camino? Bajemos el soporte "
                  "mínimo a 2: Pan ya no sale y el árbol de Cerveza se ramifica. No hay regla "
                  "nueva: se aplica el mismo procedimiento DENTRO de ese árbol (recursión). "
                  "Cada nodo de la derecha es un patrón: léalo subiendo hasta Cerveza.")
        self.start("2 · Minar", "Si el árbol se ramifica: recursión", PINK)
        left = TreeView(REC_TREE, root_pos=[-4.4, 1.55, 0], dx=2.4, dy=0.95, node_w=2.0,
                        size=21).build_all()
        left_lab = safe_text(f"árbol condicional de Cerveza (mínimo {REC_MIN})", size=19,
                             color=PURPLE).move_to([-4.4, 2.15, 0])
        self.play(FadeIn(left_lab), FadeIn(left))
        calls = {("Cerveza",): COUNTS["Cerveza"]}
        for pat, c in REC_PATTERNS:
            calls[tuple(reversed(pat))] = c
        rec = TreeView(calls, root_pos=[3.0, 1.8, 0], dx=1.85, dy=0.9, node_w=1.75, size=19,
                       root_label="∅")
        rec_lab = safe_text("llamadas recursivas = patrones", size=19, color=MUTED)
        rec_lab.move_to([3.0, 2.3, 0])
        self.play(FadeIn(rec.root), FadeIn(rec_lab))
        rec.build_all(calls)
        for d in range(1, 5):
            self.play(*[AnimationGroup(Create(rec.edges[p]), FadeIn(rec.nodes[p], scale=0.7))
                        for p in calls if len(p) == d], run_time=0.6)
        example = safe_text("ej.: Leche → Pan → Pañales → Cerveza = {Leche, Pan, Pañales, Cerveza}: 2",
                            size=19, color=MUTED, max_width=12.6).move_to([0, -2.95, 0])
        self.play(FadeIn(example))

        # ── Formalización ──────────────────────────────────────────────────
        self.beat("41–42 min. Formalice. La base condicional B_a reúne los caminos prefijo de a "
                  "con el contador del nodo a. El algoritmo completo cabe en siete líneas: "
                  "recorre la tabla de abajo hacia arriba, crece el sufijo y se llama sobre un "
                  "árbol cada vez más pequeño. No hay generación de candidatos.")
        self.start("2 · Minar", "FP-Growth, formalmente", PINK)
        definition = tex(r"\mathcal{B}_a=\{(P_j,c_j):\ P_j\ \text{camino prefijo de un nodo } a,"
                         r"\ c_j=\operatorname{count}(\text{ese nodo } a)\}", size=34,
                         max_width=12.6).move_to([0, 1.95, 0])
        code = code_block([
            "FPGrowth(árbol, sufijo):",
            "  para cada ítem a, del menos al más frecuente:",
            "    patrón = {a} ∪ sufijo      # soporte = suma de contadores de a",
            "    reportar(patrón)",
            "    base = caminos prefijo de a, con el contador del nodo a",
            "    quitar de la base los ítems con conteo < min_sup",
            "    si la base no está vacía: FPGrowth(FP-tree(base), patrón)",
        ], size=21, width=12.4).move_to([0, -0.65, 0])
        self.play(Write(definition))
        self.play(FadeIn(code[0]), LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in code[1]],
                                               lag_ratio=0.12), run_time=1.4)
