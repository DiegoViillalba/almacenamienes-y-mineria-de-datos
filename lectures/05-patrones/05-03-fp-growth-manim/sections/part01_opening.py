"""Acto 1 · Portada, objetivos e intuición (0–10 min).

Recuerda Apriori sobre los cinco tickets del capítulo 5.3 para mostrar dónde
está su costo (candidatos y lecturas por nivel), escala ese costo y propone la
idea de FP-Growth con una analogía cotidiana: palabras que comparten prefijo
se escriben una sola vez en un árbol.
"""
from __future__ import annotations

from math import comb

from manim import *

from sections.data import (APRIORI, MIN_SUP, ORDER, TICKETS, TICKETS_RAW, WORD_LETTERS,
                           WORD_TREE, WORDS)
from sections.fptree import TreeView
from sections.theme import (ABBR, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK,
                            PURPLE, RED, YELLOW, bullet_list, card, chip, chip_row, pill,
                            safe_text, table)


def lattice_node(itemset, size: float = 17) -> VGroup:
    items = [i for i in ORDER if i in itemset]
    return VGroup(*[chip(i, label=ABBR[i], width=0.46, height=0.40, size=size)
                    for i in items]).arrange(RIGHT, buff=0.04)


class OpeningSection:
    def opening(self) -> None:
        self.beat("0–2 min. Portada. Pregunte: en la clase pasada Apriori encontró los itemsets "
                  "frecuentes. ¿Qué pasaría con una tienda real de 10 000 productos y millones "
                  "de tickets? Recoja intuiciones sobre el costo; no las corrija aún.")
        eyebrow = safe_text("EXTRACCIÓN DE PATRONES · FP-GROWTH", size=22, color=BLUE, weight=BOLD)
        title = safe_text("¿Y si leemos los tickets sólo dos veces?", size=56, weight=BOLD,
                          max_width=12.6)
        subtitle = safe_text("Comprimir transacciones en un árbol y minar patrones por partes",
                             size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("2–3 min. Objetivos como acciones verificables. Anuncie que a la mitad habrá "
                  "un «tu turno» (minar Pañales) y al final un reto en parejas con datos nuevos.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "¿Qué sabremos hacer al terminar?", YELLOW)
        goals = bullet_list([
            "1. Explicar por qué Apriori se vuelve caro: candidatos y lecturas por nivel.",
            "2. Construir un FP-tree: contar, ordenar e insertar transacciones.",
            "3. Minar el árbol con bases y árboles condicionales.",
            "4. Decidir cuándo conviene FP-Growth y usarlo en Python.",
        ], size=27).move_to([0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.15) for g in goals], lag_ratio=0.15),
                  run_time=1.0)
        self.apriori_cost()
        self.explosion()
        self.words()

    # ------------------------------------------------------------------
    def apriori_cost(self) -> None:
        self.beat("3–4 min. Recordatorio con los cinco tickets del libro (soporte mínimo 3). "
                  "Apriori trabaja por niveles: fabrica candidatos y RELEE toda la base para "
                  "contarlos. Pida que sigan el contador de lecturas.")
        head = self.start("Intuición", "¿En qué gasta su tiempo Apriori?", ORANGE)
        stack = VGroup()
        for k, t in enumerate(TICKETS_RAW):
            row = VGroup(safe_text(f"T{k + 1}", size=18, color=MUTED, weight=BOLD),
                         VGroup(*[chip(i, label=ABBR[i], width=0.5, height=0.38, size=16)
                                  for i in t]).arrange(RIGHT, buff=0.05)).arrange(RIGHT, buff=0.2)
            stack.add(row)
        stack.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        box = card(stack.width + 0.5, stack.height + 0.5)
        tickets = VGroup(box, stack).move_to([-5.35, -0.15, 0])
        stack.move_to(box)
        db_lab = safe_text("la base", size=20, color=MUTED).next_to(box, UP, buff=0.12)
        legend = VGroup(*[VGroup(chip(i, label=ABBR[i], width=0.5, height=0.38, size=16),
                                 safe_text(i, size=17, color=MUTED)).arrange(RIGHT, buff=0.1)
                          for i in ORDER]).arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.12),
                                                            col_alignments="ll")
        legend.next_to(box, DOWN, buff=0.22)
        self.play(FadeIn(tickets), FadeIn(db_lab), FadeIn(legend))

        reads = ValueTracker(0)
        counted = ValueTracker(0)
        meter = VGroup(
            safe_text("lecturas", size=20, color=MUTED),
            always_redraw(lambda: safe_text(f"{int(reads.get_value())}", size=34, color=YELLOW,
                                            weight=BOLD)),
            safe_text("candidatos", size=20, color=MUTED),
            always_redraw(lambda: safe_text(f"{int(counted.get_value())}", size=34, color=YELLOW,
                                            weight=BOLD)),
        )
        meter_pos = [[4.4, 2.95, 0], [4.4, 2.45, 0], [6.1, 2.95, 0], [6.1, 2.45, 0]]
        for m, p in zip(meter, meter_pos):
            m.move_to(p)
        for m, p in zip(meter[1::2], meter_pos[1::2]):
            m.add_updater(lambda mob, p=p: mob.move_to(p))
        self.play(FadeIn(meter))

        level_y = [1.45, 0.35, -0.75, -1.85]
        level_rows: list[VGroup] = []

        def scan() -> Animation:
            beam = Rectangle(width=box.width - 0.1, height=0.10, stroke_width=0,
                             fill_color=YELLOW, fill_opacity=0.55).move_to(box.get_top() + DOWN * 0.1)
            return Succession(FadeIn(beam, run_time=0.1),
                              beam.animate(run_time=0.8).move_to(box.get_bottom() + UP * 0.1),
                              FadeOut(beam, run_time=0.1))

        def show_level(k: int) -> None:
            cands, pruned, freq = APRIORI[k]
            all_nodes = [c for c in cands] + [p for p in pruned]
            all_nodes.sort(key=lambda s: [ORDER.index(i) for i in sorted(s, key=ORDER.index)])
            row = VGroup(*[lattice_node(s) for s in all_nodes]).arrange(RIGHT, buff=0.22)
            row.move_to([2.75, level_y[k], 0])
            tag = safe_text(f"nivel {k + 1}", size=18, color=MUTED).move_to([-1.75, level_y[k], 0])
            level_rows.append(VGroup(tag, row))
            self.play(FadeIn(tag), LaggedStart(*[FadeIn(n, shift=DOWN * 0.1) for n in row],
                                               lag_ratio=0.08), run_time=0.7)
            marks = []
            for s, n in zip(all_nodes, row):
                if s in pruned:
                    marks.append(Cross(n, stroke_color=RED, stroke_width=3, scale_factor=0.8))
            if marks:
                self.play(*[Create(m) for m in marks], *[n.animate.set_opacity(0.35)
                          for s, n in zip(all_nodes, row) if s in pruned], run_time=0.5)
                level_rows[-1].add(*marks)
            if cands:
                outlines = VGroup(*[SurroundingRectangle(n, color=YELLOW, buff=0.06,
                                                         stroke_width=2)
                                    for s, n in zip(all_nodes, row) if s in cands])
                self.play(Create(outlines), run_time=0.3)
                self.play(scan(), reads.animate.increment_value(1),
                          counted.animate.increment_value(len(cands)), run_time=1.0)
                self.play(*[o.animate.set_color(GREEN if s in freq else RED)
                            for o, s in zip(outlines, [s for s in all_nodes if s in cands])],
                          run_time=0.4)
                level_rows[-1].add(outlines)

        show_level(0)
        show_level(1)

        self.beat("4–5 min. Nivel 2: {Pan, Cerveza} aparece sólo en 2 tickets y cae (rojo). "
                  "En el nivel 3 la antimonotonía poda dos tríos sin contarlos, pero los "
                  "demás exigen otra lectura completa. Pregunte: ¿cuántas lecturas haría falta "
                  "con un patrón frecuente de 10 ítems?")
        show_level(2)
        show_level(3)
        verdict = pill("Cada nivel: fabricar candidatos + releer toda la base", RED, size=21)
        verdict.move_to([2.4, -2.9, 0])
        self.play(FadeIn(verdict, shift=UP * 0.1))
        self.apriori_meter = meter

    def explosion(self) -> None:
        self.beat("5–6 min. Escale la cuenta. Con m productos frecuentes, el nivel 2 tiene "
                  "m(m−1)/2 candidatos. Pida estimar el de 10 000 antes de revelarlo. Cierre: "
                  "si existe un patrón frecuente de 20 ítems, sus 2^20 subconjuntos también "
                  "son frecuentes y Apriori los fabrica todos.")
        self.start("Intuición", "La cuenta que no escala", ORANGE)
        rows = [["productos", "pares candidatos"]]
        for m in (4, 100, 1_000, 10_000):
            rows.append([f"{m:,}".replace(",", " "), f"{comb(m, 2):,}".replace(",", " ")])
        cols = [[INK, INK], [INK, YELLOW], [INK, ORANGE], [INK, RED]]
        tbl = table(rows, col_widths=[2.9, 3.7], size=28, row_h=0.66, colors=cols)
        tbl.move_to([-3.3, 0.45, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in tbl[1:-1]],
                              lag_ratio=0.35), run_time=1.6)
        side = VGroup(
            safe_text("Un patrón frecuente", size=26, weight=BOLD),
            safe_text("de 20 ítems obliga a", size=26, weight=BOLD),
            safe_text(f"{2 ** 20 - 1:,}".replace(",", " "), size=40, color=RED, weight=BOLD),
            safe_text("subconjuntos frecuentes,", size=26, weight=BOLD),
            safe_text("todos fabricados como candidatos", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.6, 0.5, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.1) for s in side], lag_ratio=0.2))
        q = pill("¿Podemos evitar fabricar candidatos?", YELLOW, size=24).move_to([0, -2.55, 0])
        self.play(FadeIn(q, shift=UP * 0.1))

    def words(self) -> None:
        self.beat("6–7 min. Analogía. Escriba en la pizarra sol, sola, solar, sopa: 16 letras. "
                  "Pregunte: ¿hace falta escribir «so» cuatro veces? Inserte palabra por palabra "
                  "y pida predecir cuántos nodos nuevos crea cada una.")
        self.start("Intuición", "Lo que se repite, se escribe una vez", BLUE)
        wlist = VGroup(*[safe_text(w, size=40, weight=BOLD, font="Menlo") for w in WORDS])
        wlist.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([-5.3, 0.3, 0])
        letters = safe_text(f"{WORD_LETTERS} letras escritas", size=24, color=MUTED)
        letters.next_to(wlist, DOWN, buff=0.5).align_to(wlist, LEFT)
        self.play(LaggedStart(*[FadeIn(w, shift=RIGHT * 0.1) for w in wlist], lag_ratio=0.2),
                  FadeIn(letters))
        trie = TreeView(WORD_TREE, root_pos=[0.2, 2.15, 0], dx=2.2, dy=0.86, node_w=1.05,
                        node_h=0.48, size=22, colorize=False, root_label="·")
        self.play(FadeIn(trie.root))
        nodes_ct = ValueTracker(0)
        counter = always_redraw(lambda: safe_text(f"{int(nodes_ct.get_value())} nodos",
                                                  size=30, color=GREEN, weight=BOLD)
                                .move_to([4.6, 1.3, 0]))
        self.add(counter)
        for w, mob in zip(WORDS, wlist):
            before = len(trie.nodes)
            anims = trie.insert(list(w))
            created = len(trie.nodes) - before
            self.play(mob.animate.set_color(YELLOW), run_time=0.2)
            self.play(Succession(*anims), nodes_ct.animate.increment_value(created),
                      run_time=0.35 * len(anims) + 0.3)
            self.play(mob.animate.set_color(MUTED), run_time=0.2)

        self.beat("7–8 min. Lea el árbol: el número en cada nodo cuenta cuántas palabras pasan "
                  "por ahí («so» lo comparten 4). 16 letras se guardan en 7 nodos SIN perder "
                  "información: cada palabra se recupera bajando por el árbol. Ese es el FP-tree.")
        compare = VGroup(
            safe_text(f"{WORD_LETTERS} letras", size=30, color=MUTED, weight=BOLD),
            safe_text("→", size=30, color=MUTED),
            safe_text(f"{len(WORD_TREE)} nodos", size=30, color=GREEN, weight=BOLD),
        ).arrange(RIGHT, buff=0.25).move_to([4.6, 1.3, 0])
        lossless = safe_text("sin perder información", size=22, color=MUTED)
        lossless.next_to(compare, DOWN, buff=0.2)
        self.play(FadeOut(counter), FadeIn(compare), FadeIn(lossless))
        self.play(Indicate(trie.path_group(("s", "o")), color=YELLOW))

        self.beat("8–9 min. El obstáculo: un ticket NO es una palabra. {Pan, Leche} y "
                  "{Leche, Pan} son el mismo ticket; las letras tienen orden, los conjuntos no. "
                  "Pregunte: si nosotros elegimos el orden, ¿cuál comprime más? Que voten: "
                  "más frecuentes primero o al revés. No revele.")
        q = VGroup(
            safe_text("Un ticket no tiene orden:", size=26, weight=BOLD),
            VGroup(chip_row(["Pan", "Leche"], size=20), safe_text("=", size=26),
                   chip_row(["Leche", "Pan"], size=20)).arrange(RIGHT, buff=0.2),
            safe_text("¿Qué orden elegimos?", size=26, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.6, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in q], lag_ratio=0.3))

        self.beat("9–10 min. Pregunta guía de la clase (léala en voz alta) y la ruta. FP-Growth "
                  "lee la base dos veces: una para contar y otra para construir el árbol. Todo "
                  "lo demás ocurre dentro del árbol.")
        self.start("Pregunta guía", "Encontrar lo frecuente sin fabricar candidatos", YELLOW)
        guide = VGroup(
            safe_text("¿Podemos encontrar todos los itemsets frecuentes", size=34, weight=BOLD),
            safe_text("leyendo la base sólo dos veces?", size=34, weight=BOLD, color=YELLOW),
        ).arrange(DOWN, buff=0.18).move_to([0, 0.75, 0])
        route = VGroup(*[pill(t, c, size=22) for t, c in [
            ("1 Comprimir: FP-tree", BLUE), ("2 Minar: bases condicionales", PINK),
            ("3 Comparar con Apriori", ORANGE), ("4 Practicar", GREEN)]]
        ).arrange(RIGHT, buff=0.2)
        if route.width > 12.6:
            route.scale_to_fit_width(12.6)
        route.move_to([0, -1.2, 0])
        self.play(FadeIn(guide, shift=UP * 0.1))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in route], lag_ratio=0.2))
