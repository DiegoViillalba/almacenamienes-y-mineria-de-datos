"""Acto 4 · Comparar con Apriori, escalar y usar en Python (42–50 min)."""
from __future__ import annotations

from manim import *

from sections.data import (APPEARANCES, APRIORI, BASE_CERVEZA, D_APP, D_NODES, FREQUENT,
                           G_APP, G_APRIORI, G_FREQUENT, G_NODES, MIN_SUP, TICKETS, TREE)
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED,
                            YELLOW, card, code_block, pill, safe_text, table)


def stat_card(title: str, color: str, rows: list[tuple[str, str]], x: float) -> VGroup:
    head = safe_text(title, size=32, color=color, weight=BOLD)
    lines = VGroup()
    for big, small in rows:
        lines.add(VGroup(safe_text(big, size=40, color=color, weight=BOLD),
                         safe_text(small, size=22, color=MUTED, max_width=3.6)
                         ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN))
    lines.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    body = VGroup(head, lines).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    box = card(5.6, body.height + 0.7, color)
    box.move_to([x, 0.15, 0])
    body.move_to(box)
    return VGroup(box, body)


class CompareSection:
    def compare(self) -> None:
        reads_ap = sum(1 for c, _, _ in APRIORI if c)
        cands_ap = sum(len(c) for c, _, _ in APRIORI)
        pruned_ap = sum(len(p) for _, p, _ in APRIORI)
        self.beat("42–44 min. Mismo ejemplo, dos caminos. Apriori: 3 lecturas, 12 candidatos "
                  "contados y 3 podados. FP-Growth: 2 lecturas y ningún candidato; trabajó con un "
                  "árbol de 6 nodos y tres bases condicionales pequeñas. Subraye: la respuesta es "
                  "idéntica, ambos son exactos.")
        self.start("3 · Comparar", "Mismo resultado, otro camino", ORANGE)
        ap = stat_card("Apriori", ORANGE, [
            (f"{reads_ap}", "lecturas de la base"),
            (f"{cands_ap}", "candidatos contados"),
            (f"{pruned_ap}", "podados sin contar"),
        ], -3.3)
        fp = stat_card("FP-Growth", BLUE, [
            ("2", "lecturas de la base"),
            ("0", "candidatos"),
            (f"{len(TREE)}", "nodos + 3 bases condicionales"),
        ], 3.3)
        self.play(FadeIn(ap, shift=RIGHT * 0.1))
        self.play(FadeIn(fp, shift=LEFT * 0.1))
        same = pill(f"Los dos encuentran los mismos {len(FREQUENT)} itemsets: cambia el camino, "
                    "no la respuesta", GREEN, size=21).move_to([0, -2.75, 0])
        self.play(FadeIn(same, shift=UP * 0.1))

        self.beat("44–45 min. A escala: las 50 canastas del capítulo de Apriori (mínimo 5). La "
                  "línea azul cuenta las apariciones de ítems; la naranja, los nodos del árbol. "
                  "Pregunte de qué depende la distancia entre ambas: de cuánto se parecen los "
                  "tickets. En datos densos (rosa, sintéticos) casi todo es prefijo compartido.")
        self.start("3 · Comparar", "¿Cuánto comprime con más datos?", ORANGE)
        ax = Axes(x_range=[0, 50, 10], y_range=[0, 150, 50], x_length=6.8, y_length=4.2,
                  axis_config=dict(color=GRID, include_numbers=True, font_size=20,
                                   decimal_number_config=dict(color=MUTED, num_decimal_places=0)),
                  tips=False).move_to([-2.6, -0.25, 0])
        xl = safe_text("canastas insertadas", size=18, color=MUTED).next_to(ax, DOWN, buff=0.15)
        self.play(Create(ax), FadeIn(xl))
        xs = list(range(1, len(G_APP) + 1))

        def curve(ys, color):
            return ax.plot_line_graph(xs, ys, line_color=color, add_vertex_dots=False,
                                      stroke_width=4)

        app = curve(G_APP, BLUE)
        nodes = curve(G_NODES, ORANGE)
        dense = curve(D_NODES, PINK)
        self.play(Create(app), Create(nodes), run_time=1.6)
        labels = VGroup(
            safe_text(f"apariciones: {G_APP[-1]}", size=20, color=BLUE, weight=BOLD),
            safe_text(f"nodos: {G_NODES[-1]} ({G_NODES[-1] / G_APP[-1]:.0%})", size=20,
                      color=ORANGE, weight=BOLD),
        )
        labels[0].next_to(ax.c2p(50, G_APP[-1]), LEFT, buff=0.15).shift(UP * 0.2)
        labels[1].next_to(ax.c2p(50, G_NODES[-1]), UP, buff=0.15).shift(LEFT * 0.6)
        self.play(FadeIn(labels))
        ap_reads = sum(1 for c, _, _ in G_APRIORI if c)
        ap_cands = sum(len(c) for c, _, _ in G_APRIORI)
        side = VGroup(
            safe_text("50 canastas reales del libro", size=22, weight=BOLD),
            safe_text(f"{len(G_FREQUENT)} itemsets frecuentes", size=21, color=MUTED),
            safe_text(f"Apriori: {ap_cands} candidatos, {ap_reads} lecturas", size=21,
                      color=ORANGE),
            safe_text("FP-Growth: 0 candidatos, 2 lecturas", size=21, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.35, 1.0, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.1) for s in side], lag_ratio=0.2))

        self.beat("45–46 min. Ahora la curva rosa: 50 canastas sintéticas donde cada una es un "
                  "prefijo de la misma lista. Apariciones 150, nodos 5. La compresión no es una "
                  "propiedad del algoritmo sino de los datos: cuanto más se parecen los tickets, "
                  "mejor.")
        self.play(Create(dense), run_time=1.0)
        dlab = safe_text(f"densas (sintéticas): {D_APP[-1]} → {D_NODES[-1]} nodos", size=20,
                         color=PINK, weight=BOLD)
        dlab.next_to(ax.c2p(50, D_NODES[-1]), UP, buff=0.12).shift(LEFT * 1.6)
        moral = pill("Más parecidos ⇒ más compresión", PINK, size=20)
        moral.move_to([4.35, -1.3, 0])
        self.play(FadeIn(dlab), FadeIn(moral))

        self.beat("46–48 min. Cuándo conviene cada uno. FP-Growth brilla con bases grandes y "
                  "densas, donde Apriori fabrica millones de candidatos. Sus costos: memoria del "
                  "árbol, implementación más compleja y no es incremental (si llegan tickets "
                  "nuevos, se reconstruye). Ninguno arregla un umbral mal elegido: muchos "
                  "patrones son mucha salida.")
        self.start("3 · Comparar", "¿Cuándo conviene cada uno?", ORANGE)
        rows = [["", "Apriori", "FP-Growth"],
                ["Candidatos", "nivel por nivel", "ninguno"],
                ["Lecturas de la base", "una por nivel", "dos"],
                ["Memoria", "listas y conteos", "el árbol (puede crecer)"],
                ["Datos densos", "explota", "comprime bien"],
                ["Explicar e implementar", "sencillo", "más complejo"],
                ["Datos que cambian", "recontar", "reconstruir el árbol"]]
        cols = [[MUTED, ORANGE, BLUE]] * (len(rows) - 1)
        tbl = table(rows, col_widths=[4.0, 3.6, 4.4], size=25, row_h=0.62, colors=cols)
        tbl.move_to([0, -0.25, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in tbl[1:-1]],
                              lag_ratio=0.15), run_time=1.4)

        self.beat("48–50 min. En Python, mlxtend: la misma matriz one-hot de Apriori y una sola "
                  "función. Ojo: min_support es relativo, 3 / 5 = 0.6. FP-Growth entrega "
                  "itemsets; las reglas (confianza, lift) se generan después, exactamente como "
                  "en la clase pasada.")
        self.start("3 · Comparar", "En Python: una línea cambia", ORANGE)
        code = code_block([
            "from mlxtend.preprocessing import TransactionEncoder",
            "from mlxtend.frequent_patterns import fpgrowth, association_rules",
            "",
            "te = TransactionEncoder()",
            "df = pd.DataFrame(te.fit_transform(tickets), columns=te.columns_)",
            "",
            f"items = fpgrowth(df, min_support={MIN_SUP}/{len(TICKETS)}, use_colnames=True)",
            f"# → {len(FREQUENT)} itemsets, los mismos que apriori(df, ...)",
            "reglas = association_rules(items, metric=\"lift\", min_threshold=1)",
        ], size=21, width=12.4).move_to([0, 0.25, 0])
        self.play(FadeIn(code[0]), LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in code[1]],
                                               lag_ratio=0.1), run_time=1.4)
        note = pill("min_support es relativo · las reglas se evalúan después con lift", YELLOW,
                    size=20).move_to([0, -2.75, 0])
        self.play(FadeIn(note, shift=UP * 0.1))
