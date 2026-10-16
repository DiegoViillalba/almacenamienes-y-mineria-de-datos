"""Acto 5 · Reto en parejas con datos nuevos (50–57 min) y cierre (57–60 min).

Seis tickets de cafetería, soporte mínimo 3. Incluye un ítem infrecuente (té),
un empate (café = galletas) y una trampa: en la base de chocolate, café y
galletas llegan a 3 por separado pero {café, galletas, chocolate} sólo a 2.
"""
from __future__ import annotations

from manim import *

from sections.data import (P_APPEARANCES, P_BASE_AZUCAR, P_BASE_CHOC, P_COUNTS, P_FREQUENT,
                           P_MIN, P_ORDER, P_ORDERED, P_TREE, PRACTICE, PRACTICE_RAW,
                           base_counts)
from sections.fptree import TreeView
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED,
                            YELLOW, bullet_list, chip, chip_row, color_of, pill, safe_text, tex)

P_ROW_Y = [1.55, 0.9, 0.25, -0.4, -1.05, -1.7]
P_X0 = -5.75
P_FREQ = {frozenset(p): c for p, c in P_FREQUENT}


def p_fmt(items) -> str:
    return "{" + ", ".join(i for i in P_ORDER if i in items) + "}"


def base_panel(item: str, base, x: float) -> VGroup:
    bc = base_counts(base)
    title = VGroup(safe_text("base de", size=22, color=MUTED), chip(item, size=21)
                   ).arrange(RIGHT, buff=0.15)
    rows = VGroup(*[VGroup(chip_row(p, size=18), safe_text(f"× {w}", size=22, color=YELLOW,
                                                           weight=BOLD)).arrange(RIGHT, buff=0.25)
                    for p, w in base]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    counts = VGroup(*[VGroup(chip(i, size=17), safe_text(
        f"{bc[i]}", size=22, color=GREEN if bc[i] >= P_MIN else RED, weight=BOLD)
    ).arrange(RIGHT, buff=0.1) for i in P_ORDER if i in bc]).arrange(RIGHT, buff=0.25)
    panel = VGroup(title, rows, counts).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    panel.move_to([x, 0.75, 0])
    return panel


class PracticeSection:
    def practice(self) -> None:
        self.beat("50–54 min. Reto en parejas, 4 minutos, datos nuevos. Pida trabajar en papel: "
                  "primero conteos y orden, después el árbol, al final el minado. Pista para "
                  "el empate café = galletas: alfabético. Recorra y detecte quién olvida quitar "
                  "el té.")
        self.start("Reto en parejas", f"Seis tickets de cafetería, soporte mínimo {P_MIN}", YELLOW)
        names = VGroup(*[safe_text(f"T{k + 1}", size=22, color=MUTED, weight=BOLD)
                         .move_to([-6.35, y, 0]) for k, y in enumerate(P_ROW_Y)])
        rows = []
        for t, y in zip(PRACTICE_RAW, P_ROW_Y):
            r = chip_row(t, size=20)
            r.move_to([0, y, 0]).shift(RIGHT * (P_X0 - r.get_left()[0]))
            rows.append(r)
        tasks = bullet_list([
            "1. Cuenta y fija el orden global.",
            "2. Construye el FP-tree.",
            "3. Mina azúcar.",
            "4. Mina chocolate: ¿es frecuente",
            "    {café, galletas, chocolate}?",
        ], size=24, width=5.6).move_to([3.9, 0.15, 0])
        timer = pill("4 min · desempate alfabético", MUTED, size=19).move_to([3.9, -2.4, 0])
        self.play(FadeIn(names), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows],
                                             lag_ratio=0.12))
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tasks], lag_ratio=0.15),
                  FadeIn(timer))

        self.beat("54–55 min. Solución (1/3): conteos café 5, galletas 5, chocolate 4, azúcar 3, "
                  "té 1. El té sale ANTES de construir. Empate café = galletas: café primero. "
                  "Reordenamos cada ticket.")
        self.play(FadeOut(tasks), FadeOut(timer))
        counts = VGroup()
        for i in P_ORDER + ["té"]:
            ok = P_COUNTS[i] >= P_MIN
            counts.add(VGroup(chip(i, size=19), safe_text(f"{P_COUNTS[i]}", size=24,
                                                          color=GREEN if ok else RED,
                                                          weight=BOLD)).arrange(RIGHT, buff=0.12))
        counts.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.6, 0.3, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in counts], lag_ratio=0.12))
        te_chips = VGroup(*[c for r in rows for c in r if c.item == "té"])
        self.play(Create(Cross(counts[-1], stroke_color=RED, stroke_width=4)),
                  te_chips.animate.set_opacity(0.15))
        anims = []
        for r, t_ord, y in zip(rows, P_ORDERED, P_ROW_Y):
            by_item = {c.item: c for c in r}
            x = P_X0
            for i in t_ord:
                c = by_item[i]
                anims.append(c.animate.move_to([x + c.width / 2, y, 0]))
                x += c.width + 0.12
        self.play(*anims, FadeOut(te_chips), run_time=1.1)

        self.beat("55–56 min. Solución (2/3): el árbol. 17 apariciones → 9 nodos. Pregunte qué "
                  "ticket abre la rama que cuelga directamente de ∅ (T4: no tiene café). Señale "
                  "los tres nodos de azúcar: tres caminos distintos.")
        self.clear()
        self.play(FadeIn(self.heading("Solución", "El FP-tree de la cafetería", GREEN)),
                  run_time=0.3)
        tree = TreeView(P_TREE, root_pos=[1.5, 2.2, 0], dx=2.2, dy=1.05, node_w=2.05,
                        size=20).build_all()
        self.play(FadeIn(tree.root))
        for d in range(1, 5):
            self.play(*[AnimationGroup(Create(tree.edges[p]), FadeIn(tree.nodes[p], scale=0.7))
                        for p in P_TREE if len(p) == d], run_time=0.5)
        summary = VGroup(safe_text(f"{P_APPEARANCES} apariciones", size=26, color=MUTED,
                                   weight=BOLD),
                         safe_text("↓", size=26, color=MUTED),
                         safe_text(f"{len(P_TREE)} nodos", size=26, color=GREEN, weight=BOLD)
                         ).arrange(DOWN, buff=0.12).move_to([-5.4, 0.3, 0])
        self.play(FadeIn(summary), Create(tree.links("azúcar", PURPLE)))

        self.beat("56–57 min. Solución (3/3). Azúcar: tres caminos × 1; sólo café llega a 3 ⇒ "
                  "{azúcar}: 3 y {café, azúcar}: 3. Chocolate: café 3 y galletas 3, pero el "
                  "trío {café, galletas, chocolate} sólo aparece en 2 tickets (el árbol "
                  "condicional se ramifica). Que una pareja explique por qué.")
        self.clear()
        self.play(FadeIn(self.heading("Solución", "Minar azúcar y chocolate", GREEN)),
                  run_time=0.3)
        left = base_panel("azúcar", P_BASE_AZUCAR, -3.9)
        right = base_panel("chocolate", P_BASE_CHOC, 2.6)
        self.play(FadeIn(left, shift=UP * 0.1))
        res_l = VGroup(*[safe_text(f"{p_fmt(p)}: {c}", size=24, color=GREEN, weight=BOLD)
                         for p, c in P_FREQUENT if p[-1] == "azúcar"]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        res_l.next_to(left, DOWN, buff=0.45).align_to(left, LEFT)
        self.play(FadeIn(res_l))
        self.play(FadeIn(right, shift=UP * 0.1))
        res_r = VGroup(*[safe_text(f"{p_fmt(p)}: {c}", size=24, color=GREEN, weight=BOLD)
                         for p, c in P_FREQUENT if p[-1] == "chocolate"])
        trio = frozenset(["café", "galletas", "chocolate"])
        trio_ct = sum(1 for t in PRACTICE if trio <= t)
        res_r.add(safe_text(f"{p_fmt(trio)}: {trio_ct}  ✗", size=24, color=RED, weight=BOLD))
        res_r.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        res_r.next_to(right, DOWN, buff=0.45).align_to(right, LEFT)
        self.play(LaggedStart(*[FadeIn(r) for r in res_r], lag_ratio=0.25))

    def closing(self) -> None:
        self.beat("57–59 min. Síntesis en cuatro verbos. Pida a un estudiante reformular cada uno "
                  "con el ejemplo de Cerveza. Insista: FP-Growth cambia el CÓMO; los itemsets y "
                  "las reglas que salen después son los mismos que con Apriori.")
        self.start("Cierre", "FP-Growth en cuatro verbos", BLUE)
        ideas = VGroup(*[
            VGroup(pill(k, c, width=2.9, size=22), safe_text(t, size=26, max_width=8.6)
                   ).arrange(RIGHT, buff=0.4)
            for k, c, t in [
                ("Contar", BLUE, "Lectura 1: soportes, poda y orden global."),
                ("Comprimir", ORANGE, "Lectura 2: prefijos compartidos → FP-tree."),
                ("Condicionar", PINK, "Base de a: caminos prefijo × contador de a."),
                ("Crecer", PURPLE, "Sufijo + recursión: patrones sin candidatos."),
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.15, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.2),
                  run_time=1.4)

        self.beat("59–60 min. Ticket de salida individual. Respuestas: (1) [Leche, Pan] × 2: el "
                  "peso es el contador del nodo Huevos, no el de Leche ni el de Pan. (2) Porque "
                  "los ítems frecuentes, arriba, son prefijo de más tickets y se comparten.")
        self.start("Ticket de salida", "Dos preguntas, una línea cada una", PURPLE)
        q = VGroup(
            safe_text("1. El único nodo Huevos está en el camino", size=28, weight=BOLD),
            safe_text("Leche: 6 → Pan: 4 → Huevos: 2.", size=28, weight=BOLD, color=YELLOW),
            safe_text("¿Qué entra a la base condicional de Huevos?", size=28, weight=BOLD),
            safe_text("2. ¿Por qué se insertan primero los ítems más frecuentes?", size=28,
                      weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.15, 0])
        q[3].shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in q], lag_ratio=0.25))
        final = safe_text("Siguiente: itemsets cerrados y maximales", size=22, color=BLUE)
        final.move_to([0, -2.75, 0])
        self.play(FadeIn(final))
