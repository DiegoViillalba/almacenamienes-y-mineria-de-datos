"""Acto 2 · Construir el FP-tree (10–24 min).

Cinco tickets del capítulo 5.3, soporte mínimo 3. Primera lectura: contar y
fijar el orden global. Segunda lectura: ordenar cada ticket e insertarlo,
prediciendo cuántos nodos nuevos crea. Después se lee el árbol (invariantes,
tabla de encabezados) y se contrasta con el árbol que produce el orden inverso.
"""
from __future__ import annotations

from manim import *

from sections.data import (APPEARANCES, COUNTS, MIN_SUP, ORDER, ORDERED, REV_ORDER,
                           REV_TREE, TICKETS_RAW, TRACE, TREE)
from sections.fptree import TreeView
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE,
                            RED, YELLOW, card, chip, color_of, pill, safe_text, tex)

ROW_Y = [1.45, 0.75, 0.05, -0.65, -1.35]
CHIP_X0 = -5.55          # borde izquierdo de la primera ficha de cada ticket
TREE_ROOT = [2.4, 2.25, 0]
TREE_KW = dict(dx=2.7, dy=1.05, node_w=2.1, node_h=0.52, size=22)


def ticket_chips(items: list[str]) -> VGroup:
    return VGroup(*[chip(i, size=21) for i in items])


def place_row(chips: VGroup, y: float, x0: float = CHIP_X0, buff: float = 0.1) -> list[np.ndarray]:
    """Posiciones de las fichas alineadas a la izquierda en la fila ``y``."""
    out, x = [], x0
    for c in chips:
        out.append(np.array([x + c.width / 2, y, 0]))
        x += c.width + buff
    return out


def main_tree() -> TreeView:
    return TreeView(TREE, root_pos=TREE_ROOT, **TREE_KW)


class BuildSection:
    def build(self) -> None:
        self.beat("10–11 min. Volvemos a los cinco tickets del libro con soporte mínimo 3 "
                  "(60 %). Pida que, sin calcular, digan qué producto aparece más y cuál "
                  "menos.")
        head = self.start("1 · Comprimir", f"Cinco tickets, soporte mínimo {MIN_SUP}", BLUE)
        names = VGroup(*[safe_text(f"T{k + 1}", size=24, color=MUTED, weight=BOLD)
                         .move_to([-6.25, y, 0]) for k, y in enumerate(ROW_Y)])
        rows = []
        for t, y in zip(TICKETS_RAW, ROW_Y):
            chips = ticket_chips(t)
            for c, p in zip(chips, place_row(chips, y)):
                c.move_to(p)
            rows.append(chips)
        self.play(FadeIn(names), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows],
                                             lag_ratio=0.15), run_time=1.0)
        ask = pill("¿Qué ítem aparece más?", YELLOW, size=22).move_to([3.4, 1.45, 0])
        self.play(FadeIn(ask))

        # ── Primera lectura: contar ────────────────────────────────────────
        self.beat("11–12 min. Primera lectura de la base: contamos cada ítem. El umbral (línea "
                  "amarilla) es 3. Aquí ningún ítem cae; si alguno cayera, se borraría de los "
                  "tickets antes de construir el árbol (lo verán en el reto).")
        first_seen = []
        for t in TICKETS_RAW:
            for i in t:
                if i not in first_seen:
                    first_seen.append(i)
        unit = 0.78
        bar_x0 = 1.65
        bar_rows = {}
        bar_y = [1.2, 0.35, -0.5, -1.35]
        for i, y in zip(first_seen, bar_y):
            lab = chip(i, size=21, width=1.45).move_to([bar_x0 - 0.9, y, 0])
            bar = Rectangle(width=COUNTS[i] * unit, height=0.46, stroke_width=0,
                            fill_color=color_of(i), fill_opacity=0.75)
            bar.move_to([bar_x0 + COUNTS[i] * unit / 2, y, 0])
            num = safe_text(f"{COUNTS[i]}", size=26, weight=BOLD).next_to(bar, RIGHT, buff=0.15)
            bar_rows[i] = VGroup(lab, bar, num)
        thr = DashedLine([bar_x0 + MIN_SUP * unit, 1.75, 0], [bar_x0 + MIN_SUP * unit, -1.9, 0],
                         color=YELLOW, stroke_width=3)
        thr_lab = safe_text(f"mínimo {MIN_SUP}", size=20, color=YELLOW).next_to(thr, DOWN, buff=0.1)
        badge = pill("lectura 1 de 2: contar", BLUE, size=20).move_to([3.4, 2.35, 0])
        self.play(FadeOut(ask), FadeIn(badge))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                for r in bar_rows.values()], lag_ratio=0.2), run_time=1.4)
        self.play(Create(thr), FadeIn(thr_lab))

        # ── Orden global ───────────────────────────────────────────────────
        self.beat("12–13 min. Fijamos un orden global: de más a menos frecuente. Pan y Pañales "
                  "empatan con 4: desempate alfabético (cualquier regla fija sirve, pero debe "
                  "ser la misma para todos los tickets). Recuerde la votación del inicio.")
        self.play(*[bar_rows[i].animate.set_y(y) for i, y in zip(ORDER, bar_y)], run_time=0.9)
        order_txt = VGroup()
        for k, i in enumerate(ORDER):
            order_txt.add(safe_text(i, size=24, color=color_of(i), weight=BOLD))
            if k < len(ORDER) - 1:
                order_txt.add(safe_text("≻", size=24, color=MUTED))
        order_txt.arrange(RIGHT, buff=0.15).move_to([3.4, -2.55, 0])
        tie = safe_text("empate Pan = Pañales → alfabético", size=18, color=MUTED)
        tie.next_to(order_txt, DOWN, buff=0.12)
        self.play(FadeIn(order_txt, shift=UP * 0.1), FadeIn(tie))

        self.beat("13–14 min. Reescribimos cada ticket en ese orden. El contenido no cambia, sólo "
                  "la forma de escribirlo: ahora todos empiezan igual (Leche…). Pregunte: ¿qué "
                  "tickets empiezan con el mismo prefijo de dos ítems?")
        anims = []
        sorted_rows = []
        for chips, t_ord, y in zip(rows, ORDERED, ROW_Y):
            by_item = {c.item: c for c in chips}
            new = VGroup(*[by_item[i] for i in t_ord])
            for c, p in zip(new, place_row(new, y)):
                anims.append(c.animate.move_to(p))
            sorted_rows.append(new)
        self.play(*anims, run_time=1.2)
        rows = sorted_rows

        # ── Segunda lectura: insertar ──────────────────────────────────────
        self.beat("14–15 min. Segunda (y última) lectura: insertamos cada ticket ordenado desde "
                  "la raíz ∅. Regla: si el siguiente ítem ya es hijo del nodo actual, sumamos 1; "
                  "si no, creamos un nodo con 1. T1 = [Leche, Pan] crea dos nodos.")
        self.play(FadeOut(VGroup(*bar_rows.values(), thr, thr_lab, order_txt, tie, badge)),
                  run_time=0.4)
        tree = main_tree()
        badge = pill("lectura 2 de 2: insertar", BLUE, size=20).move_to([5.25, 2.25, 0])
        self.play(FadeIn(tree.root), FadeIn(badge))
        status = VGroup()
        self.tree = tree

        notes = [
            None,
            "15–16 min. Antes de insertar T2 pregunte: ¿cuántos nodos nuevos crea? Respuesta: "
            "comparte Leche → Pan (sus contadores suben a 2) y crea Pañales y Cerveza.",
            "16–17 min. T3 = [Leche, Pañales, Cerveza]. Predicción. Comparte sólo Leche: bajo "
            "Leche no hay hijo Pañales, así que abre una RAMA NUEVA. Es la única que lo hace. "
            "Error típico: «reusar» el nodo Pañales de la otra rama; no se puede, el camino es otro.",
            "17–18 min. T4 repite exactamente a T2: no crea nada, sólo suben contadores. Aquí "
            "está la compresión: un ticket repetido cuesta cero nodos.",
            "18–19 min. T5 = [Leche, Pan, Pañales]: también comparte todo su camino. Fin de la "
            "segunda lectura: ya no volveremos a tocar los tickets.",
        ]
        for k, (t, y) in enumerate(zip(ORDERED, ROW_Y)):
            if notes[k]:
                self.beat(notes[k])
            hl = SurroundingRectangle(VGroup(names[k], rows[k]), color=YELLOW, buff=0.08,
                                      stroke_width=3)
            shared, created = TRACE[k]
            new_status = VGroup(
                safe_text(f"comparte {len(shared)}", size=22, color=GREEN, weight=BOLD),
                safe_text("·", size=22, color=MUTED),
                safe_text(f"crea {len(created)}", size=22, color=ORANGE, weight=BOLD),
            ).arrange(RIGHT, buff=0.2).move_to([-4.3, -2.3, 0])
            self.play(Create(hl), FadeOut(status), run_time=0.35)
            ins = tree.insert(t)
            self.play(Succession(*ins), run_time=0.5 * len(ins) + 0.2)
            self.play(FadeIn(new_status), run_time=0.3)
            status = new_status
            self.play(FadeOut(hl), names[k].animate.set_color(GREEN), run_time=0.3)

        # ── Leer el árbol ──────────────────────────────────────────────────
        self.beat("19–20 min. Lean el árbol. 16 apariciones de ítems → 6 nodos. Cada contador "
                  "dice cuántos tickets EMPIEZAN con ese camino: Leche → Pan: 4 significa que 4 "
                  "tickets empiezan [Leche, Pan]. Al bajar, los contadores nunca crecen.")
        summary = VGroup(
            safe_text(f"{APPEARANCES} apariciones", size=28, color=MUTED, weight=BOLD),
            safe_text("→", size=28, color=MUTED),
            safe_text(f"{len(TREE)} nodos", size=28, color=GREEN, weight=BOLD),
        ).arrange(RIGHT, buff=0.2).move_to([-4.0, -2.95, 0])
        self.play(FadeOut(status), FadeIn(summary))
        path = tree.path_group(("Leche", "Pan"))
        self.play(Indicate(path, color=YELLOW, scale_factor=1.05), run_time=1.0)
        mono = safe_text("contador del hijo ≤ contador del padre", size=20, color=YELLOW)
        mono.move_to([3.6, -2.95, 0])
        self.play(FadeIn(mono))

        self.beat("20–21 min. Pregunta: ¿en cuántos tickets aparece Pañales? Dejen que busquen. "
                  "Está en DOS nodos: 3 + 1 = 4. Para no recorrer todo el árbol, FP-tree guarda "
                  "una tabla de encabezados con enlaces entre los nodos de un mismo ítem.")
        q = pill("¿En cuántos tickets aparece Pañales?", YELLOW, size=21).move_to([-3.9, -2.2, 0])
        self.play(FadeIn(q))
        pn = [tree.nodes[p] for p in tree.nodes_of("Pañales")]
        rings = VGroup(*[SurroundingRectangle(n, color=YELLOW, buff=0.06) for n in pn])
        self.play(Create(rings))
        sum_txt = tex(r"3 + 1 = 4", size=40, color=PINK).move_to([5.75, -0.4, 0])
        self.play(Write(sum_txt))

        # ── Tabla de encabezados ───────────────────────────────────────────
        self.beat("21–22 min. Tabla de encabezados: un renglón por ítem, en el orden global, con "
                  "su soporte y un puntero al primer nodo. Las líneas discontinuas enlazan las "
                  "apariciones. Seguir los enlaces de un ítem = encontrar todos sus tickets.")
        names.add(*rows)
        self.play(FadeOut(VGroup(names, rings, sum_txt, q, badge, summary, mono)), run_time=0.4)
        htitle = safe_text("tabla de encabezados", size=22, color=MUTED, weight=BOLD)
        htitle.move_to([-4.9, 2.2, 0])
        header = VGroup()
        pointers = VGroup()
        for i in ORDER:
            first = tree.nodes_of(i)[0]
            y = tree.pos[first][1]
            row = VGroup(chip(i, size=21, width=1.5).move_to([-5.6, y, 0]),
                         safe_text(f"{COUNTS[i]}", size=26, weight=BOLD).move_to([-4.3, y, 0]))
            header.add(row)
            start = np.array([-3.75, y, 0])
            end = tree.nodes[first][1].get_left()
            pointers.add(DashedLine(start, end, color=color_of(i), stroke_width=2.5,
                                    dash_length=0.1))
        sup_lab = safe_text("soporte", size=18, color=MUTED).move_to([-4.3, 1.75, 0])
        self.play(FadeIn(htitle), FadeIn(sup_lab),
                  LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in header], lag_ratio=0.15))
        self.play(LaggedStart(*[Create(p) for p in pointers], lag_ratio=0.15), run_time=1.0)
        links = VGroup(tree.links("Pañales", PINK), tree.links("Cerveza", PURPLE))
        self.play(Create(links), run_time=1.0)
        self.header_group = VGroup(htitle, sup_lab, header, pointers, links)

        # ── ¿Y con el orden inverso? ───────────────────────────────────────
        self.beat("22–23 min. Resolvemos la votación del inicio. Mismos tickets, orden inverso "
                  "(menos frecuentes primero). Pida predecir cuántos nodos tendrá. Resultado: 10 "
                  "contra 6. Los ítems frecuentes arriba se comparten entre más tickets.")
        self.clear()
        self.play(FadeIn(self.heading("1 · Comprimir", "¿Importa el orden?", BLUE)), run_time=0.3)
        good = main_tree().build_all()
        good.scale(0.66).move_to([-3.75, -0.25, 0])
        bad = TreeView(REV_TREE, root_pos=ORIGIN, dx=2.3, dy=1.05, node_w=2.1, size=22).build_all()
        bad.scale(0.66).move_to([3.15, -0.25, 0])

        def caption(order, n, col):
            line = VGroup(*[safe_text(i, size=18, color=color_of(i), weight=BOLD) for i in order])
            line.arrange(RIGHT, buff=0.22)
            return VGroup(safe_text(f"{n} nodos", size=32, color=col, weight=BOLD), line
                          ).arrange(DOWN, buff=0.12)

        cap_good = caption(ORDER, len(TREE), GREEN).move_to([-3.75, -3.0, 0])
        cap_bad = caption(REV_ORDER, len(REV_TREE), RED).move_to([3.15, -3.0, 0])
        lab_good = safe_text("más frecuentes primero", size=22, color=GREEN).move_to([-3.75, 2.25, 0])
        lab_bad = safe_text("menos frecuentes primero", size=22, color=RED).move_to([3.15, 2.25, 0])
        self.play(FadeIn(good), FadeIn(lab_good), FadeIn(cap_good))
        self.play(LaggedStart(*[FadeIn(m) for m in bad], lag_ratio=0.03), FadeIn(lab_bad),
                  run_time=1.2)
        self.play(FadeIn(cap_bad, shift=UP * 0.1))

        # ── Formalización ──────────────────────────────────────────────────
        self.beat("23–24 min. Formalice. Cada nodo v guarda un ítem y un contador: el número de "
                  "tickets (ordenados) que empiezan con el camino raíz → v. Lea los cinco pasos y "
                  "subraye: sólo DOS lecturas de la base. Advertencia: en el peor caso (sin "
                  "prefijos comunes) el árbol ocupa más memoria que los datos.")
        self.clear()
        self.play(FadeIn(self.heading("1 · Comprimir", "El FP-tree, formalmente", BLUE)),
                  run_time=0.3)
        definition = tex(r"\operatorname{count}(v)=\left|\{\,T\in\mathcal{D}:\ "
                         r"\operatorname{camino}(v)\ \text{es prefijo de}\ T\ \text{ordenada}\,\}\right|",
                         size=36, max_width=12.6).move_to([0, 1.85, 0])
        steps = VGroup(*[
            VGroup(safe_text(n, size=24, color=c, weight=BOLD), safe_text(s, size=24, max_width=7.6)
                   ).arrange(RIGHT, buff=0.25)
            for n, c, s in [
                ("1", BLUE, "Lectura 1: contar soportes y quitar los ítems infrecuentes."),
                ("2", BLUE, "Orden global: soporte descendente, desempate fijo."),
                ("3", ORANGE, "Lectura 2: ordenar cada ticket e insertarlo desde ∅."),
                ("4", ORANGE, "Prefijo existente: +1. Si no existe: nodo nuevo con 1."),
                ("5", PINK, "Tabla de encabezados: enlaza los nodos de cada ítem."),
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-2.15, -0.75, 0])
        mini = main_tree().build_all().scale(0.5).move_to([4.75, -0.75, 0])
        self.play(Write(definition))
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in steps], lag_ratio=0.2),
                  FadeIn(mini), run_time=1.4)
