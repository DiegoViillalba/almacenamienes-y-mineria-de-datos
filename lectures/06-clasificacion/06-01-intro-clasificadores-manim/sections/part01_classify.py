"""Acto 1 · ¿Qué es clasificar? (0–10 min).

De decisiones cotidianas a una tabla; de la tabla a puntos en un plano; de los
puntos a una regla que parte el plano. Distingue atributos x, etiqueta y y
predicción ŷ antes de escribir cualquier símbolo.
"""
from __future__ import annotations

from manim import *

from sections.data import (STUMP_T, TEST, TRAIN, X_RANGE, Y_RANGE, YTR, baseline_acc,
                           stump_sweep)
from sections.plane import ClientPlane
from sections.theme import (BLUE, CLASS_COLOR, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED,
                            YELLOW, bullet_list, card, legend_item, marker, pill, safe_text,
                            table, tex)


def client_table(size: float = 22) -> VGroup:
    rows = [["cliente", "ingreso", "deuda %", "¿pagó?"]]
    cols = []
    for cid, x, y, c in TRAIN:
        rows.append([str(cid), str(x), str(y), "sí" if c == 0 else "no"])
        cols.append([MUTED, INK, INK, CLASS_COLOR[c]])
    return table(rows, col_widths=[1.35, 1.35, 1.45, 1.35], size=size, row_h=0.43, colors=cols)


class ClassifySection:
    def opening(self) -> None:
        self.beat("0–1 min. Portada. Pregunte: ¿cuántas decisiones de «sí o no» toma hoy un "
                  "banco, un hospital o su correo sin que una persona las revise? Deje que den "
                  "dos o tres ejemplos.")
        eyebrow = safe_text("CLASIFICACIÓN · CLASE INTRODUCTORIA", size=22, color=BLUE,
                            weight=BOLD)
        title = safe_text("¿Pagará el préstamo?", size=62, weight=BOLD, max_width=12.6)
        subtitle = safe_text("Aprender a decidir a partir de ejemplos: k-NN y clasificadores "
                             "lineales", size=28, color=MUTED, max_width=12.0)
        course = safe_text("Almacenes y Minería de Datos", size=25, color=BLUE, weight=BOLD)
        author = safe_text("Diego Villalba · Facultad de Ciencias · UNAM", size=20, color=MUTED)
        logo = self.logo_lockup(width=3.65, height=1.0)
        cover = Group(eyebrow, title, subtitle, course, author, logo).arrange(DOWN, buff=0.26)
        self.play(FadeIn(cover, shift=UP * 0.16), run_time=0.8)

        self.beat("1–2 min. Ruta. Anuncie el hilo: un solo ejemplo (diez clientes de una "
                  "financiera) recorre toda la clase. Habrá dos «tu turno» y un ejercicio final.")
        self.clear()
        self.add_branding()
        self.start("Ruta de la clase", "Una pregunta, cinco paradas", YELLOW)
        route = VGroup(*[pill(t, c, size=24, width=5.6) for t, c in [
            ("1 · ¿Qué es clasificar?", BLUE), ("2 · ¿Cómo medimos errores?", RED),
            ("3 · Preguntar a los vecinos", GREEN), ("4 · Trazar una frontera", PURPLE),
            ("5 · Comparar y practicar", YELLOW)]]).arrange(DOWN, buff=0.22)
        route.move_to([-3.4, -0.35, 0])
        promise = VGroup(
            safe_text("Al terminar sabrás", size=26, color=MUTED),
            safe_text("· leer una matriz de confusión", size=26),
            safe_text("· clasificar a mano con k-NN", size=26),
            safe_text("· interpretar una regresión logística", size=26),
            safe_text("· decidir cuál modelo se equivoca menos", size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.4, -0.35, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in route], lag_ratio=0.12))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in promise], lag_ratio=0.12))
        self.everyday()
        self.table_to_plane()
        self.first_rule()

    # ------------------------------------------------------------------
    def everyday(self) -> None:
        self.beat("2–3 min. Tres decisiones. Pregunte qué tienen en común antes de mostrar la "
                  "línea inferior: algo que SABEMOS (izquierda) y una categoría que "
                  "QUEREMOS ANTICIPAR (derecha), elegida de una lista cerrada.")
        self.start("¿Qué es clasificar?", "Decidir una categoría a partir de lo que sabemos",
                   BLUE)
        cases = [
            ("un correo: remitente, palabras", "spam / no spam"),
            ("una radiografía: píxeles", "sano / enfermo"),
            ("un cliente: ingreso, deuda", "paga / no paga"),
        ]
        rows = VGroup()
        for know, cat in cases:
            left = card(5.4, 0.85)
            lt = safe_text(know, size=24, max_width=5.0).move_to(left)
            arrow = Arrow(LEFT * 0.6, RIGHT * 0.6, color=MUTED, stroke_width=4, buff=0)
            right = card(3.6, 0.85, YELLOW)
            rt = safe_text(cat, size=24, color=YELLOW, weight=BOLD, max_width=3.2).move_to(right)
            rows.add(VGroup(VGroup(left, lt), arrow, VGroup(right, rt)).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.35).move_to([0, 0.15, 0])
        for r in rows:
            self.play(FadeIn(r[0], shift=RIGHT * 0.1), GrowArrow(r[1]), FadeIn(r[2]),
                      run_time=0.6)
        top = VGroup(safe_text("lo que sabemos", size=22, color=MUTED).next_to(rows[0][0], UP),
                     safe_text("lo que queremos anticipar", size=22, color=MUTED)
                     .next_to(rows[0][2], UP))
        self.play(FadeIn(top))
        moral = pill("Clasificar = asignar una categoría de una lista finita", BLUE, size=24)
        moral.move_to([0, -2.6, 0])
        self.play(FadeIn(moral, shift=UP * 0.1))

    def table_to_plane(self) -> None:
        self.beat("3–4 min. La financiera prestó a diez clientes. Señale las columnas: "
                  "ingreso y deuda se conocen ANTES de prestar (atributos, x); «¿pagó?» "
                  "sólo se supo DESPUÉS (etiqueta, y). Pregunte: ¿qué columna querríamos "
                  "adivinar en un cliente nuevo?")
        head = self.start("¿Qué es clasificar?", "Diez clientes que ya conocemos", BLUE)
        tbl = client_table().move_to([-2.9, -0.75, 0])
        self.play(FadeIn(tbl[0]), Create(tbl[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.05) for r in tbl[1:-1]],
                              lag_ratio=0.06), run_time=1.2)
        feat_cols = VGroup(*[VGroup(r[1], r[2]) for r in tbl[:-1]])
        lab_col = VGroup(*[r[3] for r in tbl[:-1]])
        b1 = Brace(feat_cols, UP, color=BLUE, buff=0.08)
        b2 = Brace(lab_col, UP, color=YELLOW, buff=0.08)
        t1 = safe_text("atributos  x", size=24, color=BLUE, weight=BOLD).next_to(b1, UP, 0.08)
        t2 = safe_text("etiqueta  y", size=24, color=YELLOW, weight=BOLD).next_to(b2, UP, 0.08)
        notes = VGroup(
            VGroup(safe_text("x  · atributos", size=26, color=BLUE, weight=BOLD),
                   safe_text("lo que sabemos antes de decidir", size=21, color=MUTED)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            VGroup(safe_text("y  · etiqueta", size=26, color=YELLOW, weight=BOLD),
                   safe_text("la verdad, que llega después", size=21, color=MUTED)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.9, 0.3, 0])
        self.play(GrowFromCenter(b1), FadeIn(t1), FadeIn(notes[0]))
        self.play(GrowFromCenter(b2), FadeIn(t2), FadeIn(notes[1]))

        self.beat("4–5 min. Cada fila se vuelve un punto: ingreso a la derecha, deuda hacia "
                  "arriba. Forma y color codifican la etiqueta (círculo azul = pagó; cuadrado "
                  "naranja = no pagó). Pregunte: ¿ven ya un patrón? (Los que no pagan están "
                  "arriba: deuda alta.)")
        plane = ClientPlane()
        pts = plane.clients()
        self.play(FadeOut(VGroup(b1, b2, t1, t2, notes)), run_time=0.4)
        self.play(tbl.animate.scale(0.82).move_to([3.7, -0.3, 0]), FadeIn(plane), run_time=0.8)
        rows = tbl[1:-1]
        self.play(LaggedStart(*[TransformFromCopy(r, p) for r, p in zip(rows, pts)],
                              lag_ratio=0.12), run_time=2.4)
        legend = VGroup(legend_item(0, "pagó"), legend_item(1, "no pagó")
                        ).arrange(RIGHT, buff=0.4).next_to(plane.axes, UP, buff=0.12)
        self.play(FadeIn(legend), FadeOut(tbl))
        self.plane, self.pts, self.legend = plane, pts, legend

        self.beat("5–6 min. Llega un cliente nuevo (rombo amarillo): ingreso 18, deuda 35. "
                  "Aún no sabemos su y. Pida a la clase que vote. Lo que el modelo responda "
                  "es la PREDICCIÓN, ŷ («y gorro»): una conjetura que se compara con y cuando "
                  "la verdad llegue.")
        newc = plane.client(("?", 18, 35, None), unknown=True, r=0.15)
        side = VGroup(
            safe_text("Cliente nuevo", size=28, color=YELLOW, weight=BOLD),
            safe_text("ingreso 18 · deuda 35 %", size=24),
            safe_text("¿pagará?", size=30, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.3, 1.2, 0])
        self.play(GrowFromCenter(newc), FadeIn(side))
        self.play(Flash(newc[0], color=YELLOW, flash_radius=0.4))
        trio = VGroup(
            VGroup(safe_text("x", size=34, color=BLUE, weight=BOLD),
                   safe_text("lo que medimos", size=22, color=MUTED)),
            VGroup(safe_text("y", size=34, color=YELLOW, weight=BOLD),
                   safe_text("la verdad (llega después)", size=22, color=MUTED)),
            VGroup(safe_text("ŷ", size=34, color=PURPLE, weight=BOLD),
                   safe_text("lo que el modelo apuesta", size=22, color=MUTED)),
        )
        for g in trio:
            g.arrange(RIGHT, buff=0.3)
        trio.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.3, -1.3, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.1) for t in trio], lag_ratio=0.3))
        self.newc = VGroup(side, trio, newc)
        self.table_head = head

    def first_rule(self) -> None:
        plane, pts = self.plane, self.pts
        self.beat("6–7 min. El modelo más tonto posible: ignorar x y decir siempre «paga» "
                  "(la clase más frecuente). Acierta 6 de 10. Es la REFERENCIA: cualquier "
                  "modelo que no la supere no aprendió nada de los atributos.")
        self.play(FadeOut(self.newc), run_time=0.4)
        head = self.swap_heading(self.table_head, "¿Qué es clasificar?",
                                 "Primera regla: no mirar nada", BLUE)
        regions = plane.stump_regions(Y_RANGE[1] + 1)
        self.play(FadeIn(regions))
        rings = VGroup(*[Circle(radius=0.24, color=RED, stroke_width=4).move_to(p[0])
                         for p in pts if p.cls == 1])
        self.play(LaggedStart(*[Create(r) for r in rings], lag_ratio=0.15))
        base = VGroup(
            safe_text("«siempre paga»", size=28, color=BLUE, weight=BOLD),
            safe_text(f"aciertos: 6 / 10 = {baseline_acc(YTR):.2f}", size=26),
            safe_text("referencia (baseline)", size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.3, 1.0, 0])
        self.play(FadeIn(base))

        self.beat("7–9 min. Ahora sí miramos la deuda. Una línea horizontal sube: abajo "
                  "decimos «paga», arriba «no paga». Siga el contador de errores (círculos "
                  "rojos). Pida que griten «¡alto!» donde crean que hay menos errores. "
                  "El mínimo es en 44 %: un solo error, el cliente 6.")
        self.play(FadeOut(rings), FadeOut(regions), base.animate.set_opacity(0.4))
        head = self.swap_heading(head, "¿Qué es clasificar?", "Segunda regla: un corte", BLUE)
        sweep = stump_sweep()
        t = ValueTracker(sweep[0][0])
        line = always_redraw(lambda: plane.hline(t.get_value()))
        shade = always_redraw(lambda: plane.stump_regions(t.get_value()))
        self.add(shade, line)
        self.bring_to_front(pts)
        err_lab = safe_text("errores", size=24, color=MUTED).move_to([2.4, -0.9, 0])
        err_val = safe_text(str(sweep[0][1]), size=56, color=RED, weight=BOLD)
        err_val.next_to(err_lab, RIGHT, buff=0.35)
        t_lab = safe_text(f"corte: {sweep[0][0]:.1f} %", size=24, color=YELLOW)
        t_lab.next_to(err_lab, DOWN, aligned_edge=LEFT, buff=0.45)
        self.play(FadeIn(err_lab), FadeIn(err_val), FadeIn(t_lab))

        def mark(wrong):
            return VGroup(*[Circle(radius=0.24, color=RED, stroke_width=4).move_to(p[0])
                            for p in pts if p.row[0] in wrong])

        marks = mark(sweep[0][2])
        self.add(marks)
        order = sweep + [s for s in reversed(sweep) if s[0] >= STUMP_T][1:]
        for cut, n_err, wrong in order[1:]:
            new_marks = mark(wrong)
            new_val = safe_text(str(n_err), size=56, color=RED, weight=BOLD).move_to(err_val)
            new_t = safe_text(f"corte: {cut:.1f} %", size=24, color=YELLOW).move_to(t_lab,
                                                                                    aligned_edge=LEFT)
            self.play(t.animate.set_value(cut), Transform(marks, new_marks),
                      Transform(err_val, new_val), Transform(t_lab, new_t), run_time=0.55)
        line.clear_updaters()
        shade.clear_updaters()
        best = pill("deuda > 44 %  ⇒  no paga", YELLOW, size=24).move_to([3.3, -2.5, 0])
        self.play(FadeIn(best, shift=UP * 0.1), err_val.animate.set_color(GREEN))
        tag6 = safe_text("¿excepción o ruido?", size=20, color=RED).next_to(
            pts[5], RIGHT, buff=0.25)
        self.play(FadeIn(tag6))

        self.beat("9–10 min. Formalice lo que YA vieron. Un clasificador es una función f "
                  "que toma x y devuelve una clase. La recta amarilla es su FRONTERA DE "
                  "DECISIÓN: parte el plano en regiones. Lo aprendemos con ejemplos "
                  "etiquetados (entrenamiento). Pregunta puente: 9 de 10 aciertos… ¿basta "
                  "con contar aciertos?")
        self.play(FadeOut(VGroup(base, err_lab, err_val, t_lab, best, tag6, marks)))
        self.swap_heading(head, "¿Qué es clasificar?", "Formalizar lo que vimos", BLUE)
        f1 = tex(r"\hat{y} = f(\mathbf{x})", size=58, color=INK)
        f2 = tex(r"f:\ (\text{ingreso},\ \text{deuda}) \;\longmapsto\; "
                 r"\{\text{paga},\ \text{no paga}\}", size=32, max_width=6.6)
        f3 = tex(r"\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{10}", size=36, color=BLUE)
        lab3 = safe_text("conjunto de entrenamiento", size=20, color=MUTED)
        f4 = safe_text("frontera de decisión", size=26, color=YELLOW, weight=BOLD)
        lab4 = safe_text("donde f cambia de respuesta", size=20, color=MUTED)
        col = VGroup(f1, f2, VGroup(f3, lab3).arrange(DOWN, buff=0.1),
                     VGroup(f4, lab4).arrange(DOWN, buff=0.1)).arrange(DOWN, buff=0.42)
        col.move_to([3.3, -0.3, 0])
        for m in col:
            self.play(FadeIn(m, shift=UP * 0.1), run_time=0.55)
        self.play(Indicate(line, color=YELLOW, scale_factor=1.02))
        self.stump_line = line
