"""Acto 6 · Kernels: curvos por fuera, lineales por dentro (48–56 min).

Diez puntos en una recta con la clase +1 en el centro: ningún umbral separa.
Al elevarlos a (x, x²) una recta horizontal sí separa; de regreso, dos
umbrales. Un anillo en el plano se eleva a un paraboloide y un plano lo corta:
de regreso, un círculo. El truco del kernel K(x, z) = φ(x)ᵀφ(z) con números.
Un kernel RBF: campanas fijas que sólo se ESCALAN con α y se suman: el modelo
es lineal en α (y con pérdida cuadrática, α tiene fórmula cerrada).
"""
from __future__ import annotations

from math import cos, sin, sqrt

import numpy as np
from manim import *

from sections.data import (KQ_X, KQ_Z, LINE1D, LX, LY, PHI_CUT, PHI_ROOT, RBF_ALPHA, RBF_GAMMA,
                           RBF_LAMBDA, RING, RING_CUT, RING_R, kernel_quad, phi_quad, rbf_f)
from sections.plane import View3D
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PINK, PURPLE, RED, YELLOW,
                            card, marker, safe_text, tex)


class KernelSection:
    def kernels(self) -> None:
        self.beat("48–49 min. Diez puntos en una recta: los cuadrados naranjas en el centro, los "
                  "círculos azules a los lados. En una variable el «hiperplano» es un punto: un "
                  "umbral. Pregunte: ¿existe un umbral que separe? Deslícelo: lo mejor que logra "
                  "son 3 errores.")
        head = self.start("Kernels", "Cuando ninguna recta alcanza", PINK)
        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[0, 10, 2], x_length=6.3, y_length=4.4,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False}).move_to([-3.2, -0.45, 0])
        xl = VGroup(*[safe_text(f"{v}", size=15, color=MUTED).next_to(ax.c2p(v, 0), DOWN,
                                                                     buff=0.1)
                      for v in (-3, -2, -1, 0, 1, 2, 3)],
                    safe_text("x", size=20, color=MUTED).next_to(ax.c2p(3.5, 0), RIGHT, buff=0.1))
        pts = VGroup()
        for x, c in LINE1D:
            m = marker(c, r=0.13).move_to(ax.c2p(x, 0))
            m.x, m.c = x, c
            pts.add(m)
        self.play(Create(ax.x_axis), FadeIn(xl), LaggedStart(*[GrowFromCenter(p) for p in pts],
                                                            lag_ratio=0.06))
        t = ValueTracker(-2.5)

        def errs():
            v = t.get_value()
            e = sum((p.x > v) != (p.c > 0) for p in pts)
            return min(e, len(pts) - e)          # cualquiera de los dos sentidos

        thr = always_redraw(lambda: DashedLine(ax.c2p(t.get_value(), -0.6), ax.c2p(t.get_value(),
                                                                                   1.6),
                                               color=YELLOW, stroke_width=4, dash_length=0.08))
        cnt = always_redraw(lambda: safe_text(f"umbral: {errs()} errores", size=26, color=RED,
                                              weight=BOLD).move_to([3.6, 1.4, 0]))
        self.play(Create(thr), FadeIn(cnt))
        self.play(t.animate.set_value(-1.2), run_time=1.4)
        self.play(t.animate.set_value(1.45), run_time=1.6)
        for m in (thr, cnt):
            m.clear_updaters()
        best = safe_text("mínimo: 3 errores", size=22, color=MUTED).next_to(cnt, DOWN, buff=0.2)
        self.play(FadeIn(best))

        self.beat("49–51 min. Idea: agregar una coordenada nueva, x². Cada punto sube a la "
                  "parábola (x, x²). En este plano ampliado, una RECTA horizontal x² = 2 separa "
                  "perfectamente. Esa recta es lineal en las nuevas coordenadas. Pregunte: "
                  "¿cómo se ve esa recta cuando regresamos a la recta original?")
        self.play(FadeOut(thr), FadeOut(cnt), FadeOut(best))
        self.play(Create(ax.y_axis))
        yl = safe_text("x²", size=20, color=PINK).next_to(ax.c2p(-3.5, 10), UP, buff=0.08)
        par = ax.plot(lambda x: x * x, x_range=[-3.16, 3.16], color=PINK, stroke_width=2,
                      stroke_opacity=0.5)
        self.play(FadeIn(yl), Create(par))
        self.play(*[p.animate.move_to(ax.c2p(p.x, p.x ** 2)) for p in pts], run_time=1.8)
        cut = ax.plot(lambda x: PHI_CUT, x_range=[-3.5, 3.5], color=YELLOW, stroke_width=5)
        shade_up = Polygon(ax.c2p(-3.5, PHI_CUT), ax.c2p(3.5, PHI_CUT), ax.c2p(3.5, 10),
                           ax.c2p(-3.5, 10), stroke_width=0, fill_color=BLUE, fill_opacity=0.12)
        shade_dn = Polygon(ax.c2p(-3.5, 0), ax.c2p(3.5, 0), ax.c2p(3.5, PHI_CUT),
                           ax.c2p(-3.5, PHI_CUT), stroke_width=0, fill_color=ORANGE,
                           fill_opacity=0.14)
        self.play(FadeIn(shade_up), FadeIn(shade_dn), Create(cut))
        self.bring_to_front(pts)
        eq = VGroup(
            tex(r"\varphi(x) = (x,\ x^2)", size=36, color=PINK),
            tex(r"f(x) = w_1 x + w_2 x^2 + b", size=36, color=PURPLE),
            tex(r"\mathbf{w} = (0,\ -1),\ b = 2", size=30, color=MUTED),
            safe_text("una recta en el espacio (x, x²)", size=22, color=YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.6, 0.8, 0])
        self.play(LaggedStart(*[Write(e) for e in eq], lag_ratio=0.35), run_time=2.4)

        self.beat("51–52 min. De regreso: los puntos bajan, la recta corta a la parábola en "
                  "x = ±1.41 y esos dos cortes son la frontera en la recta original: un "
                  "intervalo. Visto en x, no es lineal; visto en (x, x²), sí. Los pesos entran "
                  "linealmente: TODO lo que sabemos de clasificadores lineales sigue valiendo.")
        drops = VGroup(*[DashedLine(ax.c2p(s * PHI_ROOT, PHI_CUT), ax.c2p(s * PHI_ROOT, 0),
                                    color=YELLOW, stroke_width=3, dash_length=0.08)
                         for s in (-1, 1)])
        self.play(Create(drops))
        self.play(*[p.animate.move_to(ax.c2p(p.x, 0)) for p in pts], run_time=1.5)
        band = Line(ax.c2p(-PHI_ROOT, 0), ax.c2p(PHI_ROOT, 0), color=ORANGE, stroke_width=12,
                    stroke_opacity=0.55)
        roots = VGroup(*[safe_text(f"{s * PHI_ROOT:+.2f}", size=18, color=YELLOW,
                                   weight=BOLD).next_to(ax.c2p(s * PHI_ROOT, 0), DOWN, buff=0.45)
                         for s in (-1, 1)])
        self.add(band)
        self.bring_to_front(pts)
        self.play(FadeIn(band), FadeIn(roots))
        rem = safe_text("curva en x, recta en φ(x)", size=24, color=PINK,
                        weight=BOLD).next_to(eq, DOWN, buff=0.45).align_to(eq, LEFT)
        self.play(FadeIn(rem))
        self.ring()
        self.trick()
        self.rbf()

    # ------------------------------------------------------------------
    def ring(self) -> None:
        self.beat("52–53 min. Lo mismo en dos variables: un anillo. Ninguna recta separa el "
                  "centro del borde. Agregamos una tercera coordenada, x₁² + x₂²: los puntos "
                  "suben a un paraboloide, los del centro quedan abajo. Un PLANO horizontal los "
                  "separa. Al girar se ve que el plano es plano.")
        head = self.start("Kernels", "Un anillo se vuelve separable al subir", PINK)
        view = View3D(origin=(-3.3, -1.2, 0), scale=0.85, az=-0.5, el=0.55)
        KZ = 0.6                       # compresión visual del eje x₃
        lift = ValueTracker(0.0)
        az = ValueTracker(-0.5)
        el = ValueTracker(1.25)        # empieza casi desde arriba: se ve el plano 2D
        cut_on = ValueTracker(0.0)

        def scene():
            a, e, s = az.get_value(), el.get_value(), lift.get_value()
            g = VGroup()
            # piso
            R = 2.3
            floor = VMobject(stroke_color=GRID, stroke_width=1.5).set_points_smoothly(
                [view(R * cos(t), R * sin(t), 0, a, e) for t in np.linspace(0, 2 * np.pi, 49)])
            g.add(floor)
            # paraboloide (alambres)
            if s > 0.02:
                for r in (0.7, 1.4, 2.0):
                    g.add(VMobject(stroke_color=PINK, stroke_width=1.4, stroke_opacity=0.5 * s)
                          .set_points_smoothly([view(r * cos(t), r * sin(t), KZ * s * r * r, a, e)
                                                for t in np.linspace(0, 2 * np.pi, 49)]))
                for t in np.linspace(0, np.pi, 6, endpoint=False):
                    g.add(VMobject(stroke_color=PINK, stroke_width=1.2, stroke_opacity=0.4 * s)
                          .set_points_smoothly([view(r * cos(t), r * sin(t), KZ * s * r * r, a, e)
                                                for r in np.linspace(-2.1, 2.1, 25)]))
            # plano de corte y su círculo
            c = cut_on.get_value()
            if c > 0.01:
                h = KZ * s * RING_CUT
                g.add(Polygon(*[view(R * cos(t), R * sin(t), h, a, e)
                                for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)],
                              stroke_color=YELLOW, stroke_width=2, stroke_opacity=c,
                              fill_color=YELLOW, fill_opacity=0.16 * c))
                g.add(VMobject(stroke_color=YELLOW, stroke_width=5, stroke_opacity=c)
                      .set_points_smoothly([view(RING_R * cos(t), RING_R * sin(t), h, a, e)
                                            for t in np.linspace(0, 2 * np.pi, 61)]))
            pts = sorted(RING, key=lambda q: view.depth(q[0], q[1], 0, a))
            for x, y, cl in pts:
                g.add(marker(cl, r=0.1).move_to(view(x, y, KZ * s * (x * x + y * y), a, e)))
            return g

        sc = always_redraw(scene)
        self.play(FadeIn(sc))
        txt = VGroup(tex(r"\varphi(\mathbf{x}) = (x_1,\ x_2,\ x_1^2 + x_2^2)", size=32,
                         color=PINK)).move_to([3.6, 1.6, 0])
        self.play(el.animate.set_value(0.5), run_time=1.6)
        self.play(lift.animate.set_value(1.0), Write(txt), run_time=2.2)

        self.beat("53–54 min. Aparece el plano x₃ = 1.8 y separa: lineal en φ. Gire la vista. "
                  "Después bajamos los puntos al piso y el plano deja su huella: un CÍRCULO de "
                  "radio 1.34. La frontera curva del espacio original es la sombra de un plano.")
        self.play(cut_on.animate.set_value(1.0), run_time=1.0)
        self.play(az.animate.set_value(0.6), run_time=2.6)
        lin = VGroup(tex(r"x_3 = 1.8", size=32, color=YELLOW),
                     safe_text("un plano en el espacio φ", size=22, color=YELLOW, weight=BOLD)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(txt, DOWN, buff=0.45)
        lin.align_to(txt, LEFT)
        self.play(FadeIn(lin))
        self.play(lift.animate.set_value(0.0), el.animate.set_value(1.2), run_time=2.4)
        sc.clear_updaters()
        back = VGroup(tex(rf"x_1^2 + x_2^2 = 1.8 \;\Rightarrow\; r = {RING_R:.2f}", size=30,
                          color=YELLOW),
                      safe_text("de regreso: un círculo", size=22, color=PINK, weight=BOLD)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(lin, DOWN, buff=0.45)
        back.align_to(txt, LEFT)
        self.play(FadeIn(back))

    # ------------------------------------------------------------------
    def trick(self) -> None:
        self.beat("54–55 min. El truco del kernel. Muchos algoritmos (SVM, logística con "
                  "kernel) sólo usan productos punto entre puntos. Si φ tiene muchísimas "
                  "coordenadas, calcularlas es caro; pero a veces φ(x)ᵀφ(z) tiene un atajo. "
                  "Hagan la cuenta a mano: 9 + 12 + 4 = 25 = (1·3 + 2·1)².")
        head = self.start("El truco del kernel", "Productos punto sin construir φ", PINK)
        x, z = KQ_X, KQ_Z
        px, pz = phi_quad(x), phi_quad(z)
        left = VGroup(
            tex(rf"\mathbf{{x}} = ({x[0]:g},\ {x[1]:g}),\quad \mathbf{{z}} = ({z[0]:g},\ "
                rf"{z[1]:g})", size=32),
            tex(r"\varphi(\mathbf{x}) = (x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)", size=32, color=PINK),
            tex(rf"\varphi(\mathbf{{x}}) = ({px[0]:g},\ {px[1]:.2f},\ {px[2]:g})", size=30,
                color=PINK),
            tex(rf"\varphi(\mathbf{{z}}) = ({pz[0]:g},\ {pz[1]:.2f},\ {pz[2]:g})", size=30,
                color=PINK),
            tex(rf"\varphi(\mathbf{{x}})^\top\varphi(\mathbf{{z}}) = 9 + 12 + 4 = "
                rf"{px @ pz:.0f}", size=32),
            tex(rf"(\mathbf{{x}}^\top\mathbf{{z}})^2 = (3 + 2)^2 = {kernel_quad(x, z):.0f}",
                size=32, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26).move_to([-3.2, -0.45, 0])
        self.play(LaggedStart(*[Write(r) for r in left], lag_ratio=0.35), run_time=4.0)
        right = VGroup(
            tex(r"K(\mathbf{x}, \mathbf{z}) = \varphi(\mathbf{x})^\top\varphi(\mathbf{z})",
                size=34, color=PINK),
            tex(r"f(\mathbf{x}) = \sum_{i=1}^{n} \alpha_i\, K(\mathbf{x}_i, \mathbf{x}) + b",
                size=34, color=PURPLE),
            safe_text("las αᵢ entran linealmente", size=24, color=YELLOW, weight=BOLD),
            safe_text("K(xᵢ, x): «parecido» con el punto i", size=22, color=MUTED),
            safe_text("RBF: φ con infinitas coordenadas", size=20, color=MUTED, max_width=5.4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, -0.3, 0])
        frame = card(6.4, 4.4, PINK).move_to(right)
        self.play(FadeIn(frame), LaggedStart(*[Write(r) for r in right], lag_ratio=0.3),
                  run_time=3.0)

    # ------------------------------------------------------------------
    def rbf(self) -> None:
        self.beat("55–56 min. El kernel RBF en la recta de antes. Cada punto de entrenamiento "
                  "pone una campana exp(−γ(x − xᵢ)²): son características FIJAS. Lo único que "
                  "aprende el modelo es cuánto escalar cada campana (αᵢ, que puede ser "
                  "negativa). Las campanas se estiran, nunca se doblan. La suma es f(x); su "
                  "signo, la clase. Pregunte: ¿es esto lineal? (En α, sí.)")
        head = self.start("Kernel RBF", "Campanas fijas, pesos lineales", PINK)
        ax = Axes(x_range=[-3.6, 3.6, 1], y_range=[-1.8, 1.8, 1], x_length=7.2, y_length=4.2,
                  tips=False, axis_config={"color": GRID, "stroke_width": 2,
                                           "include_ticks": False}).move_to([-2.7, -0.5, 0])
        pts = VGroup(*[marker(c, r=0.12).move_to(ax.c2p(x, 0)) for x, c in LINE1D])
        self.play(Create(ax), FadeIn(pts))
        amp = ValueTracker(0.0)      # 0: campanas de altura 1; 1: escaladas por α

        def bumps():
            g = VGroup()
            s = amp.get_value()
            for xi, ai, yi in zip(LX, RBF_ALPHA, LY):
                h = (1 - s) * 1.0 + s * ai
                col = ORANGE if yi > 0 else BLUE
                g.add(ax.plot(lambda x, xi=xi, h=h: h * np.exp(-RBF_GAMMA * (x - xi) ** 2),
                              x_range=[-3.6, 3.6], color=col, stroke_width=2.2,
                              stroke_opacity=0.8))
            return g

        bm = always_redraw(bumps)
        self.play(FadeIn(bm))
        k_eq = tex(r"K(x_i, x) = e^{-\gamma (x - x_i)^2}", size=32, color=PINK).move_to(
            [3.9, 1.7, 0])
        self.play(Write(k_eq))
        self.play(amp.animate.set_value(1.0), run_time=2.2)
        bm.clear_updaters()
        al = tex(r"\boldsymbol\alpha = (K + \lambda I)^{-1}\mathbf{y}", size=32,
                 color=YELLOW).next_to(k_eq, DOWN, buff=0.35)
        al2 = safe_text("pérdida cuadrática → fórmula cerrada", size=20, color=MUTED).next_to(
            al, DOWN, buff=0.15)
        self.play(Write(al), FadeIn(al2))

        self.beat("Las campanas escaladas se suman: f(x) (curva gruesa). Donde f > 0 se predice "
                  "naranja: exactamente el intervalo central. Con pérdida cuadrática, α se "
                  "obtiene con UNA fórmula, (K + λI)⁻¹y, igual que MCO: porque el problema es "
                  "lineal en α. Con pérdida logística (kernel logístico) habría que iterar, "
                  "igual que antes.")
        f = ax.plot(lambda x: float(rbf_f(x)[0]), x_range=[-3.6, 3.6], color=PURPLE,
                    stroke_width=6)
        self.play(bm.animate.set_stroke(opacity=0.3), Create(f), run_time=1.8)
        # región f > 0
        xs = np.linspace(-3.6, 3.6, 721)
        fv = rbf_f(xs)
        segs = VGroup()
        i = 0
        while i < len(xs):
            if fv[i] > 0:
                j = i
                while j + 1 < len(xs) and fv[j + 1] > 0:
                    j += 1
                segs.add(Line(ax.c2p(xs[i], 0), ax.c2p(xs[j], 0), color=ORANGE, stroke_width=12,
                              stroke_opacity=0.55))
                i = j + 1
            else:
                i += 1
        self.add(segs)
        self.bring_to_front(pts)
        self.play(FadeIn(segs))
        sum_eq = VGroup(
            tex(r"f(x) = \sum_i \alpha_i K(x_i, x)", size=32, color=PURPLE),
            tex(r"\hat y = \operatorname{signo} f(x)", size=32, color=YELLOW),
            safe_text("lineal en α · hiperplano en φ", size=22, color=PINK, weight=BOLD,
                      max_width=4.8),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(al2, DOWN, buff=0.4)
        sum_eq.align_to(al, LEFT)
        self.play(LaggedStart(*[FadeIn(s) for s in sum_eq], lag_ratio=0.3))
