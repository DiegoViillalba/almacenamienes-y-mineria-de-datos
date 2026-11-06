"""Acto 7 · Comparar, ejercicio y cierre (55–60 min).

Bagging frente a boosting en una tabla; los hiperparámetros de XGBoost como
perillas de sesgo y varianza; un ejercicio a mano (una ronda de gradient
boosting y su versión XGBoost) con solución, y el cierre con ticket de salida.
"""
from __future__ import annotations

from manim import *

from sections.data import ETA, EX_X, EX_Y, LAMBDA, exercise
from sections.theme import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, PANEL, PINK, PURPLE, RED,
                            TEAL, YELLOW, card, pill, safe_text, table, tex)


class CloseSection:
    def close(self) -> None:
        self.compare()
        self.knobs()
        self.exercise()
        self.closing()

    # ------------------------------------------------------------------
    def compare(self) -> None:
        self.beat("55–56 min. Las dos familias lado a lado. Regla práctica: un bosque aleatorio "
                  "con valores por defecto es una referencia excelente y difícil de batir con "
                  "poco esfuerzo; el boosting bien ajustado (XGBoost, LightGBM) suele ganar por un "
                  "margen pequeño cuando hay tiempo para validar sus hiperparámetros.")
        self.start("Bagging frente a boosting", "Dos maneras de combinar árboles", YELLOW)
        rows = [["", "bagging / bosques", "boosting / XGBoost"],
                ["entrenamiento", "en paralelo", "en secuencia"],
                ["modelo base", "árbol profundo (inestable)", "árbol poco profundo (débil)"],
                ["reduce", "varianza", "sesgo (y algo de varianza)"],
                ["combina", "promedio o voto", "suma ponderada"],
                ["más modelos", "no sobreajusta", "sí: rondas por validación"],
                ["etiquetas ruidosas", "robusto", "sensible (AdaBoost)"]]
        tb = table(rows, col_widths=[2.8, 4.2, 4.6], size=22, row_h=0.58,
                   colors=[[MUTED, GREEN, BLUE]] * 6).move_to([0, -0.3, 0])
        self.play(FadeIn(tb[0]), FadeIn(tb[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in tb[1:-1]], lag_ratio=0.15))

    # ------------------------------------------------------------------
    def knobs(self) -> None:
        self.beat("56–57 min. Los hiperparámetros de XGBoost son perillas de complejidad. Bajar "
                  "η exige más rondas pero generaliza mejor; el número de rondas se elige con "
                  "early stopping. Árboles poco profundos (3–8). λ y γ regularizan cada árbol; "
                  "subsample y colsample añaden el azar de los bosques.")
        self.start("Hiperparámetros de XGBoost", "Perillas de complejidad", PINK)
        rows = [["parámetro", "qué controla", "valor de partida"],
                ["n_estimators + early stopping", "número de rondas", "hasta que la validación pare"],
                ["learning_rate (η)", "tamaño de cada corrección", "0.05 – 0.3"],
                ["max_depth", "interacciones por árbol", "3 – 8"],
                ["reg_lambda (λ)", "encoge los pesos de hoja", "1"],
                ["gamma (γ)", "ganancia mínima para cortar", "0 – 5"],
                ["subsample · colsample_bytree", "azar en filas y columnas", "0.7 – 1.0"]]
        tb = table(rows, col_widths=[4.3, 4.0, 3.6], size=21, row_h=0.56,
                   colors=[[PINK, INK, MUTED]] * 6).move_to([0, -0.25, 0])
        self.play(FadeIn(tb[0]), FadeIn(tb[-1]))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in tb[1:-1]], lag_ratio=0.15))

    # ------------------------------------------------------------------
    def exercise(self) -> None:
        self.beat("57–59 min. Ejercicio individual (2 min). Cuatro puntos: x = 1, 2, 3, 4 y "
                  "y = 1, 3, 6, 8. (1) F₀ y los residuos. (2) El mejor tocón para los residuos y "
                  "F₁ con η = 0.5. (3) Los pesos de hoja de XGBoost con λ = 1 y la ganancia del "
                  "corte. Error típico: olvidar el signo de g = F − y.")
        self.start("Tu turno", "Una ronda de boosting a mano", YELLOW)
        data = table([["x"] + [f"{v:g}" for v in EX_X], ["y"] + [f"{v:g}" for v in EX_Y]],
                     col_widths=[0.8] + [0.9] * 4, size=26, row_h=0.6,
                     colors=[[MUTED] + [BLUE] * 4]).move_to([-3.6, 1.2, 0])
        tasks = VGroup(
            safe_text("1. F₀ (la mejor constante) y los residuos", size=23),
            safe_text("2. Mejor tocón para los residuos y F₁ con η = 0.5", size=23),
            safe_text("3. XGBoost (λ = 1): g = F₀ − y, pesos de hoja", size=23),
            safe_text("    −G/(H + λ) y ganancia del corte", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0, -0.85, 0])
        timer = pill("2 minutos", YELLOW, size=24).move_to([3.6, 1.2, 0])
        self.play(FadeIn(data))
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tasks], lag_ratio=0.2))
        self.play(FadeIn(timer))

        self.beat("59 min. Solución. F₀ = 4.5; residuos (−3.5, −1.5, 1.5, 3.5); el tocón corta en "
                  "2.5 con −2.5 | 2.5; F₁ = (3.25, 3.25, 5.75, 5.75) y la SSE baja de 29 a 10.25. "
                  "XGBoost: G_L = 5, G_R = −5, H = 2 y 2: pesos −1.67 y 1.67 (más chicos que "
                  "±2.5 por λ); ganancia ½(25/3 + 25/3 − 0) = 8.33.")
        e = exercise()
        self.play(FadeOut(tasks), FadeOut(timer))
        sol = VGroup(
            tex(rf"F_0 = {e['F0']:g}, \quad r = ({', '.join(f'{v:g}' for v in e['r'])})", size=30),
            tex(rf"h_1:\; x \le {e['t']:g} \Rightarrow {e['a']:g} \,|\, {e['b']:g}, \quad "
                rf"F_1 = ({', '.join(f'{v:g}' for v in e['F1'])})", size=30, color=TEAL),
            tex(rf"\text{{SSE}}: {e['sse0']:g} \rightarrow {e['sse1']:g}", size=30, color=RED),
            tex(rf"G_L = {e['GL']:g},\; G_R = {e['GR']:g} \;\Rightarrow\; w = "
                rf"{e['wl']:.2f},\; {e['wr']:.2f}", size=30, color=PINK),
            tex(rf"\text{{Ganancia}} = \tfrac12\big(\tfrac{{25}}{{3}} + \tfrac{{25}}{{3}} - 0\big)"
                rf" = {e['gain']:.2f}", size=30, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0.6, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in sol], lag_ratio=0.35))

    # ------------------------------------------------------------------
    def closing(self) -> None:
        self.beat("60 min. Cierre con cuatro ideas y el ticket de salida.")
        self.start("Cierre", "Lo que nos llevamos", YELLOW)
        ideas = VGroup(*[VGroup(safe_text(n, size=30, color=c, weight=BOLD),
                                safe_text(t, size=23)).arrange(RIGHT, buff=0.3)
                         for n, t, c in [
                             ("promediar", "baja la varianza si los modelos son diversos", PURPLE),
                             ("bosques", "bootstrap + variables al azar → menos correlación",
                              GREEN),
                             ("boosting", "cada modelo corrige los residuos (−gradiente)", BLUE),
                             ("XGBoost", "Taylor de 2º orden: w = −G/(H+λ), poda con γ", PINK),
                         ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to([0, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in ideas], lag_ratio=0.25))
        ticket = VGroup(
            safe_text("Ticket de salida", size=24, color=YELLOW, weight=BOLD),
            safe_text("¿Por qué agregar árboles a un bosque no sobreajusta y agregar rondas de "
                      "boosting sí puede?", size=21, max_width=11.5),
            safe_text("¿Qué hiperparámetro de XGBoost usarías para frenarlo?", size=21,
                      max_width=11.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = card(ticket.width + 0.6, ticket.height + 0.4, YELLOW)
        tg = VGroup(box, ticket.move_to(box)).move_to([0, -2.1, 0])
        self.play(FadeIn(tg, shift=UP * 0.1))
