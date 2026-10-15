"""Datos y métricas de reglas de asociación usados por la presentación.

Todos los números que aparecen en pantalla se derivan de aquí, de modo que la
animación y el guion docente no dependen de valores copiados a mano.
``python -m sections.data`` imprime un resumen para verificar el guion.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

# ── 1. Los cinco tickets del libro (capítulos 5.0 y 5.1) ────────────────
ITEMS = ["leche", "pan", "mantequilla", "queso"]
TICKETS = [
    {"leche", "pan", "mantequilla"},
    {"leche", "pan"},
    {"leche", "mantequilla"},
    {"pan", "mantequilla"},
    {"leche", "pan", "queso"},
]
MIN_COUNT = 2          # soporte mínimo 0.40
MIN_CONF = 0.60


def count(itemset: set[str]) -> int:
    return sum(1 for t in TICKETS if itemset <= t)


SINGLE_COUNTS = {i: count({i}) for i in ITEMS}                 # 4, 4, 3, 1
FREQ_ITEMS = [i for i in ITEMS if SINGLE_COUNTS[i] >= MIN_COUNT]
PAIR_COUNTS = {pair: count(set(pair)) for pair in combinations(FREQ_ITEMS, 2)}
FREQ_PAIRS = [p for p, c in PAIR_COUNTS.items() if c >= MIN_COUNT]
TRIPLE_COUNT = count(set(FREQ_ITEMS))                          # 1 → no frecuente


def toy_rules() -> list[dict]:
    """Las seis reglas de un ítem a un ítem generadas por los pares frecuentes."""
    n = len(TICKETS)
    rules = []
    for a, b in FREQ_PAIRS:
        for x, y in ((a, b), (b, a)):
            nab, na, nb = PAIR_COUNTS[(a, b)], SINGLE_COUNTS[x], SINGLE_COUNTS[y]
            conf = nab / na
            rules.append(dict(a=x, b=y, nab=nab, na=na, nb=nb, conf=conf,
                              lift=conf / (nb / n), ok=conf >= MIN_CONF - 1e-9))
    return rules


TOY_RULES = toy_rules()


# ── 2. La tienda de la esquina: 100 tickets ─────────────────────────────
@dataclass(frozen=True)
class Rule:
    a: str
    b: str
    n: int
    na: int
    nb: int
    nab: int

    @property
    def pa(self) -> float:
        return self.na / self.n

    @property
    def pb(self) -> float:
        return self.nb / self.n

    @property
    def pab(self) -> float:
        return self.nab / self.n

    @property
    def conf(self) -> float:            # altura de B dentro de la columna A
        return self.nab / self.na

    @property
    def h_not_a(self) -> float:         # altura de B fuera de la columna A
        return (self.nb - self.nab) / (self.n - self.na)

    @property
    def lift(self) -> float:
        return self.pab / (self.pa * self.pb)

    @property
    def lev(self) -> float:
        return self.pab - self.pa * self.pb

    @property
    def conv(self) -> float:
        return float("inf") if self.conf == 1 else (1 - self.pb) / (1 - self.conf)

    @property
    def zhang(self) -> float:
        return (self.conf - self.h_not_a) / max(self.conf, self.h_not_a)

    @property
    def lift_max(self) -> float:
        return 1 / max(self.pa, self.pb)

    @property
    def reverse(self) -> "Rule":
        return Rule(self.b, self.a, self.n, self.nb, self.na, self.nab)


STORE_N = 100
HOOK = Rule("pan dulce", "leche", STORE_N, 30, 60, 18)         # lift = 1
CAFE = Rule("café", "pan dulce", STORE_N, 40, 30, 20)          # lift = 1.67
VINO = Rule("vino", "queso", STORE_N, 5, 8, 4)                 # lift = 10
TE = Rule("té", "café", STORE_N, 20, 40, 2)                    # lift = 0.25
STORE_RULES = [CAFE, HOOK, VINO, TE]

# Reto en parejas
PRACTICE = Rule("papas", "refresco", 200, 50, 80, 30)


def fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace("-0.00", "0.00")


if __name__ == "__main__":
    print("Soportes:", SINGLE_COUNTS, "| pares:", PAIR_COUNTS, "| trío:", TRIPLE_COUNT)
    for r in TOY_RULES:
        print(f"  {r['a']:>11} ⇒ {r['b']:<11} {r['nab']}/{r['na']} conf={r['conf']:.3f} "
              f"lift={r['lift']:.4f} {'✓' if r['ok'] else '·'}")
    for r in STORE_RULES + [CAFE.reverse, PRACTICE, PRACTICE.reverse]:
        print(f"{r.a:>9} ⇒ {r.b:<9} supp={r.pab:.3f} conf={r.conf:.3f} h¬A={r.h_not_a:.3f} "
              f"lift={r.lift:.3f} lev={r.lev:+.3f} ({r.lev * r.n:+.1f} tickets) "
              f"conv={r.conv:.3f} zhang={r.zhang:+.3f} liftmax={r.lift_max:.3f}")
