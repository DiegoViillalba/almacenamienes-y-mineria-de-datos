"""Datos, FP-tree y minado usados por la presentación.

Todos los números que aparecen en pantalla se derivan de aquí, de modo que la
animación y el guion docente no dependen de valores copiados a mano.
``python -m sections.data`` imprime un resumen para verificar el guion.

Convenciones (las mismas del capítulo 5.3 y de su laboratorio):
- orden global por frecuencia descendente, desempate alfabético;
- los árboles condicionales conservan el orden global;
- un nodo se identifica por su camino desde la raíz (tupla de ítems).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

Path = tuple[str, ...]

# ── 1. Ejemplo base del libro (capítulo 5.3) ────────────────────────────
TICKETS_RAW = [                     # tal como los lista el libro (sin orden útil)
    ["Pan", "Leche"],
    ["Pan", "Leche", "Pañales", "Cerveza"],
    ["Leche", "Pañales", "Cerveza"],
    ["Pan", "Leche", "Pañales", "Cerveza"],
    ["Pan", "Leche", "Pañales"],
]
TICKETS = [set(t) for t in TICKETS_RAW]
MIN_SUP = 3


def item_counts(tickets: list[set[str]]) -> Counter:
    return Counter(i for t in tickets for i in t)


def global_order(tickets: list[set[str]], min_sup: int, *, reverse: bool = False) -> list[str]:
    """Ítems frecuentes de más a menos frecuente (``reverse``: de menos a más)."""
    c = item_counts(tickets)
    freq = [i for i in c if c[i] >= min_sup]
    sign = 1 if reverse else -1
    return sorted(freq, key=lambda i: (sign * c[i], i))


def ordered(tickets: list[set[str]], order: list[str]) -> list[list[str]]:
    rank = {i: r for r, i in enumerate(order)}
    return [sorted((i for i in t if i in rank), key=rank.get) for t in tickets]


def build_tree(transactions: list[list[str]], weights: list[int] | None = None):
    """Inserta transacciones ordenadas. Devuelve conteos por nodo y la traza.

    ``counts`` es un dict ordenado por creación: camino → contador.
    ``trace[k]`` = (caminos compartidos, caminos creados) al insertar la k-ésima.
    """
    weights = weights or [1] * len(transactions)
    counts: dict[Path, int] = {}
    trace = []
    for t, w in zip(transactions, weights):
        shared, created = [], []
        for d in range(1, len(t) + 1):
            p = tuple(t[:d])
            (shared if p in counts else created).append(p)
            counts[p] = counts.get(p, 0) + w
        trace.append((shared, created))
    return counts, trace


def node_links(counts: dict[Path, int], item: str) -> list[Path]:
    """Nodos del ítem en orden de creación (los «enlaces de nodo»)."""
    return [p for p in counts if p[-1] == item]


def conditional_base(counts: dict[Path, int], item: str) -> list[tuple[Path, int]]:
    """Caminos prefijo del ítem, cada uno con el contador del nodo del ítem."""
    return [(p[:-1], counts[p]) for p in node_links(counts, item) if len(p) > 1]


def base_counts(base: list[tuple[Path, int]]) -> Counter:
    c = Counter()
    for path, w in base:
        for i in path:
            c[i] += w
    return c


def mine(transactions, weights, order, min_sup, suffix=(), out=None, calls=None):
    """FP-Growth recursivo. ``out``: lista de (patrón, conteo) en orden de hallazgo.

    ``calls`` registra (sufijo, base condicional filtrada) de cada llamada.
    """
    out = [] if out is None else out
    counts, _ = build_tree(transactions, weights)
    support = Counter()
    for p, c in counts.items():
        support[p[-1]] += c
    for item in reversed(order):
        if support[item] < min_sup:
            continue
        pattern = (item,) + tuple(suffix)
        out.append((pattern, support[item]))
        base = conditional_base(counts, item)
        bc = base_counts(base)
        keep = [i for i in order if bc[i] >= min_sup]
        sub = [([i for i in path if i in keep], w) for path, w in base]
        sub = [(p, w) for p, w in sub if p]
        if calls is not None:
            calls.append((pattern, sub))
        if sub:
            mine([p for p, _ in sub], [w for _, w in sub], keep, min_sup, pattern, out, calls)
    return out


def all_frequent(tickets, min_sup):
    order = global_order(tickets, min_sup)
    tx = ordered(tickets, order)
    return mine(tx, None, order, min_sup)


def brute_force(tickets, min_sup):
    items = sorted(item_counts(tickets))
    res = {}
    for k in range(1, len(items) + 1):
        for combo in combinations(items, k):
            c = sum(1 for t in tickets if set(combo) <= t)
            if c >= min_sup:
                res[frozenset(combo)] = c
    return res


def apriori_trace(tickets, min_sup):
    """Por nivel: (candidatos contados, podados sin contar, frecuentes)."""
    items = sorted(item_counts(tickets))
    levels = []
    cands = [frozenset([i]) for i in items]
    prev: set[frozenset] = set()
    k = 1
    while cands:
        if k > 1:
            joined = {a | b for a in prev for b in prev if len(a | b) == k}
            pruned = [c for c in joined
                      if any(frozenset(s) not in prev for s in combinations(c, k - 1))]
            cands = [c for c in joined if c not in pruned]
        else:
            pruned = []
        if not cands:
            levels.append((cands, pruned, []))
            break
        freq = [c for c in cands if sum(1 for t in tickets if c <= t) >= min_sup]
        levels.append((cands, pruned, freq))
        prev = set(freq)
        k += 1
        cands = cands if freq else []
    return levels


COUNTS = item_counts(TICKETS)                       # Leche 5, Pan 4, Pañales 4, Cerveza 3
ORDER = global_order(TICKETS, MIN_SUP)              # Leche ≻ Pan ≻ Pañales ≻ Cerveza
ORDERED = ordered(TICKETS, ORDER)
TREE, TRACE = build_tree(ORDERED)                   # 6 nodos
APPEARANCES = sum(len(t) for t in ORDERED)          # 16

REV_ORDER = global_order(TICKETS, MIN_SUP, reverse=True)
REV_TREE, _ = build_tree(ordered(TICKETS, REV_ORDER))   # 10 nodos

FREQUENT = all_frequent(TICKETS, MIN_SUP)           # 11 itemsets
APRIORI = apriori_trace(TICKETS, MIN_SUP)

# Subproblema guiado: Cerveza, y Pañales como turno del grupo.
BASE_CERVEZA = conditional_base(TREE, "Cerveza")
BASE_PANALES = conditional_base(TREE, "Pañales")
BASE_PAN = conditional_base(TREE, "Pan")

# Recursión con soporte mínimo 2 en la base de Cerveza (árbol que se ramifica).
REC_MIN = 2
REC_TREE, _ = build_tree([p for p, _ in BASE_CERVEZA], [w for _, w in BASE_CERVEZA])
REC_PATTERNS = mine([p for p, _ in BASE_CERVEZA], [w for _, w in BASE_CERVEZA],
                    ORDER, REC_MIN, ("Cerveza",))

# Analogía inicial: palabras con prefijo común.
WORDS = ["sol", "sola", "solar", "sopa"]
WORD_TREE, _ = build_tree([list(w) for w in WORDS])
WORD_LETTERS = sum(len(w) for w in WORDS)          # 16

# ── 2. Reto en parejas: datos nuevos ────────────────────────────────────
PRACTICE_RAW = [
    ["galletas", "café", "chocolate"],
    ["azúcar", "café", "galletas"],
    ["chocolate", "azúcar", "café"],
    ["galletas", "chocolate"],
    ["café", "azúcar", "chocolate", "galletas"],
    ["té", "galletas", "café"],
]
PRACTICE = [set(t) for t in PRACTICE_RAW]
P_MIN = 3
P_COUNTS = item_counts(PRACTICE)
P_ORDER = global_order(PRACTICE, P_MIN)             # café ≻ galletas ≻ chocolate ≻ azúcar
P_ORDERED = ordered(PRACTICE, P_ORDER)
P_TREE, _ = build_tree(P_ORDERED)                   # 9 nodos
P_APPEARANCES = sum(len(t) for t in P_ORDERED)      # 17
P_FREQUENT = all_frequent(PRACTICE, P_MIN)
P_BASE_AZUCAR = conditional_base(P_TREE, "azúcar")
P_BASE_CHOC = conditional_base(P_TREE, "chocolate")

# ── 3. A escala: las 50 canastas del capítulo de Apriori (min. 5 canastas) ──
CANASTAS = [
    ["whole milk", "yogurt", "bread"], ["whole milk", "butter", "eggs"], ["whole milk", "yogurt", "vegetables"],
    ["bread", "butter", "eggs"], ["whole milk", "soda", "chips"], ["yogurt", "fruit", "bread"],
    ["whole milk", "coffee", "bread"], ["eggs", "butter", "cheese"], ["whole milk", "vegetables", "fruit"],
    ["soda", "chips", "candy"], ["whole milk", "bread", "eggs"], ["yogurt", "fruit", "granola"],
    ["whole milk", "coffee", "sugar"], ["bread", "eggs", "cheese"], ["whole milk", "yogurt", "coffee"],
    ["butter", "eggs", "bread"], ["chips", "soda", "beer"], ["whole milk", "vegetables", "butter"],
    ["yogurt", "granola", "fruit"], ["whole milk", "bread", "sugar"], ["eggs", "cheese", "vegetables"],
    ["whole milk", "soda", "bread"], ["coffee", "sugar", "bread"], ["yogurt", "whole milk", "fruit"],
    ["butter", "whole milk", "eggs"], ["bread", "chips", "soda"], ["whole milk", "granola", "yogurt"],
    ["vegetables", "fruit", "bread"], ["whole milk", "butter", "cheese"], ["eggs", "bread", "coffee"],
    ["whole milk", "yogurt", "butter"], ["chips", "candy", "soda"], ["whole milk", "coffee", "eggs"],
    ["bread", "vegetables", "fruit"], ["yogurt", "cheese", "eggs"], ["whole milk", "bread", "coffee"],
    ["soda", "beer", "chips"], ["whole milk", "vegetables", "eggs"], ["yogurt", "granola", "whole milk"],
    ["bread", "butter", "sugar"], ["whole milk", "fruit", "cheese"], ["eggs", "vegetables", "butter"],
    ["whole milk", "chips", "soda"], ["coffee", "bread", "butter"], ["yogurt", "whole milk", "bread"],
    ["eggs", "cheese", "bread"], ["whole milk", "sugar", "butter"], ["vegetables", "fruit", "yogurt"],
    ["whole milk", "bread", "butter"], ["granola", "yogurt", "fruit"],
]
G_MIN = 5


def growth_curve(tickets: list[set[str]], min_sup: int):
    order = global_order(tickets, min_sup)
    tx = ordered(tickets, order)
    app, nodes = [], []
    for k in range(1, len(tx) + 1):
        counts, _ = build_tree(tx[:k])
        app.append(sum(len(t) for t in tx[:k]))
        nodes.append(len(counts))
    return app, nodes


G_TICKETS = [set(t) for t in CANASTAS]
G_APP, G_NODES = growth_curve(G_TICKETS, G_MIN)
G_FREQUENT = all_frequent(G_TICKETS, G_MIN)
G_APRIORI = apriori_trace(G_TICKETS, G_MIN)

# Datos densos (mismo tamaño): cada canasta es un prefijo de una lista fija.
DENSE_ITEMS = ["whole milk", "bread", "eggs", "butter", "yogurt"]
DENSE = [set(DENSE_ITEMS[: 1 + (k * 7) % 5]) for k in range(50)]
D_APP, D_NODES = growth_curve(DENSE, G_MIN)


def fmt_set(s) -> str:
    return "{" + ", ".join(sorted(s, key=lambda i: ORDER.index(i) if i in ORDER else 0)) + "}"


if __name__ == "__main__":
    print("Conteos:", dict(COUNTS), "| orden:", " ≻ ".join(ORDER))
    for t, o, (sh, cr) in zip(TICKETS, ORDERED, TRACE):
        print(f"  {o}  comparte {len(sh)}, crea {len(cr)}")
    print("Árbol:", TREE, f"→ {len(TREE)} nodos / {APPEARANCES} apariciones")
    print("Orden inverso:", " ≻ ".join(REV_ORDER), f"→ {len(REV_TREE)} nodos", REV_TREE)
    print("Base Cerveza:", BASE_CERVEZA, dict(base_counts(BASE_CERVEZA)))
    print("Base Pañales:", BASE_PANALES, dict(base_counts(BASE_PANALES)))
    print("Base Pan:", BASE_PAN)
    print(f"Frecuentes ({len(FREQUENT)}):", FREQUENT)
    bf = brute_force(TICKETS, MIN_SUP)
    assert {frozenset(p): c for p, c in FREQUENT} == bf, "FP-Growth ≠ fuerza bruta"
    print("Apriori:")
    for k, (c, pr, f) in enumerate(APRIORI, 1):
        print(f"  nivel {k}: contados {len(c)}, podados {len(pr)}, frecuentes {len(f)}",
              [fmt_set(x) for x in pr])
    print("Recursión min 2: árbol", REC_TREE, "→", REC_PATTERNS)
    print("Palabras:", WORD_TREE, f"{len(WORD_TREE)} nodos / {WORD_LETTERS} letras")
    print("Reto: conteos", dict(P_COUNTS), "orden", P_ORDER)
    print("  árbol", P_TREE, f"{len(P_TREE)} nodos / {P_APPEARANCES}")
    print("  base azúcar", P_BASE_AZUCAR, dict(base_counts(P_BASE_AZUCAR)))
    print("  base chocolate", P_BASE_CHOC, dict(base_counts(P_BASE_CHOC)))
    print("  frecuentes", P_FREQUENT)
    assert {frozenset(p): c for p, c in P_FREQUENT} == brute_force(PRACTICE, P_MIN)
    print(f"50 canastas: {G_APP[-1]} apariciones → {G_NODES[-1]} nodos; "
          f"{len(G_FREQUENT)} itemsets; Apriori contados =",
          sum(len(c) for c, _, _ in G_APRIORI), "en", sum(1 for c, _, _ in G_APRIORI if c), "lecturas")
    assert {frozenset(p): c for p, c in G_FREQUENT} == brute_force(G_TICKETS, G_MIN)
    print(f"Densas: {D_APP[-1]} apariciones → {D_NODES[-1]} nodos")
