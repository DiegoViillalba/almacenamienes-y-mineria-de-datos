"""Datos y cálculos de la clase. Todo número que aparece en pantalla sale de aquí.

Ejemplos (todos del libro):

- los diez clientes de ``6-0-introduccion-clasificacion.qmd`` (ingreso, deuda,
  ¿pagó?), que ya conocen de la clase anterior;
- los quince días de béisbol de ``6-1-clasificacion_arboles.qmd`` (ID3, C4.5);
- los catorce compradores de laptop de ``6-1-clasificacion_cart.qmd`` (Gini);
- los 60 puntos del laboratorio de CART (sobreajuste y poda).

Incluye un CART mínimo en NumPy (crecimiento voraz con Gini y poda de
costo-complejidad por el eslabón más débil) que reproduce los resultados de
``scikit-learn`` reportados en el libro.

``python -m sections.data`` imprime el resumen que usa el guion docente.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log2

import numpy as np

# ============================================================================
# 1. Medidas de impureza
# ============================================================================


def entropy(counts) -> float:
    n = sum(counts)
    return float(-sum(c / n * log2(c / n) for c in counts if c))


def gini(counts) -> float:
    n = sum(counts)
    return float(1 - sum((c / n) ** 2 for c in counts))


def miscls(counts) -> float:
    n = sum(counts)
    return float(1 - max(counts) / n)


def H2(p: float) -> float:
    return entropy([p, 1 - p]) if 0 < p < 1 else 0.0


def G2(p: float) -> float:
    return 2 * p * (1 - p)


def E2(p: float) -> float:
    return min(p, 1 - p)


# Concavidad: un padre con p = 0.25 se divide en dos mitades con p = 0.05 y 0.45.
CONCAVE_PARENT, CONCAVE_KIDS = 0.25, (0.05, 0.45)


# ============================================================================
# 2. Béisbol: ID3 y C4.5
# ============================================================================
ATTRS = ["Pronóstico", "Temperatura", "Humedad", "Viento"]
DAYS = [  # (día, pronóstico, temperatura, humedad, viento, juega)
    ("D1", "Soleado", "Caluroso", "Alta", "Débil", "No"),
    ("D2", "Soleado", "Caluroso", "Alta", "Fuerte", "No"),
    ("D3", "Nublado", "Caluroso", "Alta", "Débil", "Sí"),
    ("D4", "Lluvia", "Templado", "Alta", "Débil", "Sí"),
    ("D5", "Lluvia", "Fresco", "Normal", "Débil", "Sí"),
    ("D6", "Lluvia", "Fresco", "Normal", "Fuerte", "No"),
    ("D7", "Nublado", "Fresco", "Normal", "Fuerte", "Sí"),
    ("D8", "Soleado", "Templado", "Alta", "Débil", "No"),
    ("D9", "Soleado", "Fresco", "Normal", "Débil", "Sí"),
    ("D10", "Lluvia", "Templado", "Normal", "Débil", "Sí"),
    ("D11", "Soleado", "Templado", "Normal", "Débil", "Sí"),
    ("D12", "Nublado", "Templado", "Alta", "Fuerte", "Sí"),
    ("D13", "Nublado", "Caluroso", "Normal", "Débil", "Sí"),
    ("D14", "Lluvia", "Templado", "Alta", "Fuerte", "No"),
    ("D15", "Soleado", "Fresco", "Alta", "Débil", "No"),
]
VALUES = {
    "Pronóstico": ["Soleado", "Nublado", "Lluvia"],
    "Temperatura": ["Caluroso", "Templado", "Fresco"],
    "Humedad": ["Alta", "Normal"],
    "Viento": ["Débil", "Fuerte"],
}


def yes_no(rows) -> tuple[int, int]:
    c = Counter(r[-1] for r in rows)
    return c["Sí"], c["No"]


def partition(rows, attr: str):
    j = 1 + ATTRS.index(attr) if attr in ATTRS else 0   # «Día» = columna 0
    vals = VALUES.get(attr) or [r[0] for r in rows]
    return {v: [r for r in rows if r[j] == v] for v in vals}


def cond_entropy(rows, attr: str) -> float:
    n = len(rows)
    return sum(len(s) / n * entropy(yes_no(s)) for s in partition(rows, attr).values() if s)


def gain(rows, attr: str) -> float:
    return entropy(yes_no(rows)) - cond_entropy(rows, attr)


def split_info(rows, attr: str) -> float:
    return entropy([len(s) for s in partition(rows, attr).values() if s])


def gain_ratio(rows, attr: str) -> float:
    return gain(rows, attr) / split_info(rows, attr)


H_ROOT = entropy(yes_no(DAYS))
ROOT_GAINS = {a: gain(DAYS, a) for a in ATTRS}
SUNNY = partition(DAYS, "Pronóstico")["Soleado"]
RAIN = partition(DAYS, "Pronóstico")["Lluvia"]
BRANCH_GAINS = {
    "Soleado": {a: gain(SUNNY, a) for a in ATTRS[1:]},
    "Lluvia": {a: gain(RAIN, a) for a in ATTRS[1:]},
}
RATIO_ATTRS = ["Día", "Humedad", "Pronóstico", "Viento", "Temperatura"]
RATIOS = {a: (gain(DAYS, a), split_info(DAYS, a), gain_ratio(DAYS, a)) for a in RATIO_ATTRS}

# Humedad numérica (ordenada) con la clase de cada día.
HUM_NUM = [(65, 1), (70, 0), (70, 1), (70, 1), (75, 1), (80, 1), (80, 1), (85, 0), (86, 1),
           (88, 0), (90, 0), (90, 1), (91, 0), (95, 0), (96, 1)]   # 1 = Sí


def numeric_cuts(pairs):
    """Para cada punto medio entre valores distintos: (t, (sí,no)≤t, (sí,no)>t, ganancia)."""
    xs = sorted(set(v for v, _ in pairs))
    tot = (sum(c for _, c in pairs), sum(1 - c for _, c in pairs))
    h = entropy(tot)
    out = []
    for a, b in zip(xs, xs[1:]):
        t = (a + b) / 2
        le = [c for v, c in pairs if v <= t]
        L = (sum(le), len(le) - sum(le))
        R = (tot[0] - L[0], tot[1] - L[1])
        n = len(pairs)
        g = h - (sum(L) / n) * entropy(L) - (sum(R) / n) * entropy(R)
        out.append((t, L, R, g))
    return out


HUM_CUTS = numeric_cuts(HUM_NUM)
HUM_BEST = max(HUM_CUTS, key=lambda c: c[3])


# ============================================================================
# 3. Laptops: Gini con preguntas binarias (CART categórico)
# ============================================================================
LAPTOP = [  # (edad, ingreso, estudiante, crédito, compra)
    ("joven", "alto", "no", "favorable", "no"), ("joven", "alto", "no", "excelente", "no"),
    ("adulto", "alto", "no", "favorable", "sí"), ("mayor", "medio", "no", "favorable", "sí"),
    ("mayor", "bajo", "sí", "favorable", "sí"), ("mayor", "bajo", "sí", "excelente", "no"),
    ("adulto", "bajo", "sí", "excelente", "sí"), ("joven", "medio", "no", "favorable", "no"),
    ("joven", "bajo", "sí", "favorable", "sí"), ("mayor", "medio", "sí", "favorable", "sí"),
    ("joven", "medio", "sí", "excelente", "sí"), ("adulto", "medio", "no", "excelente", "sí"),
    ("adulto", "alto", "sí", "favorable", "sí"), ("mayor", "medio", "no", "excelente", "no"),
]
LAPTOP_ATTRS = ["edad", "ingreso", "estudiante", "crédito"]


def lap_counts(rows):
    return (sum(r[-1] == "sí" for r in rows), sum(r[-1] == "no" for r in rows))


def lap_question(attr: str, value: str):
    j = LAPTOP_ATTRS.index(attr)
    yes = [r for r in LAPTOP if r[j] == value]
    no = [r for r in LAPTOP if r[j] != value]
    n = len(LAPTOP)
    w = len(yes) / n * gini(lap_counts(yes)) + len(no) / n * gini(lap_counts(no))
    return lap_counts(yes), lap_counts(no), w, gini(lap_counts(LAPTOP)) - w


G_LAPTOP_ROOT = gini(lap_counts(LAPTOP))
AGE_QUESTIONS = {v: lap_question("edad", v) for v in ("adulto", "joven", "mayor")}
BEST_QUESTIONS = [("edad", "adulto"), ("estudiante", "sí"), ("crédito", "excelente"),
                  ("ingreso", "alto")]


# ============================================================================
# 4. CART numérico mínimo (Gini, cortes en puntos medios, poda de costo-complejidad)
# ============================================================================
@dataclass
class Node:
    idx: np.ndarray                      # índices de los ejemplos del nodo
    counts: tuple[int, int]              # (clase 0, clase 1)
    depth: int
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    box: tuple = field(default=(0, 10, 0, 10))   # (x0, x1, y0, y1) en datos

    @property
    def n(self) -> int:
        return int(sum(self.counts))

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def cls(self) -> int:
        return int(self.counts[1] > self.counts[0])

    @property
    def g(self) -> float:
        return gini(self.counts)


def best_split(X, y, idx, order=(0, 1)):
    """Mejor (feature, umbral, reducción) con Gini; empates → primer atributo de ``order``
    y umbral más bajo (como scikit-learn con un orden de atributos fijo)."""
    parent = gini((np.sum(y[idx] == 0), np.sum(y[idx] == 1)))
    best = None
    n = len(idx)
    for f in order:
        vals = X[idx, f]
        srt = np.argsort(vals, kind="stable")
        v, yy = vals[srt], y[idx][srt]
        for k in range(1, n):
            if v[k] <= v[k - 1] + 1e-7:
                continue
            L = (np.sum(yy[:k] == 0), np.sum(yy[:k] == 1))
            R = (np.sum(yy[k:] == 0), np.sum(yy[k:] == 1))
            w = k / n * gini(L) + (n - k) / n * gini(R)
            imp = parent - w
            if best is None or imp > best[2] + 1e-12:
                best = (f, float(v[k - 1] / 2 + v[k] / 2), imp)
    return best


def grow(X, y, *, order=(0, 1), box=(0, 10, 0, 10), idx=None, depth=0,
         max_depth: int | None = None) -> Node:
    idx = np.arange(len(y)) if idx is None else idx
    node = Node(idx, (int(np.sum(y[idx] == 0)), int(np.sum(y[idx] == 1))), depth, box=box)
    if min(node.counts) == 0 or (max_depth is not None and depth >= max_depth):
        return node
    s = best_split(X, y, idx, order)
    if s is None:
        return node
    f, t, _ = s
    node.feature, node.threshold = f, t
    li, ri = idx[X[idx, f] <= t], idx[X[idx, f] > t]
    x0, x1, y0, y1 = box
    lb = (x0, t, y0, y1) if f == 0 else (x0, x1, y0, t)
    rb = (t, x1, y0, y1) if f == 0 else (x0, x1, t, y1)
    node.left = grow(X, y, order=order, box=lb, idx=li, depth=depth + 1, max_depth=max_depth)
    node.right = grow(X, y, order=order, box=rb, idx=ri, depth=depth + 1, max_depth=max_depth)
    return node


def leaves(node: Node) -> list[Node]:
    return [node] if node.is_leaf else leaves(node.left) + leaves(node.right)


def internal(node: Node) -> list[Node]:
    return [] if node.is_leaf else [node] + internal(node.left) + internal(node.right)


def predict_one(node: Node, x) -> int:
    while not node.is_leaf:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.cls


def predict(node: Node, X):
    return np.array([predict_one(node, x) for x in X])


def accuracy(node: Node, X, y) -> float:
    return float(np.mean(predict(node, X) == y))


def R(node: Node, n_total: int) -> float:
    return node.n / n_total * node.g


def R_tree(node: Node, n_total: int) -> float:
    return sum(R(l, n_total) for l in leaves(node))


def weakest_link(node: Node, n_total: int) -> float:
    return (R(node, n_total) - R_tree(node, n_total)) / (len(leaves(node)) - 1)


def copy_tree(node: Node) -> Node:
    c = Node(node.idx, node.counts, node.depth, node.feature, node.threshold, box=node.box)
    if not node.is_leaf:
        c.left, c.right = copy_tree(node.left), copy_tree(node.right)
    return c


def pruning_path(root: Node):
    """Secuencia anidada [(α, árbol)] podando repetidamente el eslabón más débil."""
    n = root.n
    tree = copy_tree(root)
    path = [(0.0, copy_tree(tree))]
    while not tree.is_leaf:
        gs = [(weakest_link(t, n), t) for t in internal(tree)]
        a = min(g for g, _ in gs)
        for g, t in gs:
            if g <= a + 1e-12:
                t.left = t.right = None
                t.feature = t.threshold = None
        path.append((a, copy_tree(tree)))
    return path


# -- 4a. Los diez clientes (clase anterior) ----------------------------------
CLIENTS = [  # (id, ingreso, deuda, y)  y = 1 → no paga
    (1, 20, 20, 0), (2, 25, 30, 0), (3, 32, 25, 0), (4, 38, 35, 0), (5, 15, 40, 0),
    (6, 30, 58, 0), (7, 12, 55, 1), (8, 22, 60, 1), (9, 35, 62, 1), (10, 28, 48, 1),
]
CX = np.array([[c[1], c[2]] for c in CLIENTS], float)
CY = np.array([c[3] for c in CLIENTS])
C_RANGE = (8, 48, 15, 67)          # ingreso, deuda
NEW_CLIENT = (33, 52)              # cliente nuevo que «cae» por el árbol


def client_tree(max_depth=None):
    return grow(CX, CY, box=C_RANGE, max_depth=max_depth)


def cut_scan(X, y, idx, f):
    """Gini ponderado de cada corte del atributo f entre los ejemplos idx."""
    vals = sorted(set(X[idx, f]))
    out = []
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2
        L = idx[X[idx, f] <= t]
        Rr = idx[X[idx, f] > t]
        cl = (np.sum(y[L] == 0), np.sum(y[L] == 1))
        cr = (np.sum(y[Rr] == 0), np.sum(y[Rr] == 1))
        w = len(L) / len(idx) * gini(cl) + len(Rr) / len(idx) * gini(cr)
        out.append((t, w))
    return out


# -- 4b. Los 60 puntos del laboratorio de CART --------------------------------
def zonas(X):
    return ((X[:, 0] > 4) & (X[:, 1] < 6)) | ((X[:, 0] < 2.5) & (X[:, 1] > 7))


def generar(n, semilla):
    rng = np.random.default_rng(semilla)
    X = rng.uniform(0, 10, size=(n, 2)).round(1)
    y = (rng.uniform(size=n) < np.where(zonas(X), 0.88, 0.12)).astype(int)
    return X, y


def as_float32(X):
    """scikit-learn trabaja en float32: sin esto, un punto de prueba justo en un corte
    (p. ej. x2 = 5.7 con corte 5.70) caería del otro lado."""
    return X.astype(np.float32).astype(np.float64)


LX, LY = generar(60, 7)
TX, TY = generar(300, 6)
LX, TX = as_float32(LX), as_float32(TX)


def lab_tree():
    return grow(LX, LY, order=LAB_ORDER)


LAB_ORDER = (0, 1)     # en empates gana x1, como en el árbol de scikit-learn del libro


def lab_path():
    return pruning_path(lab_tree())


# Árbol de 4 hojas de la secuencia (el de la tabla del eslabón más débil).
def lab_four_leaves():
    for a, t in lab_path():
        if len(leaves(t)) == 4:
            return a, t
    raise RuntimeError("no hay árbol de 4 hojas en la ruta")


# Inestabilidad: quitar un punto y volver a crecer el árbol.
def lab_tree_without(i: int):
    keep = np.array([k for k in range(len(LY)) if k != i])
    return grow(LX[keep], LY[keep], order=LAB_ORDER)


# ============================================================================
# 5. Ejercicio final (a mano)
# ============================================================================
EX_POINTS = [  # (x1, x2, y)
    (1, 6, 1), (2, 7, 1), (3, 2, 0), (4, 8, 1), (5, 3, 0), (6, 6, 0), (7, 1, 0), (8, 7, 0),
]


EX_X = np.array([[p[0], p[1]] for p in EX_POINTS], float)
EX_Y = np.array([p[2] for p in EX_POINTS])
EX_CANDIDATES = [(0, 2.5), (0, 4.5), (1, 4.5)]      # los tres cortes que se piden a mano


def exercise():
    X, y = EX_X, EX_Y
    idx = np.arange(len(y))
    return X, y, gini((np.sum(y == 0), np.sum(y == 1))), cut_scan(X, y, idx, 0), \
        cut_scan(X, y, idx, 1)


def exercise_weighted(f: int, t: float) -> float:
    return dict(cut_scan(EX_X, EX_Y, np.arange(len(EX_Y)), f))[t]


def exercise_tree():
    return grow(EX_X, EX_Y, box=(0, 9, 0, 9))


# Inestabilidad: el punto cuya ausencia cambia la raíz y el número de hojas.
UNSTABLE_POINT = 34


# ============================================================================
def describe(node: Node, names=("x1", "x2"), pad="") -> str:
    if node.is_leaf:
        return f"{pad}hoja {node.counts} → {node.cls}\n"
    s = f"{pad}{names[node.feature]} <= {node.threshold:.2f}  {node.counts}\n"
    return s + describe(node.left, names, pad + "  ") + describe(node.right, names, pad + "  ")


def summary() -> None:
    print(f"H(raíz) = {H_ROOT:.3f}")
    for a, g in ROOT_GAINS.items():
        print(f"  Gan({a}) = {g:.3f}   H después = {H_ROOT - g:.3f}")
    for b, d in BRANCH_GAINS.items():
        print(f"  rama {b}: " + ", ".join(f"{a} {g:.3f}" for a, g in d.items()))
    for a, (g, si, gr) in RATIOS.items():
        print(f"  C4.5 {a}: gan {g:.3f} split {si:.3f} razón {gr:.3f}")
    print("Humedad numérica:", [(t, L, Rr, round(g, 3)) for t, L, Rr, g in HUM_CUTS])
    print("  mejor:", HUM_BEST)
    pp = CONCAVE_PARENT
    for name, f in [("H", H2), ("G", G2), ("E", E2)]:
        kids = [f(p) for p in CONCAVE_KIDS]
        print(f"Concavidad {name}: padre {f(pp):.3f}, hijos {[round(k, 3) for k in kids]}, "
              f"ganancia {f(pp) - sum(kids) / 2:.3f}")
    print(f"Laptop Gini raíz {G_LAPTOP_ROOT:.3f}")
    for v, q in AGE_QUESTIONS.items():
        print(f"  ¿edad={v}? {q[0]} {q[1]} w={q[2]:.3f} ΔG={q[3]:.3f}")
    for a, v in BEST_QUESTIONS:
        q = lap_question(a, v)
        print(f"  ¿{a}={v}? w={q[2]:.3f} ΔG={q[3]:.3f}")

    t = client_tree()
    print("Clientes, CART completo:\n" + describe(t, ("ingreso", "deuda")))
    idx = np.arange(10)
    for f, nm in [(0, "ingreso"), (1, "deuda")]:
        print(f"  barrido {nm}:", [(c, round(w, 3)) for c, w in cut_scan(CX, CY, idx, f)])
    print("  nuevo cliente", NEW_CLIENT, "→", predict_one(t, NEW_CLIENT))

    full = lab_tree()
    print(f"Lab: {len(leaves(full))} hojas, train {accuracy(full, LX, LY):.3f}, "
          f"test {accuracy(full, TX, TY):.3f}")
    for a, tr in lab_path():
        print(f"  α={a:.4f} hojas={len(leaves(tr)):2d} R={R_tree(tr, 60):.3f} "
              f"train={accuracy(tr, LX, LY):.3f} test={accuracy(tr, TX, TY):.3f}")
    a4, t4 = lab_four_leaves()
    print(describe(t4))
    for nd in internal(t4):
        print(f"  nodo x{nd.feature + 1}<={nd.threshold:.2f} {nd.counts} G={nd.g:.3f} "
              f"R={R(nd, 60):.3f} RT={R_tree(nd, 60):.3f} hojas={len(leaves(nd))} "
              f"g={weakest_link(nd, 60):.3f}")
    for i in range(60):
        tw = lab_tree_without(i)
        if tw.feature != full.feature or abs(tw.threshold - full.threshold) > 1e-9:
            print(f"  quitar el punto {i} {LX[i]} y={LY[i]} cambia la raíz: "
                  f"x{tw.feature + 1}<={tw.threshold:.2f} (antes x{full.feature + 1}<="
                  f"{full.threshold:.2f}), hojas {len(leaves(tw))}")
    X, y, g0, s1, s2 = exercise()
    print("Ejercicio: Gini raíz", round(g0, 3),
          [(f, t, round(exercise_weighted(f, t), 3)) for f, t in EX_CANDIDATES])
    print(describe(exercise_tree()))
    print("  x1:", [(c, round(w, 3)) for c, w in s1])
    print("  x2:", [(c, round(w, 3)) for c, w in s2])


if __name__ == "__main__":
    summary()
