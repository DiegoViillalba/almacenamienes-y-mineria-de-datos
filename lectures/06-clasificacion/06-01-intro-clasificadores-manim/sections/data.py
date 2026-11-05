"""Datos y cálculos de la clase. Todo número que aparece en pantalla sale de aquí.

Ejemplo base: los diez clientes de ``unidades/clasificadores/6-0-introduccion-
clasificacion.qmd`` (ingreso en miles de pesos al mes, endeudamiento en % del
ingreso, ¿pagó?) y sus cinco clientes de prueba A–E.

Convención de clases: la clase **positiva** es «no paga» (impago), la que el
banco quiere detectar. ``y = 1`` significa «no paga», ``y = 0`` «paga».

``python -m sections.data`` imprime el resumen que usa el guion docente.
"""
from __future__ import annotations

from math import exp, log, sqrt

import numpy as np

# ----------------------------------------------------------------------------
# Clientes
# ----------------------------------------------------------------------------
# (id, ingreso, deuda, y)  con y = 1 → no paga (positivo)
TRAIN = [
    (1, 20, 20, 0), (2, 25, 30, 0), (3, 32, 25, 0), (4, 38, 35, 0), (5, 15, 40, 0),
    (6, 30, 58, 0), (7, 12, 55, 1), (8, 22, 60, 1), (9, 35, 62, 1), (10, 28, 48, 1),
]
TEST = [
    ("A", 18, 35, 0), ("B", 31, 60, 1), ("C", 26, 45, 1), ("D", 36, 30, 0), ("E", 40, 50, 0),
]
LABEL = {0: "paga", 1: "no paga"}

X_RANGE = (8, 48)     # ingreso
Y_RANGE = (15, 67)    # deuda


def xy(rows):
    return np.array([[r[1], r[2]] for r in rows], float), np.array([r[3] for r in rows])


XTR, YTR = xy(TRAIN)
XTE, YTE = xy(TEST)


# ----------------------------------------------------------------------------
# Referencia y tocón
# ----------------------------------------------------------------------------
def baseline_acc(y) -> float:
    return float(np.mean(y == 0))   # siempre «paga» (la clase mayoritaria)


def stump_predict(deuda, t: float):
    """Regla de un corte: deuda > t ⇒ no paga (1)."""
    return (np.asarray(deuda) > t).astype(int)


def stump_sweep():
    """Errores de entrenamiento para cada corte en el punto medio de valores vecinos."""
    vals = sorted(set(XTR[:, 1]))
    cuts = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    out = []
    for t in cuts:
        pred = stump_predict(XTR[:, 1], t)
        wrong = [TRAIN[i][0] for i in range(len(TRAIN)) if pred[i] != YTR[i]]
        out.append((t, len(wrong), wrong))
    return out


STUMP_T = 44.0


# ----------------------------------------------------------------------------
# Matriz de confusión y métricas
# ----------------------------------------------------------------------------
def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def metrics(cm):
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]
    n = tp + fn + fp + tn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def cell_of(y: int, p: int) -> str:
    return {(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[(y, p)]


THRESHOLDS = [37.5, 44.0, 51.5, 59.0]   # recorrido del corte en la matriz de confusión


def threshold_table():
    rows = []
    for t in THRESHOLDS:
        cm = confusion(YTR, stump_predict(XTR[:, 1], t))
        rows.append((t, cm, metrics(cm)))
    return rows


# Paradoja de la exactitud: 100 clientes, 3 fraudes; el modelo perezoso dice «todo bien».
LAZY_N, LAZY_POS = 100, 3


# ----------------------------------------------------------------------------
# k vecinos más cercanos
# ----------------------------------------------------------------------------
def distances(q, X=XTR, sx: float = 1.0, sy: float = 1.0):
    d = np.sqrt(((X[:, 0] - q[0]) * sx) ** 2 + ((X[:, 1] - q[1]) * sy) ** 2)
    return d


def knn_order(q, **kw):
    d = distances(q, **kw)
    return [int(i) for i in np.argsort(d, kind="stable")], d


def knn_predict_one(q, k: int, X=XTR, y=YTR, **kw) -> int:
    idx, _ = knn_order(q, X=X, **kw)
    votes = y[idx[:k]]
    return int(votes.sum() * 2 > k)        # mayoría (k impar)


def knn_predict(Q, k: int, **kw):
    return np.array([knn_predict_one(q, k, **kw) for q in Q])


def knn_train_acc(k: int) -> float:
    """Exactitud de entrenamiento: cada cliente se cuenta como su propio vecino."""
    return float(np.mean(knn_predict(XTR, k) == YTR))


QUERY = (31, 60)        # cliente B: el ejemplo guiado
TURN = (26, 45)         # cliente C: «tu turno»

# Cuadrícula para colorear regiones de decisión (centros de celda).
GRID_NX, GRID_NY = 40, 52


def grid_centers():
    xs = np.linspace(X_RANGE[0], X_RANGE[1], GRID_NX + 1)
    ys = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_NY + 1)
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    return cx, cy


def knn_grid(k: int):
    cx, cy = grid_centers()
    return np.array([[knn_predict_one((x, y), k) for x in cx] for y in cy])


# ----------------------------------------------------------------------------
# Regresión logística (Newton–Raphson sobre variables estandarizadas)
# ----------------------------------------------------------------------------
MU = XTR.mean(axis=0)
SD = XTR.std(axis=0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _design(X):
    Z = (np.asarray(X, float) - MU) / SD
    return np.column_stack([np.ones(len(Z)), Z])


def log_loss(beta, X=XTR, y=YTR) -> float:
    p = sigmoid(_design(X) @ beta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(l2: float = 0.0, iters: int = 50):
    A = _design(XTR)
    beta = np.zeros(3)
    for _ in range(iters):
        p = sigmoid(A @ beta)
        g = A.T @ (p - YTR) + l2 * np.r_[0, beta[1:]]
        H = A.T @ (A * (p * (1 - p))[:, None]) + l2 * np.diag([0, 1, 1])
        beta -= np.linalg.solve(H, g)
    return beta


BETA = fit_logistic()


def raw_coefs(beta=BETA):
    """Convierte a unidades originales: z = b + w1·ingreso + w2·deuda."""
    w = beta[1:] / SD
    b = beta[0] - float(np.sum(beta[1:] * MU / SD))
    return float(b), float(w[0]), float(w[1])


def logit_proba(X, beta=BETA):
    return sigmoid(_design(X) @ beta)


def gd_path(steps: int = 7, lr: float = 1.2):
    """Trayectoria de descenso de gradiente desde una recta mala hasta el óptimo.

    Arranca con una recta casi vertical (solo ingreso) para que el movimiento se
    vea; usa pasos crecientes en escala logarítmica y termina en BETA.
    """
    A = _design(XTR)
    beta = np.array([0.0, -1.5, 0.2])
    path = [beta.copy()]
    k = 0
    budget = [1, 2, 4, 8, 20, 60, 400]
    for n in budget[:steps - 1]:
        for _ in range(n):
            p = sigmoid(A @ beta)
            beta = beta - lr * A.T @ (p - YTR) / len(YTR)
            k += 1
        path.append(beta.copy())
    path.append(BETA.copy())
    return path


# Pesos redondeados para el ejercicio a mano (unidades originales).
EX_B, EX_W1, EX_W2 = -8.0, -0.10, 0.25
EX_CLIENT = (20, 50)


def ex_score(x1, x2):
    return EX_B + EX_W1 * x1 + EX_W2 * x2


# ----------------------------------------------------------------------------
# Comparación en prueba
# ----------------------------------------------------------------------------
K_FINAL = 3


def test_predictions():
    knn = knn_predict(XTE, K_FINAL)
    p = logit_proba(XTE)
    lr = (p >= 0.5).astype(int)
    return knn, p, lr


def summary() -> None:
    print("Referencia:", baseline_acc(YTR), "prueba", baseline_acc(YTE))
    print("Barrido del corte (t, errores, quiénes):")
    for t, e, w in stump_sweep():
        print(f"  {t:5.1f} {e} {w}")
    print("Corte y matriz (positivo = no paga):")
    for t, cm, m in threshold_table():
        print(f"  t={t:5.1f} {cm} acc={m['acc']:.2f} prec={m['prec']:.2f} "
              f"rec={m['rec']:.2f} f1={m['f1']:.2f}")
    print("Paradoja: exactitud perezosa", (LAZY_N - LAZY_POS) / LAZY_N)

    for name, q in [("B", QUERY), ("C", TURN)]:
        idx, d = knn_order(q)
        print(f"Vecinos de {name}{q}:",
              [(TRAIN[i][0], round(float(d[i]), 2), LABEL[int(YTR[i])]) for i in idx[:5]])
    idx, d = knn_order(QUERY, sy=0.01)
    print("B con deuda como proporción:",
          [(TRAIN[i][0], round(float(d[i]), 3), LABEL[int(YTR[i])]) for i in idx[:4]])
    for k in (1, 3, 5, 9):
        print(f"k={k}: train acc {knn_train_acc(k):.2f}, test pred {knn_predict(XTE, k)}, "
              f"test acc {np.mean(knn_predict(XTE, k) == YTE):.2f}")

    b, w1, w2 = raw_coefs()
    print(f"Logística: z = {b:.3f} + {w1:.4f}·ingreso + {w2:.4f}·deuda; "
          f"loss={log_loss(BETA):.3f}")
    ptr = logit_proba(XTR)
    print("  p(no paga) train:", np.round(ptr, 2), "acc", np.mean((ptr >= .5) == YTR))
    print("  frontera: deuda = ", f"{-b / w2:.2f} + {-w1 / w2:.3f}·ingreso")
    for beta in gd_path():
        bb, a1, a2 = raw_coefs(beta)
        print(f"  GD loss={log_loss(beta):.3f}  z={bb:.2f}+{a1:.3f}x1+{a2:.3f}x2")

    knn, p, lr = test_predictions()
    print("Prueba:", [r[0] for r in TEST], "real", YTE)
    print("  3-NN", knn, confusion(YTE, knn), metrics(confusion(YTE, knn)))
    print("  logit p", np.round(p, 2), lr, confusion(YTE, lr), metrics(confusion(YTE, lr)))
    print("  logit train acc", np.mean((logit_proba(XTR) >= .5) == YTR))
    z = ex_score(*EX_CLIENT)
    print("Ejercicio: z =", z, "p =", 1 / (1 + exp(-z)))
    idx, d = knn_order(EX_CLIENT)
    print("  vecinos ejercicio:",
          [(TRAIN[i][0], round(float(d[i]), 2), LABEL[int(YTR[i])]) for i in idx[:4]])


if __name__ == "__main__":
    summary()
