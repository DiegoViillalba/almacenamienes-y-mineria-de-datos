"""Datos y cálculos de la clase. Todo número que aparece en pantalla sale de aquí.

Ejemplos (todos del libro):

- las seis laptops (RAM, precio) de ``capitulo_arboles_regresion_lineal.qmd``:
  árbol de regresión, muestra bootstrap y fuera de la bolsa;
- el jurado de Condorcet (5 jueces al 70 %) y la varianza de un promedio;
- los 40 puntos del laboratorio de bagging (y = 2 sen x + 0.3 x + ruido);
- los diez puntos de AdaBoost (tres rondas de tocones);
- las cinco casas de ``capitulo_xgboost.qmd``: gradient boosting a mano y la
  primera ronda de XGBoost (gradientes, hessianos, pesos y ganancias).

``python -m sections.data`` imprime el resumen que usa el guion docente.
"""
from __future__ import annotations

from math import comb, exp, log

import numpy as np

# ============================================================================
# 1. Árbol de regresión: seis laptops
# ============================================================================
LAPTOPS = [("A", 4, 6500), ("B", 4, 7200), ("C", 8, 10500), ("D", 8, 12000),
           ("E", 16, 21000), ("F", 16, 24000)]
RAM = np.array([l[1] for l in LAPTOPS], float)
PRICE = np.array([l[2] for l in LAPTOPS], float)


def sse(y) -> float:
    y = np.asarray(y, float)
    return float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0


def ram_cut(t: float, x=RAM, y=PRICE):
    left, right = y[x <= t], y[x > t]
    return float(left.mean()), float(right.mean()), sse(left) + sse(right)


SSE0 = sse(PRICE)
RAM_CUTS = {t: ram_cut(t) for t in (6.0, 12.0)}

# Muestra bootstrap del texto y su árbol (mismo corte RAM ≤ 12).
BOOT = ["A", "A", "C", "D", "F", "F"]
OOB = [l[0] for l in LAPTOPS if l[0] not in BOOT]


def boot_tree():
    idx = [[l[0] for l in LAPTOPS].index(b) for b in BOOT]
    return ram_cut(12.0, RAM[idx], PRICE[idx])


def oob_errors():
    lm, rm, _ = boot_tree()
    out = []
    for name in OOB:
        _, ram, price = LAPTOPS[[l[0] for l in LAPTOPS].index(name)]
        pred = lm if ram <= 12 else rm
        out.append((name, price, pred, price - pred))
    return out


# ============================================================================
# 2. Sabiduría de muchos
# ============================================================================
def majority(n: int, p: float) -> float:
    """Probabilidad de que la mayoría de n jueces independientes (n impar) acierte."""
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def binom_terms(n: int, p: float):
    return [comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


def var_mean(B: float, rho: float, sigma2: float = 1.0) -> float:
    return rho * sigma2 + (1 - rho) * sigma2 / B


def oob_fraction(n: int) -> float:
    return (1 - 1 / n) ** n


# ============================================================================
# 3. Bagging en 1D: los 40 puntos del laboratorio del libro
# ============================================================================
BAG_TR = np.array([
    [0.29, 0.87], [0.7, 1.65], [0.83, 2.38], [1.29, 1.54], [1.29, 1.91], [1.3, 1.81],
    [1.38, 1.34], [1.48, 2.51], [1.48, 2.75], [2.02, 1.96], [2.05, 3.22], [2.17, 2.8],
    [2.35, 2.5], [2.75, 1.83], [2.77, 2.13], [3.53, -0.5], [3.69, 0.43], [4.3, -0.18],
    [4.67, -1.66], [4.84, -0.32], [4.99, -0.58], [5.11, 0.16], [5.12, -0.56], [5.49, 0.21],
    [5.54, 0.51], [5.92, 0.54], [6.01, 1.62], [6.22, 1.68], [6.63, 2.96], [6.7, 2.51],
    [6.73, 3.53], [7.88, 4.73], [8.02, 4.27], [8.17, 4.73], [8.67, 4.73], [8.96, 4.66],
    [9.01, 2.56], [9.28, 3.6], [9.48, 3.01], [9.81, 2.14]])
SIGMA2 = 0.36


def f_true(x):
    return 2 * np.sin(x) + 0.3 * np.asarray(x)


def reg_tree(x, y, depth: int = 0, max_depth: int = 99):
    """Árbol de regresión 1D: (umbral, izq, der) o valor constante (hoja)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if depth >= max_depth or len(y) < 2:
        return float(y.mean())
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    best = None
    base = sse(ys)
    for k in range(1, len(ys)):
        if xs[k] == xs[k - 1]:
            continue
        s = sse(ys[:k]) + sse(ys[k:])
        if best is None or s < best[0] - 1e-9:
            best = (s, (xs[k - 1] + xs[k]) / 2)
    if best is None or best[0] >= base - 1e-12:
        return float(y.mean())
    t = best[1]
    return (t, reg_tree(x[x <= t], y[x <= t], depth + 1, max_depth),
            reg_tree(x[x > t], y[x > t], depth + 1, max_depth))


def reg_predict(tree, x: float) -> float:
    while isinstance(tree, tuple):
        tree = tree[1] if x <= tree[0] else tree[2]
    return tree


def reg_steps(tree, lo: float = 0.0, hi: float = 10.0):
    """Lista de (x0, x1, valor) de la función escalonada en [lo, hi]."""
    if not isinstance(tree, tuple):
        return [(lo, hi, tree)]
    t = tree[0]
    return reg_steps(tree[1], lo, min(t, hi)) + reg_steps(tree[2], max(t, lo), hi)


BAG_B = 60
BAG_SEED = 7


def bagged_trees(B: int = BAG_B, seed: int = BAG_SEED, max_depth: int = 99):
    rng = np.random.default_rng(seed)
    n = len(BAG_TR)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(reg_tree(BAG_TR[idx, 0], BAG_TR[idx, 1], max_depth=max_depth))
    return out


def test_grid(n: int = 400):
    xs = np.linspace(0, 10, n)
    return xs, f_true(xs)


def mse_vs_truth(pred_fn) -> float:
    """Error esperado en datos nuevos = sesgo² + varianza (respecto a f) + σ²."""
    xs, fx = test_grid()
    return float(np.mean([(pred_fn(x) - f) ** 2 for x, f in zip(xs, fx)]) + SIGMA2)


# ============================================================================
# 4. AdaBoost: diez puntos en una recta
# ============================================================================
ADA_X = np.arange(1, 11)
ADA_Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])


def best_stump(w):
    """Mejor tocón (t, signo izq): predice signo si x ≤ t, −signo si no."""
    best = None
    for t in np.arange(0.5, 11, 1.0):
        for s in (1, -1):
            pred = np.where(ADA_X <= t, s, -s)
            err = float(np.sum(w[pred != ADA_Y]))
            if best is None or err < best[0] - 1e-12:
                best = (err, float(t), s)
    return best


def adaboost(rounds: int = 3):
    w = np.full(10, 0.1)
    out = []
    for _ in range(rounds):
        err, t, s = best_stump(w)
        alpha = 0.5 * log((1 - err) / err)
        pred = np.where(ADA_X <= t, s, -s)
        w_before = w.copy()
        w = w * np.exp(-alpha * ADA_Y * pred)
        w = w / w.sum()
        out.append(dict(t=t, s=s, err=err, alpha=alpha, pred=pred, w_before=w_before,
                        w_after=w.copy(), wrong=[int(x) for x in ADA_X[pred != ADA_Y]]))
    return out


ADA = adaboost()


def ada_score():
    return sum(r["alpha"] * r["pred"] for r in ADA)


# ============================================================================
# 5. Gradient boosting a mano: cinco casas
# ============================================================================
GB_X = np.array([1, 2, 3, 4, 5], float)
GB_Y = np.array([2, 4, 9, 10, 15], float)
ETA = 0.5


def stump_fit(x, r):
    """Tocón de regresión para los residuos r: (corte, media izq, media der)."""
    best = None
    for k in range(1, len(x)):
        s = sse(r[:k]) + sse(r[k:])
        if best is None or s < best[0]:
            best = (s, (x[k - 1] + x[k]) / 2, float(r[:k].mean()), float(r[k:].mean()))
    return best[1:]


def gb_rounds(rounds: int = 3, eta: float = ETA):
    F = np.full(5, GB_Y.mean())
    out = [dict(F=F.copy(), r=GB_Y - F, sse=float(np.sum((GB_Y - F) ** 2)), stump=None)]
    for _ in range(rounds):
        r = GB_Y - F
        t, a, b = stump_fit(GB_X, r)
        F = F + eta * np.where(GB_X <= t, a, b)
        out.append(dict(F=F.copy(), r=GB_Y - F, sse=float(np.sum((GB_Y - F) ** 2)),
                        stump=(t, a, b), fitted_r=r))
    return out


GB = gb_rounds()

# ---- XGBoost: primera ronda con pérdida ½(y − F)² --------------------------
LAMBDA = 1.0
F0 = float(GB_Y.mean())
G_I = F0 - GB_Y            # gradientes g_i = F − y
H_I = np.ones(5)           # hessianos h_i = 1


def leaf_weight(G: float, H: float, lam: float = LAMBDA) -> float:
    return -G / (H + lam)


def xgb_gain(k: int, lam: float = LAMBDA, gamma: float = 0.0):
    """Ganancia de cortar entre la casa k y la k+1 (k = 1…4)."""
    GL, GR = float(G_I[:k].sum()), float(G_I[k:].sum())
    HL, HR = float(H_I[:k].sum()), float(H_I[k:].sum())
    g = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam))
    return GL, GR, HL, HR, g - gamma


XGB_GAINS = [(k + 0.5, *xgb_gain(k)) for k in range(1, 5)]
XGB_BEST = max(XGB_GAINS, key=lambda z: z[-1])

# ---- Clasificación: log-loss --------------------------------------------------
CLF_Y = [1, 1, 0]
CLF_P = 0.5


def clf_leaf():
    g = [CLF_P - y for y in CLF_Y]
    h = [CLF_P * (1 - CLF_P)] * len(CLF_Y)
    G, H = sum(g), sum(h)
    return g, h, G, H, -G / (H + LAMBDA)


def logloss(F: float, y: int = 1) -> float:
    p = 1 / (1 + exp(-F))
    return -(y * log(p) + (1 - y) * log(1 - p))


def logloss_taylor(F: float, F0: float = -0.3, y: int = 1):
    p = 1 / (1 + exp(-F0))
    g, h = p - y, p * (1 - p)
    d = F - F0
    return logloss(F0, y) + g * d, logloss(F0, y) + g * d + 0.5 * h * d * d


# ============================================================================
# 6. Ejercicio final: una ronda de gradient boosting y de XGBoost a mano
# ============================================================================
EX_X = np.array([1, 2, 3, 4], float)
EX_Y = np.array([1, 3, 6, 8], float)


def exercise():
    F0 = float(EX_Y.mean())
    r = EX_Y - F0
    t, a, b = stump_fit(EX_X, r)
    F1 = F0 + ETA * np.where(EX_X <= t, a, b)
    g = F0 - EX_Y
    k = int(np.sum(EX_X <= t))
    GL, GR = float(g[:k].sum()), float(g[k:].sum())
    wl, wr = leaf_weight(GL, k), leaf_weight(GR, len(g) - k)
    gain = 0.5 * (GL ** 2 / (k + LAMBDA) + GR ** 2 / (len(g) - k + LAMBDA)
                  - (GL + GR) ** 2 / (len(g) + LAMBDA))
    return dict(F0=F0, r=r, t=t, a=a, b=b, F1=F1, sse0=float(np.sum(r ** 2)),
                sse1=float(np.sum((EX_Y - F1) ** 2)), GL=GL, GR=GR, wl=wl, wr=wr, gain=gain)


# ============================================================================
def summary() -> None:
    print(f"Laptops: media {PRICE.mean():.2f}, SSE0 = {SSE0 / 1e6:.2f}e6")
    for t, (lm, rm, s) in RAM_CUTS.items():
        print(f"  RAM ≤ {t:g}: {lm:.0f} | {rm:.0f}  SSE {s / 1e6:.2f}e6  "
              f"reducción {100 * (1 - s / SSE0):.1f} %")
    print("  bootstrap", BOOT, "→", boot_tree(), "OOB", oob_errors())
    print(f"Condorcet: 5 → {majority(5, .7):.3f}  términos {[round(t, 3) for t in binom_terms(5, .7)[3:]]}"
          f"  25 → {majority(25, .7):.3f}  101 → {majority(101, .7):.4f}")
    print("  (1-1/n)^n:", {n: round(oob_fraction(n), 3) for n in (6, 10, 40, 100, 1000)})
    one = reg_tree(BAG_TR[:, 0], BAG_TR[:, 1])
    trees = bagged_trees()
    for B in (1, 5, 20, 60):
        avg = lambda x, B=B: float(np.mean([reg_predict(t, x) for t in trees[:B]]))
        print(f"  bagging B={B}: error esperado {mse_vs_truth(avg):.3f}")
    print(f"  un árbol: error esperado {mse_vs_truth(lambda x: reg_predict(one, x)):.3f}, "
          f"hojas {len(reg_steps(one))}")
    for i, r in enumerate(ADA, 1):
        print(f"AdaBoost ronda {i}: x ≤ {r['t']} ⇒ {r['s']:+d}, ε={r['err']:.3f}, "
              f"α={r['alpha']:.3f}, falla {r['wrong']}, pesos después "
              f"{np.round(r['w_after'], 4)}")
    print("  F(x) =", np.round(ada_score(), 3), "signo ok:", bool(np.all(np.sign(ada_score()) == ADA_Y)))
    for i, r in enumerate(GB):
        print(f"GB ronda {i}: F={np.round(r['F'], 2)} SSE={r['sse']:.2f} tocón={r['stump']}")
    print(f"XGB g={G_I} hojas x≤2.5: {leaf_weight(G_I[:2].sum(), 2):.3f} "
          f"{leaf_weight(G_I[2:].sum(), 3):.3f}")
    for row in XGB_GAINS:
        print(f"  corte {row[0]}: GL={row[1]:.0f} GR={row[2]:.0f} ganancia {row[5]:.2f}")
    print("  clasificación:", clf_leaf())
    print("  Taylor en F0=-0.3:", logloss(-0.3), logloss_taylor(0.5))
    print("Ejercicio:", {k: (np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3))
                         for k, v in exercise().items()})


if __name__ == "__main__":
    summary()
