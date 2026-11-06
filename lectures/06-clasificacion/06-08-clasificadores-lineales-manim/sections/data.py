"""Todos los números de la clase. ``python -m sections.data`` imprime el resumen.

Conjuntos (pequeños a propósito, para que todo se pueda hacer a mano):

- ``TWO``: 16 puntos en el plano (8 por clase), linealmente separables.
- ``NOISY``: ``TWO`` más dos puntos «cruzados»; ya no es separable.
- ``HOURS``: seis estudiantes (horas, aprobó), el ejemplo del libro.
- ``FISHER_A`` / ``FISHER_B``: los seis puntos del capítulo de Fisher.
- ``THREE``: tres clases en el plano para la logística multinomial.
- ``LINE1D``: puntos en una recta, clase +1 en el centro (no separable).
- ``RING``: anillo en el plano, clase +1 en el centro (no separable).

Convención: y ∈ {−1, +1} para la geometría y mínimos cuadrados; y ∈ {0, 1}
para la logística (y01 = (y + 1) / 2).
"""
from __future__ import annotations

from math import atan2, cos, exp, log, sin, sqrt

import numpy as np

# ---------------------------------------------------------------------------
# 1 · Separación lineal clásica
# ---------------------------------------------------------------------------
NEG = [(1.5, 2.0), (2.5, 1.0), (2.0, 3.5), (3.5, 2.5), (3.0, 4.5), (4.5, 1.5),
       (4.2, 3.4), (1.2, 5.0)]
POS = [(6.0, 5.5), (7.5, 4.5), (6.5, 7.0), (8.5, 6.0), (5.5, 6.8), (7.8, 2.8),
       (8.8, 4.0), (6.8, 3.8)]
TWO = [(x, y, -1) for x, y in NEG] + [(x, y, +1) for x, y in POS]
X_BOX = (0.0, 10.0)
Y_BOX = (0.0, 8.0)

# Un frontera «a ojo» que separa: x1 + x2 = 9 → w = (1, 1), b = −9.
EYE_W = (1.0, 1.0)
EYE_B = -9.0
# Familia de rectas que también separan (ángulo de w en grados, b ajustado
# para que la recta pase por el punto medio entre las nubes).
SEP_CENTER = (5.2, 4.3)
SEP_ANGLES = [20, 35, 45, 60, 72]


def line_through(angle_deg: float, center=SEP_CENTER):
    """(w, b) con ‖w‖ = 1 y ángulo dado, recta por ``center``."""
    a = np.radians(angle_deg)
    w = np.array([cos(a), sin(a)])
    b = -float(w @ np.array(center))
    return (float(w[0]), float(w[1])), b


def g(w, b, x) -> float:
    return float(w[0] * x[0] + w[1] * x[1] + b)


def separates(w, b, pts=TWO) -> bool:
    return all(np.sign(g(w, b, (x, y))) == c for x, y, c in pts)


PERC_W0 = (cos(np.radians(75)), sin(np.radians(75)))   # recta inicial casi horizontal
PERC_B0 = -6.0
PERC_ETA = 0.2


def perceptron(pts=TWO, w0=PERC_W0, b0=PERC_B0, eta=PERC_ETA, max_epochs=60):
    """Perceptrón de Rosenblatt en el orden de ``pts``. Devuelve la historia de
    actualizaciones: lista de (índice del punto, w, b) después de cada error."""
    w = np.array(w0, float)
    b = float(b0)
    hist = [(None, tuple(w), b)]
    for _ in range(max_epochs):
        errors = 0
        for i, (x, y, c) in enumerate(pts):
            if c * (w @ np.array([x, y]) + b) <= 0:
                w = w + eta * c * np.array([x, y])
                b = b + eta * c
                hist.append((i, tuple(w), b))
                errors += 1
        if errors == 0:
            break
    return hist


PERC = perceptron()

# ---------------------------------------------------------------------------
# 2 · Mínimos cuadrados (con y = ±1) en una variable: seis estudiantes
# ---------------------------------------------------------------------------
HOURS = [(1, 0), (2, 0), (3, 1), (4, 0), (5, 1), (6, 1)]
HX = np.array([h for h, _ in HOURS], float)
HY01 = np.array([y for _, y in HOURS], float)
HYPM = 2 * HY01 - 1


def ols_1d(x, y):
    xb, yb = x.mean(), y.mean()
    sxy = float(((x - xb) * (y - yb)).sum())
    sxx = float(((x - xb) ** 2).sum())
    b1 = sxy / sxx
    b0 = yb - b1 * xb
    return b0, b1, sxy, sxx


OLS0, OLS1, OLS_SXY, OLS_SXX = ols_1d(HX, HYPM)          # −1.4, 0.4
OLS_CUT = -OLS0 / OLS1                                    # 3.5


def ols_sse(b0, b1, x=HX, y=HYPM) -> float:
    return float(((y - b0 - b1 * x) ** 2).sum())




# Mínimos cuadrados en el plano: puntos correctos pero lejanos arrastran la recta.
MASK_NEG = [(1.0, 2.0), (1.8, 3.2), (2.4, 1.4), (1.2, 4.2), (2.8, 2.8), (2.0, 5.0),
            (3.6, 2.2)]
MASK_POS = [(4.4, 2.4), (5.0, 3.6), (5.6, 1.8), (4.6, 5.2), (5.8, 3.0), (4.0, 3.0)]
MASK_FAR = [(12.0, 7.0), (12.8, 6.6), (13.4, 7.4), (12.4, 7.8), (13.0, 6.2)]
MASK_LAMBDA = 0.01             # base separable: la logística necesita una pizca de L2
MASK_BASE = [(x, y, -1) for x, y in MASK_NEG] + [(x, y, 1) for x, y in MASK_POS]
MASK_ALL = MASK_BASE + [(x, y, 1) for x, y in MASK_FAR]
MASK_BOX = ((0.0, 14.0), (0.0, 8.5))


def ols_2d(pts):
    X = np.array([(1.0, x, y) for x, y, _ in pts])
    Y = np.array([c for *_, c in pts], float)
    return np.linalg.solve(X.T @ X, X.T @ Y)


def n_errors(beta, pts) -> int:
    return int(sum(np.sign(beta[0] + beta[1] * x + beta[2] * y) != c for x, y, c in pts))


# ---------------------------------------------------------------------------
# 3 · Fisher (datos del capítulo)
# ---------------------------------------------------------------------------
FISHER_A = np.array([(1, 2), (2, 3), (3, 3)], float)
FISHER_B = np.array([(4, 5), (5, 6), (6, 5)], float)
M_A = FISHER_A.mean(0)
M_B = FISHER_B.mean(0)


def scatter(P):
    D = P - P.mean(0)
    return D.T @ D


S_A = scatter(FISHER_A)
S_B = scatter(FISHER_B)
S_W = S_A + S_B
W_FISHER = np.linalg.solve(S_W, M_A - M_B)
W_FISHER_U = W_FISHER / np.linalg.norm(W_FISHER)
W_MEANS_U = (M_A - M_B) / np.linalg.norm(M_A - M_B)


def fisher_J(u) -> float:
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    za, zb = FISHER_A @ u, FISHER_B @ u
    return float((za.mean() - zb.mean()) ** 2 / (((za - za.mean()) ** 2).sum()
                                                 + ((zb - zb.mean()) ** 2).sum()))


J_FISHER = fisher_J(W_FISHER_U)
J_MEANS = fisher_J(W_MEANS_U)
FISHER_ANGLE = float(np.degrees(atan2(W_FISHER_U[1], W_FISHER_U[0])))
MEANS_ANGLE = float(np.degrees(atan2(W_MEANS_U[1], W_MEANS_U[0])))
ANGLES = np.arange(0, 180.0, 0.5)
J_CURVE = np.array([fisher_J((cos(np.radians(a)), sin(np.radians(a)))) for a in ANGLES])

# ---------------------------------------------------------------------------
# 4 · Regresión logística en una variable
# ---------------------------------------------------------------------------


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss(beta, X, y01) -> float:
    p = sigma(X @ beta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p)))


def newton_logit(X, y01, beta0=None, steps=25, lam=0.0):
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, float)
    path = [beta.copy()]
    n = len(y01)
    for _ in range(steps):
        p = sigma(X @ beta)
        grad = X.T @ (p - y01) / n + lam * np.r_[0, beta[1:]]
        H = (X.T * (p * (1 - p))) @ X / n + lam * np.diag(np.r_[0, np.ones(len(beta) - 1)])
        step = np.linalg.solve(H, grad)
        beta = beta - step
        path.append(beta.copy())
        if np.abs(step).max() < 1e-10:
            break
    return beta, path


def gd_logit(X, y01, beta0=(0.0, 0.0), eta=0.5, steps=400):
    beta = np.array(beta0, float)
    path = [beta.copy()]
    for _ in range(steps):
        p = sigma(X @ beta)
        beta = beta - eta * X.T @ (p - y01) / len(y01)
        path.append(beta.copy())
    return beta, path


XH = np.column_stack([np.ones(6), HX])
LOG_BETA, LOG_NEWTON = newton_logit(XH, HY01)
LOG0, LOG1 = (float(v) for v in LOG_BETA)                # −4.25, 1.21
LOG_CUT = -LOG0 / LOG1                                     # 3.5
LOG_LOSS = logloss(LOG_BETA, XH, HY01)
LOG_P = sigma(XH @ LOG_BETA)
_, LOG_GD = gd_logit(XH, HY01, eta=0.5, steps=3000)
GRAD0 = XH.T @ (sigma(XH @ np.zeros(2)) - HY01) / 6       # (0, −0.583)



def logit_2d(pts, lam=MASK_LAMBDA):
    X = np.array([(1.0, x, y) for x, y, _ in pts])
    Y = np.array([(c + 1) / 2 for *_, c in pts], float)
    return newton_logit(X, Y, lam=lam, steps=60)[0]


MASK_OLS = ols_2d(MASK_BASE)
MASK_OLS_FAR = ols_2d(MASK_ALL)
MASK_LOG = logit_2d(MASK_BASE)
MASK_LOG_FAR = logit_2d(MASK_ALL)

# Pérdidas en el plano (β0, β1) para dibujar curvas de nivel.
# Para que ambas superficies se comparen, la de mínimos cuadrados se divide entre n.


def ols_loss(beta) -> float:
    return ols_sse(beta[0], beta[1]) / len(HX)


OLS_MIN = np.array([OLS0, OLS1])
OLS_MIN_LOSS = ols_loss(OLS_MIN)


def level_curve(f, center, level, n=120, rmax=40.0):
    """Curva de nivel {β : f(β) = level} de una función convexa con mínimo en
    ``center``: para cada ángulo, bisección sobre el radio. Coordenadas (β0, β1)."""
    center = np.asarray(center, float)
    pts = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        d = np.array([cos(t), sin(t)])
        lo, hi = 0.0, 1e-3
        while f(center + hi * d) < level and hi < rmax:
            hi *= 1.6
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if f(center + mid * d) < level:
                lo = mid
            else:
                hi = mid
        pts.append(center + 0.5 * (lo + hi) * d)
    return np.array(pts)


LOSS_B0 = (-7.0, 2.0)        # ventana del plano de parámetros
LOSS_B1 = (-0.6, 2.0)
OLS_LEVELS = [0.62, 0.75, 1.0, 1.4, 2.0, 3.0]
LOG_LEVELS = [0.42, 0.45, 0.5, 0.57, 0.65]


def log_loss_h(beta) -> float:
    return logloss(np.asarray(beta), XH, HY01)


# Separabilidad perfecta: si el estudiante de 3 h no aprobó y el de 4 h sí,
# la pérdida baja sin fin al hacer la sigmoide más empinada.
SEP_Y01 = np.array([0, 0, 0, 1, 1, 1], float)
SEP_SLOPES = [0.5, 1.0, 2.0, 4.0, 8.0]
SEP_LOSSES = [logloss(np.array([-3.5 * s, s]), XH, SEP_Y01) for s in SEP_SLOPES]

# ---------------------------------------------------------------------------
# 5 · Logística con varias variables
# ---------------------------------------------------------------------------
NOISE = [(6.4, 5.0, -1), (3.8, 4.2, +1)]
NOISY = TWO + NOISE
XN = np.array([(1.0, x, y) for x, y, _ in NOISY])
YN01 = np.array([(c + 1) / 2 for *_, c in NOISY])
LOG2_BETA, LOG2_PATH = newton_logit(XN, YN01)
LOG2_B, LOG2_W1, LOG2_W2 = (float(v) for v in LOG2_BETA)
LOG2_P = sigma(XN @ LOG2_BETA)
LOG2_LOSS = logloss(LOG2_BETA, XN, YN01)
P_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]


def logit(p: float) -> float:
    return log(p / (1 - p))


# Plano en tres variables (ilustrativo): w = (1, 1, 1.2), b = −6 en el cubo [0, 4]³.
CUBE_W = (1.0, 1.0, 1.2)
CUBE_B = -6.0
_rng = np.random.default_rng(7)
_cube = _rng.uniform(0.3, 3.7, size=(60, 3))
_marg = _cube @ np.array(CUBE_W) + CUBE_B
CUBE_PTS = [(*p, 1 if m > 0 else -1) for p, m in zip(_cube, _marg) if abs(m) > 0.9][:22]

# Tres clases (softmax)
THREE = [
    (1.5, 1.5, 0), (2.5, 1.0, 0), (1.0, 2.6, 0), (2.8, 2.2, 0), (2.0, 3.4, 0),
    (7.5, 1.2, 1), (8.5, 2.0, 1), (6.8, 2.3, 1), (8.2, 3.3, 1), (7.0, 0.8, 1),
    (4.6, 6.0, 2), (5.6, 6.8, 2), (4.0, 7.0, 2), (5.4, 5.5, 2), (6.4, 6.2, 2),
]


def fit_softmax(pts=THREE, lam=0.02, eta=0.3, steps=6000):
    X = np.array([(1.0, x, y) for x, y, _ in pts])
    mu = X[:, 1:].mean(0)
    sd = X[:, 1:].std(0)
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - mu) / sd
    Y = np.eye(3)[[c for *_, c in pts]]
    W = np.zeros((3, 3))
    for _ in range(steps):
        Z = Xs @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = Xs.T @ (P - Y) / len(pts) + lam * np.vstack([np.zeros(3), W[1:]])
        W -= eta * G
    # Volver a la escala original: z_k = b_k + w_k·x
    Wo = np.zeros((3, 3))
    Wo[1:] = W[1:] / sd[:, None]
    Wo[0] = W[0] - (mu / sd) @ W[1:]
    Wo -= Wo.mean(1, keepdims=True)          # identificabilidad: suma cero
    return Wo                                 # columnas = clases; filas = (b, w1, w2)


SOFT_W = fit_softmax()


def soft_scores(x, y):
    return np.array([1.0, x, y]) @ SOFT_W


def triple_point():
    W = SOFT_W
    A = np.array([[W[1, 0] - W[1, 1], W[2, 0] - W[2, 1]],
                  [W[1, 0] - W[1, 2], W[2, 0] - W[2, 2]]])
    rhs = -np.array([W[0, 0] - W[0, 1], W[0, 0] - W[0, 2]])
    return np.linalg.solve(A, rhs)


TRIPLE = triple_point()


def soft_rays():
    """Para cada par (k, j): el rayo desde el punto triple donde z_k = z_j es máximo."""
    W = SOFT_W
    out = []
    for k, j in [(0, 1), (0, 2), (1, 2)]:
        n = np.array([W[1, k] - W[1, j], W[2, k] - W[2, j]])
        d = np.array([-n[1], n[0]])
        d /= np.linalg.norm(d)
        l = 3 - k - j
        probe = TRIPLE + d
        s = soft_scores(*probe)
        if s[l] > s[k]:
            d = -d
        out.append(((k, j), d))
    return out


SOFT_RAYS = soft_rays()

# ---------------------------------------------------------------------------
# 6 · Kernels
# ---------------------------------------------------------------------------
LINE1D = [(-3.0, -1), (-2.3, -1), (-1.6, -1), (-0.8, 1), (-0.2, 1), (0.5, 1), (1.1, 1),
          (1.8, -1), (2.5, -1), (3.1, -1)]
PHI_CUT = 2.0                 # en el espacio (x, x²): la recta x² = 2 separa
PHI_ROOT = sqrt(PHI_CUT)      # de regreso a la recta: |x| = 1.41

_rng2 = np.random.default_rng(3)
_ang_in = np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.3
_r_in = 0.45 + 0.45 * _rng2.random(9)
_ang_out = np.linspace(0, 2 * np.pi, 14, endpoint=False)
_r_out = 1.75 + 0.35 * _rng2.random(14)
RING = ([(float(r * cos(a)), float(r * sin(a)), 1) for r, a in zip(_r_in, _ang_in)]
        + [(float(r * cos(a)), float(r * sin(a)), -1) for r, a in zip(_r_out, _ang_out)])
RING_CUT = 1.8                # plano x3 = x1² + x2² = 1.8  ⇒ círculo de radio 1.34
RING_R = sqrt(RING_CUT)

# Kernel RBF en una dimensión con pérdida cuadrática (kernel ridge): solución
# cerrada α = (K + λI)⁻¹ y. El modelo es lineal en α.
RBF_GAMMA = 0.9
RBF_LAMBDA = 0.3
LX = np.array([x for x, _ in LINE1D])
LY = np.array([c for _, c in LINE1D], float)


def rbf(a, b, gamma=RBF_GAMMA):
    return np.exp(-gamma * (np.asarray(a)[:, None] - np.asarray(b)[None, :]) ** 2)


K_LINE = rbf(LX, LX)
RBF_ALPHA = np.linalg.solve(K_LINE + RBF_LAMBDA * np.eye(len(LX)), LY)


def rbf_f(x):
    x = np.atleast_1d(np.asarray(x, float))
    return rbf(x, LX) @ RBF_ALPHA


def kernel_quad(x, z) -> float:
    """(xᵀz)² en 2D, igual a φ(x)ᵀφ(z) con φ = (x1², √2 x1x2, x2²)."""
    return float((x[0] * z[0] + x[1] * z[1]) ** 2)


def phi_quad(x):
    return np.array([x[0] ** 2, sqrt(2) * x[0] * x[1], x[1] ** 2])


KQ_X = (1.0, 2.0)
KQ_Z = (3.0, 1.0)

# ---------------------------------------------------------------------------
# 7 · Ejercicio de salida
# ---------------------------------------------------------------------------
EX_W = (2.0, -1.0)
EX_B = -3.0
EX_PTS = [("P", (3.0, 1.0)), ("Q", (1.0, 2.0)), ("R", (2.0, 1.0))]


def exercise():
    rows = []
    for name, x in EX_PTS:
        z = g(EX_W, EX_B, x)
        rows.append((name, x, z, float(sigma(z))))
    return rows


EX_ROWS = exercise()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("1 · separables a ojo:", separates(EYE_W, EYE_B))
    for a in SEP_ANGLES:
        w, b = line_through(a)
        print(f"   ángulo {a:>3}: separa = {separates(w, b)}")
    print(f"   perceptrón: {len(PERC) - 1} actualizaciones; final w = "
          f"{np.round(PERC[-1][1], 3)}, b = {PERC[-1][2]:.2f}; separa = "
          f"{separates(PERC[-1][1], PERC[-1][2])}")
    print("   primeros errores:", [(i, np.round(w, 2), round(b, 2)) for i, w, b in PERC[:6]])
    print(f"2 · MCO ±1: Sxy = {OLS_SXY:.2f}, Sxx = {OLS_SXX:.2f}, β = ({OLS0:.3f}, {OLS1:.3f}),"
          f" corte = {OLS_CUT:.3f}, SSE = {ols_sse(OLS0, OLS1):.3f}")
    for name, be, pts in [("MCO base", MASK_OLS, MASK_BASE), ("MCO lejanos", MASK_OLS_FAR, MASK_ALL),
                          ("log base", MASK_LOG, MASK_BASE), ("log lejanos", MASK_LOG_FAR, MASK_ALL)]:
        print(f"   {name}: β = {np.round(be, 3)}, errores = {n_errors(be, pts)}")
    print(f"3 · Fisher: m_A = {M_A}, m_B = {M_B}\n   S_W = {S_W.tolist()}\n"
          f"   w = {W_FISHER}, û = {W_FISHER_U}, ángulo = {FISHER_ANGLE:.1f}°, J = {J_FISHER:.3f}"
          f"\n   medias û = {W_MEANS_U}, ángulo = {MEANS_ANGLE:.1f}°, J = {J_MEANS:.3f}"
          f"\n   J máx en la curva = {J_CURVE.max():.3f} a {ANGLES[J_CURVE.argmax()]:.1f}°")
    print(f"4 · Logística 1D: β = ({LOG0:.3f}, {LOG1:.3f}), corte = {LOG_CUT:.3f}, "
          f"pérdida = {LOG_LOSS:.3f}, p = {np.round(LOG_P, 2)}")
    print(f"   gradiente en 0 = {GRAD0}, Newton pasos = {len(LOG_NEWTON) - 1}")
    print("   ruta de Newton:", [np.round(b, 2).tolist() for b in LOG_NEWTON[:7]])
    print("   GD 5 pasos:", [np.round(b, 3).tolist() for b in LOG_GD[:6]],
          " tras 3000:", np.round(LOG_GD[-1], 3))
    print("   separable: pérdidas", [round(v, 4) for v in SEP_LOSSES])
    print(f"   pérdida MCO/n mín = {OLS_MIN_LOSS:.3f}")
    print(f"5 · Logística 2D: b = {LOG2_B:.3f}, w = ({LOG2_W1:.3f}, {LOG2_W2:.3f}), "
          f"pérdida = {LOG2_LOSS:.3f}, Newton pasos = {len(LOG2_PATH) - 1}")
    print("   p de los ruidosos:", np.round(LOG2_P[-2:], 3))
    print("   logit de niveles:", [round(logit(p), 3) for p in P_LEVELS])
    print("   cubo: puntos", len(CUBE_PTS), "pos", sum(c > 0 for *_, c in CUBE_PTS))
    print("   softmax W (filas b, w1, w2):\n", SOFT_W)
    pred = [int(np.argmax(soft_scores(x, y))) for x, y, _ in THREE]
    print("   softmax aciertos:", sum(p == c for p, (_, _, c) in zip(pred, THREE)), "/ 15")
    print("   punto triple:", TRIPLE, " rayos:", [(k, np.round(d, 3)) for k, d in SOFT_RAYS])
    ok = all((x * x > PHI_CUT) == (c < 0) for x, c in LINE1D)
    print(f"6 · 1D → (x, x²) separa con x² = {PHI_CUT}: {ok}, raíz = {PHI_ROOT:.3f}")
    ok = all((x * x + y * y > RING_CUT) == (c < 0) for x, y, c in RING)
    print(f"   anillo separable por r² = {RING_CUT}: {ok}, radio = {RING_R:.3f}")
    print("   radios internos máx", max(sqrt(x*x+y*y) for x, y, c in RING if c > 0),
          " externos mín", min(sqrt(x*x+y*y) for x, y, c in RING if c < 0))
    print("   RBF α =", np.round(RBF_ALPHA, 3))
    print("   RBF signo correcto:", all(np.sign(rbf_f(LX)) == LY), " f en puntos:",
          np.round(rbf_f(LX), 2))
    print(f"   kernel (xᵀz)² = {kernel_quad(KQ_X, KQ_Z)}; φᵀφ = "
          f"{phi_quad(KQ_X) @ phi_quad(KQ_Z):.3f}; φ(x) = {phi_quad(KQ_X)}, φ(z) = {phi_quad(KQ_Z)}")
    print("7 · ejercicio:", [(n, x, z, round(p, 3)) for n, x, z, p in EX_ROWS])
