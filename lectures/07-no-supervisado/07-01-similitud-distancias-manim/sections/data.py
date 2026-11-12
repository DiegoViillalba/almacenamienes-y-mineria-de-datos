"""Datos y cálculos de la clase, todos con NumPy y semillas fijas.

Los números del ejemplo de Ana, Mahalanobis, coseno, UMAP en la recta y la
concentración de distancias reproducen los del capítulo
``unidades/clustering/capitulo_clustering_intro.qmd`` y
``unidades/clustering/capitulo_tsne_umap.qmd``. ``python -m sections.data``
imprime un resumen para verificarlos.
"""
from __future__ import annotations

import math

import numpy as np

# ── 1 · Ana y sus cuatro vecinos (variables mixtas) ─────────────────────────
CLIENTS = [  # nombre, edad, ingreso (miles), canal, tarjeta
    ("Ana", 34, 28, "app", "sí"),
    ("Beto", 36, 75, "app", "no"),
    ("Carla", 58, 30, "tienda", "sí"),
    ("Daniel", 31, 26, "web", "no"),
    ("Elena", 55, 70, "tienda", "no"),
]
_ages = np.array([c[1] for c in CLIENTS], float)
_inc = np.array([c[2] for c in CLIENTS], float)
R_AGE = _ages.max() - _ages.min()      # 27
R_INC = _inc.max() - _inc.min()        # 49


def gower_parts(i: int, j: int) -> list[float]:
    a, b = CLIENTS[i], CLIENTS[j]
    return [abs(a[1] - b[1]) / R_AGE, abs(a[2] - b[2]) / R_INC,
            float(a[3] != b[3]), float(a[4] != b[4])]


def gower(i: int, j: int, w=(1, 1, 1, 1)) -> float:
    parts = gower_parts(i, j)
    return sum(wk * p for wk, p in zip(w, parts)) / sum(w)


OTHERS = [1, 2, 3, 4]
GOWER_EQ = {CLIENTS[j][0]: gower(0, j) for j in OTHERS}
W_BIZ = (1, 1, 0.5, 0.5)
GOWER_W = {CLIENTS[j][0]: gower(0, j, W_BIZ) for j in OTHERS}

# Sólo numéricas, normalizadas por rango: Ana ↔ Daniel
_dn = np.array(gower_parts(0, 3)[:2])
ANA_DANIEL = {"L1": float(_dn.sum()), "L2": float(np.sqrt((_dn ** 2).sum())),
              "Linf": float(_dn.max()), "diffs": _dn}
NUM_ONLY = {CLIENTS[j][0]: float(np.linalg.norm(gower_parts(0, j)[:2])) for j in OTHERS}

# ── 2 · Escala como transformación ──────────────────────────────────────────
SCALE_Q = np.array([0.0, 0.0])
SCALE_PTS = np.array([[0.3, 1.1], [1.0, 0.3], [-1.5, 1.2], [-0.4, -1.8],
                      [1.6, -1.2], [-1.3, -0.7]])
STRETCH = 2.5                                   # el eje x se mide en otra unidad
_S = np.diag([STRETCH, 1.0])
SCALE_D0 = np.linalg.norm(SCALE_PTS - SCALE_Q, axis=1)
SCALE_D1 = np.linalg.norm(SCALE_PTS @ _S.T - SCALE_Q, axis=1)
SCALE_NN0 = int(SCALE_D0.argmin())              # vecino antes
SCALE_NN1 = int(SCALE_D1.argmin())              # vecino después

# ── 3 · Mahalanobis: la nube correlacionada ─────────────────────────────────
RHO = 0.8
SIGMA = np.array([[1.0, RHO], [RHO, 1.0]])
_vals, _vecs = np.linalg.eigh(SIGMA)
SIGMA_INV_HALF = _vecs @ np.diag(1 / np.sqrt(_vals)) @ _vecs.T
_rng = np.random.default_rng(7)
_cloud = _rng.multivariate_normal([0, 0], SIGMA, size=260)
_md = np.sqrt(np.einsum("ij,jk,ik->i", _cloud, np.linalg.inv(SIGMA), _cloud))
MAHA_CLOUD = _cloud[_md < 2.1][:150]
MAHA_A = np.array([1.0, 1.0])
MAHA_B = np.array([1.0, -1.0])
MAHA_DA = float(np.linalg.norm(SIGMA_INV_HALF @ MAHA_A))   # 1.05
MAHA_DB = float(np.linalg.norm(SIGMA_INV_HALF @ MAHA_B))   # 3.16

# ── 4 · Coseno: tres documentos (datos, minería, fútbol) ────────────────────
DOCS = {"d1": np.array([3.0, 2.0, 0.0]), "d2": np.array([6.0, 4.0, 0.0]),
        "d3": np.array([0.0, 1.0, 3.0])}


def cos_sim(x, y) -> float:
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


EUC_12 = float(np.linalg.norm(DOCS["d1"] - DOCS["d2"]))   # 3.61
EUC_13 = float(np.linalg.norm(DOCS["d1"] - DOCS["d3"]))   # 4.36
EUC_23 = float(np.linalg.norm(DOCS["d2"] - DOCS["d3"]))   # 7.35
COS_D13 = 1 - cos_sim(DOCS["d1"], DOCS["d3"])               # 0.825
ANGLE_13 = math.degrees(math.acos(cos_sim(DOCS["d1"], DOCS["d3"])))  # 79.9°
# Contraejemplo de la desigualdad triangular para 1 − cos: 0°, 45°, 90°
COS_TRI = {"xz": 1 - math.cos(math.pi / 2), "xy": 1 - math.cos(math.pi / 4)}

# ── 5 · Dos grupos que sólo se separan en la tercera dimensión ──────────────
_rng = np.random.default_rng(11)
PROJ_A = _rng.normal([0.0, 0.0, -1.25], [0.55, 0.45, 0.28], size=(28, 3))
PROJ_B = _rng.normal([0.25, 0.15, 1.25], [0.55, 0.45, 0.28], size=(28, 3))

# ── 6 · La esfera dentro del cubo ───────────────────────────────────────────
_rng = np.random.default_rng(3)
MC_SQUARE = _rng.uniform(-1, 1, size=(420, 2))
MC_INSIDE = np.linalg.norm(MC_SQUARE, axis=1) <= 1


def ball_fraction(p: int) -> float:
    """Volumen de la bola unitaria entre el volumen del cubo [−1, 1]^p."""
    return math.pi ** (p / 2) / math.gamma(p / 2 + 1) / 2 ** p


BALL_DIMS = [1, 2, 3, 5, 10, 20]
BALL_FRAC = {p: ball_fraction(p) for p in BALL_DIMS}
SHELL_DIMS = [2, 3, 10, 50, 100]
SHELL_FRAC = {p: 1 - 0.95 ** p for p in SHELL_DIMS}

# Concentración: mismo experimento y semilla que el capítulo
CONC_DIMS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
CONC_RATIO: dict[int, float] = {}
CONC_DIST: dict[int, np.ndarray] = {}
_rng = np.random.default_rng(0)
for _p in CONC_DIMS:
    _X = _rng.random((500, _p))
    _q = _rng.random(_p)
    _d = np.linalg.norm(_X - _q, axis=1)
    CONC_RATIO[_p] = float((_d.max() - _d.min()) / _d.min())
    CONC_DIST[_p] = _d / _d.mean()
HIST_DIMS = [2, 10, 100, 1000]
HIST_BINS = np.linspace(0, 2.0, 41)
HIST = {p: np.histogram(np.clip(CONC_DIST[p], 0, 1.999), bins=HIST_BINS)[0] / 500
        for p in HIST_DIMS}

# ── 7 · UMAP: el rollo y la regla de cada punto ─────────────────────────────
N_T, N_H = 36, 8
T_MIN, T_MAX = 1.5 * math.pi, 4.5 * math.pi
H_MAX = 18.0


def _arc(t):
    return 0.5 * (t * np.sqrt(1 + t * t) + np.arcsinh(t))


# Muestreo uniforme en longitud de arco: separación constante sobre la hoja,
# así ningún vecino k-NN salta a la vuelta siguiente del rollo.
_tgrid = np.linspace(T_MIN, T_MAX, 20000)
_sgrid = _arc(_tgrid) - _arc(T_MIN)
_step = _sgrid[-1] / (N_T - 1)
_rng = np.random.default_rng(5)
_ss, _hh = np.meshgrid(np.linspace(0, _sgrid[-1], N_T), np.linspace(0, H_MAX, N_H))
ROLL_S = (_ss.ravel() + _rng.uniform(-0.15, 0.15, N_T * N_H) * _step).clip(0, _sgrid[-1])
ROLL_H = (_hh.ravel() + _rng.uniform(-0.4, 0.4, N_T * N_H)).clip(0, H_MAX)
ROLL_T = np.interp(ROLL_S, _sgrid, _tgrid)
ROLL_3D = np.c_[ROLL_T * np.cos(ROLL_T), ROLL_H - H_MAX / 2, ROLL_T * np.sin(ROLL_T)]
# longitud sobre la hoja = ROLL_S
ROLL_FLAT = np.c_[ROLL_S - ROLL_S.max() / 2, ROLL_H - H_MAX / 2]
ROLL_COLOR_T = (ROLL_T - T_MIN) / (T_MAX - T_MIN)

K_NN = 5
_D = np.linalg.norm(ROLL_3D[:, None] - ROLL_3D[None], axis=2)
np.fill_diagonal(_D, np.inf)
_knn = np.argsort(_D, axis=1)[:, :K_NN]
ROLL_EDGES = sorted({(min(i, j), max(i, j)) for i in range(len(ROLL_3D)) for j in _knn[i]})
ROLL_MAX_DT = max(abs(ROLL_T[i] - ROLL_T[j]) for i, j in ROLL_EDGES)
assert ROLL_MAX_DT < math.pi, "el grafo k-NN atraviesa capas del rollo"

# Dos puntos del rollo: cerca en línea recta, lejos sobre la hoja
_cands = np.where(np.abs(ROLL_H - H_MAX / 2) < 2.0)[0]
_best = None
for _i in _cands:
    for _j in _cands:
        if ROLL_T[_j] - ROLL_T[_i] > 5.5:          # vuelta siguiente del rollo
            _r = _D[_i, _j] / abs(ROLL_S[_j] - ROLL_S[_i])
            if _best is None or _r < _best[0]:
                _best = (_r, int(_i), int(_j))
ROLL_PA, ROLL_PB = _best[1], _best[2]
ROLL_EUC_AB = float(_D[ROLL_PA, ROLL_PB])
ROLL_GEO_AB = float(abs(ROLL_S[ROLL_PB] - ROLL_S[ROLL_PA]))

# La recta del capítulo de t-SNE/UMAP: A, B, C, D, E en 0, 1, 3, 5, 6; k = 3
LINE_POS = {"A": 0, "B": 1, "C": 3, "D": 5, "E": 6}
UMAP_K = 3
UMAP_RHO = 1.0
_u = (-1 + math.sqrt(1 + 4 * (math.log2(UMAP_K) - 1))) / 2   # e^{−2/σ}
UMAP_SIGMA = -2 / math.log(_u)                                # 2.27
UMAP_W = {"B": 1.0, "C": _u, "D": _u ** 2}                    # 1, 0.414, 0.171


if __name__ == "__main__":
    print("Gower (pesos iguales):", {k: round(v, 3) for k, v in GOWER_EQ.items()})
    print("Gower (1,1,.5,.5):   ", {k: round(v, 3) for k, v in GOWER_W.items()})
    print("Sólo numéricas L2:    ", {k: round(v, 3) for k, v in NUM_ONLY.items()})
    print("Ana–Daniel L1/L2/Linf:", round(ANA_DANIEL["L1"], 3), round(ANA_DANIEL["L2"], 3),
          round(ANA_DANIEL["Linf"], 3))
    print("Escala: vecino antes", SCALE_NN0, SCALE_D0.round(3), "después", SCALE_NN1,
          SCALE_D1.round(3))
    print("Mahalanobis a, b:", round(MAHA_DA, 3), round(MAHA_DB, 3), "nube", len(MAHA_CLOUD))
    print("Euclidiana d1d2, d1d3, d2d3:", round(EUC_12, 2), round(EUC_13, 2), round(EUC_23, 2))
    print("Coseno d1d3:", round(COS_D13, 3), "ángulo", round(ANGLE_13, 1))
    print("Triángulo coseno: d(x,z) =", COS_TRI["xz"], " d(x,y)+d(y,z) =", 2 * COS_TRI["xy"])
    print("Fracción bola/cubo:", {p: f"{v:.3g}" for p, v in BALL_FRAC.items()})
    print("Monte Carlo 2D:", MC_INSIDE.sum(), "/", len(MC_INSIDE))
    print("Cáscara 5 %:", {p: round(v, 3) for p, v in SHELL_FRAC.items()})
    print("Contraste:", {p: round(v, 3) for p, v in CONC_RATIO.items()})
    print("Rollo:", len(ROLL_3D), "puntos,", len(ROLL_EDGES), "aristas, máx Δt en aristas",
          round(ROLL_MAX_DT, 2))
    print("A, B del rollo:", ROLL_PA, ROLL_PB, "recta", round(ROLL_EUC_AB, 2), "sobre la hoja",
          round(ROLL_GEO_AB, 2))
    print("UMAP recta: σ_A =", round(UMAP_SIGMA, 3), {k: round(v, 3) for k, v in UMAP_W.items()})
