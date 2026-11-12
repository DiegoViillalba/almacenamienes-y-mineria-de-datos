"""Datos y cálculos de la clase. Todos los números en pantalla salen de aquí.

Sólo usa NumPy y ``math`` (el entorno ``.venv-manim`` no incluye SciPy).
``python -m sections.data`` imprime un resumen para el guion docente.

Ejemplos:
- ``TOY``: Retraso (sin / con) × Satisfacción (alta / baja), n = 200,
  cuentas redondas para calcular a mano.
- ``SEAT``: Asiento (ventana / pasillo) × Satisfacción, casi independiente.
- ``STRENGTH``: Momento de compra × Satisfacción, n = 100 y n = 1,000.
- ``PRACTICE``: Wi-Fi a bordo × Satisfacción, n = 120 (reto final).
- ``airline_*``: encuesta sintética de 1,000 pasajeros, idéntica a la de
  ``lectures/04-preprocesamiento/04-05-chi-cuadrado.qmd`` (misma semilla).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Estadística básica
# ---------------------------------------------------------------------------

def expected(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    return np.outer(obs.sum(axis=1), obs.sum(axis=0)) / obs.sum()


def chi2_stat(obs) -> float:
    obs = np.asarray(obs, dtype=float)
    e = expected(obs)
    return float(((obs - e) ** 2 / e).sum())


def dof(obs) -> int:
    r, c = np.asarray(obs).shape
    return (r - 1) * (c - 1)


def cramer_v(obs) -> float:
    obs = np.asarray(obs, dtype=float)
    r, c = obs.shape
    return math.sqrt(chi2_stat(obs) / (obs.sum() * min(r - 1, c - 1)))


def chi2_pdf(x: float, k: int) -> float:
    if x <= 0:
        return 0.0
    return x ** (k / 2 - 1) * math.exp(-x / 2) / (2 ** (k / 2) * math.gamma(k / 2))


def chi2_sf(x: float, k: int) -> float:
    """P(χ²_k ≥ x) para k entero (forma cerrada)."""
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, k // 2):
            term *= (x / 2) / i
            total += term
        return math.exp(-x / 2) * total
    s = math.sqrt(x)
    total = math.erfc(s / math.sqrt(2))
    phi = math.exp(-x / 2) / math.sqrt(2 * math.pi)
    term = s
    for i in range(1, (k - 1) // 2 + 1):
        if i > 1:
            term *= x / (2 * i - 1)
        total += 2 * phi * term
    return total


def chi2_crit(alpha: float, k: int) -> float:
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if chi2_sf(mid, k) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Ejemplo de juguete (actos 2–5)
# ---------------------------------------------------------------------------

TOY_ROWS = ["Sin retraso", "Con retraso"]
TOY_COLS = ["Alta", "Baja"]
TOY = np.array([[80, 40], [20, 60]])
TOY_E = expected(TOY)                      # [[60, 60], [40, 40]]
TOY_RESID = TOY - TOY_E                    # ±20
TOY_CONTRIB = (TOY - TOY_E) ** 2 / TOY_E   # 6.67, 6.67, 10, 10
TOY_CHI2 = chi2_stat(TOY)                  # 33.33
TOY_V = cramer_v(TOY)                      # 0.408
TOY_P = chi2_sf(TOY_CHI2, 1)

SEAT_ROWS = ["Ventana", "Pasillo"]
SEAT = np.array([[61, 59], [39, 41]])
SEAT_CHI2 = chi2_stat(SEAT)                # 0.083
SEAT_P = chi2_sf(SEAT_CHI2, 1)

CRIT_1 = chi2_crit(0.05, 1)                # 3.84


def permutation_null(n_sims: int = 400, seed: int = 7):
    """Barajar satisfacción entre los 200 pasajeros conserva los totales.

    Devuelve las etiquetas originales (1 = Alta), las primeras permutaciones
    completas (para animarlas) y el χ² de todas las simulaciones.
    """
    rng = np.random.default_rng(seed)
    n_row0 = int(TOY[0].sum())
    labels = np.array([1] * TOY[0, 0] + [0] * TOY[0, 1]
                      + [1] * TOY[1, 0] + [0] * TOY[1, 1])
    perms, chis = [], []
    for s in range(n_sims):
        perm = rng.permutation(labels)
        a = int(perm[:n_row0].sum())
        table = np.array([[a, n_row0 - a],
                          [TOY[:, 0].sum() - a,
                           len(labels) - n_row0 - (TOY[:, 0].sum() - a)]])
        chis.append(chi2_stat(table))
        if s < 6:
            perms.append(perm)
    return labels, perms, np.array(chis)


# ---------------------------------------------------------------------------
# Significancia vs fuerza (acto 6) y reto final (acto 8)
# ---------------------------------------------------------------------------

STRENGTH_ROWS = ["Anticipado", "Último momento"]
STRENGTH = np.array([[30, 20], [20, 30]])  # n = 100: χ² = 4, V = 0.20; ×10 → χ² = 40

PRACTICE_ROWS = ["Usó Wi-Fi", "No usó"]
PRACTICE = np.array([[28, 32], [22, 38]])
PRACTICE_E = expected(PRACTICE)            # [[25, 35], [25, 35]]
PRACTICE_CONTRIB = (PRACTICE - PRACTICE_E) ** 2 / PRACTICE_E
PRACTICE_CHI2 = chi2_stat(PRACTICE)        # 1.23
PRACTICE_P = chi2_sf(PRACTICE_CHI2, 1)     # 0.27
PRACTICE_V = cramer_v(PRACTICE)            # 0.10


# ---------------------------------------------------------------------------
# Encuesta de aerolínea (misma generación que el .qmd)
# ---------------------------------------------------------------------------

def airline_dataset() -> dict[str, np.ndarray]:
    np.random.seed(42)
    n = 1000
    clase = np.random.choice(["Económica", "Business", "Primera"],
                             size=n, p=[0.65, 0.25, 0.10])
    tipo_map = {"Económica": [0.30, 0.70], "Business": [0.85, 0.15],
                "Primera": [0.70, 0.30]}
    tipo = np.array([np.random.choice(["Negocio", "Vacaciones"], p=tipo_map[c])
                     for c in clase])
    freq_map = {"Económica": [0.15, 0.45, 0.40], "Business": [0.65, 0.30, 0.05],
                "Primera": [0.75, 0.20, 0.05]}
    frecuencia = np.array([np.random.choice(["Frecuente", "Ocasional", "Primera vez"],
                                            p=freq_map[c]) for c in clase])
    retraso_map = {"Negocio": [0.50, 0.30, 0.20], "Vacaciones": [0.45, 0.35, 0.20]}
    retraso = np.array([np.random.choice(["Sin retraso", "Leve", "Grave"],
                                         p=retraso_map[t]) for t in tipo])
    satisf = []
    for c, r in zip(clase, retraso):
        if c == "Primera" and r == "Sin retraso":
            p = [0.05, 0.15, 0.80]
        elif c == "Business" and r != "Grave":
            p = [0.10, 0.35, 0.55]
        elif c == "Económica" and r == "Grave":
            p = [0.60, 0.30, 0.10]
        elif c == "Económica" and r == "Sin retraso":
            p = [0.20, 0.45, 0.35]
        else:
            p = [0.30, 0.40, 0.30]
        satisf.append(np.random.choice(["Baja", "Media", "Alta"], p=p))
    satisf = np.array(satisf)
    asiento = np.random.choice(["Ventana", "Pasillo", "Centro"],
                               size=n, p=[0.45, 0.40, 0.15])

    def p_norm():
        p = np.array([0.50, 0.50]) + np.random.uniform(-0.03, 0.03, 2)
        return p / p.sum()

    momento = np.array([np.random.choice(["Anticipado", "Último momento"], p=p_norm())
                        for _ in range(n)])
    return {"Clase": clase, "Tipo de viaje": tipo, "Frecuencia": frecuencia,
            "Retraso": retraso, "Satisfacción": satisf, "Asiento": asiento,
            "Momento de compra": momento}


LEVELS = {
    "Clase": ["Económica", "Business", "Primera"],
    "Tipo de viaje": ["Negocio", "Vacaciones"],
    "Frecuencia": ["Frecuente", "Ocasional", "Primera vez"],
    "Retraso": ["Sin retraso", "Leve", "Grave"],
    "Satisfacción": ["Baja", "Media", "Alta"],
    "Asiento": ["Ventana", "Pasillo", "Centro"],
    "Momento de compra": ["Anticipado", "Último momento"],
}
PREDICTORS = ["Clase", "Tipo de viaje", "Frecuencia", "Retraso",
              "Asiento", "Momento de compra"]


def crosstab(a: np.ndarray, b: np.ndarray, la: list[str], lb: list[str]) -> np.ndarray:
    return np.array([[int(np.sum((a == x) & (b == y))) for y in lb] for x in la])


AIR = airline_dataset()


def _table(v1: str, v2: str) -> np.ndarray:
    return crosstab(AIR[v1], AIR[v2], LEVELS[v1], LEVELS[v2])


AIR_RETRASO = _table("Retraso", "Satisfacción")   # 3 × 3 para grados de libertad


def _rank():
    rows = []
    for v in PREDICTORS:
        t = _table(v, "Satisfacción")
        x2 = chi2_stat(t)
        rows.append({"var": v, "chi2": x2, "dof": dof(t),
                     "p": chi2_sf(x2, dof(t)), "V": cramer_v(t)})
    return sorted(rows, key=lambda r: -r["V"])


AIR_RANK = _rank()
USEFUL = [r["var"] for r in AIR_RANK if r["p"] < 0.05]
AIR_PAIRS = {(a, b): cramer_v(_table(a, b)) for a, b in combinations(USEFUL, 2)}


def pair_v(a: str, b: str) -> float:
    return AIR_PAIRS.get((a, b), AIR_PAIRS.get((b, a), 1.0))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("TOY E\n", TOY_E, "\ncontrib\n", TOY_CONTRIB)
    print(f"TOY χ²={TOY_CHI2:.3f} p={TOY_P:.2e} V={TOY_V:.3f}")
    print(f"SEAT χ²={SEAT_CHI2:.4f} p={SEAT_P:.3f}  crit(1)={CRIT_1:.3f}")
    _, _, chis = permutation_null()
    print(f"null sims: max={chis.max():.2f} mean={chis.mean():.2f} "
          f">=3.84: {(chis >= CRIT_1).mean():.3f} >=obs: {(chis >= TOY_CHI2).sum()}")
    print(f"STRENGTH χ²={chi2_stat(STRENGTH):.2f} p={chi2_sf(chi2_stat(STRENGTH), 1):.4f} "
          f"×10 χ²={chi2_stat(STRENGTH * 10):.1f} p={chi2_sf(chi2_stat(STRENGTH * 10), 1):.2e} "
          f"V={cramer_v(STRENGTH):.3f}")
    print("PRACTICE E\n", PRACTICE_E, "\ncontrib\n", PRACTICE_CONTRIB)
    print(f"PRACTICE χ²={PRACTICE_CHI2:.3f} p={PRACTICE_P:.4f} V={PRACTICE_V:.3f}")
    print("Retraso×Satisf\n", AIR_RETRASO, f"χ²={chi2_stat(AIR_RETRASO):.2f}")
    for r in AIR_RANK:
        print(f"{r['var']:20s} χ²={r['chi2']:7.2f} gl={r['dof']} p={r['p']:.4f} V={r['V']:.3f}")
    print("USEFUL", USEFUL)
    for k, v in AIR_PAIRS.items():
        print(k, round(v, 3))
    for k in (1, 2, 4):
        print(f"crit gl={k}: {chi2_crit(0.05, k):.3f}")
