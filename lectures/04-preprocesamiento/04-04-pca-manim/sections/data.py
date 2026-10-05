"""Datos y cálculos de PCA usados por la presentación.

Todos los números que aparecen en pantalla se derivan de aquí con NumPy, de
modo que la animación y el guion docente no dependen de valores copiados.
"""
from __future__ import annotations

import numpy as np


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (autovalores descendentes, autovectores en columnas, scores)."""
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Signo convencional: la mayor componente de cada vector es positiva.
    for j in range(vecs.shape[1]):
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    return vals, vecs, Xc @ vecs


# ── 1. Dos sensores del capítulo ────────────────────────────────────────
TOY = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
TOY_MEAN = TOY.mean(axis=0)                         # (3, 4)
TOY_C = TOY - TOY_MEAN                              # (-1,-1), (0,0), (1,1)
TOY_S = TOY_C.T @ TOY_C / 2                         # [[1,1],[1,1]]
TOY_VALS, TOY_VECS, TOY_Z = pca(TOY)                # λ = 2, 0 ; z = ∓1.414, 0

# ── 2. Nube 2D con ruido para barrer direcciones ────────────────────────
_rng = np.random.default_rng(7)
_raw = _rng.normal(size=(26, 2))
_raw -= _raw.mean(axis=0)
_raw = _raw @ np.linalg.inv(np.linalg.cholesky(np.cov(_raw.T))).T  # cov = I
_ANG = np.deg2rad(32)
_R = np.array([[np.cos(_ANG), -np.sin(_ANG)], [np.sin(_ANG), np.cos(_ANG)]])
CLOUD = _raw @ np.diag([1.45, 0.48]) @ _R.T         # λ = 2.1025, 0.2304 exactos
CLOUD_VALS, CLOUD_VECS, _ = pca(CLOUD)
CLOUD_S = np.cov(CLOUD.T)
CLOUD_ANGLE = float(np.arctan2(CLOUD_VECS[1, 0], CLOUD_VECS[0, 0]))


def cloud_split(theta: float) -> tuple[float, float, float]:
    """Varianza proyectada, residual y total para la dirección theta."""
    u = np.array([np.cos(theta), np.sin(theta)])
    proj = float(u @ CLOUD_S @ u)
    total = float(np.trace(CLOUD_S))
    return proj, total - proj, total


# ── 3. Nube 3D en forma de "tortilla" ───────────────────────────────────
_rng = np.random.default_rng(11)
_raw3 = _rng.normal(size=(70, 3))
_raw3 -= _raw3.mean(axis=0)
_raw3 = _raw3 @ np.linalg.inv(np.linalg.cholesky(np.cov(_raw3.T))).T


def _rot(ax: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[ax]


PANCAKE_SD = np.array([1.9, 0.95, 0.22])
_ROT3 = _rot("z", np.deg2rad(35)) @ _rot("x", np.deg2rad(28)) @ _rot("y", np.deg2rad(-12))
PANCAKE = (_raw3 * PANCAKE_SD) @ _ROT3.T
PAN_VALS, PAN_VECS, PAN_Z = pca(PANCAKE)
PAN_PVE = PAN_VALS / PAN_VALS.sum()

# ── 4. El resorte: una masa que oscila y tres cámaras ───────────────────
SPRING_ANCHOR = np.array([-2.6, 0.0, 0.0])
SPRING_DIR = np.array([1.0, 0.0, 0.0])
SPRING_REST = 2.6          # posición de equilibrio de la masa sobre el eje x
SPRING_AMP = 1.1


def mass_position(phase: float) -> np.ndarray:
    return SPRING_ANCHOR + SPRING_DIR * (SPRING_REST + SPRING_AMP * np.cos(phase))


def _camera_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    arb = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(arb, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


CAMERAS = {
    "A": np.array([0.35, -1.0, 0.25]),
    "B": np.array([1.0, 0.55, 0.40]),
    "C": np.array([-0.75, 0.9, 0.6]),
}
CAMERA_BASES = {k: _camera_basis(n) for k, n in CAMERAS.items()}

_rng = np.random.default_rng(3)
SPRING_T = np.linspace(0, 4 * np.pi, 120)
SPRING_TRUE = np.array([mass_position(p) for p in SPRING_T])
SPRING_NOISE = 0.06
SPRING_COLS = ["xA", "yA", "xB", "yB", "xC", "yC"]
SPRING_X = np.column_stack([
    SPRING_TRUE @ b + _rng.normal(0, SPRING_NOISE, len(SPRING_T))
    for k in "ABC" for b in CAMERA_BASES[k]
])
SPRING_VALS, SPRING_VECS, SPRING_Z = pca(SPRING_X)
SPRING_PVE = SPRING_VALS / SPRING_VALS.sum()
SPRING_CORR = np.corrcoef(SPRING_X.T)

# ── 5. La hélice del capítulo (un resorte visto por tres sensores) ──────
_rng = np.random.default_rng(42)
HELIX_N = 160
HELIX_T = np.linspace(-10, 10, HELIX_N)
HELIX_TRUE = np.column_stack([HELIX_T / 3.0, np.cos(HELIX_T), np.sin(HELIX_T)])
_SENSORS = [np.array([1.0, -1.0, 0.0]), np.array([10.0, 0.0, 0.1]), np.array([1.0, 1.0, 1.0])]
HELIX_X = np.column_stack([
    HELIX_TRUE @ b + _rng.normal(0, 0.12, HELIX_N)
    for n in _SENSORS for b in _camera_basis(n)
])
HELIX_VALS, HELIX_VECS, HELIX_Z = pca(HELIX_X)
HELIX_PVE = HELIX_VALS / HELIX_VALS.sum()

# ── 6. Un valor atípico gira PC1 ────────────────────────────────────────
OUTLIER_BASE = CLOUD.copy()
OUTLIER_POINT = np.array([0.6, -3.1])
_vals_o, _vecs_o, _ = pca(np.vstack([OUTLIER_BASE, OUTLIER_POINT]))
OUTLIER_ANGLE = float(np.arctan2(_vecs_o[1, 0], _vecs_o[0, 0]))

# ── 7. Práctica ─────────────────────────────────────────────────────────
PRACTICE = np.array([[1.0, 3.0], [2.0, 6.0], [4.0, 4.0], [5.0, 7.0]])
PRACTICE_VALS, PRACTICE_VECS, PRACTICE_Z = pca(PRACTICE)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("TOY S", TOY_S, "vals", TOY_VALS, "v1", TOY_VECS[:, 0], "z", TOY_Z[:, 0])
    print("CLOUD vals", CLOUD_VALS, "angle", np.rad2deg(CLOUD_ANGLE), "S", CLOUD_S)
    print("CLOUD range", CLOUD.min(0), CLOUD.max(0))
    for d in (0, 32, 90, 122):
        print(" split", d, cloud_split(np.deg2rad(d)))
    print("PAN vals", PAN_VALS, "pve", PAN_PVE, "cum2", PAN_PVE[:2].sum())
    print("PAN range", PANCAKE.min(0), PANCAKE.max(0))
    print("SPRING pve", SPRING_PVE)
    print("SPRING v1", SPRING_VECS[:, 0])
    print("SPRING corr\n", SPRING_CORR)
    print("HELIX pve", HELIX_PVE, "cum3", HELIX_PVE[:3].sum())
    print("OUTLIER angle", np.rad2deg(OUTLIER_ANGLE))
    print("PRACTICE", PRACTICE.mean(0), PRACTICE_VALS, PRACTICE_VECS[:, 0], PRACTICE_Z[:, 0],
          PRACTICE_VALS / PRACTICE_VALS.sum())
    print("PRACTICE S", np.cov(PRACTICE.T))
