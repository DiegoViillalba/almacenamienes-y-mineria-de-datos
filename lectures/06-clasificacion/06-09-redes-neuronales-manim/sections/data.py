"""Todos los números de la clase. ``python -m sections.data`` imprime el resumen.

Contenido:

- ``NEURON_*``: una neurona con tres entradas, calculada a mano.
- ``BUMP_*``: cuatro ReLU en una recta que forman un «escalón» (1D).
- ``SQUARE`` y ``square_score``: un enclave cuadrado hecho con cuatro neuronas.
- El mapa **estilizado** de Baarle (Baarle-Hertog, Bélgica, dentro de
  Baarle-Nassau, Países Bajos): polígonos de Bélgica, enclaves y un
  contraenclave neerlandés. No es el catastro real; conserva la propiedad que
  importa: la región belga está partida en varios pedazos y uno tiene un hueco.
- Una red 2 → 16 → 16 → 1 con ReLU entrenada en NumPy (Adam, entropía cruzada)
  y una regresión logística como referencia lineal. Los pesos se guardan en
  ``baarle_mlp.json`` para que el render sea determinista; si el archivo no
  existe, se reentrena con la misma semilla.
- ``BACKPROP``: una red de una neurona oculta, hacia adelante y hacia atrás.
- ``IMG`` / ``KERNEL``: una imagen 6 × 6 con un borde vertical y su convolución.

Convención de clases: 0 = Bélgica (azul, círculo), 1 = Países Bajos (naranja,
cuadrado).
"""
from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1 · Una neurona a mano
# ---------------------------------------------------------------------------
NEURON_X = (2.0, 1.0, 3.0)
NEURON_W = (0.5, -1.0, 0.25)
NEURON_B = 0.5
NEURON_Z = sum(x * w for x, w in zip(NEURON_X, NEURON_W)) + NEURON_B   # 1.25
NEURON_RELU = max(0.0, NEURON_Z)
NEURON_SIG = 1.0 / (1.0 + exp(-NEURON_Z))                             # 0.777

# ---------------------------------------------------------------------------
# 2 · Cuatro ReLU forman un escalón en una recta
# f(x) = ReLU(x−3) − ReLU(x−4) − ReLU(x−6) + ReLU(x−7)
# ---------------------------------------------------------------------------
BUMP_KNOTS = (3.0, 4.0, 6.0, 7.0)
BUMP_SIGNS = (+1, -1, -1, +1)
BUMP_POS = [1.0, 2.0, 2.5, 8.0, 8.8, 9.5]      # Países Bajos (fuera)
BUMP_NEG = [4.2, 4.9, 5.4, 5.8]                 # Bélgica (dentro)


def relu(v):
    return np.maximum(0.0, v)


def bump(x):
    return sum(s * relu(np.asarray(x, dtype=float) - k) for k, s in zip(BUMP_KNOTS, BUMP_SIGNS))


# ---------------------------------------------------------------------------
# 3 · Un enclave cuadrado con cuatro neuronas
# h_k = ReLU(n_k · x − c_k)  (normales hacia afuera), score = 1 − 2 Σ h_k
# ---------------------------------------------------------------------------
SQUARE = (3.0, 7.0, 2.0, 5.0)          # x0, x1, y0, y1 en un lienzo de 10 × 7
SQUARE_NEURONS = [((-1.0, 0.0), -3.0),  # izquierda:  ReLU(3 − x)
                  ((1.0, 0.0), 7.0),    # derecha:    ReLU(x − 7)
                  ((0.0, -1.0), -2.0),  # abajo:      ReLU(2 − y)
                  ((0.0, 1.0), 5.0)]    # arriba:     ReLU(y − 5)
SQUARE_GAIN = 2.0


def square_hidden(x, y):
    return [relu(n[0] * x + n[1] * y - c) for n, c in SQUARE_NEURONS]


def square_score(x, y):
    """> 0 dentro del cuadrado (Bélgica), < 0 lejos de él."""
    return 1.0 - SQUARE_GAIN * sum(square_hidden(x, y))


# ---------------------------------------------------------------------------
# 4 · El mapa estilizado de Baarle (lienzo 10 × 7)
# ---------------------------------------------------------------------------
W_MAP, H_MAP = 10.0, 7.0
BE_SOUTH = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.5), (8.2, 1.1), (6.4, 1.7), (4.4, 1.2),
            (2.3, 1.6), (0.0, 1.2)]
ENCLAVES = {
    "H1": [(1.3, 3.0), (3.4, 2.6), (4.3, 3.9), (3.3, 5.5), (1.6, 5.0)],
    "H2": [(5.8, 4.4), (7.5, 4.1), (7.9, 5.8), (6.1, 6.2)],
    "H3": [(7.4, 2.3), (9.2, 2.6), (8.3, 3.7)],
    "H4": [(5.0, 2.4), (6.0, 2.3), (6.1, 3.3), (5.1, 3.4)],
}
COUNTER = {"N1": [(2.4, 3.6), (3.2, 3.4), (3.4, 4.2), (2.6, 4.4)]}   # NL dentro de H1


def _inside(poly, x, y):
    """Prueba par-impar vectorizada."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    res = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        res ^= cond & (x < xin)
    return res


def country(x, y):
    """0 = Bélgica, 1 = Países Bajos."""
    be = _inside(BE_SOUTH, x, y)
    for p in ENCLAVES.values():
        be |= _inside(p, x, y)
    for p in COUNTER.values():
        be &= ~_inside(p, x, y)
    return np.where(be, 0, 1)


_rng = np.random.default_rng(7)
N_TRAIN = 4000
TRAIN_X = np.column_stack([_rng.uniform(0, W_MAP, N_TRAIN), _rng.uniform(0, H_MAP, N_TRAIN)])
TRAIN_Y = country(TRAIN_X[:, 0], TRAIN_X[:, 1]).astype(float)

# Puntos que se dibujan: una malla ligeramente perturbada (se ve ordenada).
_gx, _gy = np.meshgrid(np.linspace(0.25, W_MAP - 0.25, 26), np.linspace(0.25, H_MAP - 0.25, 18))
_jit = np.random.default_rng(3).uniform(-0.1, 0.1, (2,) + _gx.shape)
SHOW_X = np.column_stack([(_gx + _jit[0]).ravel(), (_gy + _jit[1]).ravel()])
SHOW_Y = country(SHOW_X[:, 0], SHOW_X[:, 1])


def to_unit(P):
    """Coordenadas del mapa → entrada de la red (centradas, escala ≈ 1)."""
    P = np.asarray(P, float)
    return np.column_stack([(P[:, 0] - 5.0) / 3.5, (P[:, 1] - 3.5) / 3.5])


# ---------------------------------------------------------------------------
# 5 · Entrenamiento (NumPy, Adam)
# ---------------------------------------------------------------------------
SIZES = (2, 16, 16, 1)
EPOCHS = 6000
SNAPSHOTS = (0, 25, 100, 400, 1500, EPOCHS)
LR = 0.01
CACHE = Path(__file__).with_name("baarle_mlp.json")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def forward(params, U, *, linear: bool = False):
    """Devuelve (logit, [activaciones ocultas]). ``linear`` quita las ReLU."""
    acts = []
    a = U
    n = len(params) // 2
    for i in range(n):
        W, b = params[2 * i], params[2 * i + 1]
        z = a @ W + b
        if i < n - 1:
            a = z if linear else relu(z)
            acts.append((z, a))
        else:
            return z[:, 0], acts


def _train(sizes, *, linear=False, epochs=EPOCHS, seed=0, snapshots=()):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    U, Y = to_unit(TRAIN_X), TRAIN_Y
    snaps, losses = {}, []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(epochs + 1):
        if t in snapshots:
            snaps[t] = [p.copy() for p in params]
        # adelante
        a_list = [U]
        z_list = []
        a = U
        nl = len(params) // 2
        for i in range(nl):
            z = a @ params[2 * i] + params[2 * i + 1]
            z_list.append(z)
            a = z if (i == nl - 1 or linear) else relu(z)
            a_list.append(a)
        p = sigmoid(z_list[-1][:, 0])
        loss = -np.mean(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12))
        losses.append(float(loss))
        if t == epochs:
            break
        # atrás
        g = ((p - Y) / len(Y))[:, None]
        grads = [None] * len(params)
        for i in reversed(range(nl)):
            grads[2 * i] = a_list[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            if i:
                g = g @ params[2 * i].T
                if not linear:
                    g = g * (z_list[i - 1] > 0)
        for k, gk in enumerate(grads):
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mh = m[k] / (1 - b1 ** (t + 1))
            vh = v[k] / (1 - b2 ** (t + 1))
            params[k] = params[k] - LR * mh / (np.sqrt(vh) + eps)
    return params, snaps, losses


def _dump(params):
    return [p.tolist() for p in params]


def _load(lst):
    return [np.array(p) for p in lst]


def _build_cache():
    mlp, snaps, losses = _train(SIZES, snapshots=SNAPSHOTS)
    logit, _, _ = _train((2, 1), epochs=3000)
    deep_lin, _, _ = _train(SIZES, linear=True, epochs=3000, seed=1)
    data = {"mlp": _dump(mlp), "snapshots": {str(k): _dump(v) for k, v in snaps.items()},
            "losses": losses[::25] + [losses[-1]], "logistic": _dump(logit),
            "deep_linear": _dump(deep_lin)}
    CACHE.write_text(json.dumps(data))
    return data


_cache = json.loads(CACHE.read_text()) if CACHE.exists() else _build_cache()
MLP = _load(_cache["mlp"])
SNAPS = {int(k): _load(v) for k, v in _cache["snapshots"].items()}
LOSSES = _cache["losses"]           # una muestra cada 25 épocas
LOGISTIC = _load(_cache["logistic"])
DEEP_LINEAR = _load(_cache["deep_linear"])


def predict(params, P, *, linear=False):
    logit, _ = forward(params, to_unit(P), linear=linear)
    return np.where(logit > 0, 1, 0)


def accuracy(params, *, linear=False):
    return float(np.mean(predict(params, TRAIN_X, linear=linear) == TRAIN_Y))


def collapse(params):
    """Producto de las capas lineales: W_eq, b_eq tales que red(x) = x W_eq + b_eq."""
    W = np.eye(2)
    b = np.zeros(2)
    for i in range(len(params) // 2):
        b = b @ params[2 * i] + params[2 * i + 1]
        W = W @ params[2 * i]
    return W[:, 0], float(b[0])


def unit_line_to_map(w, b):
    """w·u + b = 0 en coordenadas de la red → (a, c, d) con a x + c y = d en el mapa."""
    # u = ((x − 5)/3.5, (y − 3.5)/3.5)
    a, c = w[0] / 3.5, w[1] / 3.5
    d = a * 5.0 + c * 3.5 - b
    return a, c, d


def first_layer_lines(params=None):
    params = MLP if params is None else params
    W, b = params[0], params[1]
    return [unit_line_to_map(W[:, k], b[k]) for k in range(W.shape[1])]


def grid(nx=500, ny=350):
    xs = (np.arange(nx) + 0.5) / nx * W_MAP
    ys = (np.arange(ny) + 0.5) / ny * H_MAP
    X, Y = np.meshgrid(xs, ys[::-1])    # fila 0 = arriba
    return X, Y


def logit_grid(params, nx=500, ny=350, *, linear=False):
    X, Y = grid(nx, ny)
    P = np.column_stack([X.ravel(), Y.ravel()])
    z, _ = forward(params, to_unit(P), linear=linear)
    return z.reshape(X.shape)


def region_ids(params=None, nx=500, ny=350, layers=2):
    """Patrón de activación (qué ReLU están encendidas) en cada píxel."""
    params = MLP if params is None else params
    X, Y = grid(nx, ny)
    P = np.column_stack([X.ravel(), Y.ravel()])
    _, acts = forward(params, to_unit(P))
    bits = np.concatenate([(z > 0) for z, _ in acts[:layers]], axis=1)
    keys = np.packbits(bits, axis=1)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    return ids.reshape(X.shape)


def representation(params=None, P=None):
    """Lleva cada punto a 2D: (logit, componente principal de h2 ⊥ a v).

    En esas coordenadas la frontera de la red es la recta vertical logit = 0.
    """
    params = MLP if params is None else params
    P = SHOW_X if P is None else P
    _, acts = forward(params, to_unit(TRAIN_X))
    H = acts[-1][1]
    v = params[-2][:, 0]
    vn = v / np.linalg.norm(v)
    R = H - np.outer(H @ vn, vn)
    mu = R.mean(0)
    _, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
    pc = Vt[0]
    logit, acts = forward(params, to_unit(P))
    H = acts[-1][1]
    second = (H - mu) @ pc
    return np.column_stack([logit, second])


# ---------------------------------------------------------------------------
# 6 · Retropropagación con números
# x → z = w x + b → a = ReLU(z) → s = v a + c → ŷ = σ(s) → L = −log ŷ  (y = 1)
# ---------------------------------------------------------------------------
def backprop_example(x=2.0, y=1.0, w=0.5, b=-0.5, v=1.5, c=-1.0, eta=0.5):
    z = w * x + b
    a = max(0.0, z)
    s = v * a + c
    yh = 1 / (1 + exp(-s))
    L = -(y * log(yh) + (1 - y) * log(1 - yh))
    dL_ds = yh - y
    dL_dv = dL_ds * a
    dL_dc = dL_ds
    dL_da = dL_ds * v
    dL_dz = dL_da * (1.0 if z > 0 else 0.0)
    dL_dw = dL_dz * x
    dL_db = dL_dz
    w2, b2, v2, c2 = w - eta * dL_dw, b - eta * dL_db, v - eta * dL_dv, c - eta * dL_dc
    z2 = w2 * x + b2
    a2 = max(0.0, z2)
    s2 = v2 * a2 + c2
    yh2 = 1 / (1 + exp(-s2))
    L2 = -log(yh2)
    return dict(x=x, y=y, w=w, b=b, v=v, c=c, eta=eta, z=z, a=a, s=s, yh=yh, L=L,
                dL_ds=dL_ds, dL_dv=dL_dv, dL_dc=dL_dc, dL_da=dL_da, dL_dz=dL_dz,
                dL_dw=dL_dw, dL_db=dL_db, w2=w2, b2=b2, v2=v2, c2=c2, yh2=yh2, L2=L2)


BACKPROP = backprop_example()

# ---------------------------------------------------------------------------
# 7 · Convolución: imagen 6 × 6 con un borde vertical
# ---------------------------------------------------------------------------
IMG = np.array([[0, 0, 0, 1, 1, 1]] * 6, dtype=int)
KERNEL = np.array([[-1, 0, 1]] * 3, dtype=int)


def conv2d(img=IMG, k=KERNEL):
    n = img.shape[0] - k.shape[0] + 1
    out = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            out[i, j] = int((img[i:i + 3, j:j + 3] * k).sum())
    return out


FEATURE = conv2d()
DENSE_IN = 224 * 224 * 3
DENSE_HIDDEN = 1000
DENSE_PARAMS = DENSE_IN * DENSE_HIDDEN
CONV_PARAMS = 3 * 3 * 3 * 64


def n_params(sizes=SIZES):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


if __name__ == "__main__":
    print(f"neurona: z = {NEURON_Z:.3f}, ReLU = {NEURON_RELU:.3f}, σ = {NEURON_SIG:.3f}")
    xs = np.array([3.5, 4, 5, 6, 6.5])
    print("escalón:", dict(zip(xs.tolist(), bump(xs).tolist())))
    print("cuadrado: centro", square_score(5, 3.5), "afuera", square_score(8, 3.5))
    print(f"proporción Bélgica en entrenamiento: {np.mean(TRAIN_Y == 0):.3f}")
    print(f"exactitud logística: {accuracy(LOGISTIC):.3f}")
    print(f"exactitud red lineal profunda: {accuracy(DEEP_LINEAR, linear=True):.3f}")
    print(f"exactitud MLP 2-16-16-1: {accuracy(MLP):.3f}  parámetros: {n_params()}")
    for k, p in SNAPS.items():
        print(f"  época {k:5d}: exactitud {accuracy(p):.3f}")
    print(f"pérdida inicial {LOSSES[0]:.3f} → final {LOSSES[-1]:.3f}")
    print("regiones lineales (capa 1):", region_ids(layers=1).max() + 1,
          " (capas 1+2):", region_ids(layers=2).max() + 1)
    bp = BACKPROP
    print("backprop:", {k: round(v, 4) for k, v in bp.items()})
    print("convolución:\n", FEATURE)
    print(f"densa: {DENSE_PARAMS:,}  conv: {CONV_PARAMS:,}")
