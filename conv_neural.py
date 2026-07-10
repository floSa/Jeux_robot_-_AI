"""Réseau convolutif 1D numpy (piste B.2 — voir STRATEGIE.md, Épisode 6).

Motivation : un MLP dense doit réapprendre chaque motif de praticabilité à
chaque position de colonne indépendamment (482 entrées, aucun paramètre
partagé). Or « un trou est un trou, où qu'il soit sur la largeur » —
exactement l'équivariance par translation qu'une convolution encode par
construction (poids partagés entre positions). L'Épisode 5 a prouvé que
« voir plus loin » (cône élargi) ne payait pas ; ce module teste l'autre
hypothèse : mieux EXPLOITER ce qui est déjà vu.

Architecture : la carte de praticabilité de `sense_grid` (6 lignes x 4
plans temporels x 19 colonnes) est vue comme un signal 1D de 24 canaux
(ligne x plan) sur 19 positions ; une convolution (noyau impair, padding
« same ») partage ses poids entre colonnes. Les scalaires par ligne
(type, vitesse) et globaux (X, stagnation) contournent la convolution et
rejoignent la tête dense après aplatissement.

Même API que `neural.MLP` (forward/backward/adam_step/get_flat/set_flat/
copy/save/load) pour rester substituable partout où `MLP` est utilisée.
Gradients vérifiés numériquement (voir .smoke_conv.sh au moment du test).
"""

from __future__ import annotations

import os

import numpy as np

from config import GRID_SENSOR_ROWS, GRID_TIME_PLANES, GRID_WIDTH

ROWS: int = len(GRID_SENSOR_ROWS)
PLANES: int = len(GRID_TIME_PLANES)
IN_CHANNELS: int = ROWS * PLANES          # 24 : une "chaîne" par (ligne, plan)
ROW_BLOCK: int = GRID_WIDTH * PLANES + 4  # 80 : bloc sense_grid d'une ligne
SCALAR_SIZE: int = ROWS * 4 + 2           # 26 : type/vitesse par ligne + X/stagnation


def split_grid_obs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sépare un batch d'observations `sense_grid` en (spatial, scalaires).

    spatial : (batch, GRID_WIDTH, IN_CHANNELS) — praticabilité, canal =
    ligne x plan, prête pour la convolution (canaux en dernier).
    scalaires : (batch, SCALAR_SIZE) — type/vitesse/X/stagnation, en aval
    de la convolution (aucune structure spatiale à exploiter).
    """
    x = np.atleast_2d(x)
    batch = x.shape[0]
    spatial = np.empty((batch, GRID_WIDTH, IN_CHANNELS))
    scalars = np.empty((batch, SCALAR_SIZE))
    for r in range(ROWS):
        base = r * ROW_BLOCK
        seg = x[:, base : base + GRID_WIDTH * PLANES].reshape(batch, PLANES, GRID_WIDTH)
        spatial[:, :, r * PLANES : (r + 1) * PLANES] = seg.transpose(0, 2, 1)
        scalars[:, r * 4 : r * 4 + 4] = x[:, base + GRID_WIDTH * PLANES : base + ROW_BLOCK]
    scalars[:, ROWS * 4 :] = x[:, ROWS * ROW_BLOCK :]
    return spatial, scalars


class ConvQNet:
    """conv1D (canaux partagés, padding same) -> tanh -> aplati + scalaires
    -> dense cachée (tanh) -> logits linéaires."""

    def __init__(
        self,
        input_size: int,
        conv_channels: int,
        kernel_size: int,
        hidden_size: int,
        n_actions: int,
        flat: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size doit être impair (padding symétrique)")
        self.input_size = input_size
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.flatten_size = GRID_WIDTH * conv_channels + SCALAR_SIZE
        # compat avec le dispatch existant (sensor_for_input(net.layout[0]))
        self.layout = (input_size, hidden_size, n_actions)

        if flat is None:
            rng = np.random.default_rng(seed)
            self.conv_w = rng.normal(0.0, 1.0 / np.sqrt(IN_CHANNELS * kernel_size),
                                      (kernel_size, IN_CHANNELS, conv_channels))
            self.conv_b = np.zeros(conv_channels)
            self.w1 = rng.normal(0.0, 1.0 / np.sqrt(self.flatten_size),
                                  (self.flatten_size, hidden_size))
            self.b1 = np.zeros(hidden_size)
            self.w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), (hidden_size, n_actions))
            self.b2 = np.zeros(n_actions)
        else:
            self.set_flat(flat)

        self._adam_m: dict[str, np.ndarray] | None = None
        self._adam_v: dict[str, np.ndarray] | None = None
        self._adam_t = 0
        self._cache: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------- paramètres

    def _param_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "conv_w": (self.kernel_size, IN_CHANNELS, self.conv_channels),
            "conv_b": (self.conv_channels,),
            "w1": (self.flatten_size, self.hidden_size),
            "b1": (self.hidden_size,),
            "w2": (self.hidden_size, self.n_actions),
            "b2": (self.n_actions,),
        }

    def get_flat(self) -> np.ndarray:
        return np.concatenate([
            getattr(self, k).ravel() for k in self._param_shapes()
        ])

    def set_flat(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.float64)
        i = 0
        for name, shape in self._param_shapes().items():
            n = int(np.prod(shape))
            setattr(self, name, flat[i : i + n].reshape(shape).copy())
            i += n
        if i != flat.size:
            raise ValueError(f"génome de taille {flat.size}, attendu {i}")

    # ---------------------------------------------------------------- calculs

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        single = x.ndim == 1
        spatial, scalars = split_grid_obs(x)  # (b,W,inC), (b,S)
        b = spatial.shape[0]
        padded = np.zeros((b, GRID_WIDTH + 2 * self.pad, IN_CHANNELS))
        padded[:, self.pad : self.pad + GRID_WIDTH, :] = spatial
        conv_out = np.zeros((b, GRID_WIDTH, self.conv_channels))
        for k in range(self.kernel_size):
            conv_out += padded[:, k : k + GRID_WIDTH, :] @ self.conv_w[k]
        conv_out += self.conv_b
        conv_h = np.tanh(conv_out)  # (b, W, convC)

        flat_conv = conv_h.reshape(b, -1)
        combined = np.concatenate((flat_conv, scalars), axis=1)  # (b, flatten_size)
        h1 = np.tanh(combined @ self.w1 + self.b1)
        logits = h1 @ self.w2 + self.b2

        if cache:
            self._cache = {
                "padded": padded, "conv_h": conv_h, "combined": combined, "h1": h1,
            }
        return logits[0] if single else logits

    def backward(self, grad_logits: np.ndarray) -> dict[str, np.ndarray]:
        assert self._cache is not None, "forward(cache=True) requis"
        c = self._cache
        g = np.atleast_2d(grad_logits)
        b = g.shape[0]

        grads = {"w2": c["h1"].T @ g, "b2": g.sum(axis=0)}
        dh1 = (g @ self.w2.T) * (1.0 - c["h1"] ** 2)
        grads["w1"] = c["combined"].T @ dh1
        grads["b1"] = dh1.sum(axis=0)

        d_combined = dh1 @ self.w1.T
        d_flat_conv = d_combined[:, : GRID_WIDTH * self.conv_channels]
        d_conv_h = d_flat_conv.reshape(b, GRID_WIDTH, self.conv_channels)
        d_conv_out = d_conv_h * (1.0 - c["conv_h"] ** 2)

        grads["conv_b"] = d_conv_out.sum(axis=(0, 1))
        conv_dw = np.empty_like(self.conv_w)
        d_padded = np.zeros_like(c["padded"])
        for k in range(self.kernel_size):
            window = c["padded"][:, k : k + GRID_WIDTH, :]  # (b, W, inC)
            conv_dw[k] = np.einsum("bwi,bwo->io", window, d_conv_out)
            d_padded[:, k : k + GRID_WIDTH, :] += d_conv_out @ self.conv_w[k].T
        grads["conv_w"] = conv_dw
        # d_input non requis (les capteurs ne sont pas appris) : on l'ignore.
        return grads

    def adam_step(
        self,
        grads: dict[str, np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        names = list(self._param_shapes())
        if self._adam_m is None:
            self._adam_m = {k: np.zeros_like(getattr(self, k)) for k in names}
            self._adam_v = {k: np.zeros_like(getattr(self, k)) for k in names}
        self._adam_t += 1
        assert self._adam_v is not None
        for k in names:
            p = getattr(self, k)
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g**2
            m_hat = self._adam_m[k] / (1 - beta1**self._adam_t)
            v_hat = self._adam_v[k] / (1 - beta2**self._adam_t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy(self) -> "ConvQNet":
        net = ConvQNet(
            self.input_size, self.conv_channels, self.kernel_size,
            self.hidden_size, self.n_actions, flat=self.get_flat(),
        )
        return net

    # ------------------------------------------------------------ persistance

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path,
            kind=np.asarray("conv"),
            arch=np.asarray(
                [self.input_size, self.conv_channels, self.kernel_size,
                 self.hidden_size, self.n_actions]
            ),
            genome=self.get_flat(),
        )

    @classmethod
    def load(cls, path: str) -> "ConvQNet":
        with np.load(path) as data:
            input_size, conv_channels, kernel_size, hidden_size, n_actions = (
                int(v) for v in data["arch"]
            )
            return cls(
                input_size, conv_channels, kernel_size, hidden_size, n_actions,
                flat=data["genome"],
            )
