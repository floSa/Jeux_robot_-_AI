"""Perceptron multicouche numpy partagé par tous les agents apprenants.

Une couche cachée tanh, sortie linéaire (logits). Trois usages :
- neuroévolution : forward seul, paramètres vus comme un génome plat ;
- DQN : forward + rétropropagation (perte TD) + Adam ;
- imitation : forward + rétropropagation (entropie croisée) + Adam.

Aucune dépendance hors numpy ; gradients vérifiés numériquement (tests).
"""

from __future__ import annotations

import os

import numpy as np

Layout = tuple[int, int, int]  # (entrée, cachée, sortie)


def genome_size(layout: Layout) -> int:
    n_in, n_hid, n_out = layout
    return n_in * n_hid + n_hid + n_hid * n_out + n_out


def random_flat(layout: Layout, rng: np.random.Generator) -> np.ndarray:
    """Génome aléatoire : poids en 1/sqrt(fan_in), biais nuls."""
    n_in, n_hid, n_out = layout
    return np.concatenate(
        (
            rng.normal(0.0, 1.0 / np.sqrt(n_in), n_in * n_hid),
            np.zeros(n_hid),
            rng.normal(0.0, 1.0 / np.sqrt(n_hid), n_hid * n_out),
            np.zeros(n_out),
        )
    )


class MLP:
    """entrée -> cachée (tanh) -> logits, avec rétropropagation et Adam."""

    def __init__(self, layout: Layout, flat: np.ndarray | None = None, seed: int = 0) -> None:
        self.layout = layout
        n_in, n_hid, n_out = layout
        if flat is None:
            flat = random_flat(layout, np.random.default_rng(seed))
        self.set_flat(flat)
        # état Adam (initialisé paresseusement au premier pas)
        self._adam_m: dict[str, np.ndarray] | None = None
        self._adam_v: dict[str, np.ndarray] | None = None
        self._adam_t = 0
        # caches du dernier forward (pour backward)
        self._x: np.ndarray | None = None
        self._h: np.ndarray | None = None

    # ------------------------------------------------------------- paramètres

    def set_flat(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.float64)
        if flat.size != genome_size(self.layout):
            raise ValueError(
                f"génome de taille {flat.size}, attendu {genome_size(self.layout)}"
            )
        n_in, n_hid, n_out = self.layout
        i = 0
        self.w1 = flat[i : i + n_in * n_hid].reshape(n_in, n_hid).copy()
        i += n_in * n_hid
        self.b1 = flat[i : i + n_hid].copy()
        i += n_hid
        self.w2 = flat[i : i + n_hid * n_out].reshape(n_hid, n_out).copy()
        i += n_hid * n_out
        self.b2 = flat[i : i + n_out].copy()

    def get_flat(self) -> np.ndarray:
        return np.concatenate(
            (self.w1.ravel(), self.b1, self.w2.ravel(), self.b2)
        )

    # ---------------------------------------------------------------- calculs

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        """Logits pour x de forme (entrée,) ou (batch, entrée)."""
        h = np.tanh(x @ self.w1 + self.b1)
        if cache:
            self._x = np.atleast_2d(x)
            self._h = np.atleast_2d(h)
        return h @ self.w2 + self.b2

    def backward(self, grad_logits: np.ndarray) -> dict[str, np.ndarray]:
        """Gradients des paramètres pour d(perte)/d(logits) donné.

        Nécessite un forward(cache=True) préalable sur le même batch.
        """
        assert self._x is not None and self._h is not None, "forward(cache=True) requis"
        g = np.atleast_2d(grad_logits)
        grads = {
            "w2": self._h.T @ g,
            "b2": g.sum(axis=0),
        }
        dh = (g @ self.w2.T) * (1.0 - self._h**2)  # dérivée de tanh
        grads["w1"] = self._x.T @ dh
        grads["b1"] = dh.sum(axis=0)
        return grads

    def adam_step(
        self,
        grads: dict[str, np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        params = {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}
        if self._adam_m is None:
            self._adam_m = {k: np.zeros_like(v) for k, v in params.items()}
            self._adam_v = {k: np.zeros_like(v) for k, v in params.items()}
        self._adam_t += 1
        assert self._adam_v is not None
        for k, p in params.items():
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g**2
            m_hat = self._adam_m[k] / (1 - beta1**self._adam_t)
            v_hat = self._adam_v[k] / (1 - beta2**self._adam_t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy(self) -> "MLP":
        return MLP(self.layout, flat=self.get_flat())

    # ------------------------------------------------------------ persistance

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path, kind=np.asarray("mlp"), layout=np.asarray(self.layout), genome=self.get_flat()
        )

    @classmethod
    def load(cls, path: str) -> "MLP":
        with np.load(path) as data:
            layout = tuple(int(v) for v in data["layout"])
            return cls((layout[0], layout[1], layout[2]), flat=data["genome"])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
