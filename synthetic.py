"""Postures whose subspace you chose, so you can check the decomposition before trusting it.

Real hands come later. The point of this file is that you know the answer: postures are built from
a `rank`-dimensional synergy you generated, so the first `rank` components must recover it.
"""

from __future__ import annotations

import numpy as np

DOF = 22  # a ShadowHand's joint count, which is why the subspace question is interesting


def synergy_basis(rank: int = 2, *, dof: int = DOF, seed: int = 0) -> np.ndarray:
    """An orthonormal dof x rank basis: the synergies the hand "really" uses."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dof, rank)))
    return basis


def postures(n: int = 800, *, rank: int = 2, noise: float = 0.01, dof: int = DOF, seed: int = 0):
    """Return (X, basis) with X of shape (n, dof), drawn from a `rank`-dimensional synergy."""
    rng = np.random.default_rng(seed + 1)
    basis = synergy_basis(rank, dof=dof, seed=seed)
    coefficients = rng.normal(scale=1.0, size=(n, rank))
    offset = rng.normal(scale=0.5, size=dof)
    x = coefficients @ basis.T + offset
    if noise:
        x = x + rng.normal(scale=noise, size=x.shape)
    return x, basis
