"""A floor and a ceiling, so a number from `method.py` means something on day one."""

from __future__ import annotations

import numpy as np


def single_joints(dof: int, rank: int = 2) -> np.ndarray:
    """"The hand grasps by moving joints 0 and 1." The floor."""
    return np.eye(dof)[:, :rank]


def random_subspace(dof: int, rank: int = 2, *, seed: int = 0) -> np.ndarray:
    """A random rank-dimensional subspace. A fairer floor than single joints."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dof, rank)))
    return q


def oracle(basis: np.ndarray) -> np.ndarray:
    """The synergy the postures were built from. The ceiling."""
    return basis
