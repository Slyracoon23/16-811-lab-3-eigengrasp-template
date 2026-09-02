"""The ruler's own tests. These pass on a fresh clone."""

from __future__ import annotations

import numpy as np

from evaluate import reconstruction_error, subspace_distance, variance_held
from synthetic import postures, synergy_basis


def test_a_subspace_matches_itself():
    basis = synergy_basis(2, seed=0)
    assert subspace_distance(basis, basis) < 1e-9


def test_the_true_subspace_holds_the_variance():
    x, basis = postures(seed=0, noise=0.0)
    assert variance_held(x, basis) > 0.999


def test_the_floor_holds_little():
    """Two arbitrary joints must NOT hold the variance, or "we found the synergy" means nothing."""
    x, _ = postures(seed=1)
    assert variance_held(x, np.eye(x.shape[1])[:, :2]) < 0.5


def test_perfect_reconstruction_scores_zero():
    x, _ = postures(seed=2)
    assert reconstruction_error(x, x) < 1e-12
