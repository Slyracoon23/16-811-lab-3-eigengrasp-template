"""Your to-do list, as tests.

These fail against the placeholder in `method.py` and pass when it is PCA. Run them with
`make check`. The order is the order to fix them in.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import reconstruction_error, subspace_distance, variance_held
from method import eigengrasps, reconstruct
from synthetic import postures


def test_recovers_a_known_synergy():
    """Step 1. The postures were built from a 2-D subspace. Find it."""
    x, basis = postures(rank=2, noise=0.0, seed=0)
    components, _ = eigengrasps(x, rank=2)
    assert subspace_distance(components, basis) < 1e-6


def test_components_are_orthonormal():
    """Step 2. The spectral theorem promises it; a decomposition that loses it is not PCA."""
    x, _ = postures(seed=1)
    components, _ = eigengrasps(x, rank=3)
    assert np.allclose(components.T @ components, np.eye(3), atol=1e-8)


def test_two_components_hold_the_variance():
    """Step 3. The 1998 claim, on data where it is true by construction."""
    x, _ = postures(rank=2, seed=2)
    components, ratio = eigengrasps(x, rank=2)
    assert variance_held(x, components) > 0.95
    assert float(np.sum(ratio[:2])) > 0.95


def test_is_unaffected_by_where_the_hand_rests():
    """Step 4. Centring. Add a constant posture offset; the subspace must not move."""
    x, basis = postures(rank=2, noise=0.0, seed=3)
    shifted = x + 5.0
    near, _ = eigengrasps(x, rank=2)
    far, _ = eigengrasps(shifted, rank=2)
    assert subspace_distance(near, basis) < 1e-6
    assert subspace_distance(far, basis) < 1e-6, "the mean posture is leaking into the components"


def test_reconstruction_improves_with_rank():
    """Step 5. More components, less error — monotonically, or something is wrong."""
    x, _ = postures(rank=4, seed=4)
    errors = [reconstruction_error(x, reconstruct(x, eigengrasps(x, rank=k)[0])) for k in (1, 2, 4, 8)]
    assert all(b <= a + 1e-9 for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < errors[0]


@pytest.mark.xfail(reason="Step 6: needs DexGraspNet. See scripts/fetch_data.py. Remove the mark when it passes.")
def test_holds_on_real_grasps():
    """The actual question: does the 1998 result survive a modern dataset?"""
    raise AssertionError("Load DexGraspNet postures and check whether two components still hold 80%.")
