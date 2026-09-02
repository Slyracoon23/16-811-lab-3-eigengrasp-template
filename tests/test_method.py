"""Your to-do list, as tests.

The first five are algebra on data whose answer you set. The last two are the paper's claim on a
real robot hand, and the question 2026 asks of a 2009 method.
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
    near, _ = eigengrasps(x, rank=2)
    far, _ = eigengrasps(x + 5.0, rank=2)
    assert subspace_distance(near, basis) < 1e-6
    assert subspace_distance(far, basis) < 1e-6, "the mean posture is leaking into the components"


def test_reconstruction_improves_with_rank():
    """Step 5. More components, less error — monotonically, or something is wrong."""
    x, _ = postures(rank=4, seed=4)
    errors = [reconstruction_error(x, reconstruct(x, eigengrasps(x, rank=k)[0])) for k in (1, 2, 4, 8)]
    assert all(b <= a + 1e-9 for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < errors[0]


# Below here the hand is real and MuJoCo has to fetch it. Slower, and worth it.
@pytest.mark.slow
def test_the_synergy_is_in_the_coordination_not_the_hand():
    """Step 6. The finding, and it is not what the paper leads you to expect.

    Santello's 80% is a fact about how people *coordinate* their fingers, not about what a hand can
    mechanically do. Sample a real Shadow Hand's actuators independently and two components hold
    about a quarter of the variance. Sample a few coordinated grasp strategies and they hold most
    of it. Same hand, same joint limits, same decomposition — the structure is in the behaviour.
    """
    import hand

    coordinated = hand.postures(240, seed=0, coordinated=True)
    independent = hand.postures(240, seed=0, coordinated=False)

    held_coordinated = variance_held(coordinated, eigengrasps(coordinated, rank=2)[0])
    held_independent = variance_held(independent, eigengrasps(independent, rank=2)[0])

    assert held_coordinated > 0.80, f"coordinated grasps held only {held_coordinated:.3f}"
    assert held_independent < 0.45, f"independent joints held {held_independent:.3f}"


@pytest.mark.slow
def test_a_learned_latent_barely_beats_the_linear_one():
    """Step 7. The 2026 question, and the answer is a judgement rather than a win.

    PCA finds the best *linear* subspace — that is a theorem, not a hope. An autoencoder is not
    bound to be linear, so at equal latent dimension it should win. It does, by a few per cent.
    Which is the result worth writing down: the extra machinery buys almost nothing here, and
    knowing when not to reach for a network is the skill.
    """
    import hand
    import latent

    x = hand.postures(600, seed=0, coordinated=True)
    train, test = x[:450], x[450:]

    components, _ = eigengrasps(train, rank=2)
    mean = train.mean(axis=0)
    linear = ((test - mean) @ components) @ components.T + mean
    linear_error = reconstruction_error(test, linear)

    net, ae_mean, device = latent.fit(train, latent=2, epochs=600, seed=0)
    learned_error = reconstruction_error(test, latent.reconstruct(net, ae_mean, test, device))

    assert learned_error <= linear_error, "a nonlinear latent should not lose to a linear one"
    assert learned_error > linear_error * 0.5, "if the gap is this large, suspect the PCA"
