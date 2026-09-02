"""The technique. This is the only file in the repository you have to write.

What ships here runs and is wrong: it returns the first `rank` joints as if a hand grasped by
moving one finger at a time. The harness is proved before you touch it.

What you are implementing (Gallier & Quaintance ch. 21.4-21.5; Ciocarlie & Allen 2009):

    Xc = X - mean(X)                 centre the postures. This is half of what PCA is.
    Xc = U S Vt                      the SVD of the centred matrix
    components = Vt[:rank].T         the leading right singular vectors: the eigengrasps
    explained  = S**2 / sum(S**2)    each component's share of the variance

Centring is not a formality. A version that skips it still returns something, and what it returns
is the direction the hand sits in rather than the directions it moves in.
"""

from __future__ import annotations

import numpy as np

ASSUMPTIONS: list[str] = [
    "Rows of X are postures, columns are joints, all in the same units (radians).",
    "Every posture is a grasp somebody would actually make.",
]


def eigengrasps(x: np.ndarray, *, rank: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (components, explained_variance_ratio).

    `components` is (dof, rank) with orthonormal columns; `explained_variance_ratio` is the full
    spectrum's share, longest first — you want the whole curve, not just the first two numbers.
    """
    x = np.asarray(x, dtype=float)

    # PLACEHOLDER — runs, and is wrong: "the hand grasps with joints 0 and 1".
    components = np.eye(x.shape[1])[:, :rank]
    ratio = np.zeros(min(x.shape))
    ratio[:rank] = 1.0 / rank
    return components, ratio


def project(x: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Coordinates of each posture in the subspace. Planning happens in here."""
    centred = x - x.mean(axis=0)
    return centred @ components


def reconstruct(x: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Back to joint angles from the subspace."""
    mean = x.mean(axis=0)
    return project(x, components) @ components.T + mean
