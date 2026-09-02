"""The ruler. Complete, and it does not import `method`."""

from __future__ import annotations

import argparse

import numpy as np

from synthetic import postures, synergy_basis


def subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest principal angle between two subspaces, in radians. 0 means identical.

    Comparing components entry-by-entry would be wrong: a subspace has no preferred basis, and two
    correct answers can differ by a rotation within it. Principal angles are the honest measure.
    """
    qa, _ = np.linalg.qr(np.asarray(a))
    qb, _ = np.linalg.qr(np.asarray(b))
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.arccos(np.clip(singular.min(), -1.0, 1.0)))


def reconstruction_error(x: np.ndarray, rebuilt: np.ndarray) -> float:
    """Mean per-posture Euclidean error in joint space, in radians."""
    return float(np.mean(np.linalg.norm(np.asarray(x) - np.asarray(rebuilt), axis=1)))


def variance_held(x: np.ndarray, components: np.ndarray) -> float:
    """Fraction of posture variance captured by this subspace."""
    centred = np.asarray(x) - np.asarray(x).mean(axis=0)
    total = float((centred**2).sum())
    kept = float(((centred @ components) ** 2).sum())
    return kept / total if total else 0.0


def self_test() -> None:
    x, basis = postures(seed=0)
    print(f"perfect subspace distance : {subspace_distance(basis, basis):.3e}  (expect 0)")
    print(f"perfect variance held     : {variance_held(x, basis):.4f}     (expect ~1)")
    wrong = np.eye(x.shape[1])[:, :2]
    print(f"joint-0/1 variance held   : {variance_held(x, wrong):.4f}     (expect small)")
    print(f"perfect reconstruction    : {reconstruction_error(x, x):.3e}  (expect 0)")
    print("\nIf any 'perfect' line is off the ruler is wrong and every later number is fiction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        self_test()
