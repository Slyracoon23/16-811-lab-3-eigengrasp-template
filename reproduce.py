"""Sweep the rank across seeds and write results.json.

Report the whole spectrum, not the first two numbers. Where the curve knees is the actual finding.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import baselines
from evaluate import reconstruction_error, subspace_distance, variance_held
from method import ASSUMPTIONS, eigengrasps, reconstruct
from synthetic import postures

RANKS = [1, 2, 3, 5, 8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--true-rank", type=int, default=2)
    args = parser.parse_args()

    records = []
    for seed in range(args.seeds):
        x, basis = postures(rank=args.true_rank, seed=seed)
        floor = baselines.single_joints(x.shape[1], args.true_rank)
        for rank in RANKS:
            components, _ = eigengrasps(x, rank=rank)
            records.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "variance_held": variance_held(x, components),
                    "recon_err": reconstruction_error(x, reconstruct(x, components)),
                    "subspace_dist": subspace_distance(components[:, : args.true_rank], basis),
                    "floor_variance": variance_held(x, floor),
                }
            )

    with open("results.json", "w") as handle:
        json.dump({"assumptions": ASSUMPTIONS, "records": records}, handle, indent=2)

    print(f"{'rank':>5} {'variance':>10} {'recon err':>11} {'subspace':>10} {'floor':>8}")
    for rank in RANKS:
        rows = [r for r in records if r["rank"] == rank]
        print(
            f"{rank:>5} {np.mean([r['variance_held'] for r in rows]):>10.4f} "
            f"{np.mean([r['recon_err'] for r in rows]):>11.3e} "
            f"{np.mean([r['subspace_dist'] for r in rows]):>10.3e} "
            f"{np.mean([r['floor_variance'] for r in rows]):>8.4f}"
        )
    print("\nOn rank-2 synthetic data, rank 2 should hold ~1.0 once method.py is yours.")


if __name__ == "__main__":
    main()
