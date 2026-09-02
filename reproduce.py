"""The experiment: the same hand, sampled two ways, decomposed the same way.

The headline number of this lab is not a number. It is that Santello's 80% survives on a real
Shadow Hand *only when the postures are coordinated* — and collapses to about a quarter when every
actuator is drawn independently. Same hand, same joint limits, same decomposition.

Writes `results.json` and `eigengrasps.npz`; `publish.py` turns the latter into a Hub artifact.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from evaluate import reconstruction_error, variance_held
from method import ASSUMPTIONS, eigengrasps, reconstruct

RANKS = [1, 2, 3, 5, 8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postures", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import hand

    records, keep = [], None
    for coordinated in (True, False):
        x = hand.postures(args.postures, seed=args.seed, coordinated=coordinated)
        for rank in RANKS:
            components, ratio = eigengrasps(x, rank=rank)
            records.append(
                {
                    "coordinated": coordinated,
                    "rank": rank,
                    "variance_held": variance_held(x, components),
                    "recon_err_rad": reconstruction_error(x, reconstruct(x, components)),
                }
            )
            if coordinated and rank == 2:
                keep = (components, ratio)

    with open("results.json", "w") as handle:
        json.dump({"assumptions": ASSUMPTIONS, "records": records}, handle, indent=2)
    if keep is not None:
        np.savez("eigengrasps.npz", components=keep[0], variance=keep[1], coordinated=True)

    print(f"{'sampling':>14} {'rank':>5} {'variance':>10} {'recon (rad)':>12}")
    for coordinated in (True, False):
        for rank in RANKS:
            row = next(r for r in records if r["coordinated"] == coordinated and r["rank"] == rank)
            label = "coordinated" if coordinated else "independent"
            print(f"{label:>14} {rank:>5} {row['variance_held']:>10.3f} {row['recon_err_rad']:>12.4f}")
    print("\nresults.json and eigengrasps.npz written. `python publish.py --repo you/name` puts them on the Hub.")


if __name__ == "__main__":
    main()
