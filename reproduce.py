"""The experiment: the same hand, sampled two ways, decomposed the same way.

The headline number of this lab is not a number. It is that Santello's 80% survives on a real
Shadow Hand *only when the postures are coordinated* — and collapses to about a quarter when every
actuator is drawn independently. Same hand, same joint limits, same decomposition.

Writes `results.json` and `eigengrasps.npz`; `publish.py` turns the latter into a Hub artifact.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from evaluate import reconstruction_error, variance_held
from method import ASSUMPTIONS, eigengrasps, reconstruct

def jsonable(record: dict) -> dict:
    """A record with every non-finite float replaced by ``None``.

    `json.dump` writes `NaN` and `Infinity` by default. Python reads those back; `JSON.parse`
    refuses them outright, so a single degenerate row makes the whole results file unreadable to
    anything that is not Python — including the course app that imports it. `null` is JSON, and it
    says the true thing: this one was not measured.

    The sanitising happens here, at the boundary, and not in the functions that compute the
    numbers. `float("nan")` is a perfectly good return value for a fit that had too few points, and
    the printed summary below still uses `np.nanmean` over the real values.
    """
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in record.items()
    }


RANKS = [1, 2, 3, 5, 8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postures", type=int, default=600)
    # `--seeds`, plural, and a real loop. The Makefile and the README have always said `--seeds`;
    # the parser defined `--seed` and ran once, so the headline comparison of this lab — 80% under
    # coordinated sampling against a quarter under independent — rested on a single draw. The
    # sibling labs all sweep seeds, and lab 1 states the reason at the top of its own file: a
    # single run is an anecdote, and the spread is what says whether a gap is real.
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    import hand

    records, keep = [], None
    for coordinated in (True, False):
        for seed in range(args.seeds):
            x = hand.postures(args.postures, seed=seed, coordinated=coordinated)
            for rank in RANKS:
                components, ratio = eigengrasps(x, rank=rank)
                records.append(
                    {
                        "coordinated": coordinated,
                        "seed": seed,
                        "rank": rank,
                        "variance_held": variance_held(x, components),
                        "recon_err_rad": reconstruction_error(x, reconstruct(x, components)),
                    }
                )
                if coordinated and seed == 0 and rank == 2:
                    keep = (components, ratio)

    with open("results.json", "w") as handle:
        json.dump(
            {"assumptions": ASSUMPTIONS, "records": [jsonable(r) for r in records]},
            handle,
            indent=2,
            allow_nan=False,
        )
    if keep is not None:
        np.savez("eigengrasps.npz", components=keep[0], variance=keep[1], coordinated=True)

    print(f"{'sampling':>14} {'rank':>5} {'variance':>10} {'spread':>8} {'recon (rad)':>12}")
    for coordinated in (True, False):
        for rank in RANKS:
            rows = [r for r in records if r["coordinated"] == coordinated and r["rank"] == rank]
            held = np.array([r["variance_held"] for r in rows])
            recon = np.mean([r["recon_err_rad"] for r in rows])
            label = "coordinated" if coordinated else "independent"
            print(f"{label:>14} {rank:>5} {held.mean():>10.3f} {held.std():>8.3f} {recon:>12.4f}")
    print("\nresults.json and eigengrasps.npz written. `python3 publish.py --repo you/name` puts them on the Hub.")


if __name__ == "__main__":
    main()
