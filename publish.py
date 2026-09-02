"""Put your eigengrasps on the Hugging Face Hub, with a card that says what they are.

The last step of the lab and the one a stranger can see. A basis in a notebook is a result; a basis
on the Hub with its variance curve, its sampling method and its limitations written down is an
artifact somebody else can use and check — which is the difference the lab is training for.

Needs `huggingface_hub` and a token (`huggingface-cli login`). Optional: nothing else here depends
on it, and the lab is complete without it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def card(repo: str, variance: np.ndarray, rank: int, coordinated: bool) -> str:
    return f"""---
license: mit
tags: [robotics, grasping, shadow-hand, pca, eigengrasp]
---

# {repo}

Eigengrasps for the Shadow Dexterous Hand: the leading {rank} principal components of hand posture,
after Ciocarlie & Allen (IJRR 2009) and Santello et al. (J. Neurosci. 1998).

- **Hand:** Shadow Hand, MuJoCo Menagerie, 24 joints / 20 actuators.
- **Postures:** {'coordinated grasp strategies with noise' if coordinated else 'independently sampled joints'}.
- **Variance held:** {', '.join(f'{k + 1}: {variance[: k + 1].sum():.3f}' for k in range(min(5, len(variance))))}

## The caveat that matters

The variance figure is a property of **how the postures were sampled**, not of the hand. Sampling
every actuator independently puts roughly a quarter of the variance in two components; sampling a
few coordinated grasp strategies puts most of it there. Santello measured people coordinating their
fingers, so the second is the comparable experiment — and any number quoted from this basis is
meaningless without saying which sampling produced it.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. yourname/shadow-hand-eigengrasps")
    parser.add_argument("--from-file", default="eigengrasps.npz")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    bundle = np.load(args.from_file)
    components, variance = bundle["components"], bundle["variance"]
    out = pathlib.Path("hf_upload")
    out.mkdir(exist_ok=True)
    np.savez(out / "eigengrasps.npz", components=components, variance=variance)
    (out / "README.md").write_text(card(args.repo, variance, components.shape[1], bool(bundle["coordinated"])))
    (out / "config.json").write_text(json.dumps({"dof": int(components.shape[0]), "rank": int(components.shape[1])}, indent=2))

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
    print(f"pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
