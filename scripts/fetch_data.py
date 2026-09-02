"""Real grasps, when you are ready for them.

Deliberately fetches nothing. Everything here should first be proved on `synthetic.py`, where you
built the subspace — so the first run needs no download at all.

DexGraspNet: 1.32M ShadowHand grasps on 5355 objects, validated in Isaac Gym.
    https://github.com/PKU-EPIC/DexGraspNet   ·   https://pku-epic.github.io/DexGraspNet/
Isaac Lab, for evaluating a planned grasp:
    https://github.com/isaac-sim/IsaacLab

Note the circularity before you measure anything: DexGraspNet's grasps were themselves synthesised
by searching in eigengrasp space, and its own paper names the resulting narrow distribution as a
limitation. Finding two dimensions in data born in two dimensions proves very little. That is the
whole of the extension.
"""

from __future__ import annotations

if __name__ == "__main__":
    print(__doc__)
