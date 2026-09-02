"""A real hand: the Shadow Dexterous Hand, from DeepMind's MuJoCo Menagerie.

Twenty-four joints and twenty actuators. That gap is not a modelling shortcut — the distal joint of
each finger is tendon-coupled to the one below it, exactly as a real Shadow Hand is built. So the
postures this hand can reach already lie on a lower-dimensional surface inside joint space, for an
anatomical reason rather than a statistical one.

Which is the whole question of the lab, restated: Santello measured how many dimensions a *human*
hand really uses. Here you can measure how many a real robot hand uses, on the model roboticists
actually run.

The model is fetched once by `robot_descriptions` and cached (a few hundred megabytes, a minute the
first time). Loading is lazy and inside the functions, deliberately, so importing this module — as
the tests and CI do — costs nothing.
"""

from __future__ import annotations

import functools

import numpy as np

DOF = 24


@functools.lru_cache(maxsize=1)
def model():
    """The Shadow Hand, cached. Fetched on first use."""
    from robot_descriptions.loaders.mujoco import load_robot_description

    return load_robot_description("shadow_hand_mj_description")


def joint_limits() -> np.ndarray:
    m = model()
    return np.array(m.jnt_range[: m.nq])


def joint_names() -> list[str]:
    import mujoco

    m = model()
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"j{i}" for i in range(m.nq)]


# Six grasp strategies, as fractions along each actuator's range. Not anatomy — a stand-in for the
# small set of things a hand is asked to do, which is what Santello's subjects were miming when
# they shaped their hands for 57 objects. The whole finding of this lab turns on the difference
# between sampling these and sampling every joint independently.
GRASPS = {
    "open": 0.05,
    "power": 0.85,
    "precision": 0.45,
    "lateral": 0.60,
    "hook": 0.75,
    "tripod": 0.50,
}


def postures(n: int = 800, *, seed: int = 0, coordinated: bool = True, spread: float = 0.12) -> np.ndarray:
    """`n` reachable hand postures, shape (n, 24), in radians.

    Sampled in *actuator* space and pushed through the hand's own transmission, so the tendon
    coupling is applied by MuJoCo rather than assumed by us.

    `coordinated` is the experiment. When true, each posture is one of a few grasp strategies with
    noise around it — fingers move together, as they do when a hand is shaped for an object. When
    false, every actuator is drawn independently, which is a hand with no coordination at all.

    Run it both ways before you conclude anything. A hand sampled independently has almost no
    low-dimensional structure, and that is the point: Santello measured how people *coordinate*
    their fingers, not what a hand mechanically can do.
    """
    import mujoco

    m = model()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    ctrl_low, ctrl_high = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    strategies = np.array(list(GRASPS.values()))

    out = []
    for index in range(n):
        if coordinated:
            centre = strategies[index % len(strategies)]
            amount = np.clip(centre + rng.normal(scale=spread, size=m.nu), 0.0, 1.0)
        else:
            amount = rng.uniform(0.0, 1.0, size=m.nu)
        d.ctrl[:] = ctrl_low + amount * (ctrl_high - ctrl_low)
        for _ in range(60):
            mujoco.mj_step(m, d)
        out.append(d.qpos[: m.nq].copy())
        mujoco.mj_resetData(m, d)
    return np.array(out)
