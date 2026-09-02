# 16-811 · Lab 3 · Find the two numbers a hand actually grasps with

Take the principal components of a hand-posture dataset, plan grasps in the two-dimensional
subspace they span, and find out whether the 1998 result still holds on a dataset with 1.32M
grasps in it.

**Technique:** Ciocarlie & Allen, *Hand Posture Subspaces for Dexterous Robotic Grasping*,
IJRR 28(7):851–867, 2009 — after Santello et al., J. Neurosci. 18(23), 1998.
**Checked against:** a learned latent of the same dimension, and [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet).
**Built on:** the [Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie) in MuJoCo,
[PyTorch](https://pytorch.org) for the autoencoder, the [Hugging Face Hub](https://huggingface.co) for the result.
**From the book:** Gallier & Quaintance ch. 21.4–21.5 (PCA, best affine approximation), 20 (SVD), 5 (rank), 16 (spectral theorem).

## Start

```bash
make check              # 11 tests. 3 fail on the fast subset. Those are the job.
make check -- -m "not slow"   # skip the two that fetch the hand and train a net
make reproduce          # runs now, with a deliberately wrong subspace. Beat that number.
```

The Shadow Hand is fetched once by `robot_descriptions` (a few hundred megabytes, a minute) and
cached. A GPU is used if you have one and is not needed.

`method.py` ships something that runs and is wrong: it claims the hand grasps by moving joints 0
and 1. Your first edit moves a real number.

## What you write

```
Xc = X - mean(X)              centre the postures — half of what PCA is
Xc = U S Vᵀ                   the SVD of the centred matrix
components = Vᵀ[:rank].ᵀ      the leading right singular vectors: the eigengrasps
explained  = S² / ΣS²         each component's share of the variance
```

Skipping the centring still returns something. What it returns is the direction the hand *sits* in
rather than the directions it *moves* in — and one of the tests exists only to catch that.

## Why the ruler uses principal angles

A subspace has no preferred basis: two correct answers can differ by a rotation inside the plane
they span, so comparing components entry-by-entry would fail a correct implementation. The largest
principal angle is the honest measure, and `evaluate.py` uses it.

## The files

| | |
|---|---|
| `method.py` | **Yours.** `eigengrasps` is four lines of NumPy when finished |
| `evaluate.py` | The ruler. Complete, never imports `method` |
| `synthetic.py` | Postures drawn from a synergy you generated, so the answer is known |
| `baselines.py` | Floors (two joints, a random plane) and a ceiling (the true synergy) |
| `reproduce.py` | Rank sweep across seeds → `results.json` |
| `extend.py` | Your own ideas, as switches |
| `tests/` | The to-do list |

## The finding, which is not the one the paper sets you up for

Run `make reproduce` and you get the same hand decomposed twice:

```
      sampling  rank   variance  recon (rad)
   coordinated     2      0.845       0.6190
   independent     2      0.253       1.0122
```

**Santello's 80% is a fact about coordination, not about hands.** Draw every actuator on a real
Shadow Hand independently and two components hold about a quarter of the variance. Draw a handful
of coordinated grasp strategies and they hold most of it. Same joints, same limits, same SVD.

That reframes the whole 2009 result: eigengrasps work because people use their fingers together,
so any variance figure quoted without saying how the postures were sampled means nothing. It is
also the trap in DexGraspNet, whose grasps were *synthesised* in eigengrasp space.

## And the 2026 question

`latent.py` trains an autoencoder at the same latent dimension. PCA is the best **linear**
subspace — chapter 21 proves it — so a nonlinear latent should win. It does, by about 4%:

```
latent 2:  PCA 0.6190 rad   autoencoder 0.5963 rad   (1.04x)
```

Four per cent, for a network, a training loop and a GPU dependency. Writing that down — and
deciding it is not worth it here — is the skill the lab is actually training.

## Past the paper

The 80% is a fact about 57 objects mimed by human hands in 1998. DexGraspNet's grasps were
themselves synthesised *inside* an eigengrasp space, so measuring its spectrum and finding two
dimensions is very nearly circular — and its authors say so. The honest test needs postures that
were not born in the subspace you are testing for.

1. Re-run the decomposition on grasps generated **without** an eigengrasp prior, and report whether
   two dimensions still hold 80%.
2. Fit a subspace **per object category** and ask whether the union beats one global basis at equal
   total rank.
3. Repeat the whole thing on a **different hand morphology** to see how much of any drop is the
   hand rather than the data.

**How you would know:** variance held at rank *k*, and simulated grasp success at rank *k* — the
same two axes `reproduce.py` already reports.
