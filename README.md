# 16-811 · Lab 3 · Find the two numbers a hand actually grasps with

Take the principal components of a hand-posture dataset, plan grasps in the two-dimensional
subspace they span, and find out whether the 1998 result still holds on a dataset with 1.32M
grasps in it.

**Technique:** Ciocarlie & Allen, *Hand Posture Subspaces for Dexterous Robotic Grasping*,
IJRR 28(7):851–867, 2009 — after Santello et al., J. Neurosci. 18(23), 1998.
**Checked against:** [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) ([arXiv:2210.02697](https://arxiv.org/abs/2210.02697)).
**From the book:** Gallier & Quaintance ch. 21.4–21.5 (PCA, best affine approximation), 20 (SVD), 5 (rank), 16 (spectral theorem).

## Start

```bash
make check        # 10 tests. 3 fail. Those 3 are the job.
make reproduce    # runs now, with a deliberately wrong subspace. Beat that number.
```

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
