"""The 2026 answer to the 2009 question: a learned latent space instead of a linear subspace.

PCA finds the best *linear* subspace — chapter 21 says so, and "best" there is a theorem, not a
hope. An autoencoder is not bound to be linear, so at the same latent dimension it can only do
better on reconstruction, and the interesting question is by how much and whether it is worth it.

Written for you. Your job is `method.py`; this is the thing you measure yourself against, the way
TEASER++ is in lab 1. Runs on the CPU in seconds and on a GPU if you have one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, dof: int, latent: int, width: int = 64) -> None:
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(dof, width), nn.Tanh(), nn.Linear(width, latent))
        self.decode = nn.Sequential(nn.Linear(latent, width), nn.Tanh(), nn.Linear(width, dof))

    def forward(self, x):
        return self.decode(self.encode(x))


def fit(postures: np.ndarray, *, latent: int = 2, epochs: int = 400, seed: int = 0, device: str | None = None):
    """Train an autoencoder on centred postures and return (model, mean, device)."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(postures, dtype=np.float32)
    mean = x.mean(axis=0)
    data = torch.tensor(x - mean, device=device)

    net = Autoencoder(x.shape[1], latent).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = nn.functional.mse_loss(net(data), data)
        loss.backward()
        optimiser.step()
    return net, mean, device


def reconstruct(net, mean, postures: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Round-trip postures through the learned latent, back in radians."""
    x = torch.tensor(np.asarray(postures, dtype=np.float32) - mean, device=device)
    with torch.no_grad():
        return net(x).cpu().numpy() + mean
