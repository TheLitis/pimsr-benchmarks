"""Thin inference wrapper around a trained pimsr-inversion checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pimsr_inversion.data import PHASE_SCALE, NormStats
from pimsr_inversion.network import PimsrNet

__all__ = ["NeuralInverter", "NeuralPrediction"]


@dataclass
class NeuralPrediction:
    log10_rho: np.ndarray  # (n_depth,)
    sigma_log10_rho: np.ndarray  # (n_depth,) aleatoric std
    density: np.ndarray  # (n_depth,) contrast-scaled units
    scenario_probs: np.ndarray  # (n_scenarios,)
    wall_time_s: float


class NeuralInverter:
    """Loads best.pt from pimsr-inversion and inverts observation vectors."""

    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
        self.device = dev
        self.stats = NormStats.from_dict(ckpt["norm_stats"])
        self.periods = np.asarray(ckpt["periods"])
        self.depth_grid = np.asarray(ckpt["depth_grid"])
        self.n_obs = int(ckpt["n_obs"])
        self.model = PimsrNet(n_obs=self.n_obs, n_depth=int(ckpt["n_depth"]))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(dev).eval()
        # observation vector layout: [log10 rho_a | phase/45 | gravity]
        self.n_periods = self.periods.size
        self.n_grav = self.n_obs - 2 * self.n_periods

    def _pack(
        self,
        log_rho_a: np.ndarray,
        phase: np.ndarray,
        gravity: np.ndarray | None,
    ) -> np.ndarray:
        if gravity is None:
            # No gravity survey (e.g. MT-only real data): use the training
            # mean, i.e. zero after normalisation - maximally uninformative.
            gravity = self.stats.obs_mean[2 * self.n_periods :]
        obs = np.concatenate([log_rho_a, phase / PHASE_SCALE, gravity]).astype(
            np.float32
        )
        return (obs - self.stats.obs_mean.astype(np.float32)) / self.stats.obs_std.astype(
            np.float32
        )

    def invert(
        self,
        log_rho_a: np.ndarray,
        phase: np.ndarray,
        gravity: np.ndarray | None = None,
    ) -> NeuralPrediction:
        t0 = perf_counter()
        x = torch.from_numpy(self._pack(log_rho_a, phase, gravity)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x.to(self.device))
        sigma = torch.exp(0.5 * out["log_sigma_rho"])
        return NeuralPrediction(
            log10_rho=out["log_rho"].squeeze(0).cpu().numpy(),
            sigma_log10_rho=sigma.squeeze(0).cpu().numpy(),
            density=out["density"].squeeze(0).cpu().numpy(),
            scenario_probs=torch.softmax(out["scenario_logits"], dim=1)
            .squeeze(0)
            .cpu()
            .numpy(),
            wall_time_s=perf_counter() - t0,
        )
