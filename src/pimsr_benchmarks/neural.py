"""Thin inference wrapper around a trained pimsr-inversion checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from pimsr_inversion.contracts1d import (
    PHASE_SCALE_DEGREES,
    Contract1DError,
    Dataset1DContract,
    validate_checkpoint1d,
    validate_checkpoint1d_heldout,
    validate_dataset1d,
)
from pimsr_inversion.data import NormStats
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
        if not isinstance(ckpt, dict):
            raise TypeError("1D checkpoint root must be a dictionary")
        contract = validate_checkpoint1d(ckpt)
        if "dataset_identities" not in ckpt:
            raise Contract1DError(
                "1D benchmark checkpoint lacks exact train/validation identities"
            )
        heldout_keys = (
            "checkpoint_schema",
            "checkpoint_schema_version",
            "n_obs",
            "n_depth",
            "n_scenarios",
            "norm_stats",
            "periods",
            "depth_grid",
            "data_contract",
            "input_contract",
            "epoch",
            "dataset_identities",
        )
        self._heldout_checkpoint = {key: ckpt[key] for key in heldout_keys}
        # The public validator requires the inference shape but not weight bytes;
        # avoid retaining a second model-sized state dictionary in memory.
        self._heldout_checkpoint["model_state"] = {}
        self.device = dev
        self.data_contract = contract
        self.stats = NormStats.from_dict(ckpt["norm_stats"])
        self.periods = contract.periods.copy()
        self.depth_grid = contract.depth_grid.copy()
        self.n_obs = int(ckpt["n_obs"])
        self.model = PimsrNet(
            n_obs=self.n_obs,
            n_depth=int(ckpt["n_depth"]),
            n_scenarios=int(ckpt["n_scenarios"]),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(dev).eval()
        # Post-hoc temperature scaling fitted on the val split (if present).
        self.sigma_temperature = float(ckpt.get("sigma_temperature_rho", 1.0))
        # observation vector layout: [log10 rho_a | phase/45 | gravity]
        self.n_periods = self.periods.size
        self.n_grav = contract.grav_offsets.size

    def require_dataset(self, file: h5py.File) -> Dataset1DContract:
        """Validate a synthetic split and require exact checkpoint compatibility."""
        dataset_contract = validate_dataset1d(file)
        self.data_contract.require_same(
            dataset_contract,
            "1D checkpoint and benchmark dataset",
        )
        validate_checkpoint1d_heldout(self._heldout_checkpoint, dataset_contract)
        return dataset_contract

    def _pack(
        self,
        log_rho_a: np.ndarray,
        phase: np.ndarray,
        gravity: np.ndarray | None,
    ) -> np.ndarray:
        log_rho_a = np.asarray(log_rho_a, dtype=np.float32)
        phase = np.asarray(phase, dtype=np.float32)
        if log_rho_a.shape != (self.n_periods,) or phase.shape != (self.n_periods,):
            raise ValueError(f"MT inputs must each have shape ({self.n_periods},)")
        if np.isinf(log_rho_a).any() or np.isinf(phase).any():
            raise ValueError("MT inputs must not contain infinities")
        missing_log_rho = np.isnan(log_rho_a)
        missing_phase = np.isnan(phase)
        if not np.array_equal(missing_log_rho, missing_phase):
            raise ValueError("unsupported MT periods must use paired NaN values")
        valid = ~missing_phase
        if np.any((phase[valid] < 0.0) | (phase[valid] >= 180.0)):
            raise ValueError("MT phase must use the checkpoint [0, 180) degree convention")
        if gravity is None:
            # No gravity survey (e.g. MT-only real data): use the training
            # mean, i.e. zero after normalisation - maximally uninformative.
            gravity = self.stats.obs_mean[2 * self.n_periods :]
        else:
            gravity = np.asarray(gravity, dtype=np.float32)
            if gravity.shape != (self.n_grav,):
                raise ValueError(f"gravity input must have shape ({self.n_grav},)")
            if not np.isfinite(gravity).all():
                raise ValueError("gravity input must be finite")
        obs = np.concatenate(
            [log_rho_a, phase / PHASE_SCALE_DEGREES, gravity]
        ).astype(np.float32)
        mean = self.stats.obs_mean.astype(np.float32)
        std = self.stats.obs_std.astype(np.float32)
        # EMTF resampling represents unsupported periods as NaN.  Mean-fill
        # them explicitly (zero after normalisation) instead of inventing an
        # edge-extrapolated response.
        obs = np.where(np.isfinite(obs), obs, mean)
        return (obs - mean) / std

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
        sigma = torch.exp(0.5 * out["log_sigma_rho"]) * self.sigma_temperature
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
