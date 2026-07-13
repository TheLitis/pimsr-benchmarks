"""Score a trained PIMSR 3D checkpoint on held-out 3D samples.

Reports volume RMSE and 68% sigma coverage with bootstrap intervals so the
first A100 cycle lands directly in the frozen statistical framework.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pimsr_benchmarks.statistics import bootstrap_ci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="directory of held-out sample_*.h5")
    parser.add_argument("--preset", default="a100-40gb")
    parser.add_argument("--out", default="results/3d/bench3d.json")
    args = parser.parse_args()

    from pimsr_inversion.network3d import Model3DConfig, PimsrNet3D
    from pimsr_inversion.train3d import Volume3DDataset

    config = Model3DConfig.preset(args.preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Volume3DDataset(args.data)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    sample_obs, _ = dataset[0]
    model = PimsrNet3D(sample_obs.shape[0], config.width, checkpoint_blocks=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    rmses, coverages = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            obs, target = dataset[index]
            obs = obs.unsqueeze(0).to(device)
            pred = model(obs, output_shape=target.shape[-3:])
            mu = pred["log_rho"].squeeze(0).cpu().numpy()
            sigma = np.exp(0.5 * pred["log_sigma_rho"].squeeze(0).cpu().numpy())
            residual = mu - target.numpy()
            rmses.append(float(np.sqrt(np.mean(residual**2))))
            coverages.append(float(np.mean(np.abs(residual) <= sigma)))

    summary = {
        "n_samples": len(dataset),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "preset": args.preset,
        "rmse": bootstrap_ci(rmses),
        "coverage68": bootstrap_ci(coverages),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
