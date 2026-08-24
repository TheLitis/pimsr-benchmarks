"""Run one fixed MT2DInv-DenseNet v1.2 common-retraining seed.

The held-out argument is deliberately limited to the truth-free observation
payload.  Training schedule, architecture, and campaign seeds are fixed in the
adapter and cannot be tuned from this command line.
"""

from __future__ import annotations

import argparse
import json
import sys

from pimsr_benchmarks.densenet2d import COMMON_RETRAIN_SEEDS, run_common_retrain


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed pinned MT2DInv-DenseNet common-retraining adapter"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="clean MT2DInv-DenseNet v1.2 clone at the pinned commit",
    )
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--validation-h5", required=True)
    parser.add_argument(
        "--observations-npz",
        required=True,
        help="truth-free pimsr-sota-2d-observations payload",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=COMMON_RETRAIN_SEEDS,
    )
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--runtime-out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_arguments = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(raw_arguments)
    command = [sys.executable, str(__file__), *raw_arguments]
    result = run_common_retrain(
        repository_path=args.repo,
        train_h5=args.train_h5,
        validation_h5=args.validation_h5,
        observations_npz=args.observations_npz,
        seed=args.seed,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        predictions_out=args.predictions_out,
        runtime_out=args.runtime_out,
        command=command,
        runner_source=__file__,
    )
    print(
        json.dumps(
            {
                "method_id": result["method_id"],
                "method": result["method"],
                "seed": result["seed"],
                "outputs": result["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
