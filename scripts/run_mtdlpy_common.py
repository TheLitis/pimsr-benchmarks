"""Run one preregistered MTDLPy DinkNet50 common-retraining seed.

Held-out input is accepted only as the truth-free observations NPZ emitted by
``materialize_sota_2d.py``.  Hyperparameters and seed choices are intentionally
fixed in the adapter instead of being test-tunable CLI options.
"""

from __future__ import annotations

import argparse
import json
import sys

from pimsr_benchmarks.mtdlpy import (
    COMMON_RETRAIN_SEEDS,
    IMAGENET_RESNET50_V1_SHA256,
    run_common_retrain,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed pinned MTDLPy common-retraining adapter"
    )
    parser.add_argument("--repo", required=True, help="clean pinned MTDLPy clone")
    parser.add_argument("--imagenet-weights", required=True)
    parser.add_argument(
        "--imagenet-weights-sha256",
        required=True,
        help=f"required reviewed hash: {IMAGENET_RESNET50_V1_SHA256}",
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
        imagenet_weights_path=args.imagenet_weights,
        imagenet_weights_sha256=args.imagenet_weights_sha256,
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
