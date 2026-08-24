"""Train or infer with the pinned MT2DInv-DenseNet common-retrain adapter."""

from __future__ import annotations

import argparse
import json
import sys

from pimsr_benchmarks.densenet2d import (
    COMMON_RETRAIN_SEEDS,
    run_common_inference,
    train_common_retrain,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed pinned MT2DInv-DenseNet common-retrain adapter"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    train = subparsers.add_parser(
        "train",
        help="train one reusable seed checkpoint from train and validation only",
    )
    train.add_argument(
        "--repo",
        required=True,
        help="clean MT2DInv-DenseNet v1.2 clone at the pinned commit",
    )
    train.add_argument("--train-h5", required=True)
    train.add_argument("--validation-h5", required=True)
    train.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=COMMON_RETRAIN_SEEDS,
    )
    train.add_argument("--device", required=True, choices=("cpu", "cuda"))
    train.add_argument("--checkpoint-out", required=True)

    infer = subparsers.add_parser(
        "infer",
        help="infer one truth-free campaign from an existing seed checkpoint",
    )
    infer.add_argument(
        "--repo",
        required=True,
        help="clean MT2DInv-DenseNet v1.2 clone at the pinned commit",
    )
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--expected-checkpoint-sha256")
    infer.add_argument(
        "--observations-npz",
        required=True,
        help="truth-free pimsr-sota-2d-observations payload",
    )
    infer.add_argument("--expected-observations-sha256")
    infer.add_argument("--device", required=True, choices=("cpu", "cuda"))
    infer.add_argument("--predictions-out", required=True)
    infer.add_argument("--runtime-out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_arguments = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(raw_arguments)
    command = [sys.executable, str(__file__), *raw_arguments]
    if args.operation == "train":
        result = train_common_retrain(
            repository_path=args.repo,
            train_h5=args.train_h5,
            validation_h5=args.validation_h5,
            seed=args.seed,
            device=args.device,
            checkpoint_out=args.checkpoint_out,
            command=command,
            runner_source=__file__,
        )
        summary = {
            "method_id": result["method_id"],
            "method": result["method"],
            "seed": result["seed"],
            "checkpoint": result["checkpoint"],
        }
    else:
        result = run_common_inference(
            repository_path=args.repo,
            checkpoint_path=args.checkpoint,
            observations_npz=args.observations_npz,
            device=args.device,
            predictions_out=args.predictions_out,
            runtime_out=args.runtime_out,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_observations_sha256=args.expected_observations_sha256,
            command=command,
            runner_source=__file__,
        )
        summary = {
            "method_id": result["method_id"],
            "method": result["method"],
            "seed": result["seed"],
            "outputs": result["outputs"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
