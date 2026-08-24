"""Train once and infer campaigns with pinned MTDLPy DinkNet50."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pimsr_benchmarks.mtdlpy import (
    COMMON_RETRAIN_SEEDS,
    DEFAULT_RECIPE_ID,
    IMAGENET_RESNET50_V1_SHA256,
    TRAINING_RECIPES,
    infer_common_retrain,
    train_common_retrain,
)


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
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
        "--seed",
        required=True,
        type=int,
        choices=COMMON_RETRAIN_SEEDS,
    )
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE_ID,
        choices=tuple(sorted(TRAINING_RECIPES)),
        help="closed-set public-validation recipe; free hyperparameters are forbidden",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed pinned MTDLPy adapter with campaign-independent training"
        )
    )
    operations = parser.add_subparsers(dest="operation", required=True)

    train = operations.add_parser(
        "train",
        help="train and publish one reusable checkpoint for a preregistered seed",
    )
    _add_common_inputs(train)
    train.add_argument("--checkpoint-out", required=True)
    train.add_argument("--runtime-out", required=True)

    infer = operations.add_parser(
        "infer",
        help="infer one truth-free observation campaign from a reusable checkpoint",
    )
    _add_common_inputs(infer)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument(
        "--observations-npz",
        required=True,
        help="truth-free pimsr-sota-2d-observations payload",
    )
    infer.add_argument("--predictions-out", required=True)
    infer.add_argument("--runtime-out", required=True)
    return parser


def _common_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repository_path": args.repo,
        "imagenet_weights_path": args.imagenet_weights,
        "imagenet_weights_sha256": args.imagenet_weights_sha256,
        "train_h5": args.train_h5,
        "validation_h5": args.validation_h5,
        "seed": args.seed,
        "device": args.device,
        "recipe_id": args.recipe,
    }


def main(argv: list[str] | None = None) -> None:
    raw_arguments = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(raw_arguments)
    command = [sys.executable, str(__file__), *raw_arguments]
    common = _common_arguments(args)
    if args.operation == "train":
        result = train_common_retrain(
            **common,
            checkpoint_out=args.checkpoint_out,
            runtime_out=args.runtime_out,
            command=command,
            runner_source=__file__,
        )
    else:
        result = infer_common_retrain(
            **common,
            checkpoint=args.checkpoint,
            observations_npz=args.observations_npz,
            predictions_out=args.predictions_out,
            runtime_out=args.runtime_out,
            command=command,
            runner_source=__file__,
        )
    print(
        json.dumps(
            {
                "method": result["method"],
                "operation": result["operation"],
                "seed": result["seed"],
                "outputs": result["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
