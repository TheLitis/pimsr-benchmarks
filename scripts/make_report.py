"""Render the retired four-file 1D comparison as a diagnostic report.

The input files predate immutable benchmark manifests. In particular, the
synthetic neural result consumes MT plus gravity while the Occam result
consumes MT only. This renderer therefore cannot produce a ranking or a
headline result. It exists only to preserve the historical measurements in
an explicitly non-comparable document.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPARISON_STATUS = "diagnostic_non_comparable"
RANKING_ALLOWED = False
_DIAGNOSTIC_STATUSES = {COMPARISON_STATUS, "legacy_non_comparable"}
_SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}
_INPUT_FILES = {
    "occam_synthetic": "occam_synthetic.json",
    "neural_synthetic": "neural_synthetic.json",
    "occam_real": "occam_real.json",
    "neural_real": "neural_real.json",
}


def fmt(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def _stable_json_object(path: Path) -> dict[str, Any]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after or len(payload) != before.st_size:
        raise RuntimeError(f"benchmark input changed while being read: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON benchmark input: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"benchmark input must be a JSON object: {path}")
    return value


def _number(value: object, *, where: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{where} must be a {qualifier} number")
    return result


def _positive_count(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be an object")
    return value


def _summary(
    result: Mapping[str, Any], key: str, *, where: str, expected_count: int
) -> Mapping[str, Any]:
    summary = _mapping(result.get(key), where=f"{where}.{key}")
    for statistic in ("mean", "median", "p90"):
        if statistic in summary:
            _number(summary[statistic], where=f"{where}.{key}.{statistic}")
    if "mean" not in summary or "median" not in summary:
        raise ValueError(f"{where}.{key} must declare mean and median")
    if (
        "n" in summary
        and _positive_count(summary["n"], where=f"{where}.{key}.n") != expected_count
    ):
        raise ValueError(f"{where}.{key}.n does not match the result count")
    return summary


def _validate_status(result: Mapping[str, Any], *, where: str) -> None:
    ranking = result.get("ranking_allowed")
    if ranking is not None and ranking is not False:
        raise ValueError(f"{where}.ranking_allowed must be false")

    comparison_status = result.get("comparison_status")
    if comparison_status is not None and comparison_status not in _DIAGNOSTIC_STATUSES:
        raise ValueError(
            f"{where}.comparison_status must mark a non-comparable diagnostic"
        )

    status = result.get("status")
    if status is not None and status not in _SUCCESS_STATUSES | _DIAGNOSTIC_STATUSES:
        raise ValueError(f"{where}.status is not a successful diagnostic input")


def _declared_budget(result: Mapping[str, Any], *, where: str) -> tuple[str, ...] | None:
    value = result.get("inverse_observation_budget")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where}.inverse_observation_budget must be a list")
    if not value or any(not isinstance(mode, str) or not mode for mode in value):
        raise ValueError(
            f"{where}.inverse_observation_budget must contain observation names"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"{where}.inverse_observation_budget contains duplicates")
    return tuple(value)


def _validate_pair_declarations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    pair_name: str,
    expected_budgets: tuple[tuple[str, ...], tuple[str, ...]],
) -> str:
    metrics = (left.get("metric_id"), right.get("metric_id"))
    if any(metric is not None for metric in metrics):
        if any(not isinstance(metric, str) or not metric for metric in metrics):
            raise ValueError(f"{pair_name} metric_id must be declared by both inputs")
        if metrics[0] != metrics[1]:
            raise ValueError(f"{pair_name} inputs declare different metric_id values")
        metric_note = f"declared identically as `{metrics[0]}`"
    else:
        metric_note = "not declared in either legacy input"

    budgets = (
        _declared_budget(left, where=f"{pair_name}.left"),
        _declared_budget(right, where=f"{pair_name}.right"),
    )
    if any(budget is not None for budget in budgets):
        if any(budget is None for budget in budgets):
            raise ValueError(
                f"{pair_name} observation budget must be declared by both inputs"
            )
        if budgets != expected_budgets:
            raise ValueError(f"{pair_name} declares an unexpected observation budget")
    return metric_note


def _validate_inputs(results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    occam_s = results["occam_synthetic"]
    neural_s = results["neural_synthetic"]
    occam_r = results["occam_real"]
    neural_r = results["neural_real"]

    expected_methods = {
        "occam_synthetic": "occam1d",
        "neural_synthetic": "pimsr-neural",
        "occam_real": "occam1d",
        "neural_real": "pimsr-neural",
    }
    for name, result in results.items():
        _validate_status(result, where=name)
        if result.get("method") != expected_methods[name]:
            raise ValueError(f"{name}.method is not the expected legacy method")

    synthetic_count = _positive_count(occam_s.get("n"), where="occam_synthetic.n")
    if _positive_count(neural_s.get("n"), where="neural_synthetic.n") != synthetic_count:
        raise ValueError("synthetic inputs do not contain the same sample count")
    _summary(
        occam_s,
        "rmse_log10_res",
        where="occam_synthetic",
        expected_count=synthetic_count,
    )
    _summary(
        neural_s,
        "rmse_log10_res",
        where="neural_synthetic",
        expected_count=synthetic_count,
    )
    _number(
        occam_s.get("time_per_station_s"),
        where="occam_synthetic.time_per_station_s",
        positive=True,
    )
    _number(
        neural_s.get("time_per_station_s"),
        where="neural_synthetic.time_per_station_s",
        positive=True,
    )

    occam_scenarios = _mapping(
        occam_s.get("per_scenario_rmse"),
        where="occam_synthetic.per_scenario_rmse",
    )
    neural_scenarios = _mapping(
        neural_s.get("per_scenario_rmse"),
        where="neural_synthetic.per_scenario_rmse",
    )
    if set(occam_scenarios) != set(neural_scenarios):
        raise ValueError("synthetic inputs do not contain the same scenario set")
    for method, scenarios in (
        ("occam_synthetic", occam_scenarios),
        ("neural_synthetic", neural_scenarios),
    ):
        for scenario, value in scenarios.items():
            _number(value, where=f"{method}.per_scenario_rmse.{scenario}")

    real_count = _positive_count(occam_r.get("n_stations"), where="occam_real.n_stations")
    if (
        _positive_count(neural_r.get("n_stations"), where="neural_real.n_stations")
        != real_count
    ):
        raise ValueError("real inputs do not contain the same station count")
    _summary(occam_r, "nrms", where="occam_real", expected_count=real_count)
    _summary(neural_r, "nrms", where="neural_real", expected_count=real_count)
    if occam_r.get("dataset") != neural_r.get("dataset"):
        raise ValueError("real inputs do not identify the same dataset")

    for name, result in (("occam_real", occam_r), ("neural_real", neural_r)):
        stations = result.get("stations")
        if not isinstance(stations, list) or len(stations) != real_count:
            raise ValueError(f"{name}.stations does not match n_stations")
    occam_ids = [station.get("station") for station in occam_r["stations"]]
    neural_ids = [station.get("station") for station in neural_r["stations"]]
    if occam_ids != neural_ids or any(not isinstance(item, str) for item in occam_ids):
        raise ValueError("real inputs do not contain the same ordered station identities")

    return {
        "synthetic_metric": _validate_pair_declarations(
            occam_s,
            neural_s,
            pair_name="synthetic",
            expected_budgets=(("mt",), ("mt", "gravity")),
        ),
        "real_metric": _validate_pair_declarations(
            occam_r,
            neural_r,
            pair_name="real",
            expected_budgets=(("mt",), ("mt",)),
        ),
    }


def render_diagnostic_report(results_dir: str | Path) -> str:
    directory = Path(results_dir)
    results = {
        name: _stable_json_object(directory / filename)
        for name, filename in _INPUT_FILES.items()
    }
    declarations = _validate_inputs(results)
    occam_s = results["occam_synthetic"]
    neural_s = results["neural_synthetic"]
    occam_r = results["occam_real"]
    neural_r = results["neural_real"]
    occam_rmse = occam_s["rmse_log10_res"]
    neural_rmse = neural_s["rmse_log10_res"]

    lines = [
        "# Legacy PIMSR diagnostic report (non-comparable)",
        "",
        f"- comparison_status: `{COMPARISON_STATUS}`",
        f"- ranking_allowed: `{str(RANKING_ALLOWED).lower()}`",
        "",
        "> **Scope of this notice:** every numerical cross-method table and every",
        "> numerical comparison in this document is a historical diagnostic only.",
        "> This includes the 1D synthetic, per-scenario and real-station tables.",
        "> None of these values may be used as a ranking, headline, superiority,",
        "> state-of-the-art, production-readiness or deployment claim.",
        "",
        "Reasons this report is non-comparable:",
        "",
        "- the synthetic neural inversion consumed MT plus gravity, while Occam",
        "  consumed MT only; the inverse observation budgets are unequal;",
        "- the real-data producer scripts did not share a versioned error-floor",
        "  contract, so their nRMS values are not proven to use identical weights;",
        "- the legacy JSON files lack immutable dataset/checkpoint identities and",
        "  complete runtime provenance.",
        "",
        "Validated declarations:",
        "",
        f"- synthetic metric: {declarations['synthetic_metric']};",
        f"- real metric: {declarations['real_metric']};",
        "- known synthetic inverse budgets: Occam `MT`; neural `MT + gravity`;",
        "- known real inverse budgets: both `MT` (neural gravity mean-filled).",
        "",
        "## Archived synthetic diagnostic values",
        "",
        f"Samples: {occam_s['n']} per method.",
        "",
        "| metric | Occam 1D (MT-only) | PIMSR neural (MT+gravity) |",
        "|---|---:|---:|",
        f"| RMSE log10(rho) mean | {fmt(occam_rmse['mean'])} | {fmt(neural_rmse['mean'])} |",
        f"| RMSE log10(rho) median | {fmt(occam_rmse['median'])} | {fmt(neural_rmse['median'])} |",
        f"| RMSE log10(rho) p90 | {fmt(occam_rmse['p90'])} | {fmt(neural_rmse['p90'])} |",
        f"| observed time/station (s) | {fmt(occam_s['time_per_station_s'], 4)} | {fmt(neural_s['time_per_station_s'], 4)} |",
    ]
    if "scenario_accuracy" in neural_s:
        scenario_accuracy = _number(
            neural_s["scenario_accuracy"],
            where="neural_synthetic.scenario_accuracy",
        )
        lines.append(f"| scenario accuracy | n/a | {fmt(scenario_accuracy)} |")
    if "sigma_coverage_1" in neural_s:
        coverage = _number(
            neural_s["sigma_coverage_1"],
            where="neural_synthetic.sigma_coverage_1",
        )
        lines.append(f"| 1-sigma coverage (reference 0.683) | n/a | {fmt(coverage)} |")

    lines += [
        "",
        "### Archived per-scenario diagnostic values",
        "",
        "| scenario | Occam 1D (MT-only) | PIMSR neural (MT+gravity) |",
        "|---|---:|---:|",
    ]
    for scenario in sorted(occam_scenarios := occam_s["per_scenario_rmse"]):
        lines.append(
            f"| {scenario} | {fmt(occam_scenarios[scenario])} | "
            f"{fmt(neural_s['per_scenario_rmse'][scenario])} |"
        )

    lines += [
        "",
        "## Archived real-station diagnostic values",
        "",
        f"Stations: {occam_r['n_stations']} ({occam_r['dataset']}).",
        "",
        "| metric | Occam 1D | PIMSR neural |",
        "|---|---:|---:|",
        f"| reported nRMS mean | {fmt(occam_r['nrms']['mean'])} | {fmt(neural_r['nrms']['mean'])} |",
        f"| reported nRMS median | {fmt(occam_r['nrms']['median'])} | {fmt(neural_r['nrms']['median'])} |",
        "",
        "Ground truth is unavailable for these field stations. The reported nRMS",
        "values are retained as observations from their original producer scripts;",
        "because their metric/error-floor contract was not shared, even their",
        "relative ordering is not a valid cross-method conclusion.",
        "",
        "## Reproduction boundary",
        "",
        "This renderer requires `--allow-legacy-diagnostic` and refuses to replace",
        "an existing output. Publishable comparisons must instead follow",
        "`docs/SOTA_PROTOCOL.md` with equal observation budgets, immutable manifests",
        "and one shared metric contract.",
    ]
    return "\n".join(lines) + "\n"


def _publish_text_no_overwrite(text: str, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(f".{output.name}.{uuid.uuid4().hex}.part")
    payload = text.encode("utf-8")
    try:
        with part.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(part, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing diagnostic report: {output}"
            ) from error
        part.unlink()
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default="REPORT.md")
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="render an explicitly non-comparable historical diagnostic",
    )
    args = parser.parse_args(argv)
    if not args.allow_legacy_diagnostic:
        parser.error(
            "the four-file report is legacy and mixed-budget; pass "
            "--allow-legacy-diagnostic only to render a non-rankable diagnostic"
        )

    report = render_diagnostic_report(args.results_dir)
    output = _publish_text_no_overwrite(report, args.out)
    print(f"wrote non-comparable diagnostic to {output}")


if __name__ == "__main__":
    main()
