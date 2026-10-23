"""Consolidated comparison report for inspection approaches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def build_comparison(
    detector_metrics: Path,
    robustness_report: Path,
    *,
    detector_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Combine detector validation and classical robustness evidence."""

    metrics = _read_json(detector_metrics)
    robustness = _read_json(robustness_report)
    detector_values = metrics.get("metrics")
    conditions = robustness.get("conditions")
    if not isinstance(detector_values, dict):
        raise ValueError("detector metrics report has no 'metrics' object")
    if not isinstance(conditions, dict):
        raise ValueError("robustness report has no 'conditions' object")
    return {
        "detector": {
            "checkpoint": str(
                detector_checkpoint.resolve()
                if detector_checkpoint is not None
                else metrics.get("checkpoint", "")
            ),
            "dataset": metrics.get("dataset", ""),
            "precision": detector_values.get("metrics/precision(B)"),
            "recall": detector_values.get("metrics/recall(B)"),
            "map50": detector_values.get("metrics/mAP50(B)"),
            "map50_95": detector_values.get("metrics/mAP50-95(B)"),
        },
        "classical_baseline": {
            "samples": robustness.get("samples"),
            "threshold": robustness.get("threshold"),
            "min_area": robustness.get("min_area"),
            "mitigate": robustness.get("mitigate"),
            "conditions": conditions,
        },
        "interpretation": {
            "detector_role": "Primary six-class defect detector",
            "baseline_role": "Reference for template-difference sensitivity",
            "deployment_caveat": (
                "Validation and controlled perturbation results do not replace "
                "a production-line acceptance study."
            ),
        },
    }


def save_comparison(result: dict[str, Any], output_path: Path) -> Path:
    """Write the consolidated comparison as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path
