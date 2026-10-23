"""Controlled robustness benchmarks for the classical inspection baseline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import numpy as np

from .baseline import (
    Evaluation,
    detect,
    estimate_translation,
    evaluate,
    load_grayscale,
    normalize_illumination,
    translate_image,
)
from .dataset import Annotation, DatasetSample, discover_samples


def _shift_annotations(
    annotations: tuple[Annotation, ...], shift: tuple[int, int], shape: tuple[int, int]
) -> tuple[Annotation, ...]:
    dx, dy = shift
    height, width = shape
    shifted: list[Annotation] = []
    for annotation in annotations:
        x1 = max(0, annotation.x1 + dx)
        y1 = max(0, annotation.y1 + dy)
        x2 = min(width, annotation.x2 + dx)
        y2 = min(height, annotation.y2 + dy)
        if x2 > x1 and y2 > y1:
            shifted.append(replace(annotation, x1=x1, y1=y1, x2=x2, y2=y2))
    return tuple(shifted)


def _aggregate(total: Evaluation, current: Evaluation) -> Evaluation:
    return Evaluation(
        total.true_positives + current.true_positives,
        total.false_positives + current.false_positives,
        total.false_negatives + current.false_negatives,
    )


def _condition(
    template: np.ndarray, tested: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if name == "baseline":
        return template, tested, (0, 0)
    if name == "brightness":
        return template, np.clip(tested.astype(np.int16) + 40, 0, 255).astype(np.uint8), (0, 0)
    if name == "contrast":
        adjusted = (tested.astype(np.float32) - 128) * 0.7 + 128
        return template, np.clip(adjusted, 0, 255).astype(np.uint8), (0, 0)
    if name == "noise":
        rng = np.random.default_rng(42)
        noisy = tested.astype(np.int16) + rng.normal(0, 10, tested.shape)
        return template, np.clip(noisy, 0, 255).astype(np.uint8), (0, 0)
    if name == "shift":
        shift = (3, 2)
        return template, translate_image(tested, shift), shift
    raise ValueError(f"unknown robustness condition: {name}")


def run_robustness(
    root: Path,
    *,
    conditions: tuple[str, ...] = ("baseline", "brightness", "contrast", "noise", "shift"),
    limit: int | None = 25,
    threshold: int = 30,
    min_area: int = 4,
    mitigate: bool = False,
) -> dict[str, object]:
    """Evaluate baseline localization under controlled image perturbations."""

    valid_conditions = {"baseline", "brightness", "contrast", "noise", "shift"}
    invalid_conditions = set(conditions) - valid_conditions
    if invalid_conditions:
        raise ValueError(
            "unknown robustness condition(s): " + ", ".join(sorted(invalid_conditions))
        )
    samples, issues = discover_samples(root)
    selected = samples[:limit] if limit is not None else samples
    results: dict[str, dict[str, float | int]] = {}
    for condition in conditions:
        total = Evaluation(0, 0, 0)
        for sample in selected:
            template = load_grayscale(sample.template_image)
            original_tested = load_grayscale(sample.test_image)
            template, tested, shift = _condition(template, original_tested, condition)
            annotations = _shift_annotations(sample.annotations, shift, tested.shape)
            if mitigate and condition in {"brightness", "contrast", "noise"}:
                template, tested = normalize_illumination(template, tested)
            if mitigate and condition == "shift":
                estimated = estimate_translation(template, tested, max_shift=5)
                tested = translate_image(tested, estimated)
                annotations = _shift_annotations(annotations, estimated, tested.shape)
            total = _aggregate(
                total,
                evaluate(
                    detect(template, tested, threshold=threshold, min_area=min_area),
                    annotations,
                ),
            )
        results[condition] = {
            "true_positives": total.true_positives,
            "false_positives": total.false_positives,
            "false_negatives": total.false_negatives,
            "precision": total.precision,
            "recall": total.recall,
            "f1": total.f1,
        }
    return {
        "samples": len(selected),
        "issues": len(issues),
        "threshold": threshold,
        "min_area": min_area,
        "mitigate": mitigate,
        "conditions": results,
    }


def save_robustness(result: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path
