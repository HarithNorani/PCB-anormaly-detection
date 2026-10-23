"""Per-class error analysis for YOLO validation predictions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .dataset import DefectClass


CLASS_NAMES = tuple(item.label for item in DefectClass)


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def _iou(left: Box, right: Box) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _read_labels(label_path: Path, image_size: int = 640) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id, center_x, center_y, width, height = map(float, fields)
        if not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number}: invalid class ID")
        boxes.append(
            Box(
                int(class_id),
                (center_x - width / 2) * image_size,
                (center_y - height / 2) * image_size,
                (center_x + width / 2) * image_size,
                (center_y + height / 2) * image_size,
            )
        )
    return boxes


def _read_predictions(path: Path) -> dict[str, list[Box]]:
    predictions = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[Box]] = {}
    for item in predictions:
        x1, y1, width, height = item["bbox"]
        image_id = str(item["image_id"])
        grouped.setdefault(image_id, []).append(
            Box(
                int(item["category_id"]) - 1,
                float(x1),
                float(y1),
                float(x1 + width),
                float(y1 + height),
                float(item["score"]),
            )
        )
    return grouped


def _match(
    predictions: Iterable[Box], targets: Iterable[Box], iou_threshold: float
) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    remaining = list(targets)
    matches: list[tuple[Box, Box]] = []
    false_positives: list[Box] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (index, _iou(prediction, target))
            for index, target in enumerate(remaining)
        ]
        if not candidates:
            false_positives.append(prediction)
            continue
        best_index, best_iou = max(candidates, key=lambda item: item[1])
        if best_iou >= iou_threshold:
            matches.append((prediction, remaining.pop(best_index)))
        else:
            false_positives.append(prediction)
    return matches, false_positives, remaining


def analyze(
    prepared_dir: Path,
    predictions_path: Path,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    image_size: int = 640,
) -> dict[str, object]:
    """Compute per-class counts and a class-confusion matrix."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    predictions = _read_predictions(predictions_path)
    counts = {
        name: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for name in CLASS_NAMES
    }
    matrix = [[0 for _ in range(len(CLASS_NAMES) + 1)] for _ in range(len(CLASS_NAMES) + 1)]
    for image_id, image_predictions in predictions.items():
        label_path = prepared_dir / "labels" / "val" / f"{image_id}.txt"
        targets = _read_labels(label_path, image_size)
        selected = [
            item for item in image_predictions if item.score >= confidence_threshold
        ]
        matches, false_positives, false_negatives = _match(
            selected, targets, iou_threshold
        )
        for prediction, target in matches:
            if prediction.class_id == target.class_id:
                counts[CLASS_NAMES[target.class_id]]["true_positives"] += 1
            else:
                counts[CLASS_NAMES[prediction.class_id]]["false_positives"] += 1
                counts[CLASS_NAMES[target.class_id]]["false_negatives"] += 1
            matrix[target.class_id][prediction.class_id] += 1
        for prediction in false_positives:
            counts[CLASS_NAMES[prediction.class_id]]["false_positives"] += 1
            matrix[len(CLASS_NAMES)][prediction.class_id] += 1
        for target in false_negatives:
            counts[CLASS_NAMES[target.class_id]]["false_negatives"] += 1
            matrix[target.class_id][len(CLASS_NAMES)] += 1
    for values in counts.values():
        tp = values["true_positives"]
        fp = values["false_positives"]
        fn = values["false_negatives"]
        values["precision"] = tp / (tp + fp) if tp + fp else 0.0
        values["recall"] = tp / (tp + fn) if tp + fn else 0.0
        values["f1"] = (
            2 * values["precision"] * values["recall"]
            / (values["precision"] + values["recall"])
            if values["precision"] + values["recall"]
            else 0.0
        )
    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "classes": counts,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": [*CLASS_NAMES, "background"],
    }


def save_analysis(result: dict[str, object], output_dir: Path) -> Path:
    """Save JSON and a compact per-class CSV report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "error-analysis.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-class-metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["class", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in result["classes"].items():
            writer.writerow({"class": name, **values})
    return json_path


def tune_thresholds(
    prepared_dir: Path,
    predictions_path: Path,
    *,
    iou_threshold: float = 0.5,
    thresholds: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    image_size: int = 640,
) -> dict[str, object]:
    """Find the best F1 threshold independently for each defect class."""

    candidate_thresholds = tuple(thresholds)
    if not candidate_thresholds or any(
        not 0 <= threshold <= 1 for threshold in candidate_thresholds
    ):
        raise ValueError("thresholds must contain values between 0 and 1")
    per_class: dict[str, list[dict[str, float]]] = {name: [] for name in CLASS_NAMES}
    for threshold in candidate_thresholds:
        result = analyze(
            prepared_dir,
            predictions_path,
            iou_threshold=iou_threshold,
            confidence_threshold=threshold,
            image_size=image_size,
        )
        for name, values in result["classes"].items():
            per_class[name].append(
                {
                    "threshold": threshold,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                }
            )
    recommendations = {}
    for name, values in per_class.items():
        recommendations[name] = max(
            values, key=lambda item: (item["f1"], item["recall"])
        )
    return {
        "iou_threshold": iou_threshold,
        "thresholds": candidate_thresholds,
        "classes": per_class,
        "recommended": recommendations,
    }


def save_threshold_tuning(result: dict[str, object], output_dir: Path) -> Path:
    """Save threshold sweep results as JSON and CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "threshold-tuning.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "threshold-tuning.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["class", "threshold", "precision", "recall", "f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in result["classes"].items():
            for row in values:
                writer.writerow({"class": name, **row})
    return json_path


def export_failure_cases(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    limit: int | None = 50,
) -> Path:
    """Export annotated images containing false positives or false negatives."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 or omitted")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    prediction_groups = _read_predictions(predictions_path)
    label_dir = prepared_dir / "labels" / "val"
    image_dir = prepared_dir / "images" / "val"
    cases: list[dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = label_path.stem
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            continue
        targets = _read_labels(label_path)
        predictions = [
            item
            for item in prediction_groups.get(image_id, [])
            if item.score >= confidence_threshold
        ]
        matches, false_positives, false_negatives = _match(
            predictions, targets, iou_threshold
        )
        mismatches = [
            (prediction, target)
            for prediction, target in matches
            if prediction.class_id != target.class_id
        ]
        if not false_positives and not false_negatives and not mismatches:
            continue
        with Image.open(image_path) as source:
            annotated = source.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        for target in false_negatives:
            draw.rectangle(
                (target.x1, target.y1, target.x2, target.y2),
                outline=(255, 165, 0),
                width=3,
            )
            draw.text(
                (target.x1, max(0, target.y1 - 14)),
                f"FN {CLASS_NAMES[target.class_id]}",
                fill=(255, 165, 0),
            )
        for prediction in false_positives:
            draw.rectangle(
                (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                outline=(220, 30, 30),
                width=3,
            )
            draw.text(
                (prediction.x1, max(0, prediction.y1 - 14)),
                f"FP {CLASS_NAMES[prediction.class_id]} {prediction.score:.2f}",
                fill=(220, 30, 30),
            )
        for prediction, target in mismatches:
            draw.rectangle(
                (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                outline=(160, 32, 240),
                width=3,
            )
            draw.text(
                (prediction.x1, max(0, prediction.y1 - 14)),
                f"WRONG {CLASS_NAMES[prediction.class_id]} -> "
                f"{CLASS_NAMES[target.class_id]}",
                fill=(160, 32, 240),
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}_failure.png"
        annotated.save(output_path)
        cases.append(
            {
                "image_id": image_id,
                "image": str(image_path),
                "output": str(output_path),
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
                "class_mismatches": len(mismatches),
            }
        )
        if limit is not None and len(cases) >= limit:
            break
    index_path = output_dir / "failure-cases.json"
    index_path.write_text(
        json.dumps(
            {
                "iou_threshold": iou_threshold,
                "confidence_threshold": confidence_threshold,
                "cases": cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path
