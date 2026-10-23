"""Prepare DeepPCB samples for six-class detector training."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dataset import DatasetSample, discover_samples


@dataclass(frozen=True)
class PreparationResult:
    """Summary of a detector-dataset export."""

    exported_samples: int
    exported_annotations: int
    issues: int
    output_dir: Path


def _yolo_record(sample: DatasetSample) -> str:
    """Convert one DeepPCB sample to normalized YOLO records."""

    lines: list[str] = []
    for annotation in sample.annotations:
        center_x = (annotation.x1 + annotation.x2) / 2 / 640
        center_y = (annotation.y1 + annotation.y2) / 2 / 640
        width = annotation.width / 640
        height = annotation.height / 640
        class_id = annotation.defect.value - 1
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_yolo_dataset(
    root: Path, output_dir: Path, *, validation_ratio: float = 0.2
) -> PreparationResult:
    """Export tested images and zero-based YOLO labels.

    The export keeps the tested image as the detector input. Template images
    remain available in the source dataset for the classical comparison
    baseline and are not silently concatenated into the model input.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be greater than 0 and less than 1")
    samples, issues = discover_samples(root)
    validation_start = round(len(samples) * (1 - validation_ratio))
    exported_annotations = 0
    exported_samples = 0
    for index, sample in enumerate(samples):
        split = "train" if index < validation_start else "val"
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.test_image, image_dir / sample.test_image.name)
        (label_dir / f"{sample.test_image.stem}.txt").write_text(
            _yolo_record(sample), encoding="utf-8"
        )
        exported_samples += 1
        exported_annotations += len(sample.annotations)

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": [
            "open",
            "short",
            "mousebite",
            "spur",
            "copper",
            "pin-hole",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.yaml").write_text(
        json.dumps(dataset_yaml, indent=2) + "\n", encoding="utf-8"
    )
    return PreparationResult(
        exported_samples, exported_annotations, len(issues), output_dir
    )
