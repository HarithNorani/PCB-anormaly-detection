"""Visual and machine-readable inspection reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .baseline import Detection, detect_sample, evaluate
from .dataset import DatasetSample


@dataclass(frozen=True)
class ReportDetection:
    """Serializable representation of one detected region."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass(frozen=True)
class InspectionReport:
    """Serializable result for one inspected sample."""

    sample_id: str
    passed: bool
    threshold: int
    min_area: int
    registered: bool
    normalized: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    detections: tuple[ReportDetection, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report mapping."""

        return asdict(self)


def draw_detections(
    image: Image.Image, detections: Iterable[Detection]
) -> Image.Image:
    """Return an RGB image with detected regions drawn in red."""

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        box = (detection.x1, detection.y1, detection.x2 - 1, detection.y2 - 1)
        draw.rectangle(box, outline=(220, 30, 30), width=2)
        draw.text(
            (detection.x1, max(0, detection.y1 - 12)),
            f"{detection.score:.0f}",
            fill=(220, 30, 30),
        )
    return annotated


def create_report(
    sample: DatasetSample,
    *,
    threshold: int = 30,
    min_area: int = 4,
    register: bool = False,
    normalize: bool = False,
    max_shift: int = 5,
) -> tuple[InspectionReport, Image.Image]:
    """Inspect one sample and return its report and annotated test image."""

    detections = detect_sample(
        sample,
        threshold=threshold,
        min_area=min_area,
        register=register,
        normalize=normalize,
        max_shift=max_shift,
    )
    evaluation = evaluate(detections, sample.annotations)
    report = InspectionReport(
        sample_id=sample.sample_id,
        passed=evaluation.false_positives == 0 and evaluation.false_negatives == 0,
        threshold=threshold,
        min_area=min_area,
        registered=register,
        normalized=normalize,
        true_positives=evaluation.true_positives,
        false_positives=evaluation.false_positives,
        false_negatives=evaluation.false_negatives,
        detections=tuple(
            ReportDetection(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.score,
            )
            for detection in detections
        ),
    )
    with Image.open(sample.test_image) as image:
        annotated = draw_detections(image, detections)
    return report, annotated


def save_report(
    report: InspectionReport, annotated: Image.Image, output_dir: Path
) -> tuple[Path, Path]:
    """Save one annotated image and JSON report, returning both paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{report.sample_id}_inspection.png"
    json_path = output_dir / f"{report.sample_id}_inspection.json"
    annotated.save(image_path)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return image_path, json_path
