"""Classical template-difference baseline for DeepPCB samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .dataset import Annotation, DatasetSample


@dataclass(frozen=True)
class Detection:
    """A candidate defect region produced by the baseline."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class Evaluation:
    """Localization metrics for one detector run."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as a two-dimensional uint8 grayscale array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def estimate_translation(
    template: np.ndarray, tested: np.ndarray, *, max_shift: int = 5
) -> tuple[int, int]:
    """Estimate the tested-image translation with minimum overlap MSE."""

    _validate_pair(template, tested)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    best_shift = (0, 0)
    best_error = float("inf")
    height, width = template.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            template_y = slice(max(0, dy), min(height, height + dy))
            template_x = slice(max(0, dx), min(width, width + dx))
            tested_y = slice(max(0, -dy), min(height, height - dy))
            tested_x = slice(max(0, -dx), min(width, width - dx))
            difference = (
                template[template_y, template_x].astype(np.float32)
                - tested[tested_y, tested_x].astype(np.float32)
            )
            error = float(np.mean(difference * difference))
            if error < best_error:
                best_error = error
                best_shift = (dx, dy)
    return best_shift


def translate_image(image: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Translate an image without wrapping pixels around its borders."""

    if image.ndim != 2:
        raise ValueError("image must be a grayscale array")
    dx, dy = shift
    height, width = image.shape
    result = np.empty_like(image)
    pad_y = abs(dy)
    pad_x = abs(dx)
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    start_y = pad_y - dy
    start_x = pad_x - dx
    result[:] = padded[start_y : start_y + height, start_x : start_x + width]
    return result


def normalize_illumination(
    template: np.ndarray, tested: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match tested-image mean and contrast to the template."""

    _validate_pair(template, tested)
    template_float = template.astype(np.float32)
    tested_float = tested.astype(np.float32)
    tested_std = float(tested_float.std())
    template_std = float(template_float.std())
    if tested_std < 1e-6:
        normalized = tested_float - tested_float.mean() + template_float.mean()
    else:
        normalized = (
            (tested_float - tested_float.mean())
            * (template_std / max(tested_std, 1e-6))
            + template_float.mean()
        )
    return template_float, np.clip(normalized, 0, 255)


def _validate_pair(template: np.ndarray, tested: np.ndarray) -> None:
    if template.shape != tested.shape:
        raise ValueError(
            f"template and tested images must have the same shape, "
            f"got {template.shape} and {tested.shape}"
        )
    if template.ndim != 2 or tested.ndim != 2:
        raise ValueError("template and tested images must be grayscale arrays")


def difference_mask(
    template: np.ndarray, tested: np.ndarray, *, threshold: int = 30
) -> np.ndarray:
    """Return pixels whose absolute template difference exceeds ``threshold``."""

    _validate_pair(template, tested)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    return np.abs(template.astype(np.int16) - tested.astype(np.int16)) > threshold


def _component_boxes(mask: np.ndarray) -> Iterable[tuple[int, int, int, int, int]]:
    """Yield connected-component boxes and pixel areas using 8-connectivity."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        area = 0
        while stack:
            current_y, current_x = stack.pop()
            area += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if not offset_x and not offset_y:
                        continue
                    next_y = current_y + offset_y
                    next_x = current_x + offset_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        yield min_x, min_y, max_x + 1, max_y + 1, area


def detect(
    template: np.ndarray,
    tested: np.ndarray,
    *,
    threshold: int = 30,
    min_area: int = 4,
) -> tuple[Detection, ...]:
    """Detect connected regions in a thresholded template difference."""

    if min_area < 1:
        raise ValueError("min_area must be at least 1")
    mask = difference_mask(template, tested, threshold=threshold)
    detections = [
        Detection(x1, y1, x2, y2, float(area))
        for x1, y1, x2, y2, area in _component_boxes(mask)
        if area >= min_area
    ]
    return tuple(sorted(detections, key=lambda item: item.score, reverse=True))


def detect_sample(
    sample: DatasetSample,
    *,
    threshold: int = 30,
    min_area: int = 4,
    register: bool = False,
    normalize: bool = False,
    max_shift: int = 5,
) -> tuple[Detection, ...]:
    """Run the baseline on one discovered DeepPCB sample."""

    template = load_grayscale(sample.template_image)
    tested = load_grayscale(sample.test_image)
    if register:
        dx, dy = estimate_translation(template, tested, max_shift=max_shift)
        tested = translate_image(tested, (dx, dy))
    if normalize:
        template, tested = normalize_illumination(template, tested)
    return detect(
        template,
        tested,
        threshold=threshold,
        min_area=min_area,
    )


def intersection_over_union(
    left: Detection | Annotation, right: Detection | Annotation
) -> float:
    """Calculate IoU for two axis-aligned boxes."""

    intersection_x1 = max(left.x1, right.x1)
    intersection_y1 = max(left.y1, right.y1)
    intersection_x2 = min(left.x2, right.x2)
    intersection_y2 = min(left.y2, right.y2)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union_area = left_area + right_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


def evaluate(
    detections: Iterable[Detection],
    annotations: Iterable[Annotation],
    *,
    iou_threshold: float = 0.33,
) -> Evaluation:
    """Evaluate class-agnostic localization with one-to-one matching."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    remaining = list(annotations)
    true_positives = 0
    false_positives = 0
    for detection in detections:
        matches = [
            (index, intersection_over_union(detection, annotation))
            for index, annotation in enumerate(remaining)
        ]
        if not matches:
            false_positives += 1
            continue
        best_index, best_iou = max(matches, key=lambda item: item[1])
        if best_iou >= iou_threshold:
            true_positives += 1
            remaining.pop(best_index)
        else:
            false_positives += 1
    return Evaluation(true_positives, false_positives, len(remaining))
