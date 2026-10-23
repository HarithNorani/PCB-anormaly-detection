import numpy as np
import pytest

from pcb_inspection.baseline import (
    detect,
    difference_mask,
    evaluate,
    estimate_translation,
    intersection_over_union,
    normalize_illumination,
    translate_image,
)
from pcb_inspection.dataset import Annotation, DefectClass


def test_difference_detects_changed_pixels() -> None:
    template = np.zeros((5, 5), dtype=np.uint8)
    tested = template.copy()
    tested[1:3, 2:4] = 100

    mask = difference_mask(template, tested, threshold=30)

    assert int(mask.sum()) == 4


def test_detect_groups_changed_pixels_into_a_box() -> None:
    template = np.zeros((6, 7), dtype=np.uint8)
    tested = template.copy()
    tested[1:3, 2:5] = 100

    detections = detect(template, tested, threshold=30, min_area=2)

    assert [(item.x1, item.y1, item.x2, item.y2) for item in detections] == [
        (2, 1, 5, 3)
    ]
    assert detections[0].score == 6.0


def test_evaluate_matches_each_annotation_once() -> None:
    annotation = Annotation(2, 1, 5, 3, DefectClass.OPEN)
    detections = detect(
        np.zeros((6, 7), dtype=np.uint8),
        np.pad(np.full((2, 3), 100, dtype=np.uint8), ((1, 3), (2, 2))),
        threshold=30,
        min_area=2,
    )

    result = evaluate((*detections, detections[0]), (annotation,))

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == pytest.approx(0.5)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(2 / 3)


def test_intersection_over_union() -> None:
    annotation = Annotation(0, 0, 4, 4, DefectClass.OPEN)
    detection = detect(
        np.zeros((4, 4), dtype=np.uint8),
        np.full((4, 4), 100, dtype=np.uint8),
        threshold=30,
        min_area=1,
    )[0]

    assert intersection_over_union(detection, annotation) == 1.0


def test_registration_estimates_and_corrects_translation() -> None:
    template = np.zeros((9, 9), dtype=np.uint8)
    template[3:6, 2:5] = 100
    tested = translate_image(template, (2, -1))

    shift = estimate_translation(template, tested, max_shift=3)

    assert shift == (-2, 1)
    np.testing.assert_array_equal(translate_image(tested, shift), template)


def test_illumination_normalization_removes_global_brightness_change() -> None:
    template = np.arange(25, dtype=np.uint8).reshape(5, 5)
    tested = np.clip(template.astype(np.int16) + 40, 0, 255).astype(np.uint8)

    normalized_template, normalized_tested = normalize_illumination(template, tested)

    assert float(np.abs(normalized_template - normalized_tested).mean()) < 1
