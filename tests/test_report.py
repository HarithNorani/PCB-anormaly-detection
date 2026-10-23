import json
from pathlib import Path

import numpy as np
from PIL import Image

from pcb_inspection.dataset import Annotation, DatasetSample, DefectClass
from pcb_inspection.report import (
    InspectionReport,
    ReportDetection,
    draw_detections,
    save_report,
)
from pcb_inspection.baseline import Detection


def test_draw_detections_returns_rgb_image() -> None:
    image = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))

    annotated = draw_detections(image, [Detection(2, 3, 7, 8, 12)])

    assert annotated.mode == "RGB"
    assert annotated.size == (10, 10)
    assert annotated.getpixel((2, 3)) == (220, 30, 30)


def test_save_report_writes_image_and_json(tmp_path: Path) -> None:
    report = InspectionReport(
        sample_id="sample",
        passed=False,
        threshold=30,
        min_area=4,
        registered=True,
        normalized=True,
        true_positives=1,
        false_positives=1,
        false_negatives=0,
        detections=(ReportDetection(1, 2, 4, 5, 9.0),),
    )
    image = Image.new("RGB", (4, 4))

    image_path, json_path = save_report(report, image, tmp_path)

    assert image_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["passed"] is False
