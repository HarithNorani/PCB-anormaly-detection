import json
from pathlib import Path

from PIL import Image

from pcb_inspection.error_analysis import export_failure_cases


def test_export_failure_cases_writes_annotated_image(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "val"
    images = tmp_path / "images" / "val"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (labels / "sample_test.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )
    Image.new("RGB", (640, 640), "white").save(images / "sample_test.jpg")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": "sample_test",
                    "category_id": 2,
                    "bbox": [0, 0, 10, 10],
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    index_path = export_failure_cases(tmp_path, predictions, tmp_path / "failures")

    assert index_path.is_file()
    assert (tmp_path / "failures" / "sample_test_failure.png").is_file()
    assert json.loads(index_path.read_text())["cases"][0]["false_positives"] == 1
