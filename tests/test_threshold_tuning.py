import json
from pathlib import Path

from pcb_inspection.error_analysis import tune_thresholds


def test_tune_thresholds_returns_per_class_recommendations(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "val"
    labels.mkdir(parents=True)
    (labels / "sample_test.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": "sample_test",
                    "category_id": 1,
                    "bbox": [240, 240, 160, 160],
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = tune_thresholds(
        tmp_path, predictions, thresholds=(0.25, 0.95)
    )

    assert result["recommended"]["open"]["threshold"] == 0.25
    assert len(result["classes"]["open"]) == 2
