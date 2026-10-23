import json
from pathlib import Path

from pcb_inspection.error_analysis import analyze


def test_analyze_counts_matching_and_unmatched_boxes(tmp_path: Path) -> None:
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
                },
                {
                    "image_id": "sample_test",
                    "category_id": 2,
                    "bbox": [0, 0, 10, 10],
                    "score": 0.8,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = analyze(tmp_path, predictions)

    assert result["classes"]["open"]["true_positives"] == 1
    assert result["classes"]["short"]["false_positives"] == 1
    assert result["classes"]["open"]["recall"] == 1.0
