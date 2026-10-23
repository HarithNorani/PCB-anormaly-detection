import json
from pathlib import Path

from pcb_inspection.comparison import build_comparison


def test_build_comparison_combines_reports(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "checkpoint": "/model/best.pt",
                "dataset": "/data/dataset.yaml",
                "metrics": {
                    "metrics/precision(B)": 0.9,
                    "metrics/recall(B)": 0.8,
                    "metrics/mAP50(B)": 0.95,
                    "metrics/mAP50-95(B)": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    robustness = tmp_path / "robustness.json"
    robustness.write_text(
        json.dumps({"samples": 2, "conditions": {"baseline": {"f1": 0.1}}}),
        encoding="utf-8",
    )

    result = build_comparison(metrics, robustness)

    assert result["detector"]["map50"] == 0.95
    assert result["classical_baseline"]["conditions"]["baseline"]["f1"] == 0.1
