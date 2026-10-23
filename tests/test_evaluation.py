from pathlib import Path

import pytest

from pcb_inspection.evaluation import evaluate_detector


def test_evaluate_rejects_missing_checkpoint(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        evaluate_detector(tmp_path / "best.pt", dataset_yaml, tmp_path / "out")
