from pathlib import Path

import pytest

from pcb_inspection.training import TrainingConfig, validate_prepared_dataset


def test_validate_prepared_dataset_rejects_incomplete_export(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing directory"):
        validate_prepared_dataset(dataset_yaml)


def test_training_config_rejects_invalid_epochs(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("{}\n", encoding="utf-8")
    config = TrainingConfig(dataset_yaml, tmp_path / "runs", epochs=0)

    with pytest.raises(ValueError, match="epochs"):
        config.validate()
