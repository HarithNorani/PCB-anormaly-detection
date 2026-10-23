"""Optional YOLO training entry point for the six-class detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit configuration for a reproducible detector run."""

    dataset_yaml: Path
    output_dir: Path
    model: str = "yolo11n.pt"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if not self.dataset_yaml.is_file():
            raise FileNotFoundError(f"dataset configuration not found: {self.dataset_yaml}")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


def validate_prepared_dataset(dataset_yaml: Path) -> None:
    """Check the exported dataset structure before invoking a trainer."""

    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"dataset configuration not found: {dataset_yaml}")
    root = dataset_yaml.parent
    required = (
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "prepared dataset is incomplete; missing directory(s): " + ", ".join(missing)
        )


def train_detector(config: TrainingConfig) -> Any:
    """Train a YOLO detector using the optional Ultralytics dependency."""

    config.validate()
    validate_prepared_dataset(config.dataset_yaml)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "AI training requires the optional dependency; install it with "
            "'python -m pip install -e \".[ai]\""
        ) from error

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.model)
    train_kwargs: dict[str, object] = {
        "data": str(config.dataset_yaml),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "seed": config.seed,
        "project": str(config.output_dir),
        "name": "pcb-defect-detector",
    }
    if config.device != "auto":
        train_kwargs["device"] = config.device
    return model.train(**train_kwargs)
