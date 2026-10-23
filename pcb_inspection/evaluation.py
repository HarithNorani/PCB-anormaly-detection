"""Evaluation entry point for a trained YOLO detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .training import validate_prepared_dataset


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def evaluate_detector(
    checkpoint: Path,
    dataset_yaml: Path,
    output_dir: Path,
    *,
    image_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
) -> Path:
    """Run validation and save Ultralytics metrics as JSON."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
    validate_prepared_dataset(dataset_yaml)
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "AI evaluation requires the optional dependency; install it with "
            "'python -m pip install -e \".[ai]\""
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    val_kwargs: dict[str, object] = {
        "data": str(dataset_yaml),
        "split": "val",
        "imgsz": image_size,
        "batch": batch_size,
        "project": str(output_dir),
        "name": "validation",
        "plots": True,
        "save_json": True,
    }
    if device != "auto":
        val_kwargs["device"] = device
    metrics = model.val(**val_kwargs)
    results = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": str(dataset_yaml.resolve()),
        "image_size": image_size,
        "batch_size": batch_size,
        "metrics": _json_value(getattr(metrics, "results_dict", {})),
        "save_dir": str(Path(str(metrics.save_dir)).resolve()),
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return report_path
