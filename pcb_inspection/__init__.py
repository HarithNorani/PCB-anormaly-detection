"""Core data and inspection utilities for the PCB visual-inspection project."""

from .dataset import (
    Annotation,
    DatasetIssue,
    DefectClass,
    DatasetSample,
    discover_samples,
    parse_annotation_file,
)

__all__ = [
    "Annotation",
    "DatasetIssue",
    "DatasetSample",
    "DefectClass",
    "discover_samples",
    "parse_annotation_file",
]
