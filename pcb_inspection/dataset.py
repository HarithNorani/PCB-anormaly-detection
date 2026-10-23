"""DeepPCB file discovery and annotation parsing."""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable


class DefectClass(IntEnum):
    """Class IDs defined by the DeepPCB annotation format."""

    OPEN = 1
    SHORT = 2
    MOUSEBITE = 3
    SPUR = 4
    COPPER = 5
    PIN_HOLE = 6

    @property
    def label(self) -> str:
        return {
            DefectClass.OPEN: "open",
            DefectClass.SHORT: "short",
            DefectClass.MOUSEBITE: "mousebite",
            DefectClass.SPUR: "spur",
            DefectClass.COPPER: "copper",
            DefectClass.PIN_HOLE: "pin-hole",
        }[self]


@dataclass(frozen=True)
class Annotation:
    """One axis-aligned DeepPCB defect annotation."""

    x1: int
    y1: int
    x2: int
    y2: int
    defect: DefectClass

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("annotation coordinates must be non-negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("annotation must have positive width and height")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DatasetSample:
    """A paired DeepPCB template/test image and its annotations."""

    sample_id: str
    test_image: Path
    template_image: Path
    annotation_file: Path
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True)
class DatasetIssue:
    """A non-fatal issue found while discovering a dataset."""

    sample_id: str
    message: str


def _parse_integer(value: str, *, field: str, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"line {line_number}: {field} must be an integer, got {value!r}"
        ) from error


def parse_annotation_lines(lines: Iterable[str]) -> tuple[Annotation, ...]:
    """Parse DeepPCB records in ``x1,y1,x2,y2,type`` format."""

    annotations: list[Annotation] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = (
            [field.strip() for field in line.split(",")]
            if "," in line
            else line.split()
        )
        if len(fields) != 5:
            raise ValueError(
                f"line {line_number}: expected 5 comma-separated fields, "
                f"got {len(fields)}"
            )
        x1, y1, x2, y2, class_id = (
            _parse_integer(fields[index], field=name, line_number=line_number)
            for index, name in enumerate(("x1", "y1", "x2", "y2", "type"))
        )
        try:
            defect = DefectClass(class_id)
        except ValueError as error:
            valid_ids = ", ".join(str(item.value) for item in DefectClass)
            raise ValueError(
                f"line {line_number}: unsupported defect class {class_id}; "
                f"expected one of {valid_ids}"
            ) from error
        try:
            annotations.append(Annotation(x1, y1, x2, y2, defect))
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
    return tuple(annotations)


def parse_annotation_file(path: Path) -> tuple[Annotation, ...]:
    """Read and parse one DeepPCB annotation file."""

    return parse_annotation_lines(path.read_text(encoding="utf-8").splitlines())


def discover_samples(root: Path) -> tuple[tuple[DatasetSample, ...], tuple[DatasetIssue, ...]]:
    """Discover complete DeepPCB samples below ``root``.

    Incomplete or malformed samples are returned as issues instead of being
    silently discarded.
    """

    if not root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist: {root}")

    search_root = root / "PCBData" if (root / "PCBData").is_dir() else root
    samples: list[DatasetSample] = []
    issues: list[DatasetIssue] = []
    for test_image in sorted(search_root.rglob("*_test.jpg")):
        sample_id = test_image.name.removesuffix("_test.jpg")
        template_image = test_image.with_name(f"{sample_id}_temp.jpg")
        annotation_candidates = (
            test_image.with_name(f"{sample_id}.txt"),
            test_image.parent.parent
            / f"{test_image.parent.name}_not"
            / f"{sample_id}.txt",
        )
        annotation_file = next(
            (path for path in annotation_candidates if path.is_file()),
            annotation_candidates[0],
        )
        missing = [str(path.name) for path in (template_image, annotation_file) if not path.is_file()]
        if missing:
            issues.append(
                DatasetIssue(sample_id, f"missing paired file(s): {', '.join(missing)}")
            )
            continue
        try:
            annotations = parse_annotation_file(annotation_file)
        except (OSError, ValueError) as error:
            issues.append(DatasetIssue(sample_id, str(error)))
            continue
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                test_image=test_image,
                template_image=template_image,
                annotation_file=annotation_file,
                annotations=annotations,
            )
        )
    return tuple(samples), tuple(issues)
