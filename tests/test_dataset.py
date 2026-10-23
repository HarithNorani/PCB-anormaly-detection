from pathlib import Path

import pytest

from pcb_inspection.dataset import (
    DefectClass,
    discover_samples,
    parse_annotation_lines,
)


def test_parse_annotation_lines() -> None:
    annotations = parse_annotation_lines(
        ["1, 2, 11, 22, 6", "", "0 0 4 5 1"]
    )

    assert len(annotations) == 2
    assert annotations[0].defect is DefectClass.PIN_HOLE
    assert annotations[0].width == 10
    assert annotations[0].height == 20


@pytest.mark.parametrize(
    "line, message",
    [
        ("1,2,3,4", "expected 5"),
        ("1,2,3,4,99", "unsupported defect class"),
        ("1,2,1,4,1", "positive width"),
        ("-1,2,3,4,1", "non-negative"),
        ("a,2,3,4,1", "x1 must be an integer"),
    ],
)
def test_parse_annotation_lines_rejects_invalid_records(
    line: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_annotation_lines([line])


def test_discover_samples_reports_incomplete_and_malformed_samples(
    tmp_path: Path,
) -> None:
    (tmp_path / "good_test.jpg").touch()
    (tmp_path / "good_temp.jpg").touch()
    (tmp_path / "good.txt").write_text("1,2,11,12,1\n", encoding="utf-8")

    (tmp_path / "missing_test.jpg").touch()
    (tmp_path / "missing.txt").write_text("1,2,11,12,1\n", encoding="utf-8")

    (tmp_path / "bad_test.jpg").touch()
    (tmp_path / "bad_temp.jpg").touch()
    (tmp_path / "bad.txt").write_text("1,2,3,4,99\n", encoding="utf-8")

    samples, issues = discover_samples(tmp_path)

    assert [sample.sample_id for sample in samples] == ["good"]
    assert {issue.sample_id for issue in issues} == {"bad", "missing"}


def test_discover_samples_supports_deeppcb_sibling_annotation_directory(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "group44000" / "44000"
    annotation_dir = tmp_path / "group44000" / "44000_not"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    (image_dir / "44000022_test.jpg").touch()
    (image_dir / "44000022_temp.jpg").touch()
    (annotation_dir / "44000022.txt").write_text("1,2,11,12,1\n", encoding="utf-8")

    samples, issues = discover_samples(tmp_path)

    assert len(issues) == 0
    assert samples[0].annotation_file == annotation_dir / "44000022.txt"
