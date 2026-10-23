import json
from pathlib import Path

from pcb_inspection.dataset import Annotation, DatasetSample, DefectClass
from pcb_inspection.training_data import _yolo_record, prepare_yolo_dataset


def _sample(root: Path) -> DatasetSample:
    test_image = root / "0001_test.jpg"
    template_image = root / "0001_temp.jpg"
    annotation_file = root / "0001.txt"
    test_image.write_bytes(b"test")
    template_image.write_bytes(b"template")
    annotation_file.write_text("64,128,192,256,6\n", encoding="utf-8")
    return DatasetSample(
        "0001",
        test_image,
        template_image,
        annotation_file,
        (Annotation(64, 128, 192, 256, DefectClass.PIN_HOLE),),
    )


def test_yolo_record_uses_zero_based_class_and_normalized_box() -> None:
    record = _yolo_record(_sample(Path("/tmp")))

    assert record == "5 0.200000 0.300000 0.200000 0.200000\n"


def test_prepare_yolo_dataset_exports_files_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sample = _sample(source)
    output = tmp_path / "prepared"

    result = prepare_yolo_dataset(source, output)

    assert result.exported_samples == 1
    assert result.exported_annotations == 1
    assert (output / "images" / "train" / sample.test_image.name).is_file()
    assert (output / "labels" / "train" / "0001_test.txt").read_text() == (
        "5 0.200000 0.300000 0.200000 0.200000\n"
    )
    metadata = json.loads((output / "dataset.yaml").read_text())
    assert metadata["names"][5] == "pin-hole"
