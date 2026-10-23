"""Command-line entry points for dataset inspection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .baseline import Evaluation, detect_sample, evaluate
from .dataset import DefectClass, discover_samples
from .report import create_report, save_report
from .training_data import prepare_yolo_dataset
from .training import TrainingConfig, train_detector
from .robustness import run_robustness, save_robustness
from .comparison import build_comparison, save_comparison
from .evaluation import evaluate_detector
from .error_analysis import (
    analyze,
    export_failure_cases,
    save_analysis,
    save_threshold_tuning,
    tune_thresholds,
)


def _summary(root: Path) -> int:
    samples, issues = discover_samples(root)
    counts = Counter(
        annotation.defect.label
        for sample in samples
        for annotation in sample.annotations
    )
    print(f"complete samples: {len(samples)}")
    print(f"issues: {len(issues)}")
    print(f"annotations: {sum(counts.values())}")
    for defect in DefectClass:
        print(f"{defect.label}: {counts[defect.label]}")
    for issue in issues[:10]:
        print(f"issue[{issue.sample_id}]: {issue.message}")
    if len(issues) > 10:
        print(f"... and {len(issues) - 10} more issue(s)")
    return 1 if issues else 0


def _baseline(
    root: Path,
    *,
    threshold: int,
    min_area: int,
    limit: int | None,
    register: bool,
    normalize: bool,
    max_shift: int,
) -> int:
    samples, issues = discover_samples(root)
    selected_samples = samples[:limit] if limit is not None else samples
    total = Evaluation(0, 0, 0)
    for sample in selected_samples:
        result = evaluate(
            detect_sample(
                sample,
                threshold=threshold,
                min_area=min_area,
                register=register,
                normalize=normalize,
                max_shift=max_shift,
            ),
            sample.annotations,
        )
        total = Evaluation(
            total.true_positives + result.true_positives,
            total.false_positives + result.false_positives,
            total.false_negatives + result.false_negatives,
        )
    print(f"samples evaluated: {len(selected_samples)}")
    print(f"issues skipped: {len(issues)}")
    print(f"true positives: {total.true_positives}")
    print(f"false positives: {total.false_positives}")
    print(f"false negatives: {total.false_negatives}")
    print(f"precision: {total.precision:.4f}")
    print(f"recall: {total.recall:.4f}")
    print(f"f1: {total.f1:.4f}")
    return 1 if issues else 0


def _report(
    root: Path,
    output_dir: Path,
    *,
    threshold: int,
    min_area: int,
    limit: int | None,
    register: bool,
    normalize: bool,
    max_shift: int,
) -> int:
    samples, issues = discover_samples(root)
    selected_samples = samples[:limit] if limit is not None else samples
    for sample in selected_samples:
        report, annotated = create_report(
            sample,
            threshold=threshold,
            min_area=min_area,
            register=register,
            normalize=normalize,
            max_shift=max_shift,
        )
        image_path, json_path = save_report(report, annotated, output_dir)
        print(f"{sample.sample_id}: {'PASS' if report.passed else 'FAIL'}")
        print(f"  image: {image_path}")
        print(f"  report: {json_path}")
    print(f"samples reported: {len(selected_samples)}")
    print(f"issues skipped: {len(issues)}")
    return 1 if issues else 0


def _prepare(root: Path, output_dir: Path, validation_ratio: float) -> int:
    result = prepare_yolo_dataset(
        root, output_dir, validation_ratio=validation_ratio
    )
    print(f"samples exported: {result.exported_samples}")
    print(f"annotations exported: {result.exported_annotations}")
    print(f"issues skipped: {result.issues}")
    print(f"output: {result.output_dir}")
    return 1 if result.issues else 0


def _train(
    dataset_yaml: Path,
    output_dir: Path,
    *,
    model: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    seed: int,
    device: str,
) -> int:
    result = train_detector(
        TrainingConfig(
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            model=model,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    )
    save_dir = getattr(result, "save_dir", "trainer output")
    print(f"training complete: {save_dir}")
    return 0


def _evaluate(
    checkpoint: Path,
    dataset_yaml: Path,
    output_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    device: str,
) -> int:
    report_path = evaluate_detector(
        checkpoint,
        dataset_yaml,
        output_dir,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    print(f"evaluation report: {report_path}")
    return 0


def _analyze(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    iou_threshold: float,
    confidence_threshold: float,
) -> int:
    result = analyze(
        prepared_dir,
        predictions_path,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    report_path = save_analysis(result, output_dir)
    print(f"error-analysis report: {report_path}")
    print(f"per-class metrics: {output_dir / 'per-class-metrics.csv'}")
    return 0


def _tune(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    iou_threshold: float,
) -> int:
    result = tune_thresholds(
        prepared_dir,
        predictions_path,
        iou_threshold=iou_threshold,
    )
    report_path = save_threshold_tuning(result, output_dir)
    print(f"threshold report: {report_path}")
    for name, recommendation in result["recommended"].items():
        print(
            f"{name}: threshold={recommendation['threshold']:.2f}, "
            f"precision={recommendation['precision']:.3f}, "
            f"recall={recommendation['recall']:.3f}, "
            f"f1={recommendation['f1']:.3f}"
        )
    return 0


def _failures(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    iou_threshold: float,
    confidence_threshold: float,
    limit: int | None,
) -> int:
    index_path = export_failure_cases(
        prepared_dir,
        predictions_path,
        output_dir,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        limit=limit,
    )
    print(f"failure-case index: {index_path}")
    return 0


def _robustness(
    root: Path,
    output_path: Path,
    *,
    limit: int | None,
    threshold: int,
    min_area: int,
    mitigate: bool,
) -> int:
    result = run_robustness(
        root,
        limit=limit,
        threshold=threshold,
        min_area=min_area,
        mitigate=mitigate,
    )
    path = save_robustness(result, output_path)
    print(f"robustness report: {path}")
    for name, values in result["conditions"].items():
        print(f"{name}: precision={values['precision']:.3f}, recall={values['recall']:.3f}, f1={values['f1']:.3f}")
    return 1 if result["issues"] else 0


def _compare(
    detector_metrics: Path,
    robustness_report: Path,
    output: Path,
    checkpoint: Path | None,
) -> int:
    result = build_comparison(
        detector_metrics,
        robustness_report,
        detector_checkpoint=checkpoint,
    )
    path = save_comparison(result, output)
    detector = result["detector"]
    print(f"comparison report: {path}")
    print(f"detector mAP50: {detector['map50']}")
    print(f"detector mAP50-95: {detector['map50_95']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PCB inspection utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary_parser = subparsers.add_parser(
        "summary", help="validate and summarize a DeepPCB directory"
    )
    summary_parser.add_argument("dataset_root", type=Path)
    baseline_parser = subparsers.add_parser(
        "baseline", help="evaluate the classical template-difference baseline"
    )
    baseline_parser.add_argument("dataset_root", type=Path)
    baseline_parser.add_argument("--threshold", type=int, default=30)
    baseline_parser.add_argument("--min-area", type=int, default=4)
    baseline_parser.add_argument("--limit", type=int)
    baseline_parser.add_argument(
        "--register", action="store_true", help="align test images before differencing"
    )
    baseline_parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize test-image brightness and contrast",
    )
    baseline_parser.add_argument("--max-shift", type=int, default=5)
    report_parser = subparsers.add_parser(
        "report", help="create annotated images and JSON inspection reports"
    )
    report_parser.add_argument("dataset_root", type=Path)
    report_parser.add_argument("output_dir", type=Path)
    report_parser.add_argument("--threshold", type=int, default=30)
    report_parser.add_argument("--min-area", type=int, default=4)
    report_parser.add_argument("--limit", type=int)
    report_parser.add_argument("--register", action="store_true")
    report_parser.add_argument("--normalize", action="store_true")
    report_parser.add_argument("--max-shift", type=int, default=5)
    prepare_parser = subparsers.add_parser(
        "prepare", help="export DeepPCB labels for a six-class detector"
    )
    prepare_parser.add_argument("dataset_root", type=Path)
    prepare_parser.add_argument("output_dir", type=Path)
    prepare_parser.add_argument("--validation-ratio", type=float, default=0.2)
    train_parser = subparsers.add_parser(
        "train", help="train the optional six-class YOLO detector"
    )
    train_parser.add_argument("dataset_yaml", type=Path)
    train_parser.add_argument("output_dir", type=Path)
    train_parser.add_argument("--model", default="yolo11n.pt")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a trained six-class YOLO detector"
    )
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("dataset_yaml", type=Path)
    evaluate_parser.add_argument("output_dir", type=Path)
    evaluate_parser.add_argument("--image-size", type=int, default=640)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--device", default="auto")
    analysis_parser = subparsers.add_parser(
        "analyze", help="analyze per-class detector errors"
    )
    analysis_parser.add_argument("prepared_dir", type=Path)
    analysis_parser.add_argument("predictions", type=Path)
    analysis_parser.add_argument("output_dir", type=Path)
    analysis_parser.add_argument("--iou-threshold", type=float, default=0.5)
    analysis_parser.add_argument("--confidence-threshold", type=float, default=0.25)
    tune_parser = subparsers.add_parser(
        "tune", help="tune per-class confidence thresholds"
    )
    tune_parser.add_argument("prepared_dir", type=Path)
    tune_parser.add_argument("predictions", type=Path)
    tune_parser.add_argument("output_dir", type=Path)
    tune_parser.add_argument("--iou-threshold", type=float, default=0.5)
    failures_parser = subparsers.add_parser(
        "failures", help="export annotated detector failure cases"
    )
    failures_parser.add_argument("prepared_dir", type=Path)
    failures_parser.add_argument("predictions", type=Path)
    failures_parser.add_argument("output_dir", type=Path)
    failures_parser.add_argument("--iou-threshold", type=float, default=0.5)
    failures_parser.add_argument("--confidence-threshold", type=float, default=0.25)
    failures_parser.add_argument("--limit", type=int, default=50)
    robustness_parser = subparsers.add_parser(
        "robustness", help="benchmark classical inspection robustness"
    )
    robustness_parser.add_argument("dataset_root", type=Path)
    robustness_parser.add_argument("output", type=Path)
    robustness_parser.add_argument("--limit", type=int, default=25)
    robustness_parser.add_argument("--threshold", type=int, default=30)
    robustness_parser.add_argument("--min-area", type=int, default=4)
    robustness_parser.add_argument("--mitigate", action="store_true")
    compare_parser = subparsers.add_parser(
        "compare", help="combine detector and classical-baseline reports"
    )
    compare_parser.add_argument("detector_metrics", type=Path)
    compare_parser.add_argument("robustness_report", type=Path)
    compare_parser.add_argument("output", type=Path)
    compare_parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "summary":
        return _summary(args.dataset_root)
    if args.command == "baseline":
        return _baseline(
            args.dataset_root,
            threshold=args.threshold,
            min_area=args.min_area,
            limit=args.limit,
            register=args.register,
            normalize=args.normalize,
            max_shift=args.max_shift,
        )
    if args.command == "report":
        return _report(
            args.dataset_root,
            args.output_dir,
            threshold=args.threshold,
            min_area=args.min_area,
            limit=args.limit,
            register=args.register,
            normalize=args.normalize,
            max_shift=args.max_shift,
        )
    if args.command == "prepare":
        return _prepare(args.dataset_root, args.output_dir, args.validation_ratio)
    if args.command == "train":
        return _train(
            args.dataset_yaml,
            args.output_dir,
            model=args.model,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
    if args.command == "evaluate":
        return _evaluate(
            args.checkpoint,
            args.dataset_yaml,
            args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.command == "analyze":
        return _analyze(
            args.prepared_dir,
            args.predictions,
            args.output_dir,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold,
        )
    if args.command == "tune":
        return _tune(
            args.prepared_dir,
            args.predictions,
            args.output_dir,
            iou_threshold=args.iou_threshold,
        )
    if args.command == "failures":
        return _failures(
            args.prepared_dir,
            args.predictions,
            args.output_dir,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold,
            limit=args.limit,
        )
    if args.command == "robustness":
        return _robustness(
            args.dataset_root,
            args.output,
            limit=args.limit,
            threshold=args.threshold,
            min_area=args.min_area,
            mitigate=args.mitigate,
        )
    if args.command == "compare":
        return _compare(
            args.detector_metrics,
            args.robustness_report,
            args.output,
            args.checkpoint,
        )
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
