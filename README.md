# PCB Visual Inspection with Visual AI

An end-to-end research prototype for automated PCB defect inspection. The
project combines a classical template-difference baseline with a six-class
YOLO detector, evaluation tooling, threshold tuning, robustness experiments,
and visual failure analysis.

It is designed as a portfolio project for industrial computer vision and
optical quality inspection roles.

> **Scope:** DeepPCB contains defects in PCB traces and copper patterns. It is
> not a complete dataset for populated-component inspection, solder defects, or
> component-placement verification.

## What the system does

```text
Defect-free template + tested PCB image
                  |
                  v
       Dataset validation and preparation
                  |
          +-------+--------+
          |                |
          v                v
 Classical reference   Six-class YOLO detector
 template difference   open / short / mousebite /
          |             spur / copper / pin-hole
          +-------+--------+
                  |
                  v
 Evaluation, threshold tuning, robustness tests
                  |
                  v
 Annotated reports, failure cases, comparison evidence
```

The same workflow is available as a graphic in
[docs/workflow.svg](docs/workflow.svg).

## Results achieved

The trained detector was evaluated on the project-generated 300-image
validation holdout:

| Metric | Result |
|---|---:|
| Precision | **92.17%** |
| Recall | **89.82%** |
| mAP@0.50 | **95.62%** |
| mAP@0.50:0.95 | **74.97%** |

Per-class analysis identified two useful engineering findings:

- `short` has the weakest recall and needs targeted analysis.
- `pin-hole` has high recall but initially produced many false positives;
  class-specific threshold tuning improved its precision from 61.6% to 90.8%
  on the analyzed validation predictions.

These are validation results, not a production acceptance claim. DeepPCB
contains artificial defects and does not represent all camera, board, fixture,
or lighting conditions on a manufacturing line.

## Example detection

The image below is an Ultralytics validation montage produced by the trained
detector. Each colored box identifies a predicted defect class and its
confidence score:

![Example PCB defect detections](runs/detect/evaluation-results/validation/val_batch0_pred.jpg)

The detector identifies examples of `open`, `short`, `spur`, `copper`,
`mousebite`, and `pin-hole` defects across the validation batch. The montage
contains multiple images so it is useful for a quick qualitative check; the
per-image failure-case exports are better for investigating individual false
positives and missed defects.

For an annotated failure example, see
[77000001_test_failure.png](failure-cases/77000001_test_failure.png). In that
image, red boxes are false positives and orange boxes are false negatives.

## Project layout

```text
pcb_inspection/
├── dataset.py          # DeepPCB discovery and annotation parsing
├── baseline.py         # Template difference, registration, metrics
├── training_data.py    # DeepPCB -> YOLO export
├── training.py         # Optional Ultralytics training
├── evaluation.py       # Checkpoint validation and metrics.json
├── error_analysis.py   # Per-class metrics, tuning, failure images
├── robustness.py       # Controlled classical-baseline perturbations
├── comparison.py       # Consolidated benchmark report
└── cli.py              # All command-line workflows
tests/                  # 26 focused tests
```

## Requirements

- Python 3.9 or newer
- macOS, Linux, or Windows
- NumPy and Pillow for the core pipeline
- Optional: Ultralytics/PyTorch for detector training and evaluation

The DeepPCB dataset is available from
[tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB) and is
restricted to research use by its repository notice.

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools
python -m pip install -e '.[dev]'
```

The pip upgrade matters on older macOS Command Line Tools Python
installations. Older pip versions cannot reliably install this modern
`pyproject.toml` project in editable mode.

Install the optional AI backend when you are ready to train or evaluate:

```bash
python -m pip install -e '.[ai]'
```

If your shell has no `python` command, use the environment explicitly:

```bash
./.venv/bin/python -m pcb_inspection.cli --help
```

Run the tests:

```bash
python -m pytest
```

## 1. Download and validate DeepPCB

Clone the dataset outside the project output directories:

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git
```

Validate the downloaded repository:

```bash
pcb-inspect summary ./DeepPCB
```

Expected result for the downloaded repository:

```text
complete samples: 1500
issues: 0
annotations: 10013
```

The loader supports the repository's actual layout:

```text
DeepPCB/PCBData/groupXXXXX/XXXXX/*_test.jpg
DeepPCB/PCBData/groupXXXXX/XXXXX/*_temp.jpg
DeepPCB/PCBData/groupXXXXX/XXXXX_not/*.txt
```

It also supports annotations beside the image and both comma-separated and
whitespace-separated annotation records.

## 2. Run the classical reference baseline

The baseline compares a tested image with its defect-free template, optionally
registers small translations, normalizes illumination, thresholds the
difference, and extracts connected components.

```bash
pcb-inspect baseline ./DeepPCB \
  --register \
  --normalize \
  --max-shift 5 \
  --threshold 30 \
  --min-area 4 \
  --limit 10
```

This is deliberately class-agnostic. It is a reference for measuring why
learned visual features are needed; it is not the primary detector.

Generate annotated PNG and JSON reports:

```bash
pcb-inspect report ./DeepPCB ./reports \
  --register \
  --normalize \
  --limit 10
```

## 3. Prepare YOLO training data

Export tested images and zero-based YOLO labels:

```bash
pcb-inspect prepare ./DeepPCB ./prepared-data
```

This creates:

```text
prepared-data/
├── dataset.yaml
├── images/train/       # 1,200 images by default
├── images/val/         # 300 images by default
├── labels/train/
└── labels/val/
```

Labels use the image stem, for example:

```text
images/train/44000022_test.jpg
labels/train/44000022_test.txt
```

This matching is required by Ultralytics. The split is deterministic and uses
20% validation by default. Keep the official DeepPCB evaluation set separate
when making benchmark claims.

## 4. Train the detector

Install the optional backend, then run:

```bash
pcb-inspect train ./prepared-data/dataset.yaml ./runs \
  --model yolo11n.pt \
  --epochs 50 \
  --image-size 640 \
  --batch-size 16
```

The command validates the prepared dataset before starting. It uses seed `42`
by default and writes the best checkpoint under the Ultralytics run directory.
On a CPU-only laptop, training can take a while; a one-epoch smoke test is
useful first:

```bash
pcb-inspect train ./prepared-data/dataset.yaml ./runs-smoke \
  --epochs 1 \
  --image-size 320 \
  --batch-size 4
```

## 5. Evaluate a trained checkpoint

```bash
pcb-inspect evaluate \
  ./runs/detect/runs/pcb-defect-detector-2/weights/best.pt \
  ./prepared-data/dataset.yaml \
  ./evaluation-results
```

Important outputs:

```text
evaluation-results/metrics.json
runs/detect/evaluation-results/validation/
├── predictions.json
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxPR_curve.png
└── BoxF1_curve.png
```

Example generated visuals:

- [Confusion matrix](runs/detect/evaluation-results/validation/confusion_matrix.png)
- [Precision-recall curve](runs/detect/evaluation-results/validation/BoxPR_curve.png)
- [F1 curve](runs/detect/evaluation-results/validation/BoxF1_curve.png)

## 6. Analyze and tune errors

Generate per-class precision, recall, F1, and a confusion matrix:

```bash
pcb-inspect analyze \
  ./prepared-data \
  ./runs/detect/evaluation-results/validation/predictions.json \
  ./error-analysis
```

Tune confidence thresholds independently:

```bash
pcb-inspect tune \
  ./prepared-data \
  ./runs/detect/evaluation-results/validation/predictions.json \
  ./threshold-tuning
```

Export annotated false positives, false negatives, and class mismatches:

```bash
pcb-inspect failures \
  ./prepared-data \
  ./runs/detect/evaluation-results/validation/predictions.json \
  ./failure-cases \
  --confidence-threshold 0.25 \
  --limit 50
```

Color coding:

- **Red:** false positive
- **Orange:** false negative
- **Purple:** class mismatch

Example generated failure image:

[Open a failure-case example](failure-cases/77000001_test_failure.png)

## 7. Test robustness and compare approaches

Run controlled perturbations against the classical baseline:

```bash
pcb-inspect robustness ./DeepPCB ./robustness.json \
  --limit 25 \
  --mitigate
```

The benchmark covers brightness, contrast, Gaussian noise, and small
translation. It is a controlled engineering experiment, not a substitute for
real production camera data.

Create the consolidated project report:

```bash
pcb-inspect compare \
  ./evaluation-results/metrics.json \
  ./robustness.json \
  ./comparison.json \
  --checkpoint ./runs/detect/runs/pcb-defect-detector-2/weights/best.pt
```

The resulting [comparison.json](comparison.json) keeps detector metrics,
classical-baseline robustness results, provenance, and deployment caveats in
one artifact.

## Reproducibility checklist

Before presenting results:

1. Record the checkpoint path and training arguments.
2. Keep the validation split separate from threshold tuning data where
   possible.
3. Report per-class metrics, not only aggregate mAP.
4. Show false positives and false negatives.
5. Measure inference speed on the target hardware.
6. State that DeepPCB is research data with artificial defects.
7. Do not claim production readiness without representative camera and line
   data.

## Why this fits industrial Visual AI work

The project demonstrates the complete inspection loop:

- Camera-like paired image inputs
- Image registration and illumination handling
- Classical computer-vision baseline
- AI-based defect localization and classification
- Production-oriented precision/recall trade-offs
- Per-class threshold tuning
- Visual failure analysis
- Reproducible reports and documentation

That is substantially stronger than presenting a detector trained on a dataset
with only one mAP number.
