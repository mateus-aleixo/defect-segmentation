# defect-segmentation

[![ci](https://github.com/mateus-aleixo/defect-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/defect-segmentation/actions/workflows/ci.yml)

Defect segmentation for an inspection line, where each part either passes automatically or
goes to a person. A torchvision DeepLabV3 (MobileNetV3) model is fine-tuned on five MVTec AD
categories, and its mask threshold is set with conformal risk control, so the masks miss at
most a chosen share α of the defects on average. It runs without PyTorch on AWS Lambda.

![Naive threshold vs conformal threshold on both categories](docs/figures/naive_vs_conformal.png)

*Top: a category where the method works. The calibrated mask is slightly larger and misses a
third as many defect pixels. Bottom: a category the model never learned, where it keeps its
10% bound by covering most of the frame. Each row shows the test image closest to that
category's held-out mean.*

## Results

The guarantee covers missed defects. It says nothing about false alarms on good parts,
which is what decides whether a line could use it, so I measured those separately on
MVTec's defect-free `test/good/` images. At α = 0.10:

| category | validation IoU | false alarms, pixel loss | false alarms, instance loss |
|---|---|---|---|
| `tile` | 0.729 | 0.061 | 0.000 |
| `metal_nut` | 0.621 | 0.045 | 0.000 |
| `carpet` | 0.520 | 0.143 | 0.036 |
| `leather` | 0.393 | 0.344 | 0.062 |
| `grid` | 0.068 | 1.000 | 1.000 |

Bounding missed pixels was the wrong target. On `leather` it stopped 34% of good parts,
even though its held-out miss rate (0.063) was well inside α. Bounding missed defect
instances instead, using connected components, cut false alarms in every category where
the model had learned something, and to zero on `tile` and `metal_nut`.

`grid` is there on purpose. The model never learned it, so the bound only holds by
escalating every part: a guarantee can be valid and useless at the same time. Retraining at
640 px raised its validation IoU from 0.068 to 0.359 and still left every clean part
escalated.

More in [docs/results.md](docs/results.md).

## Live API

`https://6s8ozlsit4.execute-api.eu-west-1.amazonaws.com/models` lists both published
categories, including the one that doesn't work: `metal_nut` with a false-alarm rate of
0.045 next to `grid` at 1.000. `POST /predict` returns pass or escalate for one image, with
the threshold, the held-out miss rate and the false-alarm rate behind the decision. The
first request after a quiet spell waits about 12 s for a cold start, warm requests take
about 0.45 s, and the stage is throttled to 5 requests per second. A sample request is in
[docs/deploy.md](docs/deploy.md).

## How it's built

- MVTec AD only has defect masks in its test split, so the masked images are re-split
  60/20/20 into training, calibration and test sets.
- The model is exported to ONNX with a parity check and served by FastAPI on onnxruntime.
  CI builds the container and checks that `import torch` fails inside it.
- Training and serving share one preprocessing module, because a threshold calibrated on
  one resize would drift under another.
- Terraform for ECR, Lambda and the HTTP API, deployed by GitHub Actions through OIDC.

[docs/architecture.md](docs/architecture.md), [docs/deploy.md](docs/deploy.md) and the
[model card](docs/model-card.md) have the details.

## Running it

    uv sync --all-extras
    uv run pytest                             # synthetic fixtures, CPU, no downloads
    uv run python scripts/fetch_mvtec.py      # about 5.3 GB once
    uv run python -m conformal_seg.train --category metal_nut --unfreeze --lr 1e-4 --epochs 60
    uv run python -m conformal_seg.calibrate --category metal_nut --alpha 0.1

MIT for the code. MVTec AD is CC BY-NC-SA 4.0 and isn't redistributed here.
