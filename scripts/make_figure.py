"""Render the headline figure: naive threshold vs conformal threshold.

The point of the figure is the contrast between the two categories, so it is
built as two rows sharing one set of columns:

  metal_nut  the method working — the calibrated mask is a little larger and
             misses far less
  grid       the method degenerating — the model never learned this category,
             so the only threshold honouring alpha floods the frame

Picks the test image whose behaviour is closest to the reported held-out mean,
rather than the most flattering one — a figure chosen for looks is a figure
that lies.

    python scripts/make_figure.py   ->  docs/figures/naive_vs_conformal.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from conformal_seg.calibrate import predict_probs
from conformal_seg.data import DefectSegDataset, discover_items, load_pair, split_items
from conformal_seg.metrics import fnr, iou
from conformal_seg.model import DefectSeg

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "figures" / "naive_vs_conformal.png"
NAIVE = 0.5
SIZE = 320


def pick_representative(probs, masks, thr):
    """The test image whose conformal FNR is nearest the set's mean.

    Deliberately not the best-looking one: the figure has to represent the
    reported numbers, not beat them.
    """
    f = np.array([fnr(p >= thr, m) for p, m in zip(probs, masks, strict=True)])
    return int(np.argmin(np.abs(f - f.mean())))


def overlay(ax, rgb, mask, colour, title, subtitle=None):
    ax.imshow(rgb)
    if mask is not None and mask.any():
        tint = np.zeros((*mask.shape, 4))
        tint[mask.astype(bool)] = colour
        ax.imshow(tint)
    ax.set_title(title, fontsize=9.5, pad=4)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8, labelpad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cfd3d8")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.6))

    for row, cat in enumerate(["metal_nut", "grid"]):
        calib = json.loads((ROOT / "runs" / cat / "calibration.json").read_text())
        thr = calib["calibration"]["threshold"]
        alpha = calib["calibration"]["alpha"]

        ck = torch.load(ROOT / "runs" / cat / "best.pt", map_location=device, weights_only=True)
        model = DefectSeg(pretrained=False).to(device)
        model.load_state_dict(ck["state_dict"])

        items = split_items(discover_items(ROOT / "data" / "mvtec", cat), seed=17)["test"]
        probs, masks = predict_probs(model, DefectSegDataset(items, SIZE), device)
        i = pick_representative(probs, masks, thr)
        prob, truth, item = probs[i], masks[i], items[i]
        rgb, _ = load_pair(item, SIZE)

        m_naive, m_conf = prob >= NAIVE, prob >= thr
        stats = {
            "naive": (fnr(m_naive, truth), iou(m_naive, truth), m_naive.mean()),
            "conf": (fnr(m_conf, truth), iou(m_conf, truth), m_conf.mean()),
        }

        overlay(axes[row, 0], rgb, None, None, f"{cat}  ·  {item.defect}",
                "input image")
        overlay(axes[row, 1], rgb, truth, (0.11, 0.65, 0.35, 0.55), "ground truth",
                f"{truth.mean():.1%} of pixels are defect")
        f, j, a = stats["naive"]
        overlay(axes[row, 2], rgb, m_naive, (0.85, 0.42, 0.10, 0.55),
                f"naive threshold {NAIVE}",
                f"misses {f:.0%} of defect  ·  IoU {j:.2f}  ·  area {a:.0%}")
        f, j, a = stats["conf"]
        overlay(axes[row, 3], rgb, m_conf, (0.20, 0.40, 0.80, 0.55),
                f"conformal threshold {thr:.2f}",
                f"misses {f:.0%} of defect  ·  IoU {j:.2f}  ·  area {a:.0%}")

        verdict = ("the guarantee is worth having"
                   if cat == "metal_nut" else
                   "the guarantee holds and the output is useless")
        axes[row, 0].text(-0.09, 0.5, verdict, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=8.6,
                          color="#444")

    fig.suptitle("Conformal risk control bounds the missed-defect rate; "
                 f"it says nothing about whether the mask is useful   (α = {alpha:.2f})",
                 fontsize=11.5, y=0.975)
    fig.text(0.5, 0.018,
             "Each row shows the test image whose conformal miss-rate is closest to that "
             "category's held-out mean, not the best-looking one.",
             ha="center", fontsize=8.2, color="#666")

    # No legend: the column headings already name each mask and the colours never
    # change between them, so a legend only adds something to collide with.

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.012, 0.035, 1, 0.955])
    fig.savefig(OUT, dpi=110, facecolor="white")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
