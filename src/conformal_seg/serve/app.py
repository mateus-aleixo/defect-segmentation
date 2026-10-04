"""FastAPI service over the exported ONNX models.

Torch-free by design: inference is onnxruntime, preprocessing is openCV and numpy,
and the conformal state is a small JSON manifest. MODEL_ROOT points at the
`models/` registry baked into the container image. The module exposes `handler`
for AWS Lambda via Mangum and runs locally with uvicorn, the conformal-rul
pattern.

The endpoint returns a decision rather than a mask, because that is the thing the
guarantee is about. It also returns the false-alarm rate measured on defect-free
parts, next to alpha, every single time. Those two numbers are the trade, and the
one this service can prove is not the one that decides whether it is deployable.
"""

from __future__ import annotations

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from conformal_seg import __version__
from conformal_seg.metrics import flagged_fraction, image_flagged
from conformal_seg.preprocess import decode_image, normalize_chw
from conformal_seg.registry import SERVING_MANIFEST

from .schemas import Guarantee, ModelInfo, PredictResponse

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))

app = FastAPI(
    title="defect-segmentation",
    version=__version__,
    description="Industrial defect segmentation whose masks provably miss at most "
    "alpha of defect pixels, served as a pass/escalate decision.",
)


class Bundle:
    def __init__(self, category: str):
        cat_dir = MODEL_ROOT / category
        manifest_path = cat_dir / SERVING_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no serving model for {category} under {MODEL_ROOT}")

        import onnxruntime as ort

        self.manifest = json.loads(manifest_path.read_text())
        self.session = ort.InferenceSession(
            str(cat_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size: int = self.manifest["input_size"]
        self.threshold: float = self.manifest["threshold"]
        self.control: dict = self.manifest.get("control", {})

    def guarantee(self) -> Guarantee:
        return Guarantee(
            alpha=self.manifest["alpha"],
            loss=self.manifest.get("loss", "pixel"),
            threshold=self.threshold,
            n_calibration=self.manifest["n_calibration"],
            held_out_fnr=self.manifest["held_out"]["fnr_mean"],
            false_alarm_rate=self.control.get("false_alarm_rate"),
        )

    def probabilities(self, rgb01: np.ndarray) -> np.ndarray:
        x = normalize_chw(rgb01)[None].astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0][0, 0]


@lru_cache(maxsize=8)
def bundle(category: str) -> Bundle:
    try:
        return Bundle(category)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _mask_png_b64(mask: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    """Read the manifests directly. Do NOT build Bundles here.

    Everything below comes from JSON, but a Bundle also opens an onnxruntime
    session, so listing five categories used to load five networks to answer a
    metadata request. On a cold Lambda that exceeded the timeout and the deploy
    smoke test caught it: /health passed while /models did not.
    """
    out = []
    for manifest_path in sorted(MODEL_ROOT.glob(f"*/{SERVING_MANIFEST}")):
        m = json.loads(manifest_path.read_text())
        control = m.get("control", {})
        out.append(
            ModelInfo(
                category=m["category"],
                source_run=m["source_run"],
                loss=m.get("loss", "pixel"),
                input_size=m["input_size"],
                threshold=m["threshold"],
                alpha=m["alpha"],
                held_out_fnr=m["held_out"]["fnr_mean"],
                held_out_mask_fraction=m["held_out"]["mask_fraction"],
                n_calibration=m["n_calibration"],
                n_control=control.get("n", 0),
                false_alarm_rate=control.get("false_alarm_rate"),
                min_area_frac=control.get("min_area_frac", 1e-3),
            )
        )
    if not out:
        raise HTTPException(status_code=503, detail=f"no models under {MODEL_ROOT}")
    return out


@app.post("/predict", response_model=PredictResponse)
async def predict(
    category: str = Query(..., description="Registry category, from GET /models."),
    image: UploadFile = File(..., description="PNG or JPEG of one part."),
    min_area_frac: float | None = Query(
        None, ge=0.0, le=1.0,
        description="Override the escalation trigger. Defaults to the value the "
        "control split was measured at.",
    ),
    return_mask: bool = Query(False, description="Include the binary mask as base64 PNG."),
) -> PredictResponse:
    b = bundle(category)

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image exceeds {MAX_UPLOAD_BYTES} bytes"
        )
    try:
        rgb01 = decode_image(data, b.size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    prob = b.probabilities(rgb01)
    mask = prob >= b.threshold

    trigger = (
        min_area_frac if min_area_frac is not None
        else b.control.get("min_area_frac", 1e-3)
    )
    return PredictResponse(
        category=b.manifest["category"],
        decision="escalate" if image_flagged(mask, trigger) else "pass",
        flagged_fraction=round(flagged_fraction(mask), 6),
        min_area_frac=trigger,
        input_size=b.size,
        guarantee=b.guarantee(),
        mask_png=_mask_png_b64(mask) if return_mask else None,
    )


try:  # Lambda entrypoint; absent locally unless the serve extra is installed
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
