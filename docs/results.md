# Results

Produced on code at `3ca4d33` (plus the `aux_loss` and ONNX fixes made
during this run). Hardware: RTX 3060 Laptop, 6 GB, CUDA 12.6, torch 2.13.
Reproduce: `scripts/fetch_mvtec.py` → `train` → `calibrate` → `onnx_export`.

## The headline

**Conformal risk control delivers exactly what it promises, and nothing more:
validity, not usefulness.** On a model that works, it costs ~5 IoU points and fixes
a miss rate that would otherwise have broken the 10% promise. On a model that
doesn't, it still honours the promise: by flagging 81% of every image.

Both cases are below. The second one is the reason this file exists.

## Setup

Supervised re-split of the mask-annotated defect images, seed 17, 60/20/20:

| category | train | calibration | test | defect types |
|---|---|---|---|---|
| `metal_nut` | 55 | 18 | 20 | bent, color, flip, scratch |
| `grid` | 34 | 11 | 12 | bent, broken, glue, metal_contamination, thread |

n is small. That is the honest condition of the dataset under a supervised
protocol, and it is precisely the regime conformal calibration is built for:
the guarantee is finite-sample, not asymptotic.

## Training

`deeplabv3_mobilenet_v3_large`, pretrained backbone, 1-channel head, 320 px,
BCE + Dice, AdamW + cosine. Best validation IoU (validation = the calibration
split; the test split is untouched until the table after this one):

| category | backbone | epochs | lr | best val IoU |
|---|---|---|---|---|
| `metal_nut` | frozen | 40 | 1e-3 | 0.442 |
| **`metal_nut`** | **unfrozen** | **60** | **1e-4** | **0.621** |
| `grid` | frozen | 40 | 1e-3 | 0.048 |
| **`grid`** | **unfrozen** | **60** | **1e-4** | **0.068** |
| `grid` @ 640 px | unfrozen | 60 | 1e-4 | **0.359** |

Unfreezing the backbone is worth ~0.18 IoU on `metal_nut` and nothing that
matters on `grid`. Roughly 1.6 s/epoch: the whole table is under 6 minutes of
compute, which is a fair description of how small this dataset is.

## Conformal calibration (α = 0.10), on the held-out test split

Threshold fitted on the calibration split only; every number below is from data
the threshold never saw.

### `metal_nut`: the guarantee is worth having

| threshold | IoU | precision | recall | **FNR** | mask area |
|---|---|---|---|---|---|
| 0.500 (naive) | **0.691** | 0.818 | 0.833 | **0.167 ✗** | 0.173 |
| 0.130 (conformal) | 0.644 | 0.672 | 0.945 | **0.055 ✓** | 0.207 |

The naive threshold looks better on IoU and **misses 16.7% of defect pixels**: it
would have violated a 10% promise nobody checked. The conformal threshold gives up
4.7 IoU points and 15 points of precision to buy a verified miss rate of 5.5%, with
masks 20% larger. For "can this part ship unseen?", that is the correct trade.

### `grid`: the guarantee holds and the output is useless

| threshold | IoU | precision | recall | **FNR** | mask area |
|---|---|---|---|---|---|
| 0.500 (naive) | 0.184 | 0.292 | 0.615 | 0.385 ✗ | 0.024 |
| 0.090 (conformal) | 0.015 | 0.015 | 0.990 | **0.010 ✓** | **0.814** |

The model never learned `grid`: thin, low-contrast texture defects, 34 training
images, downsampled to 320 px. Calibration did its job perfectly and the result is
absurd: to miss at most 10% of defect pixels with a model this weak, the only
admissible threshold flags **81% of the image**. FNR 0.0096, IoU 0.015.

This is the point worth taking away. Conformal risk control is a *validity*
guarantee, not a quality guarantee: it will always find a threshold that honours α,
and when the underlying model is bad that threshold is "flag almost everything".
The correct reading of a `grid` deployment is not "we have a guarantee" but
"mask area 0.81 means there is no usable signal here: fix the model, or don't
automate this category". Reporting mask area beside FNR is what makes that legible;
a paper reporting only coverage would have called this a success.

What would actually fix `grid`: higher resolution or tiled inference (its defects
are a few pixels wide), more annotated data, and a texture-anomaly approach rather
than supervised segmentation, which is what MVTec AD is designed for in the first
place.

## The defect-free control split

Everything above measures the model on **defective** parts. That is what the
conformal loss is defined on: FNR is the fraction of true defect pixels the mask
misses, so a part with no defects contributes a loss of exactly 0 no matter what the
mask does. The guarantee is therefore blind, by construction, to the question an
inspection line actually asks: **of the parts that are fine, how many get stopped?**

MVTec ships those parts under `test/good/` and nothing here was using them. They are
the control: 22 defect-free `metal_nut` images and 21 `grid` images, scored by the
same model at the same two thresholds. A part is counted as escalated when the mask
flags at least 0.1% of the frame (~100 px at 320x320), so a single stray pixel does
not stop the belt.

![The curve calibration reads and the one it ignores](figures/operating_point.png)

| category | threshold | false-alarm rate | mean flagged area on a clean part |
|---|---|---|---|
| `metal_nut` | 0.500 (naive) | **0.000** | 0.000% |
| `metal_nut` | 0.130 (conformal) | **0.045** (1 of 22) | 0.016% |
| `grid` | 0.500 (naive) | 0.429 (9 of 21) | 13.9% |
| `grid` | 0.090 (conformal) | **1.000** (21 of 21) | 89.8% |

**`metal_nut`: the guarantee is close to free on this side.** Buying the drop from a
16.7% miss rate to 5.5% costs one false alarm in 22 clean parts, and the mask it
draws on that one part covers 0.016% of the frame. A line running this model
auto-passes 21 of 22 good parts and catches 94.5% of defect pixels on bad ones. That
is a system worth deploying, and until now the repo could not say so: it had measured
only half of it.

**`grid`: the control split is what turns "useless" from an opinion into a number.**
The earlier sections said the guarantee held by flagging 81% of the image, which
*looks* bad. The control says exactly how bad: at λ̂ = 0.09 **every single
defect-free part is escalated**. The model does not merely produce large masks, it
sends 100% of the line to a human. Its throughput contribution is zero, and no
statement about coverage or validity changes that.

Note the naive threshold already fails `grid` on both counts, at 0.429 false alarms
*and* a 38.5% miss rate. Conformal calibration did not break this category; it
inherited a model that never learned it and honoured its promise the only way left.

### Why this belongs in a conformal repo

Conformal risk control guarantees **marginal validity on the loss you hand it**. Hand
it FNR and it will bound FNR, perfectly, forever, including in the regime where the
answer is to flag everything. Nothing in the theory is violated by the `grid` row and
nothing in the theory warns you about it either. The guard is empirical: measure the
other side, on data the loss cannot see, and report both.

## Does resolution fix `grid`?

The obvious reading of the `grid` failure is that 320 px is too coarse: its defects
are hairline threads and bent wires, and an INTER_AREA resize from 1024 px averages
them into the background. That is a hypothesis, and the control split gives it a
sharp test. Retrained at **640 px**, batch 4, everything else identical:

| | `grid` @ 320 px | `grid` @ 640 px |
|---|---|---|
| best val IoU | 0.068 | **0.359** |
| λ̂ at α = 0.10 | 0.090 | **0.030** |
| held-out FNR | 0.010 ✓ | 0.009 ✓ |
| false alarms at the **naive** 0.5 | 0.429 | **0.095** |
| flagged area at the naive 0.5 | 13.9% | **0.013%** |
| false alarms at **λ̂** | 1.000 | **1.000** |
| flagged area at λ̂ | 89.8% | 75.6% |

**Resolution was a real limitation of the model and not the binding constraint on the
guarantee.** Four times the pixels bought a 5x better model by IoU, and at the naive
threshold it is a different system: false alarms fall from 43% to 9.5%, and the mask
it draws on a clean part shrinks from 13.9% of the frame to 0.013%. That is most of
the way to something deployable.

The calibrated operating point did not move at all. It still escalates **every**
defect-free part. Note the direction λ̂ travelled: it went *down*, 0.090 to 0.030. A
sharper model did not let the threshold relax, it forced it lower, because catching
90% of the pixels in a one-pixel-wide thread means accepting anything faintly
thread-like, and at 640 px there is more of the image to be faintly thread-like in.

So the gap between the two operating points **widened**. Before, both were bad; now
the naive threshold is good and the calibrated one is unchanged, which is a worse
failure to have, because it is the one a summary statistic hides. Reporting IoU alone
would have called this experiment a success.

What it actually says: a pixel-level FNR target of 0.10 is the wrong contract for
hairline defects at any resolution tested here. The next thing to try is not more
pixels but a different loss, something at the level of the defect instance rather
than the pixel, so that "found the thread" does not require "found 90% of the
thread's pixels".

## Changing the contract: an instance-level loss

The resolution experiment said the pixel-level FNR target is the wrong contract
for hairline defects. This tests that directly rather than asserting it.

`instance_fnr` labels connected components of the ground truth and counts an
instance as found when the predicted mask covers at least 10% of it. Missing a
thread entirely still costs; clipping its ends does not. Like pixel FNR it is
per-image, in [0, 1], and nondecreasing in the threshold, so conformal risk
control applies unchanged: only the quantity being bounded changes.

At α = 0.10, calibrated and evaluated exactly as before:

| category | loss | λ̂ | held-out risk | mask area | false alarms on clean parts |
|---|---|---|---|---|---|
| `metal_nut` | pixel | 0.130 | 0.055 | 0.207 | 0.045 (1 of 22) |
| **`metal_nut`** | **instance** | **0.290** | **0.025** | **0.186** | **0.000 (0 of 22)** |
| `grid` @ 320 | pixel | 0.090 | 0.010 | 0.814 | 1.000 |
| `grid` @ 320 | instance | 0.110 | 0.028 | 0.639 | 1.000 |
| `grid` @ 640 | pixel | 0.030 | 0.009 | 0.734 | 1.000 |
| **`grid` @ 640** | **instance** | **0.070** | **0.063** | **0.105** | **1.000** |

### `metal_nut`: strictly better

The instance loss more than doubles the admissible threshold (0.130 to 0.290),
tightens the mask, and takes the false-alarm rate to **zero**. The system now
auto-passes all 22 clean parts and still finds 97.5% of defect instances. The
earlier operating point cost one false alarm in 22 to buy its guarantee; this one
costs none.

Nothing was lost. The bound changed from "misses at most 10% of defect pixels"
to "misses at most 10% of defect instances", which is the question an inspection
line was asking in the first place.

### `grid`: the contract was wrong, and the model is still wrong

At 640 px the instance loss shrinks the mask sevenfold, 0.734 to **0.105** of the
frame, and on clean parts 0.756 to 0.167. That is a real improvement and it is
not enough, because the false-alarm rate stays pinned at 1.000.

The reason is visible in the distributions rather than the summary. At λ̂ = 0.070
the median flagged area is **0.038 on defect images and 0.020 on clean ones**.
They overlap. Sweeping the escalation trigger finds no operating point:

| trigger | clean parts escalated | defect images flagged |
|---|---|---|
| 0.001 | 100.0% | 100.0% |
| 0.05 | 28.6% | 33.3% |
| 0.10 | 23.8% | 8.3% |
| 0.20 | 19.0% | 8.3% |

At a trigger of 0.10 the system escalates clean parts *more often than defective
ones*. There is no threshold and no trigger that separates them, because the
model has no discriminative signal on this category to separate with.

### Does it replicate? Five categories

The `metal_nut` result above is n = 1. Three more categories were trained under
the identical recipe (unfrozen backbone, 60 epochs, lr 1e-4, 320 px) and
calibrated under both losses, to test whether the instance loss's advantage is
real or was luck on one class.

| category | best val IoU | loss | λ̂ | held-out risk | mask area | false alarms |
|---|---|---|---|---|---|---|
| `metal_nut` | 0.621 | pixel | 0.130 | 0.055 | 0.207 | 0.045 |
| | | **instance** | **0.290** | **0.025** | **0.186** | **0.000** |
| `tile` | 0.729 | pixel | 0.130 | 0.025 | 0.125 | 0.061 |
| | | **instance** | **0.540** | **0.000** | **0.084** | **0.000** |
| `leather` | 0.393 | pixel | 0.180 | 0.063 | 0.032 | 0.344 |
| | | **instance** | **0.260** | **0.000** | **0.018** | **0.062** |
| `carpet` | 0.520 | pixel | 0.250 | 0.085 | 0.033 | 0.143 |
| | | **instance** | **0.540** | **0.053** | **0.024** | **0.036** |
| `grid` | 0.068 | pixel | 0.090 | 0.010 | 0.814 | 1.000 |
| | | instance | 0.110 | 0.028 | 0.639 | 1.000 |

**It replicates, on four of four categories where the model learned anything.**
In every one the instance loss admits a higher threshold, draws a smaller mask,
holds α, and cuts the false-alarm rate. On `metal_nut` and `tile` it removes
false alarms entirely. On `leather` it is the difference between escalating a
third of all clean parts and escalating one in sixteen.

`leather` is the clearest case for the argument. Its pixel-loss operating point
escalates **34.4%** of defect-free parts, which is not a deployable system, and
nothing in the conformal report would have said so: held-out FNR is 0.063,
comfortably inside α = 0.10. Changing what the guarantee is *about* takes the
same model, the same images and the same α to a 6.2% false-alarm rate.

`grid` is the control, and behaves as a control should: the loss change moves the
threshold and the mask a little and the decision not at all, because there is no
signal for a better contract to exploit.

### What the two experiments together establish

The pixel-level contract was genuinely wrong: fixing it improved every category
it was applied to, and on `metal_nut` it removed the last false alarm. It also
cannot manufacture signal that is not there. `grid` has now been given the
benefit of the doubt three times in sequence, more resolution, a better loss, and
a swept decision trigger, and it fails each time for the same underlying reason:
34 training images of a low-contrast texture is not enough to learn from.

That is the honest end of this line. The next thing that would move `grid` is
more labelled data or a different problem formulation (anomaly detection against
a defect-free reference, which is what MVTec AD was built for), not another
adjustment to the guarantee.

## ONNX export

Parity against torch on random inputs, max |Δ|:

| category | max abs diff | file |
|---|---|---|
| `metal_nut` | 2.26e-06 | 493 KB |
| `grid` | 3.70e-06 | 493 KB |

Both under the 1e-4 gate. Inference path is onnxruntime only: no torch at serving
time, same pattern as [turbofan-rul](https://github.com/mateus-aleixo/turbofan-rul).

## Bugs this run found

Two, both invisible to the test suite until real data and real weights showed up:

1. **`aux_loss`**: the pretrained VOC checkpoint was trained with an auxiliary
   head and torchvision refuses to load it under `aux_loss=False`. Every test built
   with `pretrained=False`, so the failure only appeared on the first real training
   run. Fixed, and the argument logic is now asserted directly in a CI-safe test.
2. **ONNX export crashed on Windows**: torch's exporter prints a ✅ progress banner
   that a cp1252 console cannot encode. Fixed with `verbose=False` plus a stdout
   reconfigure.
