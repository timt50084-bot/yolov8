# Stage 4 Small-Object Optimization

## Goal

Stage 4 closes the small-object optimization loop on top of the explicit Stage 3 RGB-IR train-assist branch without
changing the default OBB or tracking baselines. This stage adds four opt-in pieces that work together:

1. small-object-oriented image sampling
2. scale-aware loss weighting
3. small-object-only validation metrics
4. a dedicated high-resolution small-object model yaml

The stage still does **not** consume `img_prev`. Temporal modeling remains reserved for Stage 5.

## Why This Is a Closed Loop

Only adding a sampler is incomplete because training still optimizes the same loss and only reports overall metrics.
Only adding a metric is incomplete because it does not change optimization pressure. Only adding a high-resolution yaml
is incomplete because the training distribution and loss remain unchanged.

Stage 4 keeps these four pieces aligned under the same small-object definition so the training branch can:

- see small-object-heavy images more often
- give those batches slightly more gradient weight
- report whether small-object quality changes
- optionally use a higher-resolution P2 detection path

## Small-Object Definition

Stage 4 uses one shared normalized OBB area threshold:

- `small_object_area_thr_norm`

By default it is `0.005`, meaning the oriented box area is less than or equal to 0.5% of the normalized image area.

The three Stage 4 subsystems map to that definition as follows:

- sampler: uses normalized polygon area from Stage 1 canonical OBB labels
- loss weighting: uses normalized `xywhr` area from the training batch (`w * h`)
- small metrics: uses normalized `xywhr` area from validator ground truth

For rectangular OBB polygons these are equivalent in practice, so the definition stays reproducible across the whole
Stage 4 loop.

## Small-Object Sampling

Stage 4 sampling is implemented as an explicit `WeightedRandomSampler` and is only active in the Stage 4 trainer when:

- `use_small_object_sampling=true`

Each image receives a weight derived from the sum of small-object severities in that image:

- objects below the threshold contribute
- smaller objects contribute more strongly
- weights are clipped by `small_object_sampling_min_weight` and `small_object_sampling_max_weight`

This keeps the default baseline dataloader unchanged while giving small-object-heavy images a higher sampling
probability on the explicit Stage 4 branch.

## Scale-Aware Loss Weighting

Stage 4 does not rewrite the native `v8OBBLoss`. Instead it applies a conservative batch-level multiplier to selected
aggregated OBB loss components after the native criterion runs.

Relevant knobs:

- `use_small_object_loss_weighting`
- `small_object_loss_gain`
- `small_object_loss_on`

Current tradeoff:

- this is a stable approximation, not per-target reweighting
- it keeps the native OBB criterion intact
- it avoids large structural risk while still increasing optimization pressure on small-object-heavy batches

The RGB-IR auxiliary loss from Stage 3 is left untouched so Stage 3 train-assist and Stage 4 small-object weighting do
not fight each other.

## Small-Object Metrics

Stage 4 adds an explicit validator subclass that preserves overall OBB metrics and computes an additional small-object
subset in parallel.

Additional keys:

- `metrics/precision_small(B)`
- `metrics/recall_small(B)`
- `metrics/mAP50_small(B)`
- `metrics/mAP50-95_small(B)`

The subset is defined by filtering ground-truth OBBs with the same `small_object_area_thr_norm` used by sampler and
loss weighting. Overall metrics remain unchanged and keep driving the normal fitness key.

## High-Resolution Small Model

Stage 4 adds:

- `ultralytics/cfg/models/v8/yolov8-rgbir-obb-small.yaml`

This is a dedicated P2 OBB model that adds a higher-resolution detection branch while keeping the Stage 3 RGB-IR
train-assist design:

- train: can still consume `img_ir`
- val / predict / deploy: still RGB-only
- temporal fields remain disabled

The small-object yaml is explicit opt-in and does not replace the Stage 3 default yaml.

## How To Enable

Use the explicit Stage 4 script:

```powershell
D:\Anaconda\envs\yolo\python.exe D:\project\ultralytics-main\tools\train_uav_rgbir_obb_small.py `
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml `
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-obb-small.yaml `
  --epochs 1 --batch 2 --imgsz 256 --device cpu --workers 0 --val
```

Key Stage 4 toggles:

- `--disable-small-object-sampling`
- `--disable-small-object-loss-weighting`
- `--disable-small-object-metrics`
- `--small-object-area-thr-norm`
- `--small-object-loss-gain`

This means Stage 4 can still be run as:

- RGB-only small-object training: disable RGB-IR train assist
- RGB-IR + small-object training: keep both enabled

## Relationship To Stage 3

Stage 4 is layered on top of the Stage 3 branch:

- Stage 3 owns train-time RGB-IR cooperation
- Stage 4 adds small-object sampling, loss emphasis, metrics, and a P2 yaml

When all Stage 4 switches are off, the behavior falls back to the Stage 3 branch.

## Why `img_prev` Is Still Not Consumed

Adding temporal inputs at the same time as small-object sampling, loss weighting, and a P2 branch would multiply the
failure surface. Stage 4 keeps `img_prev` unused on purpose so the small-object loop can be validated in isolation.

Temporal feature use remains a Stage 5 responsibility.
