# Stage 5 Light Temporal Detection

## Goal

Stage 5 adds a lightweight two-frame temporal branch on top of the explicit Stage 3 and Stage 4 branches without
touching the default OBB or tracking baselines.

This stage is responsible for:

- consuming `img_prev` during training
- supporting optional temporal refinement at validation time
- supporting explicit single-frame and sequence-style inference entry points
- staying compatible with Stage 3 RGB-IR train assist
- staying compatible with Stage 4 small-object sampling, loss weighting, metrics, and small-model yaml

Tracking is still out of scope. That remains a Stage 6 concern.

## Why Two-Frame / One-Step

Stage 5 deliberately uses a one-step previous-frame design instead of long memory or video-transformer style models.
The reasons are practical:

- lower numerical risk
- easier RGB-only fallback
- easier boundary-frame handling
- easier deployment because single-frame inference still works
- cleaner future handoff to tracking, where motion/state logic belongs

## Temporal Module Structure

Stage 5 temporal refine is implemented as:

1. a lightweight previous-frame adapter that projects `img_prev` to selected stage resolutions
2. a temporal refine block that updates current-frame features using previous-frame context

Supported temporal refine modes:

- `diff_gate`
- `gated_add`
- `align_only`
- `none`

Current default is `diff_gate`, which uses the absolute feature difference as a conservative gate for residual refine.

## Temporal Stages

Stage 5 uses selected middle/high stages only:

- `temporal_feature_stages: [6, 9]` by default

This mirrors the Stage 3 RGB-IR assist philosophy:

- keep the deployable RGB graph intact
- avoid rewriting the full backbone or neck
- confine new behavior to a small number of high-value feature locations

## Boundary-Frame Strategy

`temporal_valid=False` means the previous-frame signal must not influence the current feature.

Current behavior:

- dataset still returns `img_prev`
- first frame or boundary frame marks `temporal_valid=False`
- temporal refine sees the mask and bypasses the current feature unchanged
- the temporal auxiliary loss also becomes zero on invalid temporal positions

This keeps first-frame and boundary behavior explicit and stable.

## Relationship to Stage 3 RGB-IR Train Assist

Stage 3 and Stage 5 are additive:

- Stage 3: optional train-time IR assist on selected stages
- Stage 5: optional two-frame temporal refine on selected stages

They can coexist on the same stage ids. The current feature is updated by the IR branch first and then refined by the
temporal branch when enabled.

Inference and deployment remain RGB-only with respect to IR. `img_ir` is not required for val/predict/export.

## Relationship to Stage 4 Small-Object Optimization

Stage 5 does not replace Stage 4. The Stage 5 trainer inherits Stage 4 capabilities:

- small-object sampling
- small-object loss weighting
- small-object metrics
- compatibility with the Stage 4 P2 small yaml

Conservative recommended combinations:

- RGB-only + small: Stage 4 trainer or Stage 5 trainer with temporal disabled
- RGB-IR + small: Stage 4 trainer or Stage 5 trainer with temporal disabled
- RGB-IR + small + temporal: Stage 5 trainer with Stage 4 small yaml plus temporal overrides

## Temporal Auxiliary Loss

Stage 5 includes a low-weight temporal consistency loss directly inside the temporal module path.

Current behavior:

- applied only when `use_temporal=true`
- applied only when `img_prev` is present
- applied only on `temporal_valid=True` samples
- weight controlled by `temporal_loss_weight`

The default is intentionally small so it does not dominate the native OBB losses.

## Why Tracking Is Still Deferred

Temporal refine and tracking solve different problems:

- Stage 5 improves frame-to-frame feature stability for detection
- Stage 6 will handle persistent identity, motion state, reactivation, and association

Mixing those responsibilities now would increase risk and blur the model/data contract.

## How to Enable Temporal Training

Use the explicit Stage 5 script:

```powershell
D:\Anaconda\envs\yolo\python.exe D:\project\ultralytics-main\tools\train_uav_rgbir_temporal_obb.py `
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml `
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-temporal-obb.yaml `
  --epochs 1 --batch 2 --imgsz 256 --device cpu --workers 0 --val
```

Important knobs:

- `--use-temporal` / `--disable-temporal`
- `--temporal-fusion-type`
- `--temporal-feature-stages`
- `--temporal-loss-weight`
- `--use-rgbir-train-assist`
- Stage 4 small-object flags

## How to Enable Temporal Inference

Use the explicit Stage 5 predict/demo script:

```powershell
D:\Anaconda\envs\yolo\python.exe D:\project\ultralytics-main\tools\predict_uav_temporal_obb.py `
  --source D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\images\rgb\val `
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-obb-small.yaml `
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml `
  --sequence --use-temporal --temporal-mode two_frame
```

Behavior:

- single-frame mode: no previous frame, temporal path bypassed
- sequence mode: first frame bypasses, later frames reuse a one-step previous-frame cache
- no IR input is required for inference
