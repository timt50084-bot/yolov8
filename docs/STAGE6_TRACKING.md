# Stage 6: UAV OBB Tracking

## Goal

Stage 6 adds an explicit, opt-in UAV-oriented multi-object tracking branch on top of the existing OBB detector stack.
The tracker is designed for RGB-only deployment and does not replace the native `yolo track` path.

The Stage 6 deliverable is a complete tracking-by-detection loop:

- OBB-aware matching,
- track state management,
- short-term lost buffering,
- reactivation within a configurable buffer,
- optional lightweight appearance support,
- an explicit tracking script for sequential images or video.

## Why Tracking-by-Detection

The repository already has strong detection branches after Stages 3-5:

- Stage 3: RGB-IR training-time assist,
- Stage 4: small-object optimization,
- Stage 5: optional two-frame temporal detection.

Stage 6 therefore builds tracking as an upper-layer consumer of detector outputs instead of coupling IDs into the
detector forward or validation loops. This keeps tracking isolated and reduces regression risk for detector loss,
overall P/R/mAP, and small-object metrics.

## Why Baseline Track Is Not Modified

The native tracker registry in [track.py](D:/project/ultralytics-main/ultralytics/trackers/track.py) still only serves
ByteTrack and BoT-SORT. Stage 6 does not extend `TRACKER_MAP`, does not intercept `yolo track`, and does not alter the
default tracker YAMLs.

Instead, Stage 6 is enabled explicitly through:

- [uav_obb_tracker.yaml](D:/project/ultralytics-main/ultralytics/cfg/trackers/uav_obb_tracker.yaml)
- [track_uav_obb.py](D:/project/ultralytics-main/tools/track_uav_obb.py)

This keeps baseline tracking behavior unchanged.

## OBB Matching Design

The Stage 6 matcher lives in [obb_matching.py](D:/project/ultralytics-main/ultralytics/trackers/utils/obb_matching.py).

The association cost is a conservative weighted combination of:

- OBB overlap:
  - implemented with Ultralytics `batch_probiou` on `xywhr`
  - this is used instead of a heavier polygon-intersection dependency
- normalized center distance
- optional lightweight appearance cosine distance
- additive class mismatch penalty

Supported cost controls:

- `iou_weight`
- `center_weight`
- `appearance_weight`
- `class_mismatch_penalty`
- `match_iou_thresh`
- `match_thresh`
- `max_center_distance`

`match_iou_thresh` acts as a conservative gate so obviously unrelated detections are filtered before assignment.

## State Management

The tracker implementation lives in [uav_obb_tracker.py](D:/project/ultralytics-main/ultralytics/trackers/uav_obb_tracker.py).

Track states:

- `tentative`
- `tracked`
- `lost`
- `removed`

Lifecycle summary:

1. frame 1 or unmatched detections above `new_track_thresh` create tentative tracks
2. repeated matches promote them to `tracked`
3. matched tracks update geometry, score, and optional appearance
4. unmatched tracked tracks move to `lost`
5. lost tracks stay in memory for `track_buffer` frames
6. if a new detection matches a lost track within the buffer, the track is reactivated with the same `track_id`
7. expired lost tracks become `removed`

`reset()` clears the entire sequence state and must be called between unrelated sequences.

## Short-Term Lost And Reactivation

Stage 6 intentionally keeps reactivation short-term and buffer-based:

- no long memory bank
- no independent ReID model
- no tracking state machine inside the detector

This is sufficient for short UAV occlusions or brief detector dropouts while keeping the system easy to reason about.

## Lightweight Appearance Support

Appearance is implemented conservatively and remains optional.

Current Stage 6 appearance descriptor:

- crops the RGB image using each OBB's axis-aligned enclosure
- computes a lightweight color-histogram + mean/std embedding
- smooths embeddings per track with EMA
- uses cosine distance with a small configurable weight

This avoids introducing a heavy ReID network while still providing a useful auxiliary signal.

If `use_appearance=false`, the tracker continues to operate on OBB overlap, center distance, and class constraints only.

## Relation To Stage 5 Temporal Detector

The tracker can consume:

- normal single-frame detector outputs
- Stage 5 temporal-detector outputs

But the tracker itself does not require temporal mode.

This is deliberate:

- temporal detection is optional
- tracking remains RGB-only
- detector and tracker stay loosely coupled

In the Stage 6 tracking script:

- single-frame mode uses standard Stage 5 detector prediction with temporal disabled
- temporal mode uses the Stage 5 one-step previous-frame cache
- the first frame always degrades safely to single-frame prediction

## Why Tracking Remains RGB-Only

Stage 3 uses IR only during training-time detector assistance.
Stage 6 tracking never requires IR input:

- detections come from RGB-only inference
- appearance embeddings are cropped from RGB frames only
- temporal detector mode, when enabled, uses previous RGB frames only

This preserves the deployment constraint that tracking on video can run from RGB streams alone.

## Explicit Enablement

Single-frame detector + tracking:

```bash
D:\Anaconda\envs\yolo\python.exe D:\project\ultralytics-main\tools\track_uav_obb.py ^
  --source D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\images\rgb\train ^
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-temporal-obb.yaml ^
  --weights D:\project\ultralytics-main\runs\stage5_smoke\stage5_temporal_small_compare\weights\best.pt ^
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml ^
  --tracker D:\project\ultralytics-main\ultralytics\cfg\trackers\uav_obb_tracker.yaml ^
  --disable-temporal-detector
```

Temporal detector + tracking:

```bash
D:\Anaconda\envs\yolo\python.exe D:\project\ultralytics-main\tools\track_uav_obb.py ^
  --source D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\images\rgb\train ^
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-temporal-obb.yaml ^
  --weights D:\project\ultralytics-main\runs\stage5_smoke\stage5_temporal_small_compare\weights\best.pt ^
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml ^
  --tracker D:\project\ultralytics-main\ultralytics\cfg\trackers\uav_obb_tracker.yaml ^
  --use-temporal-detector
```

Optional outputs:

- `--save-json path/to/results.json`
- `--track-low-thresh`
- `--new-track-thresh`
- `--match-thresh`
- `--match-iou-thresh`
- `--use-appearance` / `--disable-appearance`

## Stage 6 Boundary

Stage 6 intentionally stops at tracking-by-detection.
It does not implement:

- full default CLI unification
- default tracker registry takeover
- long-term memory tracking
- heavy ReID
- track-aware detector training

Those concerns remain reserved for Stage 7 integration work.
