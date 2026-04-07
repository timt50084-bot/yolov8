# Stage 1 Data Preprocess Pipeline

## Goal

Stage 1 builds an offline preprocessing toolchain for UAV RGB/IR OBB data. It does not modify the default training,
validation, prediction, or tracking code paths. The output of this phase is a repeatable, disk-backed dataset asset
that later stages can consume directly.

## Input Assumptions

The tools are designed around the following common inputs:

1. RGB images.
2. IR images.
3. Raw labels stored as TXT or XML.

The default local layout used in this repository is:

- `train/images/img`
- `train/images/imgr`
- `train/labels/merged`
- `val/images/img`
- `val/images/imgr`
- `val/labels/merged`

The scripts also support:

- unsplit roots with random train/val splitting,
- modality-specific label folders,
- stem normalization rules for pairing,
- XML labels containing polygon, rotated box, or axis-aligned box geometry.

## Output Structure

The preprocessing script writes a structured dataset root such as:

```text
output_root/
  images/
    rgb/
      train/
      val/
    ir/
      train/
      val/
  labels/
    obb/
      train/
      val/
    rgb/
      train/
      val/
  index/
    train_pairs.json
    val_pairs.json
    train_temporal.json
    val_temporal.json
  reports/
    preprocess_summary.json
    anomalies.json
    class_mapping.json
  data/
    uav_rgb_obb.yaml
```

Notes:

- `labels/obb` is the canonical merged OBB label output for later RGB-IR and temporal stages.
- `labels/rgb` is a compatibility mirror for current Ultralytics OBB training. It exists because the current dataset
  scanner resolves labels by mirroring `images/...` to `labels/...`, so a direct `images/rgb/train -> labels/rgb/train`
  view is required for zero-intrusion training compatibility.

## Label Standardization Strategy

The preprocessing tool normalizes raw annotations into one internal representation:

- absolute-pixel 4-point OBB polygons.

Supported raw formats:

1. TXT `class cx cy w h angle` (xywha).
2. TXT `class x1 y1 x2 y2 x3 y3 x4 y4`.
3. XML polygon labels.
4. XML rotated box labels.
5. XML axis-aligned box labels.

Class names can be normalized through:

- explicit `--names`,
- alias mapping files,
- case-insensitive and punctuation-insensitive normalization.

All mappings are recorded in `reports/class_mapping.json`.

## Why Final Training Labels Use the Current-Compatible Format

The repository currently rejects 6-column xywha labels during its standard label verification flow. Because Stage 1
must remain zero-intrusion, it does not patch the core validator. Instead, the final training labels are written as:

- `class x1 y1 x2 y2 x3 y3 x4 y4`

with normalized polygon coordinates.

This format is accepted by the current Ultralytics OBB dataset scan and allows Stage 1 outputs to be used by the
existing OBB pipeline without modifying core code.

## White-Border Crop and 100x100 Target Protection

Stage 1 only performs one crop type:

- white-border trimming.

The crop box is computed from the union of RGB and IR valid-content regions so that:

1. both modalities keep spatial alignment,
2. content present in either modality is preserved.

Target protection is applied conservatively:

- when an object is close to a crop boundary, the crop is expanded using a 100x100 protection window centered on the
  object region.

This keeps the crop logic stable, explainable, and easy to audit.

## RGB / IR Label Fusion Strategy

If both RGB and IR labels exist for the same pair:

1. same-class objects with sufficiently high overlap are matched,
2. matched polygons are merged into one OBB,
3. unmatched objects are retained and logged,
4. class-conflict overlaps are recorded instead of silently overwritten.

This avoids the unsafe behavior of blindly choosing one modality and dropping the other without traceability.

## Temporal Index Purpose

Stage 1 does not run temporal modeling, but it prepares the metadata needed for later stages. Each split gets a
temporal index that records:

1. current sample id,
2. previous sample id within the split,
3. current RGB/IR/label paths.

This is a minimal and stable entry point for later temporal datasets and sequence-aware training logic.

## Why Stage 1 Does Not Touch the Training Mainline

Stage 1 is an offline data-engineering phase. Pushing preprocessing into the runtime dataset path would increase
baseline risk, make data mutations harder to reproduce, and complicate debugging. Keeping preprocessing in `tools/`
maintains:

- baseline isolation,
- repeatability,
- easier anomaly reporting,
- clear separation between data preparation and model logic.

## How Later Phases Use These Outputs

Stage 2 and later can consume the Stage 1 outputs as follows:

1. Current OBB baseline training can read `data/uav_rgb_obb.yaml`.
2. Future RGB-IR dataset work can read `images/rgb`, `images/ir`, and `index/*_pairs.json`.
3. Future temporal work can read `index/*_temporal.json`.
4. Future debugging can inspect `reports/anomalies.json` and `reports/class_mapping.json`.

## Minimal Usage

Example preprocessing command:

```powershell
D:\Anaconda\envs\yolo\python.exe tools\prepare_uav_rgbir_obb_dataset.py `
  --input-root D:\DataSet\DroneVehicle_Processed `
  --output-root D:\DataSet\DroneVehicle_Processed_stage1 `
  --names car truck bus van "Freight Car" `
  --allow-empty-labels `
  --overwrite
```

Example check command:

```powershell
D:\Anaconda\envs\yolo\python.exe tools\check_uav_rgbir_dataset.py `
  --dataset-root D:\DataSet\DroneVehicle_Processed_stage1
```

The generated `data/uav_rgb_obb.yaml` is the minimal template for current OBB training. Extra IR and temporal index
paths are stored there only as future metadata; the current baseline ignores them safely.
