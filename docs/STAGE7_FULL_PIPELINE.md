# Stage 7: Full UAV Pipeline Entry

## 1. Goal

Stage 7 does not add new detection, temporal, or tracking algorithms. It converges the Stage 1-6 assets into one explicit engineering launcher so the full pipeline can be executed, demonstrated, and smoke-checked without changing the default Ultralytics baseline entry points.

The Stage 7 launcher is:

- `tools/uav_pipeline.py`

It is intentionally opt-in. If you do not call it, the repository keeps the existing baseline behavior.

## 2. Why Stage 7 Focuses on Orchestration

By Stage 6, the repository already contains:

- Stage 1 preprocessing and dataset checking
- Stage 3 RGB-IR train-assist training
- Stage 4 small-object optimization
- Stage 5 temporal prediction
- Stage 6 explicit UAV OBB tracking

The remaining engineering gap is not another algorithm. The gap is:

- clear task routing
- stable mode naming
- consistent output directories
- reproducible smoke commands

Stage 7 therefore adds a thin launcher layer instead of rewriting the existing stage scripts.

## 3. Supported Subcommands

`tools/uav_pipeline.py` currently supports:

- `preprocess`
- `check-data`
- `train`
- `val`
- `predict`
- `temporal-predict`
- `track`

Use `--help` on the launcher or any subcommand to inspect the routed arguments.

## 4. Supported Modes

The launcher keeps a small, explicit mode registry.

### `baseline`

Use this when you want the native Ultralytics OBB path.

- train/val/predict route to the native `YOLO(...)` API
- track routes to native tracker yaml usage
- no RGB-IR
- no small-object extras
- no temporal

### `rgbir`

Use this for Stage 3 training-time RGB-IR assistance.

- train routes to `tools/train_uav_rgbir_obb.py`
- val/predict remain RGB-only
- no temporal
- no Stage 6 tracking

### `rgbir-small`

Use this for Stage 4 small-object optimization.

- train routes to `tools/train_uav_rgbir_obb_small.py`
- uses the Stage 4 small-object model yaml
- val/predict remain RGB-only

### `rgbir-temporal`

Use this for Stage 5 lightweight temporal detection.

- train routes to `tools/train_uav_rgbir_temporal_obb.py`
- `temporal-predict` routes to `tools/predict_uav_temporal_obb.py`
- val/predict remain RGB-only unless the explicit temporal predictor is used

### `rgbir-temporal-track`

Use this for Stage 6 UAV OBB tracking-by-detection.

- training still uses the Stage 5 detector path
- `temporal-predict` stays available
- `track` routes to `tools/track_uav_obb.py`
- tracking remains RGB-only at deployment time

## 5. Mode-to-Route Mapping

Stage 7 keeps the mapping in:

- `tools/utils_uav_pipeline.py`

Each mode declares:

- default model yaml
- staged training script or native path
- whether temporal prediction is supported
- whether Stage 6 tracking is supported
- whether RGB-IR, small-object, temporal, or tracking are expected

This avoids large unstructured `if/else` routing blocks.

## 6. Output Directory Convention

By default, Stage 7 writes into:

- `outputs/uav_pipeline/<mode>/<subtask>/<run_name>/`

Examples:

- `outputs/uav_pipeline/data/preprocess/<run_name>/`
- `outputs/uav_pipeline/baseline/train/<run_name>/`
- `outputs/uav_pipeline/rgbir/val/<run_name>/`
- `outputs/uav_pipeline/rgbir-temporal/temporal_predict/<run_name>/`
- `outputs/uav_pipeline/rgbir-temporal-track/track/<run_name>/`

Each routed run writes a `route.json` manifest so the resolved mode, script/API path, model source, and output directory remain traceable.

Tracking results use a stable JSON naming convention:

- default: `<run_dir>/tracks.json`

## 7. End-to-End Usage

The intended sequence is:

1. preprocess raw UAV RGB/IR data
2. check the prepared dataset
3. train a chosen detector mode
4. validate the detector
5. run image prediction
6. run temporal sequence prediction if needed
7. run tracking if needed

This keeps the earlier stage tools intact while exposing a single top-level engineering entry.

## 8. Recommended Commands

### Data Preparation

```powershell
python tools/uav_pipeline.py preprocess --input-root D:\data\uav_raw --name prep_run -- --names car truck bus van "Freight Car"
```

```powershell
python tools/uav_pipeline.py check-data --dataset-root D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run --name check_run
```

### Training

#### Baseline

```powershell
python tools/uav_pipeline.py train --mode baseline --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --epochs 100 --batch 8 --imgsz 1024 --device 0
```

#### Stage 3 RGB-IR

```powershell
python tools/uav_pipeline.py train --mode rgbir --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --epochs 100 --batch 8 --imgsz 1024 --device 0
```

#### Stage 4 RGB-IR + Small Object

```powershell
python tools/uav_pipeline.py train --mode rgbir-small --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --epochs 100 --batch 8 --imgsz 1024 --device 0
```

#### Stage 5 RGB-IR + Temporal

```powershell
python tools/uav_pipeline.py train --mode rgbir-temporal --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --epochs 100 --batch 8 --imgsz 1024 --device 0
```

### Validation

#### Baseline

```powershell
python tools/uav_pipeline.py val --mode baseline --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --weights D:\path\to\baseline_best.pt --imgsz 1024 --device 0
```

#### RGB-IR

```powershell
python tools/uav_pipeline.py val --mode rgbir --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --weights D:\path\to\rgbir_best.pt --imgsz 1024 --device 0
```

#### RGB-IR + Small

```powershell
python tools/uav_pipeline.py val --mode rgbir-small --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --weights D:\path\to\rgbir_small_best.pt --imgsz 1024 --device 0
```

#### RGB-IR + Temporal

```powershell
python tools/uav_pipeline.py val --mode rgbir-temporal --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --weights D:\path\to\rgbir_temporal_best.pt --imgsz 1024 --device 0
```

### Prediction

#### Single-Image RGB-only Prediction

```powershell
python tools/uav_pipeline.py predict --mode baseline --source D:\data\demo\frame_0001.jpg --weights D:\path\to\baseline_best.pt --imgsz 1024 --device 0
```

### Temporal Sequence Prediction

```powershell
python tools/uav_pipeline.py temporal-predict --mode rgbir-temporal --source D:\data\demo_sequence --weights D:\path\to\rgbir_temporal_best.pt --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --imgsz 1024 --device 0 --max-frames 100
```

### Tracking

```powershell
python tools/uav_pipeline.py track --mode rgbir-temporal-track --source D:\data\demo_sequence --weights D:\path\to\rgbir_temporal_best.pt --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\prep_run\data\uav_rgb_obb.yaml --imgsz 1024 --device 0 --max-frames 200
```

## 9. Minimal Smoke Commands

The following are the recommended Stage 7 smoke commands:

### Preprocess + Check

```powershell
python tools/uav_pipeline.py preprocess --input-root D:\project\ultralytics-main\runs\stage1_preprocess_smoke\raw_sample --name smoke_prepared -- --names car truck bus van "Freight Car"
```

```powershell
python tools/uav_pipeline.py check-data --dataset-root D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared --name smoke_check
```

### Train Route Smoke

```powershell
python tools/uav_pipeline.py train --mode baseline --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --epochs 1 --batch 2 --imgsz 256 --workers 0 --device cpu --val
```

```powershell
python tools/uav_pipeline.py train --mode rgbir --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --epochs 1 --batch 2 --imgsz 256 --workers 0 --device cpu --val
```

```powershell
python tools/uav_pipeline.py train --mode rgbir-temporal --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --epochs 1 --batch 2 --imgsz 256 --workers 0 --device cpu --val
```

### Validation / Prediction / Temporal Prediction / Tracking Smoke

```powershell
python tools/uav_pipeline.py val --mode baseline --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --weights D:\path\to\best.pt --batch 2 --imgsz 256 --workers 0 --device cpu
```

```powershell
python tools/uav_pipeline.py predict --mode baseline --source D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\images\rgb\val\00001.jpg --weights D:\path\to\best.pt --imgsz 256 --device cpu
```

```powershell
python tools/uav_pipeline.py temporal-predict --mode rgbir-temporal --source D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\images\rgb\val --weights D:\path\to\rgbir_temporal_best.pt --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --imgsz 256 --device cpu --max-frames 3
```

```powershell
python tools/uav_pipeline.py track --mode rgbir-temporal-track --source D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\images\rgb\train --weights D:\path\to\rgbir_temporal_best.pt --data D:\project\ultralytics-main\outputs\uav_pipeline\data\preprocess\smoke_prepared\data\uav_rgb_obb.yaml --imgsz 256 --device cpu --max-frames 3 --conf 0.001 -- --track-low-thresh 0.0 --new-track-thresh 0.0 --match-iou-thresh 0.01 --disable-appearance
```

## 10. Baseline Relationship

Stage 7 does not replace the baseline.

- If you keep using `yolo obb train/val/predict/track`, baseline behavior stays as before.
- If you use `tools/uav_pipeline.py --mode baseline`, the launcher still routes to the native baseline path instead of a pseudo-baseline reimplementation.

This is why Stage 7 keeps the earlier stage scripts and only adds an orchestration layer above them.
