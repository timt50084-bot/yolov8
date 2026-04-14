from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolo.obb.rgbir_obb_train import RGBIRTrainAssistOBBModel
from ultralytics.models.yolo.obb.rgbir_small_obb_train import RGBIRSmallObjectOBBModel
from ultralytics.models.yolo.obb.rgbir_small_val import SmallObjectOBBValidator
from ultralytics.models.yolo.obb.rgbir_temporal_obb_train import RGBIRTemporalOBBModel
from ultralytics.models.yolo.obb.rgbir_temporal_val import TemporalOBBValidator
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.tasks import torch_safe_load, yaml_model_load


MODE_DEFAULT_MODELS = {
    "rgbir": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb.yaml",
    "rgbir-small": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb-small.yaml",
    "small": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb-small.yaml",
    "rgbir-small-temporal": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb-small.yaml",
    "temporal_small": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-small-temporal-obb.yaml",
    "temporal": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-temporal-obb.yaml",
    "rgbir-temporal": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml",
    "rgbir-temporal-track": REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit non-baseline Stage 7 validation entry.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=list(MODE_DEFAULT_MODELS),
        help="Stage mode whose explicit validation semantics should be used.",
    )
    parser.add_argument("--data", required=True, type=str, help="Prepared dataset yaml.")
    parser.add_argument("--weights", required=True, type=str, help="Checkpoint to validate.")
    parser.add_argument("--model", default=None, type=str, help="Optional model yaml override.")
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    parser.add_argument("--max-det", default=300, type=int)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "stage7_val"), type=str)
    parser.add_argument("--name", default="val", type=str)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--enable-small-object-metrics", dest="enable_small_object_metrics", action="store_true")
    parser.add_argument("--disable-small-object-metrics", dest="enable_small_object_metrics", action="store_false")
    parser.set_defaults(enable_small_object_metrics=None)
    parser.add_argument("--small-object-area-thr-norm", default=0.005, type=float)
    parser.add_argument("--use-temporal-val", dest="use_temporal_val", action="store_true")
    parser.add_argument("--disable-temporal-val", dest="use_temporal_val", action="store_false")
    parser.set_defaults(use_temporal_val=False)
    return parser.parse_args()


def load_data_cfg(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset yaml at {path} to load into a mapping.")
    return data


def normalize_names(raw: Any) -> dict[int, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    raise TypeError(f"Unsupported names type: {type(raw)!r}")


def default_model_for_mode(mode: str) -> str:
    return str(MODE_DEFAULT_MODELS[mode])


def build_model(args: argparse.Namespace, data_cfg: dict[str, Any]):
    model_cfg = yaml_model_load(args.model or default_model_for_mode(args.mode))
    if args.mode in {"rgbir-small-temporal", "temporal_small"}:
        model_cfg["use_temporal"] = True
        model_cfg["temporal_mode"] = str(model_cfg.get("temporal_mode", "two_frame") or "two_frame")
    names = normalize_names(data_cfg.get("names"))
    nc = data_cfg.get("nc", len(names) if names else None)

    if args.mode == "rgbir":
        model = RGBIRTrainAssistOBBModel(model_cfg, nc=nc, ch=3, verbose=False)
    elif args.mode in {"rgbir-small", "small"}:
        model = RGBIRSmallObjectOBBModel(model_cfg, nc=nc, ch=3, verbose=False)
    else:
        model = RGBIRTemporalOBBModel(model_cfg, nc=nc, ch=3, verbose=False)

    ckpt, _ = torch_safe_load(args.weights)
    model.load(ckpt, verbose=False)
    if names:
        model.names = names
    return model


def build_validator(args: argparse.Namespace, save_dir: Path):
    validator_args = {
        "data": args.data,
        "model": args.weights,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": args.plots,
        "task": "obb",
        "split": "val",
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "augment": False,
        "half": False,
        "dnn": False,
    }
    enable_small_metrics = args.enable_small_object_metrics
    if enable_small_metrics is None:
        enable_small_metrics = args.mode in {"rgbir-small", "small", "rgbir-small-temporal", "temporal_small"}

    if args.mode == "rgbir":
        return OBBValidator(save_dir=save_dir, args=validator_args)
    if args.mode in {"rgbir-small", "small"}:
        return SmallObjectOBBValidator(
            save_dir=save_dir,
            args=validator_args,
            enable_small_object_metrics=bool(enable_small_metrics),
            small_object_area_thr_norm=float(args.small_object_area_thr_norm),
        )
    return TemporalOBBValidator(
        save_dir=save_dir,
        args=validator_args,
        use_temporal=bool(args.use_temporal_val),
        enable_small_object_metrics=bool(enable_small_metrics),
        small_object_area_thr_norm=float(args.small_object_area_thr_norm),
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.project) / args.name
    data_cfg = load_data_cfg(args.data)
    model = build_model(args, data_cfg)
    validator = build_validator(args, save_dir=save_dir)
    results = validator(model=model)
    if results is None:
        results = {}
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
