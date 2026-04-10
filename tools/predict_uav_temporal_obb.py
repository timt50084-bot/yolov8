from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import cv2
    import numpy as np
    import torch
    import yaml

    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.obb.rgbir_temporal_obb_train import RGBIRTemporalOBBModel
    from ultralytics.nn.tasks import torch_safe_load, yaml_model_load
    from ultralytics.utils import nms, ops
except ModuleNotFoundError as exc:
    cv2 = None
    np = None
    torch = None
    yaml = None
    Results = Any
    RGBIRTemporalOBBModel = Any
    torch_safe_load = None
    yaml_model_load = None
    nms = None
    ops = None
    _IMPORT_ERROR = exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Runtime dependencies for tools/predict_uav_temporal_obb.py are missing. "
            "Install/activate an environment with torch, opencv-python, pyyaml, and ultralytics before running inference."
        ) from _IMPORT_ERROR


def parse_stage_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def normalize_device_arg(device: str) -> str:
    """Keep common CLI shorthands like '0' compatible with torch.device."""
    device = str(device).strip()
    return f"cuda:{device}" if device.isdigit() else device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit Stage 5 temporal OBB predict/demo entry.")
    parser.add_argument("--source", required=True, type=str, help="Single image path or a directory of sequential images.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml"),
        type=str,
        help="Stage 5 temporal model yaml. Can also point to Stage 4 small yaml when combined with temporal overrides.",
    )
    parser.add_argument("--weights", default=None, type=str, help="Optional PyTorch checkpoint to load.")
    parser.add_argument("--data", default=None, type=str, help="Optional dataset yaml for nc/names override.")
    parser.add_argument("--imgsz", default=256, type=int, help="Inference resize size.")
    parser.add_argument("--device", default="cpu", type=str, help="Inference device.")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold for OBB NMS.")
    parser.add_argument("--iou", default=0.45, type=float, help="IoU threshold for OBB NMS.")
    parser.add_argument("--max-det", default=300, type=int, help="Maximum detections per frame.")
    parser.add_argument("--sequence", action="store_true", help="Enable one-step previous-frame cache across frames.")
    parser.add_argument("--max-frames", default=0, type=int, help="Optional max frames to process from a directory.")
    parser.add_argument("--save-dir", default=None, type=str, help="Optional output directory for rendered images and txt labels.")
    parser.add_argument("--save-json", default=None, type=str, help="Optional JSON path for structured predictions.")
    parser.add_argument("--save-vis", dest="save_vis", action="store_true", help="Save rendered OBB visualizations.")
    parser.add_argument("--no-save-vis", dest="save_vis", action="store_false", help="Disable rendered visualization saving.")
    parser.set_defaults(save_vis=False)
    parser.add_argument("--save-txt", dest="save_txt", action="store_true", help="Save per-frame YOLO-format txt labels.")
    parser.add_argument("--no-save-txt", dest="save_txt", action="store_false", help="Disable per-frame txt label saving.")
    parser.set_defaults(save_txt=False)
    parser.add_argument("--line-width", default=None, type=int, help="Optional visualization line width override.")
    parser.add_argument("--use-temporal", dest="use_temporal", action="store_true")
    parser.add_argument("--disable-temporal", dest="use_temporal", action="store_false")
    parser.set_defaults(use_temporal=True)
    parser.add_argument("--temporal-mode", default="two_frame", type=str, help="Temporal mode. Stage 5 supports two_frame.")
    parser.add_argument("--temporal-fusion-type", default="diff_gate", type=str, help="Temporal refine mode.")
    parser.add_argument("--temporal-feature-stages", default="6,9", type=str, help="Comma-separated temporal stage ids.")
    parser.add_argument("--temporal-branch-width", default=0.25, type=float, help="Width multiplier for previous-frame adapters.")
    parser.add_argument("--temporal-loss-weight", default=0.02, type=float, help="Temporal aux loss weight. Only matters during training.")
    parser.add_argument("--temporal-residual-scale", default=0.1, type=float, help="Residual scale for temporal refine.")
    return parser.parse_args()


def load_data_cfg(path: str | None) -> dict[str, Any]:
    ensure_runtime_dependencies()
    if path is None:
        return {}
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


def build_model(args: argparse.Namespace) -> RGBIRTemporalOBBModel:
    ensure_runtime_dependencies()
    model_cfg = yaml_model_load(args.model)
    model_cfg["use_temporal"] = args.use_temporal
    model_cfg["temporal_mode"] = args.temporal_mode
    model_cfg["temporal_fusion_type"] = args.temporal_fusion_type
    model_cfg["temporal_feature_stages"] = parse_stage_list(args.temporal_feature_stages)
    model_cfg["temporal_branch_width"] = args.temporal_branch_width
    model_cfg["temporal_loss_weight"] = args.temporal_loss_weight
    model_cfg["temporal_residual_scale"] = args.temporal_residual_scale

    data_cfg = load_data_cfg(args.data)
    names = normalize_names(data_cfg.get("names"))
    nc = data_cfg.get("nc", len(names) if names else None)

    model = RGBIRTemporalOBBModel(model_cfg, nc=nc, ch=3, verbose=False)
    if args.weights:
        ckpt, _ = torch_safe_load(args.weights)
        model.load(ckpt, verbose=False)
    if names:
        model.names = names
    elif not hasattr(model, "names") or not model.names:
        model.names = {i: str(i) for i in range(int(nc or 1))}
    model = model.to(torch.device(normalize_device_arg(args.device))).eval()
    return model


def iter_frames(source: str, max_frames: int = 0) -> list[Path]:
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image source: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    frames = [p for p in sorted(path.iterdir(), key=lambda item: item.name.lower()) if p.suffix.lower() in IMAGE_EXTS]
    if not frames:
        raise ValueError(f"No image frames found in source directory: {source}")
    return frames[:max_frames] if max_frames > 0 else frames


def load_image_bgr(path: Path) -> np.ndarray:
    ensure_runtime_dependencies()
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return frame_bgr


def preprocess_frame_bgr(frame_bgr: np.ndarray, imgsz: int, device: torch.device) -> torch.Tensor:
    ensure_runtime_dependencies()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous().float() / 255.0
    return tensor.to(device)


def postprocess_obb_predictions(
    model: RGBIRTemporalOBBModel,
    preds: Any,
    img_tensor: torch.Tensor,
    orig_shape: tuple[int, int],
    *,
    conf: float,
    iou: float,
    max_det: int,
) -> np.ndarray:
    ensure_runtime_dependencies()
    outputs = nms.non_max_suppression(
        preds,
        conf_thres=conf,
        iou_thres=iou,
        max_det=max_det,
        nc=len(model.names),
        agnostic=False,
        rotated=True,
        end2end=getattr(model, "end2end", False),
    )
    pred = outputs[0]
    if pred.numel() == 0:
        return np.zeros((0, 7), dtype=np.float32)
    rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
    rboxes[:, :4] = ops.scale_boxes(img_tensor.shape[2:], rboxes[:, :4], orig_shape, xywh=True)
    obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
    return obb.detach().cpu().numpy().astype(np.float32, copy=False)


def build_results(frame_bgr: np.ndarray, frame_path: Path, names: dict[int, str], detections: np.ndarray) -> Results:
    ensure_runtime_dependencies()
    obb = torch.from_numpy(detections.copy()) if detections.size else torch.zeros((0, 7), dtype=torch.float32)
    return Results(orig_img=frame_bgr, path=str(frame_path), names=names, obb=obb)


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    if save_dir is None and (args.save_vis or args.save_txt):
        raise ValueError("--save-dir is required when visualization or txt saving is enabled.")
    json_path = Path(args.save_json).expanduser().resolve() if args.save_json else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = None
    txt_dir = None
    if save_dir and args.save_vis:
        vis_dir = save_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    if save_dir and args.save_txt:
        txt_dir = save_dir / "labels"
        txt_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, vis_dir, txt_dir


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_dependencies()
    frames = iter_frames(args.source, max_frames=args.max_frames)
    device = torch.device(normalize_device_arg(args.device))
    model = build_model(args)
    model.reset_temporal_cache()
    _, vis_dir, txt_dir = resolve_output_paths(args)

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for i, frame_path in enumerate(frames):
            frame_bgr = load_image_bgr(frame_path)
            img_tensor = preprocess_frame_bgr(frame_bgr, imgsz=args.imgsz, device=device)
            if args.sequence:
                preds = model.predict_with_prev_cache(img_tensor)
                temporal_valid = i > 0
            else:
                preds = model.predict(img_tensor, img_prev=None, temporal_valid=None)
                temporal_valid = False

            detections = postprocess_obb_predictions(
                model,
                preds,
                img_tensor,
                frame_bgr.shape[:2],
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
            )
            result = build_results(frame_bgr, frame_path, model.names, detections)

            vis_path = None
            if vis_dir is not None:
                vis_path = vis_dir / frame_path.name
                result.save(filename=str(vis_path), line_width=args.line_width)

            txt_path = None
            if txt_dir is not None:
                txt_path = txt_dir / f"{frame_path.stem}.txt"
                result.save_txt(txt_path, save_conf=True)

            record = {
                "frame_id": int(i),
                "frame_name": frame_path.name,
                "temporal_valid_input": bool(temporal_valid),
                "temporal_used": bool(model.last_temporal_used),
                "temporal_stages": [int(x) for x in model.last_temporal_stage_ids],
                "num_detections": int(len(result)),
                "detections": result.summary(normalize=False, decimals=5),
                "visualization_path": str(vis_path) if vis_path else None,
                "txt_path": str(txt_path) if txt_path else None,
            }
            records.append(record)
            print(
                f"frame={i} file={frame_path.name} detections={record['num_detections']} "
                f"sequence={args.sequence} temporal_valid_input={temporal_valid} "
                f"temporal_used={model.last_temporal_used} temporal_stages={record['temporal_stages']}"
            )

    payload = {
        "source": str(Path(args.source).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "weights": None if args.weights is None else str(Path(args.weights).expanduser()),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "sequence": bool(args.sequence),
        "use_temporal": bool(args.use_temporal),
        "frames": records,
    }

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
