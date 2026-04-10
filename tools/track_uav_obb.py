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

    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.obb.rgbir_obb_train import _parse_stage_list
    from ultralytics.models.yolo.obb.rgbir_temporal_obb_train import RGBIRTemporalOBBModel
    from ultralytics.nn.tasks import torch_safe_load, yaml_model_load
    from ultralytics.trackers.uav_obb_tracker import UAVOBBTracker
    from ultralytics.utils import YAML, nms, ops
except ModuleNotFoundError as exc:
    cv2 = None
    np = None
    torch = None
    yaml = None
    LetterBox = Any
    Results = Any
    _parse_stage_list = None
    RGBIRTemporalOBBModel = Any
    torch_safe_load = None
    yaml_model_load = None
    UAVOBBTracker = Any
    YAML = None
    nms = None
    ops = None
    _IMPORT_ERROR = exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}


def ensure_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Runtime dependencies for tools/track_uav_obb.py are missing. "
            "Install/activate an environment with torch, opencv-python, pyyaml, and ultralytics before running tracking."
        ) from _IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit Stage 6 UAV OBB tracking entry.")
    parser.add_argument("--source", required=True, type=str, help="Image, image directory, or video path.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml"),
        type=str,
        help="Stage 5 detector yaml. Can also point to the Stage 4 small yaml when temporal overrides are supplied.",
    )
    parser.add_argument("--weights", default=None, type=str, help="Optional checkpoint to load into the Stage 5 detector.")
    parser.add_argument("--data", default=None, type=str, help="Optional dataset yaml for nc/names override.")
    parser.add_argument(
        "--tracker",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "trackers" / "uav_obb_tracker.yaml"),
        type=str,
        help="Stage 6 tracker yaml.",
    )
    parser.add_argument("--imgsz", default=256, type=int, help="Inference image size.")
    parser.add_argument("--device", default="cpu", type=str, help="Inference device.")
    parser.add_argument("--conf", default=0.10, type=float, help="Confidence threshold for OBB NMS.")
    parser.add_argument("--iou", default=0.45, type=float, help="IoU threshold for OBB NMS.")
    parser.add_argument("--max-det", default=100, type=int, help="Maximum detections per frame.")
    parser.add_argument("--max-frames", default=0, type=int, help="Optional maximum number of frames to process.")
    parser.add_argument("--save-json", default=None, type=str, help="Optional JSON path for tracked results.")
    parser.add_argument("--save-video", default=None, type=str, help="Optional rendered output video path.")
    parser.add_argument("--line-width", default=None, type=int, help="Optional visualization line width override.")
    parser.add_argument("--track-low-thresh", default=-1.0, type=float, help="Optional override for tracker low threshold.")
    parser.add_argument("--new-track-thresh", default=-1.0, type=float, help="Optional override for tracker initialization threshold.")
    parser.add_argument("--match-thresh", default=-1.0, type=float, help="Optional override for tracker assignment cost threshold.")
    parser.add_argument("--match-iou-thresh", default=-1.0, type=float, help="Optional override for OBB IoU gating threshold.")
    parser.add_argument("--use-appearance", dest="use_appearance", action="store_true")
    parser.add_argument("--disable-appearance", dest="use_appearance", action="store_false")
    parser.set_defaults(use_appearance=None)
    parser.add_argument("--use-temporal-detector", dest="use_temporal_detector", action="store_true")
    parser.add_argument("--disable-temporal-detector", dest="use_temporal_detector", action="store_false")
    parser.set_defaults(use_temporal_detector=False)
    parser.add_argument("--temporal-mode", default="two_frame", type=str, help="Temporal mode. Stage 6 only uses Stage 5's one-step temporal detector.")
    parser.add_argument("--temporal-fusion-type", default="diff_gate", type=str, help="Temporal fusion type override.")
    parser.add_argument("--temporal-feature-stages", default="6,9", type=str, help="Comma-separated temporal stage ids.")
    parser.add_argument("--temporal-branch-width", default=0.25, type=float, help="Previous-frame adapter width multiplier.")
    parser.add_argument("--temporal-loss-weight", default=0.02, type=float, help="Temporal auxiliary loss weight. Only matters during training.")
    parser.add_argument("--temporal-residual-scale", default=0.10, type=float, help="Temporal residual scale.")
    return parser.parse_args()


def normalize_device_arg(device: str) -> str:
    """Keep common CLI shorthands like '0' compatible with torch.device."""
    device = str(device).strip()
    return f"cuda:{device}" if device.isdigit() else device


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


def build_detector(args: argparse.Namespace) -> RGBIRTemporalOBBModel:
    ensure_runtime_dependencies()
    cfg = yaml_model_load(args.model)
    cfg["use_temporal"] = bool(args.use_temporal_detector)
    cfg["temporal_mode"] = args.temporal_mode
    cfg["temporal_fusion_type"] = args.temporal_fusion_type
    cfg["temporal_feature_stages"] = _parse_stage_list(args.temporal_feature_stages)
    cfg["temporal_branch_width"] = float(args.temporal_branch_width)
    cfg["temporal_loss_weight"] = float(args.temporal_loss_weight)
    cfg["temporal_residual_scale"] = float(args.temporal_residual_scale)

    data_cfg = load_data_cfg(args.data)
    names = normalize_names(data_cfg.get("names"))
    nc = data_cfg.get("nc", len(names) if names else None)
    model = RGBIRTemporalOBBModel(cfg, nc=nc, ch=3, verbose=False)
    if args.weights:
        ckpt, _ = torch_safe_load(args.weights)
        model.load(ckpt, verbose=False)
    if names:
        model.names = names
    elif not hasattr(model, "names") or not model.names:
        model.names = {i: str(i) for i in range(int(nc or 1))}
    model = model.to(torch.device(normalize_device_arg(args.device))).eval()
    model.reset_temporal_cache()
    return model


def build_tracker(args: argparse.Namespace) -> UAVOBBTracker:
    ensure_runtime_dependencies()
    tracker_cfg = YAML.load(args.tracker)
    if not isinstance(tracker_cfg, dict):
        raise TypeError(f"Expected tracker yaml at {args.tracker} to load into a mapping.")
    if args.track_low_thresh >= 0:
        tracker_cfg["track_low_thresh"] = float(args.track_low_thresh)
    if args.new_track_thresh >= 0:
        tracker_cfg["new_track_thresh"] = float(args.new_track_thresh)
    if args.match_thresh >= 0:
        tracker_cfg["match_thresh"] = float(args.match_thresh)
    if args.match_iou_thresh >= 0:
        tracker_cfg["match_iou_thresh"] = float(args.match_iou_thresh)
    if args.use_appearance is not None:
        tracker_cfg["use_appearance"] = bool(args.use_appearance)
    return UAVOBBTracker(args=tracker_cfg, frame_rate=30)


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def iter_image_paths(source: str, max_frames: int = 0) -> list[Path]:
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


def frame_generator(source: str, max_frames: int = 0):
    ensure_runtime_dependencies()
    path = Path(source)
    if is_video(path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video source: {source}")
        count = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                yield count, frame_bgr, f"{path.stem}_{count:06d}"
                count += 1
                if max_frames > 0 and count >= max_frames:
                    break
        finally:
            cap.release()
        return

    for idx, img_path in enumerate(iter_image_paths(source, max_frames=max_frames)):
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        yield idx, frame_bgr, img_path.name


def preprocess_frame(frame_bgr: np.ndarray, imgsz: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    ensure_runtime_dependencies()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    resized = letterbox(image=rgb)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).contiguous().float() / 255.0
    return tensor.to(device), rgb


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


def summarize_states(tracks: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for track in tracks:
        state = str(track["state"])
        summary[state] = summary.get(state, 0) + 1
    return summary


def build_results(frame_bgr: np.ndarray, frame_name: str, names: dict[int, str], track_rows: np.ndarray) -> Results:
    ensure_runtime_dependencies()
    obb = torch.from_numpy(track_rows.copy()) if track_rows.size else torch.zeros((0, 8), dtype=torch.float32)
    return Results(orig_img=frame_bgr, path=frame_name, names=names, obb=obb)


def resolve_output_frame_rate(source: str) -> float:
    ensure_runtime_dependencies()
    path = Path(source)
    if not is_video(path):
        return 30.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video source: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else 30.0


def create_video_writer(save_path: Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    ensure_runtime_dependencies()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if suffix == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (frame_shape[1], frame_shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {save_path}")
    return writer


def run_tracking(args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_dependencies()
    device = torch.device(normalize_device_arg(args.device))
    detector = build_detector(args)
    tracker = build_tracker(args)
    save_json_path = Path(args.save_json).expanduser().resolve() if args.save_json else None
    save_video_path = Path(args.save_video).expanduser().resolve() if args.save_video else None
    if save_json_path:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
    if save_video_path:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)

    output_fps = resolve_output_frame_rate(args.source) if save_video_path else 0.0
    video_writer: cv2.VideoWriter | None = None
    records: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for frame_idx, frame_bgr, frame_name in frame_generator(args.source, max_frames=args.max_frames):
                img_tensor, frame_rgb = preprocess_frame(frame_bgr, imgsz=args.imgsz, device=device)
                if args.use_temporal_detector:
                    preds = detector.predict_with_prev_cache(img_tensor)
                else:
                    preds = detector.predict(img_tensor, img_prev=None, temporal_valid=None)

                detections = postprocess_obb_predictions(
                    detector,
                    preds,
                    img_tensor,
                    frame_rgb.shape[:2],
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                )
                track_rows = tracker.update(detections, frame_rgb, frame_id=frame_idx)
                active_tracks = tracker.get_active_tracks()
                result = build_results(frame_bgr, frame_name, detector.names, track_rows)

                if save_video_path:
                    if video_writer is None:
                        video_writer = create_video_writer(save_video_path, frame_bgr.shape, output_fps)
                    rendered = result.plot(line_width=args.line_width)
                    video_writer.write(rendered)

                records.append(
                    {
                        "frame_id": int(frame_idx),
                        "frame_name": frame_name,
                        "num_detections": int(detections.shape[0]),
                        "num_tracks": int(len(active_tracks)),
                        "temporal_detector": bool(args.use_temporal_detector),
                        "temporal_used": bool(detector.last_temporal_used),
                        "tracks": active_tracks,
                    }
                )
                print(
                    f"frame={frame_idx} file={frame_name} detections={detections.shape[0]} tracks={len(active_tracks)} "
                    f"temporal_detector={args.use_temporal_detector} temporal_used={detector.last_temporal_used} "
                    f"states={summarize_states(active_tracks)}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()

    if not records:
        raise ValueError(f"No frames found in source: {args.source}")

    payload = {
        "source": str(Path(args.source).expanduser()),
        "tracker": str(Path(args.tracker).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "weights": None if args.weights is None else str(Path(args.weights).expanduser()),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "use_temporal_detector": bool(args.use_temporal_detector),
        "save_video": None if save_video_path is None else str(save_video_path),
        "frames": records,
    }

    if save_json_path:
        save_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    args = parse_args()
    run_tracking(args)


if __name__ == "__main__":
    main()
