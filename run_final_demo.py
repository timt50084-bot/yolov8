from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.predict_uav_temporal_obb import IMAGE_EXTS, run_prediction
from tools.track_uav_obb import VIDEO_EXTS, run_tracking


# Centralized defaults for the final demo/test route.
DEFAULT_FINAL_MODEL = REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-small-temporal-obb.yaml"
DEFAULT_FINAL_WEIGHTS: Path | None = None
DEFAULT_TRACKER = REPO_ROOT / "ultralytics" / "cfg" / "trackers" / "uav_obb_tracker.yaml"
DEFAULT_IMGSZ = 1024
DEFAULT_DEVICE = "0"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5
DEFAULT_MAX_DET = 300
DEFAULT_SAVE_JSON = True
DEFAULT_SAVE_TXT = False
DEFAULT_LINE_WIDTH: int | None = None
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "final_demo"
AUTO_FINAL_TRAIN_ROOT = REPO_ROOT / "outputs" / "uav_pipeline" / "final" / "train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified final demo/test entry for image, folder, and video sources.")
    parser.add_argument("--source", default=None, type=str, help="Image file, image folder, or video file.")
    parser.add_argument("--weights", default=None, type=str, help="Optional checkpoint override for the final detector.")
    parser.add_argument("--model", default=None, type=str, help="Optional model yaml override. Defaults to final small-temporal yaml.")
    parser.add_argument("--tracker", default=None, type=str, help="Optional tracker yaml override for video mode.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, type=str, help="Inference device, for example 0 or cpu.")
    parser.add_argument("--imgsz", default=DEFAULT_IMGSZ, type=int, help="Inference image size.")
    parser.add_argument("--conf", default=DEFAULT_CONF, type=float, help="Confidence threshold.")
    parser.add_argument("--iou", default=DEFAULT_IOU, type=float, help="IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", default=DEFAULT_MAX_DET, type=int, help="Maximum detections per frame.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=str, help="Root directory for final demo outputs.")
    parser.add_argument("--name", default=None, type=str, help="Optional run name prefix.")
    parser.add_argument("--line-width", default=DEFAULT_LINE_WIDTH, type=int, help="Optional visualization line width override.")
    parser.add_argument("--mode", default="auto", choices=["auto", "image", "folder", "video"], help="Input mode override.")

    save_json_group = parser.add_mutually_exclusive_group()
    save_json_group.add_argument("--save-json", dest="save_json", action="store_true", help="Save structured JSON results.")
    save_json_group.add_argument("--no-save-json", dest="save_json", action="store_false", help="Disable structured JSON results.")

    save_txt_group = parser.add_mutually_exclusive_group()
    save_txt_group.add_argument("--save-txt", dest="save_txt", action="store_true", help="Save per-image txt labels.")
    save_txt_group.add_argument("--no-save-txt", dest="save_txt", action="store_false", help="Disable per-image txt labels.")

    parser.set_defaults(save_json=DEFAULT_SAVE_JSON, save_txt=DEFAULT_SAVE_TXT)
    return parser


def prompt_for_source_and_mode(current_mode: str) -> tuple[str, str]:
    if not sys.stdin.isatty():
        raise ValueError("--source is required when stdin is not interactive.")

    options = {
        "1": "image",
        "2": "folder",
        "3": "video",
        "4": "auto",
        "": current_mode if current_mode != "auto" else "auto",
    }
    print("No --source provided.")
    print("Select input mode: 1=image 2=folder 3=video 4=auto")
    selected = options.get(input("Mode [4]: ").strip(), "auto")
    source = input("Source path: ").strip()
    if not source:
        raise ValueError("Source path cannot be empty.")
    return source, selected


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def collect_folder_images(folder: Path) -> list[Path]:
    images = [p for p in sorted(folder.iterdir(), key=lambda item: item.name.lower()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError(f"No supported images found in folder: {folder}")
    return images


def detect_source_mode(source: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if source.is_dir():
        collect_folder_images(source)
        return "folder"
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    raise ValueError(
        "Unsupported source suffix: "
        f"{source.suffix}. Supported images: {sorted(IMAGE_EXTS)}. Supported videos: {sorted(VIDEO_EXTS)}."
    )


def resolve_source_and_mode(args: argparse.Namespace) -> tuple[Path, str]:
    source_text = args.source
    requested_mode = args.mode
    if not source_text:
        source_text, requested_mode = prompt_for_source_and_mode(requested_mode)
    source_path = Path(source_text).expanduser()
    actual_mode = detect_source_mode(source_path)
    if requested_mode != "auto" and requested_mode != actual_mode:
        raise ValueError(f"--mode {requested_mode!r} does not match source type {actual_mode!r} for {source_path}.")
    return source_path.resolve(), actual_mode


def resolve_model_path(raw_model: str | None) -> Path:
    model_path = Path(raw_model).expanduser() if raw_model else DEFAULT_FINAL_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Model yaml not found: {model_path}")
    return model_path.resolve()


def resolve_tracker_path(raw_tracker: str | None) -> Path:
    tracker_path = Path(raw_tracker).expanduser() if raw_tracker else DEFAULT_TRACKER
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker yaml not found: {tracker_path}")
    return tracker_path.resolve()


def latest_checkpoint(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.glob(f"**/weights/{filename}") if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_weights_path(raw_weights: str | None) -> Path:
    if raw_weights:
        weights_path = Path(raw_weights).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        return weights_path.resolve()

    if DEFAULT_FINAL_WEIGHTS is not None:
        if not DEFAULT_FINAL_WEIGHTS.exists():
            raise FileNotFoundError(f"Configured default weights not found: {DEFAULT_FINAL_WEIGHTS}")
        return DEFAULT_FINAL_WEIGHTS.resolve()

    best_ckpt = latest_checkpoint(AUTO_FINAL_TRAIN_ROOT, "best.pt")
    if best_ckpt is not None:
        return best_ckpt.resolve()

    last_ckpt = latest_checkpoint(AUTO_FINAL_TRAIN_ROOT, "last.pt")
    if last_ckpt is not None:
        return last_ckpt.resolve()

    raise FileNotFoundError(
        "No final weights were found.\n"
        f"Expected one under: {AUTO_FINAL_TRAIN_ROOT / '<run>' / 'weights' / 'best.pt'}\n"
        f"or under: {AUTO_FINAL_TRAIN_ROOT / '<run>' / 'weights' / 'last.pt'}\n"
        "You can also pass a checkpoint explicitly, for example:\n"
        "python run_final_demo.py --weights path/to/best.pt --source path/to/image.jpg"
    )


def make_run_dir(output_root: Path, mode: str, source_path: Path, name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = sanitize_name(name or (source_path.stem if source_path.is_file() else source_path.name))
    run_dir = output_root / mode / f"{base_name}_{timestamp}"
    index = 2
    while run_dir.exists():
        run_dir = output_root / mode / f"{base_name}_{timestamp}_{index}"
        index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_predict_args(
    *,
    source_path: Path,
    model_path: Path,
    weights_path: Path,
    run_dir: Path,
    sequence: bool,
    args: argparse.Namespace,
) -> argparse.Namespace:
    json_path = run_dir / "predictions.json" if args.save_json else None
    return argparse.Namespace(
        source=str(source_path),
        model=str(model_path),
        weights=str(weights_path),
        data=None,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        sequence=sequence,
        max_frames=0,
        save_dir=str(run_dir),
        save_json=None if json_path is None else str(json_path),
        save_vis=True,
        save_txt=bool(args.save_txt),
        line_width=args.line_width,
        use_temporal=True,
        temporal_mode="two_frame",
        temporal_fusion_type="diff_gate",
        temporal_feature_stages="6,9",
        temporal_branch_width=0.25,
        temporal_loss_weight=0.02,
        temporal_residual_scale=0.1,
    )


def build_track_args(
    *,
    source_path: Path,
    model_path: Path,
    weights_path: Path,
    tracker_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    json_path = run_dir / "tracking_results.json" if args.save_json else None
    video_path = run_dir / f"{source_path.stem}_tracked.mp4"
    return argparse.Namespace(
        source=str(source_path),
        model=str(model_path),
        weights=str(weights_path),
        data=None,
        tracker=str(tracker_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        max_frames=0,
        save_json=None if json_path is None else str(json_path),
        save_video=str(video_path),
        line_width=args.line_width,
        track_low_thresh=-1.0,
        new_track_thresh=-1.0,
        match_thresh=-1.0,
        match_iou_thresh=-1.0,
        use_appearance=None,
        use_temporal_detector=True,
        temporal_mode="two_frame",
        temporal_fusion_type="diff_gate",
        temporal_feature_stages="6,9",
        temporal_branch_width=0.25,
        temporal_loss_weight=0.02,
        temporal_residual_scale=0.1,
    )


def print_route_banner(kind: str, source_path: Path, model_path: Path, weights_path: Path, run_dir: Path) -> None:
    print(f"Mode: {kind}")
    print(f"Source: {source_path}")
    print(f"Model: {model_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {run_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_path, resolved_mode = resolve_source_and_mode(args)
    model_path = resolve_model_path(args.model)
    weights_path = resolve_weights_path(args.weights)
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = make_run_dir(output_root, resolved_mode, source_path, args.name)

    if resolved_mode == "image":
        print_route_banner("image detect", source_path, model_path, weights_path, run_dir)
        predict_args = build_predict_args(
            source_path=source_path,
            model_path=model_path,
            weights_path=weights_path,
            run_dir=run_dir,
            sequence=False,
            args=args,
        )
        run_prediction(predict_args)
        print(f"Saved visualizations: {run_dir / 'visualizations'}")
        if args.save_json:
            print(f"Saved JSON: {run_dir / 'predictions.json'}")
        if args.save_txt:
            print(f"Saved TXT labels: {run_dir / 'labels'}")
        return

    if resolved_mode == "folder":
        image_list = collect_folder_images(source_path)
        use_sequence = len(image_list) > 1
        kind = "folder temporal detect" if use_sequence else "folder detect"
        print_route_banner(kind, source_path, model_path, weights_path, run_dir)
        predict_args = build_predict_args(
            source_path=source_path,
            model_path=model_path,
            weights_path=weights_path,
            run_dir=run_dir,
            sequence=use_sequence,
            args=args,
        )
        run_prediction(predict_args)
        print(f"Saved visualizations: {run_dir / 'visualizations'}")
        if args.save_json:
            print(f"Saved JSON: {run_dir / 'predictions.json'}")
        if args.save_txt:
            print(f"Saved TXT labels: {run_dir / 'labels'}")
        return

    tracker_path = resolve_tracker_path(args.tracker)
    if args.save_txt:
        print("Note: --save-txt is ignored for video tracking.")
    print_route_banner("video track", source_path, model_path, weights_path, run_dir)
    track_args = build_track_args(
        source_path=source_path,
        model_path=model_path,
        weights_path=weights_path,
        tracker_path=tracker_path,
        run_dir=run_dir,
        args=args,
    )
    run_tracking(track_args)
    print(f"Saved video: {run_dir / f'{source_path.stem}_tracked.mp4'}")
    if args.save_json:
        print(f"Saved JSON: {run_dir / 'tracking_results.json'}")


if __name__ == "__main__":
    main()
