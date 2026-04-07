from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolo.obb.rgbir_temporal_obb_train import RGBIRTemporalOBBModel
from ultralytics.nn.tasks import torch_safe_load, yaml_model_load


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    parser.add_argument("--sequence", action="store_true", help="Enable one-step previous-frame cache across frames.")
    parser.add_argument("--max-frames", default=0, type=int, help="Optional max frames to process from a directory.")
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


def load_data_cfg(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset yaml at {path} to load into a mapping.")
    return data


def build_model(args: argparse.Namespace) -> RGBIRTemporalOBBModel:
    model_cfg = yaml_model_load(args.model)
    model_cfg["use_temporal"] = args.use_temporal
    model_cfg["temporal_mode"] = args.temporal_mode
    model_cfg["temporal_fusion_type"] = args.temporal_fusion_type
    model_cfg["temporal_feature_stages"] = parse_stage_list(args.temporal_feature_stages)
    model_cfg["temporal_branch_width"] = args.temporal_branch_width
    model_cfg["temporal_loss_weight"] = args.temporal_loss_weight
    model_cfg["temporal_residual_scale"] = args.temporal_residual_scale
    data_cfg = load_data_cfg(args.data)
    nc = data_cfg.get("nc")
    model = RGBIRTemporalOBBModel(model_cfg, nc=nc, ch=3, verbose=False)
    if args.weights:
        ckpt, _ = torch_safe_load(args.weights)
        model.load(ckpt, verbose=False)
    model = model.to(torch.device(normalize_device_arg(args.device))).eval()
    return model


def iter_frames(source: str, max_frames: int = 0) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    frames = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return frames[:max_frames] if max_frames > 0 else frames


def load_rgb_tensor(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous().float() / 255.0
    return tensor.to(device)


def summarize_output(output) -> str:
    if isinstance(output, dict):
        keys = ",".join(sorted(output.keys()))
        return f"dict[{keys}]"
    if isinstance(output, (list, tuple)):
        return f"{type(output).__name__}(len={len(output)})"
    if isinstance(output, torch.Tensor):
        return f"tensor{tuple(output.shape)}"
    return type(output).__name__


def main() -> None:
    args = parse_args()
    frames = iter_frames(args.source, max_frames=args.max_frames)
    if not frames:
        raise ValueError(f"No image frames found in source: {args.source}")

    device = torch.device(normalize_device_arg(args.device))
    model = build_model(args)
    model.reset_temporal_cache()

    with torch.no_grad():
        for i, frame_path in enumerate(frames):
            img = load_rgb_tensor(frame_path, imgsz=args.imgsz, device=device)
            if args.sequence:
                output = model.predict_with_prev_cache(img)
                temporal_valid = i > 0
            else:
                output = model.predict(img, img_prev=None, temporal_valid=None)
                temporal_valid = False
            print(
                f"frame={i} path={frame_path.name} sequence={args.sequence} "
                f"temporal_valid_input={temporal_valid} temporal_used={model.last_temporal_used} "
                f"temporal_stages={list(model.last_temporal_stage_ids)} output={summarize_output(output)}"
            )


if __name__ == "__main__":
    main()
