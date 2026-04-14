from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_uav_augment_args import add_train_augment_args, augment_overrides_from_args
from ultralytics.models.yolo.obb.rgbir_train import RGBIROBBTrainer


def parse_stage_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit Stage 3 RGB-IR train-assist OBB training entry.")
    parser.add_argument("--data", required=True, type=str, help="Stage 1 prepared dataset yaml.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb.yaml"),
        type=str,
        help="Stage 3 RGB-IR OBB model yaml.",
    )
    parser.add_argument("--pretrained", default=None, type=str, help="Optional pretrained weights checkpoint.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--batch", default=2, type=int, help="Batch size.")
    parser.add_argument("--imgsz", default=640, type=int, help="Image size.")
    parser.add_argument("--device", default="cpu", type=str, help="Training device, e.g. cpu or 0.")
    parser.add_argument("--workers", default=0, type=int, help="Dataloader workers.")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "stage3_rgbir"), type=str, help="Output project directory.")
    parser.add_argument("--name", default="train", type=str, help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing save dir.")
    parser.add_argument("--close-mosaic", default=0, type=int, help="Disable mosaic from the beginning for small smoke tests.")
    parser.add_argument("--plots", action="store_true", help="Enable training/validation plots.")
    parser.add_argument("--val", action="store_true", help="Run validation during training.")
    parser.add_argument("--use-rgbir-train-assist", dest="use_rgbir_train_assist", action="store_true")
    parser.add_argument("--disable-rgbir-train-assist", dest="use_rgbir_train_assist", action="store_false")
    parser.set_defaults(use_rgbir_train_assist=True)
    parser.add_argument("--fusion-type", default="gated_add", type=str, help="Fusion mode: gated_add|weighted_sum|align_only|none.")
    parser.add_argument("--ir-feature-stages", default="6,9", type=str, help="Comma-separated assisted stage ids.")
    parser.add_argument("--ir-branch-width", default=0.25, type=float, help="Width multiplier for the lightweight IR adapters.")
    parser.add_argument("--rgbir-aux-loss-weight", default=0.05, type=float, help="Weight for the RGB-IR auxiliary alignment loss.")
    parser.add_argument("--rgbir-residual-scale", default=0.1, type=float, help="Residual fusion scale for IR assistance.")
    add_train_augment_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "model": args.model,
        "data": args.data,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "close_mosaic": args.close_mosaic,
        "plots": args.plots,
        "val": args.val,
        "use_rgbir_train_assist": args.use_rgbir_train_assist,
        "fusion_type": args.fusion_type,
        "ir_feature_stages": parse_stage_list(args.ir_feature_stages),
        "ir_branch_width": args.ir_branch_width,
        "rgbir_aux_loss_weight": args.rgbir_aux_loss_weight,
        "rgbir_residual_scale": args.rgbir_residual_scale,
    }
    overrides.update(augment_overrides_from_args(args))
    trainer = RGBIROBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
