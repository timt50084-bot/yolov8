from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolo.obb.rgbir_temporal_train import RGBIRTemporalOBBTrainer


def parse_stage_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_loss_on(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit Stage 5 RGB-IR + small-object + temporal OBB training entry.")
    parser.add_argument("--data", required=True, type=str, help="Stage 1 prepared dataset yaml.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml"),
        type=str,
        help="Stage 5 temporal OBB model yaml. Can also point to the Stage 4 small yaml.",
    )
    parser.add_argument("--pretrained", default=None, type=str, help="Optional pretrained weights checkpoint.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--batch", default=2, type=int, help="Batch size.")
    parser.add_argument("--imgsz", default=640, type=int, help="Image size.")
    parser.add_argument("--device", default="cpu", type=str, help="Training device, e.g. cpu or 0.")
    parser.add_argument("--workers", default=0, type=int, help="Dataloader workers.")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "stage5_temporal"), type=str, help="Output project directory.")
    parser.add_argument("--name", default="train", type=str, help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing save dir.")
    parser.add_argument("--close-mosaic", default=0, type=int, help="Disable mosaic from the beginning for smoke tests.")
    parser.add_argument("--plots", action="store_true", help="Enable training/validation plots.")
    parser.add_argument("--val", action="store_true", help="Run validation during training.")

    parser.add_argument("--use-rgbir-train-assist", dest="use_rgbir_train_assist", action="store_true")
    parser.add_argument("--disable-rgbir-train-assist", dest="use_rgbir_train_assist", action="store_false")
    parser.set_defaults(use_rgbir_train_assist=True)
    parser.add_argument("--fusion-type", default="gated_add", type=str, help="IR fusion mode: gated_add|weighted_sum|align_only|none.")
    parser.add_argument("--ir-feature-stages", default="6,9", type=str, help="Comma-separated IR-assisted stage ids.")
    parser.add_argument("--ir-branch-width", default=0.25, type=float, help="Width multiplier for lightweight IR adapters.")
    parser.add_argument("--rgbir-aux-loss-weight", default=0.05, type=float, help="Weight for the RGB-IR auxiliary alignment loss.")
    parser.add_argument("--rgbir-residual-scale", default=0.1, type=float, help="Residual fusion scale for IR assistance.")

    parser.add_argument("--use-small-object-sampling", dest="use_small_object_sampling", action="store_true")
    parser.add_argument("--disable-small-object-sampling", dest="use_small_object_sampling", action="store_false")
    parser.set_defaults(use_small_object_sampling=True)
    parser.add_argument("--use-small-object-loss-weighting", dest="use_small_object_loss_weighting", action="store_true")
    parser.add_argument("--disable-small-object-loss-weighting", dest="use_small_object_loss_weighting", action="store_false")
    parser.set_defaults(use_small_object_loss_weighting=True)
    parser.add_argument("--enable-small-object-metrics", dest="enable_small_object_metrics", action="store_true")
    parser.add_argument("--disable-small-object-metrics", dest="enable_small_object_metrics", action="store_false")
    parser.set_defaults(enable_small_object_metrics=True)
    parser.add_argument("--small-object-area-thr-norm", default=0.005, type=float, help="Shared normalized OBB area threshold for Stage 4/5.")
    parser.add_argument("--small-object-sampling-power", default=1.0, type=float, help="Weight emphasis power for small-object sampling.")
    parser.add_argument("--small-object-sampling-min-weight", default=1.0, type=float, help="Minimum image sampling weight.")
    parser.add_argument("--small-object-sampling-max-weight", default=3.0, type=float, help="Maximum image sampling weight.")
    parser.add_argument("--small-object-loss-gain", default=0.25, type=float, help="Conservative gain for small-object loss weighting.")
    parser.add_argument("--small-object-loss-on", default="box,cls,dfl,angle", type=str, help="Comma-separated loss components to scale.")

    parser.add_argument("--use-temporal", dest="use_temporal", action="store_true")
    parser.add_argument("--disable-temporal", dest="use_temporal", action="store_false")
    parser.set_defaults(use_temporal=True)
    parser.add_argument("--temporal-mode", default="two_frame", type=str, help="Temporal mode. Stage 5 supports two_frame.")
    parser.add_argument("--temporal-fusion-type", default="diff_gate", type=str, help="Temporal refine mode: diff_gate|gated_add|align_only|none.")
    parser.add_argument("--temporal-feature-stages", default="6,9", type=str, help="Comma-separated temporal refine stage ids.")
    parser.add_argument("--temporal-branch-width", default=0.25, type=float, help="Width multiplier for previous-frame adapters.")
    parser.add_argument("--temporal-loss-weight", default=0.02, type=float, help="Low-weight temporal consistency loss.")
    parser.add_argument("--temporal-residual-scale", default=0.1, type=float, help="Residual scale for temporal refine.")
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
        "use_small_object_sampling": args.use_small_object_sampling,
        "small_object_area_thr_norm": args.small_object_area_thr_norm,
        "small_object_sampling_power": args.small_object_sampling_power,
        "small_object_sampling_min_weight": args.small_object_sampling_min_weight,
        "small_object_sampling_max_weight": args.small_object_sampling_max_weight,
        "use_small_object_loss_weighting": args.use_small_object_loss_weighting,
        "small_object_loss_gain": args.small_object_loss_gain,
        "small_object_loss_on": parse_loss_on(args.small_object_loss_on),
        "enable_small_object_metrics": args.enable_small_object_metrics,
        "use_temporal": args.use_temporal,
        "temporal_mode": args.temporal_mode,
        "temporal_fusion_type": args.temporal_fusion_type,
        "temporal_feature_stages": parse_stage_list(args.temporal_feature_stages),
        "temporal_branch_width": args.temporal_branch_width,
        "temporal_loss_weight": args.temporal_loss_weight,
        "temporal_residual_scale": args.temporal_residual_scale,
    }
    trainer = RGBIRTemporalOBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
