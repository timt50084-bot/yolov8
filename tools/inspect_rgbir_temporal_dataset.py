from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.data.build import build_rgbir_temporal_obb_dataset
from ultralytics.data.rgbir_temporal_obb_dataset import RGBIRTemporalOBBDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the opt-in RGBIRTemporal OBB dataset from Stage 1 outputs.")
    parser.add_argument("--data", type=str, default=None, help="Path to the Stage 1 data yaml.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the Stage 1 prepared dataset root.")
    parser.add_argument("--mode", type=str, default="train", choices=("train", "val"), help="Dataset split to inspect.")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size for synchronized letterbox.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dataloader smoke.")
    parser.add_argument("--inspect-samples", type=int, default=3, help="Number of individual samples to inspect.")
    parser.add_argument("--augment", action="store_true", help="Enable conservative synchronized flip augmentation.")
    parser.add_argument(
        "--use-builder",
        action="store_true",
        help="Instantiate through ultralytics.data.build.build_rgbir_temporal_obb_dataset.",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save a JSON summary.")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> RGBIRTemporalOBBDataset:
    kwargs = {"data": args.data, "data_root": args.data_root, "mode": args.mode, "imgsz": args.imgsz, "augment": args.augment}
    if args.use_builder:
        return build_rgbir_temporal_obb_dataset(**kwargs)
    return RGBIRTemporalOBBDataset(**kwargs)


def summarize_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "sample_id": sample["sample_id"],
        "frame_id": sample["frame_id"],
        "seq_id": sample["seq_id"],
        "temporal_valid": bool(sample["temporal_valid"].item()),
        "img_shape": list(sample["img"].shape),
        "img_ir_shape": list(sample["img_ir"].shape),
        "img_prev_shape": list(sample["img_prev"].shape),
        "cls_shape": list(sample["cls"].shape),
        "bboxes_shape": list(sample["bboxes"].shape),
        "segments_shape": list(sample["segments"].shape),
        "im_file": sample["im_file"],
        "im_file_ir": sample["im_file_ir"],
        "im_file_prev": sample["im_file_prev"],
        "label_file": sample["label_file"],
    }


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "img_shape": list(batch["img"].shape),
        "img_ir_shape": list(batch["img_ir"].shape),
        "img_prev_shape": list(batch["img_prev"].shape),
        "cls_shape": list(batch["cls"].shape),
        "bboxes_shape": list(batch["bboxes"].shape),
        "segments_shape": list(batch["segments"].shape),
        "batch_idx_shape": list(batch["batch_idx"].shape),
        "temporal_valid_shape": list(batch["temporal_valid"].shape),
        "temporal_valid": [bool(x) for x in batch["temporal_valid"].tolist()],
        "sample_id": batch["sample_id"],
        "frame_id": batch["frame_id"],
        "seq_id": batch["seq_id"],
    }


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    inspect_count = min(args.inspect_samples, len(dataset))
    sample_summaries = [summarize_sample(i, dataset[i]) for i in range(inspect_count)]

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batch_summary = summarize_batch(next(iter(loader)))
    summary = {
        "dataset_type": type(dataset).__name__,
        "mode": args.mode,
        "length": len(dataset),
        "names": dataset.names,
        "samples": sample_summaries,
        "batch": batch_summary,
        "status": "ok",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
