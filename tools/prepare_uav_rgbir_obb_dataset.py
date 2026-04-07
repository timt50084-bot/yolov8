from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils_preprocess_uav import (
    ClassMapper,
    IMAGE_SUFFIXES,
    LABEL_SUFFIXES,
    OBBObject,
    RawPair,
    apply_target_protection,
    clamp_polygon,
    clip_polygon_to_box,
    copy_text_file,
    detect_valid_bbox,
    ensure_dir,
    find_label_path,
    fuse_label_sets,
    load_alias_map,
    load_image,
    natural_sort_key,
    parse_label_file,
    polygon_area,
    relative_posix,
    save_image,
    scan_files_by_key,
    stable_rectangle,
    union_box,
    write_json,
    write_label_file,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired RGB/IR UAV OBB data into a current-Ultralytics-compatible training asset."
    )
    parser.add_argument("--input-root", required=True, type=Path, help="Root of the raw dataset.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for processed assets.")
    parser.add_argument(
        "--split-mode",
        choices=("auto", "existing", "random"),
        default="auto",
        help="Use existing train/val folders or split one unsplit source root.",
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "legacy_prepared", "dronevehicle_raw"),
        default="auto",
        help="Input directory layout. 'auto' tries the legacy prepared layout first, then DroneVehicle raw layout.",
    )
    parser.add_argument("--train-split-name", default="train", help="Split folder name for train data.")
    parser.add_argument("--val-split-name", default="val", help="Split folder name for val data.")
    parser.add_argument("--rgb-subdir", default="images/img", help="RGB image subdirectory inside each split root.")
    parser.add_argument("--ir-subdir", default="images/imgr", help="IR image subdirectory inside each split root.")
    parser.add_argument(
        "--label-subdir",
        default="labels/merged",
        help="Shared label subdirectory. Used when modality-specific label folders are absent.",
    )
    parser.add_argument("--rgb-label-subdir", default=None, help="Optional RGB-only label subdirectory.")
    parser.add_argument("--ir-label-subdir", default=None, help="Optional IR-only label subdirectory.")
    parser.add_argument(
        "--rgb-key-remove",
        action="append",
        default=[],
        help="Tokens removed from RGB file stems before pairing. Repeat as needed.",
    )
    parser.add_argument(
        "--ir-key-remove",
        action="append",
        default=[],
        help="Tokens removed from IR file stems before pairing. Repeat as needed.",
    )
    parser.add_argument(
        "--label-key-remove",
        action="append",
        default=[],
        help="Tokens removed from label file stems before pairing. Repeat as needed.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when split-mode=random.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split-mode=random.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional class names list. Numeric labels use these names when available.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="JSON/YAML class alias mapping. Supports alias->canonical and canonical->aliases.",
    )
    parser.add_argument(
        "--angle-unit",
        choices=("auto", "rad", "deg"),
        default="auto",
        help="Interpretation of xywha label angles.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "fixed"),
        default="auto",
        help="Crop mode. 'auto' keeps the current white-border detection flow, 'fixed' applies explicit margins.",
    )
    parser.add_argument("--crop-left", type=int, default=100, help="Left crop margin used when --crop-mode=fixed.")
    parser.add_argument("--crop-top", type=int, default=100, help="Top crop margin used when --crop-mode=fixed.")
    parser.add_argument("--crop-right", type=int, default=100, help="Right crop margin used when --crop-mode=fixed.")
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=100,
        help="Bottom crop margin used when --crop-mode=fixed.",
    )
    parser.add_argument("--white-thresh", type=int, default=245, help="Pixels >= threshold are treated as white.")
    parser.add_argument(
        "--protect-size",
        type=int,
        default=100,
        help="Target protection window size for crop expansion near crop edges.",
    )
    parser.add_argument(
        "--fusion-iou-thresh",
        type=float,
        default=0.6,
        help="IoU threshold for RGB/IR label fusion when both modalities have labels.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="Minimum absolute polygon area kept after crop and normalization.",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Keep negative/background samples even when both label sources are missing or empty.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root before writing new results.",
    )
    return parser.parse_args()


def add_anomaly(
    anomalies: list[dict[str, Any]],
    split: str,
    key: str,
    anomaly_type: str,
    detail: str,
    **extra: Any,
) -> None:
    anomalies.append({"split": split, "id": key, "type": anomaly_type, "detail": detail, **extra})


def select_one(indexed: dict[str, list[Path]], key: str) -> Path | None:
    paths = indexed.get(key, [])
    if not paths:
        return None
    return sorted(paths, key=lambda item: item.name.lower())[0]


@dataclass(frozen=True)
class SplitSourceDirs:
    split: str
    root: Path
    rgb_dir: Path
    ir_dir: Path
    shared_label_dir: Path | None = None
    rgb_label_dir: Path | None = None
    ir_label_dir: Path | None = None


def build_split_source_dirs(
    split: str,
    root: Path,
    rgb_subdir: str,
    ir_subdir: str,
    label_subdir: str | None,
    rgb_label_subdir: str | None,
    ir_label_subdir: str | None,
) -> SplitSourceDirs:
    return SplitSourceDirs(
        split=split,
        root=root,
        rgb_dir=root / rgb_subdir,
        ir_dir=root / ir_subdir,
        shared_label_dir=root / label_subdir if label_subdir else None,
        rgb_label_dir=root / rgb_label_subdir if rgb_label_subdir else None,
        ir_label_dir=root / ir_label_subdir if ir_label_subdir else None,
    )


def build_legacy_split_dirs(split: str, root: Path, args: argparse.Namespace) -> SplitSourceDirs:
    return build_split_source_dirs(
        split=split,
        root=root,
        rgb_subdir=args.rgb_subdir,
        ir_subdir=args.ir_subdir,
        label_subdir=args.label_subdir,
        rgb_label_subdir=args.rgb_label_subdir,
        ir_label_subdir=args.ir_label_subdir,
    )


def build_dronevehicle_split_dirs(split: str, root: Path) -> SplitSourceDirs:
    split_prefix = root.name
    return build_split_source_dirs(
        split=split,
        root=root,
        rgb_subdir=f"{split_prefix}img",
        ir_subdir=f"{split_prefix}imgr",
        label_subdir=None,
        rgb_label_subdir=f"{split_prefix}label",
        ir_label_subdir=f"{split_prefix}labelr",
    )


def validate_split_source_dirs(source_dirs: SplitSourceDirs) -> list[str]:
    missing: list[str] = []
    if not source_dirs.root.is_dir():
        missing.append(f"{source_dirs.split} root: {source_dirs.root}")
    if not source_dirs.rgb_dir.is_dir():
        missing.append(f"{source_dirs.split} RGB dir: {source_dirs.rgb_dir}")
    if not source_dirs.ir_dir.is_dir():
        missing.append(f"{source_dirs.split} IR dir: {source_dirs.ir_dir}")
    return missing


def format_layout_missing(layout_name: str, split_dirs: dict[str, SplitSourceDirs]) -> str:
    missing: list[str] = []
    for source_dirs in split_dirs.values():
        missing.extend(validate_split_source_dirs(source_dirs))
    if not missing:
        return f"{layout_name}: OK"
    return f"{layout_name}: missing " + "; ".join(missing)


def resolve_existing_layout(args: argparse.Namespace) -> tuple[str, dict[str, SplitSourceDirs]]:
    split_roots = {
        "train": args.input_root / args.train_split_name,
        "val": args.input_root / args.val_split_name,
    }
    candidates = {
        "legacy_prepared": {
            split: build_legacy_split_dirs(split, root, args) for split, root in split_roots.items()
        },
        "dronevehicle_raw": {
            split: build_dronevehicle_split_dirs(split, root) for split, root in split_roots.items()
        },
    }

    layout_order = ["legacy_prepared", "dronevehicle_raw"]
    if args.input_layout == "auto":
        candidate_names = layout_order
    else:
        candidate_names = [args.input_layout]

    for layout_name in candidate_names:
        split_dirs = candidates[layout_name]
        missing: list[str] = []
        for source_dirs in split_dirs.values():
            missing.extend(validate_split_source_dirs(source_dirs))
        if not missing:
            return layout_name, split_dirs

    if args.input_layout == "auto":
        detail = " | ".join(format_layout_missing(name, candidates[name]) for name in layout_order)
        raise FileNotFoundError(
            f"Could not detect a supported input layout under '{args.input_root}'. {detail}"
        )

    detail = format_layout_missing(args.input_layout, candidates[args.input_layout])
    raise FileNotFoundError(
        f"input-layout={args.input_layout} does not match '{args.input_root}'. {detail}"
    )


def resolve_unsplit_source_dirs(args: argparse.Namespace) -> SplitSourceDirs:
    if args.input_layout == "dronevehicle_raw":
        raise ValueError(
            "input-layout=dronevehicle_raw is only supported with split-mode=existing. "
            "Use the dataset root that already contains train/val folders."
        )

    source_dirs = build_legacy_split_dirs("unsplit", args.input_root, args)
    missing = validate_split_source_dirs(source_dirs)
    if missing:
        raise FileNotFoundError(
            f"Could not resolve unsplit input directories under '{args.input_root}'. "
            + "; ".join(missing)
        )
    return source_dirs


def collect_pairs_for_root(
    split: str,
    source_dirs: SplitSourceDirs,
    args: argparse.Namespace,
    anomalies: list[dict[str, Any]],
) -> list[RawPair]:
    rgb_dir = source_dirs.rgb_dir
    ir_dir = source_dirs.ir_dir
    shared_label_dir = source_dirs.shared_label_dir
    rgb_label_dir = source_dirs.rgb_label_dir
    ir_label_dir = source_dirs.ir_label_dir

    rgb_index = scan_files_by_key(rgb_dir, IMAGE_SUFFIXES, args.rgb_key_remove)
    ir_index = scan_files_by_key(ir_dir, IMAGE_SUFFIXES, args.ir_key_remove)
    shared_label_index = (
        scan_files_by_key(shared_label_dir, LABEL_SUFFIXES, args.label_key_remove)
        if shared_label_dir is not None
        else {}
    )
    rgb_label_index = (
        scan_files_by_key(rgb_label_dir, LABEL_SUFFIXES, args.label_key_remove)
        if rgb_label_dir is not None
        else {}
    )
    ir_label_index = (
        scan_files_by_key(ir_label_dir, LABEL_SUFFIXES, args.label_key_remove)
        if ir_label_dir is not None
        else {}
    )

    for key, paths in rgb_index.items():
        if len(paths) > 1:
            add_anomaly(anomalies, split, key, "duplicate_rgb", f"Multiple RGB files resolved to key '{key}'.", paths=[str(p) for p in paths])
    for key, paths in ir_index.items():
        if len(paths) > 1:
            add_anomaly(anomalies, split, key, "duplicate_ir", f"Multiple IR files resolved to key '{key}'.", paths=[str(p) for p in paths])
    for key, paths in shared_label_index.items():
        if len(paths) > 1:
            add_anomaly(
                anomalies,
                split,
                key,
                "duplicate_shared_label",
                f"Multiple shared labels resolved to key '{key}'.",
                paths=[str(p) for p in paths],
            )

    pairs: list[RawPair] = []
    all_keys = sorted(set(rgb_index) | set(ir_index), key=natural_sort_key)
    for key in all_keys:
        rgb_image = select_one(rgb_index, key)
        ir_image = select_one(ir_index, key)
        if rgb_image is None or ir_image is None:
            add_anomaly(
                anomalies,
                split,
                key,
                "missing_pair",
                "RGB/IR pair is incomplete.",
                rgb_image=str(rgb_image) if rgb_image else None,
                ir_image=str(ir_image) if ir_image else None,
            )
            continue
        shared_label = find_label_path(shared_label_index, key)
        rgb_label = find_label_path(rgb_label_index, key) or shared_label
        ir_label = find_label_path(ir_label_index, key) or shared_label
        pairs.append(RawPair(split=split, key=key, rgb_image=rgb_image, ir_image=ir_image, rgb_label=rgb_label, ir_label=ir_label))
    return pairs


def split_unsplit_pairs(
    pairs: list[RawPair], val_ratio: float, seed: int
) -> dict[str, list[RawPair]]:
    import random

    ordered = sorted(pairs, key=lambda pair: natural_sort_key(pair.key))
    random.Random(seed).shuffle(ordered)
    val_count = int(round(len(ordered) * val_ratio))
    val_ids = {pair.key for pair in ordered[:val_count]}
    split_pairs = {"train": [], "val": []}
    for pair in sorted(pairs, key=lambda item: natural_sort_key(item.key)):
        split_pairs["val" if pair.key in val_ids else "train"].append(pair)
    return split_pairs


def resolve_split_pairs(args: argparse.Namespace, anomalies: list[dict[str, Any]]) -> dict[str, list[RawPair]]:
    train_root = args.input_root / args.train_split_name
    val_root = args.input_root / args.val_split_name
    if args.split_mode == "auto":
        split_mode = "existing" if train_root.exists() and val_root.exists() else "random"
    else:
        split_mode = args.split_mode
    if split_mode == "existing":
        if not train_root.exists() or not val_root.exists():
            raise FileNotFoundError(
                f"split-mode=existing requires '{train_root}' and '{val_root}' to exist."
            )
        _, split_sources = resolve_existing_layout(args)
        return {
            "train": collect_pairs_for_root("train", split_sources["train"], args, anomalies),
            "val": collect_pairs_for_root("val", split_sources["val"], args, anomalies),
        }
    unsplit_pairs = collect_pairs_for_root("unsplit", resolve_unsplit_source_dirs(args), args, anomalies)
    return split_unsplit_pairs(unsplit_pairs, args.val_ratio, args.seed)


def compute_crop_box(
    rgb_image: Image.Image,
    ir_image: Image.Image,
    objects: list[OBBObject],
    crop_mode: str,
    white_thresh: int,
    protect_size: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> tuple[int, int, int, int]:
    if crop_mode == "fixed":
        return compute_fixed_crop_box(
            image_width=rgb_image.width,
            image_height=rgb_image.height,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )
    crop_box = union_box(detect_valid_bbox(rgb_image, white_thresh), detect_valid_bbox(ir_image, white_thresh))
    return apply_target_protection(crop_box, objects, rgb_image.width, rgb_image.height, protect_size)


def compute_fixed_crop_box(
    image_width: int,
    image_height: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> tuple[int, int, int, int]:
    if crop_left < 0 or crop_top < 0 or crop_right < 0 or crop_bottom < 0:
        raise ValueError(
            "Fixed crop margins must be >= 0, got "
            f"left={crop_left}, top={crop_top}, right={crop_right}, bottom={crop_bottom}."
        )
    if crop_left + crop_right >= image_width:
        raise ValueError(
            f"Fixed crop is invalid for image width {image_width}: "
            f"crop_left + crop_right must be < image_width, got {crop_left} + {crop_right}."
        )
    if crop_top + crop_bottom >= image_height:
        raise ValueError(
            f"Fixed crop is invalid for image height {image_height}: "
            f"crop_top + crop_bottom must be < image_height, got {crop_top} + {crop_bottom}."
        )

    x0 = crop_left
    y0 = crop_top
    x1 = image_width - crop_right
    y1 = image_height - crop_bottom
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            f"Fixed crop produced non-positive output size {crop_width}x{crop_height} "
            f"from image size {image_width}x{image_height}."
        )
    return x0, y0, x1, y1


def infer_single_size(size_counter: Counter[tuple[int, int]]) -> list[int] | None:
    if len(size_counter) != 1:
        return None
    width, height = next(iter(size_counter))
    return [width, height]


def crop_objects(
    objects: list[OBBObject],
    crop_box: tuple[int, int, int, int],
    output_width: int,
    output_height: int,
    min_area: float,
) -> tuple[list[OBBObject], list[str]]:
    events: list[str] = []
    x0, y0, _, _ = crop_box
    kept: list[OBBObject] = []
    for obj in objects:
        clipped = clip_polygon_to_box(obj.polygon, crop_box)
        if len(clipped) == 0:
            events.append(f"dropped:{obj.class_name}:{obj.source}:outside_crop")
            continue
        polygon = stable_rectangle(clipped)
        polygon[:, 0] -= x0
        polygon[:, 1] -= y0
        polygon = clamp_polygon(polygon, output_width, output_height)
        if polygon_area(polygon) < min_area:
            events.append(f"dropped:{obj.class_name}:{obj.source}:degenerate_after_crop")
            continue
        kept.append(
            OBBObject(
                class_id=obj.class_id,
                class_name=obj.class_name,
                polygon=polygon,
                source=obj.source,
                raw_class=obj.raw_class,
                meta=obj.meta,
            )
        )
    return kept, events


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "rgb_train": output_root / "images" / "rgb" / "train",
        "rgb_val": output_root / "images" / "rgb" / "val",
        "ir_train": output_root / "images" / "ir" / "train",
        "ir_val": output_root / "images" / "ir" / "val",
        "obb_train": output_root / "labels" / "obb" / "train",
        "obb_val": output_root / "labels" / "obb" / "val",
        "rgb_label_train": output_root / "labels" / "rgb" / "train",
        "rgb_label_val": output_root / "labels" / "rgb" / "val",
        "index": output_root / "index",
        "reports": output_root / "reports",
        "data": output_root / "data",
    }
    for path in paths.values():
        ensure_dir(path)
    return paths


def create_data_yaml(output_root: Path, class_names: list[str]) -> Path:
    data_yaml_path = output_root / "data" / "uav_rgb_obb.yaml"
    write_yaml(
        data_yaml_path,
        {
            "path": str(output_root.resolve()),
            "train": "images/rgb/train",
            "val": "images/rgb/val",
            "names": class_names,
            "nc": len(class_names),
            "channels": 3,
            "ir_train": "images/ir/train",
            "ir_val": "images/ir/val",
            "labels_train_compat": "labels/rgb/train",
            "labels_val_compat": "labels/rgb/val",
            "labels_obb_canonical": "labels/obb",
            "pair_index_train": "index/train_pairs.json",
            "pair_index_val": "index/val_pairs.json",
            "temporal_index_train": "index/train_temporal.json",
            "temporal_index_val": "index/val_temporal.json",
        },
    )
    return data_yaml_path


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output root '{args.output_root}' is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)

    alias_map = load_alias_map(args.class_map)
    class_mapper = ClassMapper(names=args.names, alias_map=alias_map)
    anomalies: list[dict[str, Any]] = []
    split_pairs = resolve_split_pairs(args, anomalies)
    output_dirs = prepare_output_dirs(args.output_root)

    pair_indexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_summaries: dict[str, Counter[str]] = defaultdict(Counter)
    observed_input_sizes: Counter[tuple[int, int]] = Counter()
    observed_output_sizes: Counter[tuple[int, int]] = Counter()

    for split, pairs in split_pairs.items():
        rgb_output_dir = output_dirs[f"rgb_{split}"]
        ir_output_dir = output_dirs[f"ir_{split}"]
        obb_output_dir = output_dirs[f"obb_{split}"]
        rgb_label_output_dir = output_dirs[f"rgb_label_{split}"]

        for pair in sorted(pairs, key=lambda item: natural_sort_key(item.key)):
            split_summaries[split]["discovered_pairs"] += 1
            try:
                rgb_image = load_image(pair.rgb_image)
                ir_image = load_image(pair.ir_image)
            except Exception as error:
                add_anomaly(anomalies, split, pair.key, "image_read_error", str(error), rgb_image=str(pair.rgb_image), ir_image=str(pair.ir_image))
                split_summaries[split]["skipped_pairs"] += 1
                continue

            if rgb_image.size != ir_image.size:
                add_anomaly(
                    anomalies,
                    split,
                    pair.key,
                    "size_mismatch",
                    f"RGB/IR sizes differ: {rgb_image.size} vs {ir_image.size}",
                    rgb_image=str(pair.rgb_image),
                    ir_image=str(pair.ir_image),
                )
                split_summaries[split]["skipped_pairs"] += 1
                continue

            image_width, image_height = rgb_image.size
            rgb_objects, rgb_issues = parse_label_file(
                pair.rgb_label, image_width, image_height, class_mapper, args.angle_unit, "rgb"
            )
            ir_objects, ir_issues = parse_label_file(
                pair.ir_label, image_width, image_height, class_mapper, args.angle_unit, "ir"
            )
            for issue in rgb_issues:
                add_anomaly(anomalies, split, pair.key, "rgb_label_issue", issue, label=str(pair.rgb_label) if pair.rgb_label else None)
            for issue in ir_issues:
                add_anomaly(anomalies, split, pair.key, "ir_label_issue", issue, label=str(pair.ir_label) if pair.ir_label else None)

            if pair.rgb_label is None and pair.ir_label is None:
                add_anomaly(
                    anomalies,
                    split,
                    pair.key,
                    "missing_labels",
                    "Both RGB and IR labels are missing.",
                    rgb_label=None,
                    ir_label=None,
                )

            combined_objects = rgb_objects + ir_objects
            try:
                crop_box = compute_crop_box(
                    rgb_image,
                    ir_image,
                    combined_objects,
                    args.crop_mode,
                    args.white_thresh,
                    args.protect_size,
                    args.crop_left,
                    args.crop_top,
                    args.crop_right,
                    args.crop_bottom,
                )
            except ValueError as error:
                raise ValueError(
                    f"Failed to compute crop box for pair '{pair.key}' with image size "
                    f"{image_width}x{image_height}: {error}"
                ) from error
            rgb_cropped = rgb_image.crop(crop_box)
            ir_cropped = ir_image.crop(crop_box)
            output_width, output_height = rgb_cropped.size
            observed_input_sizes[(image_width, image_height)] += 1
            observed_output_sizes[(output_width, output_height)] += 1

            rgb_objects, rgb_crop_events = crop_objects(rgb_objects, crop_box, output_width, output_height, args.min_area)
            ir_objects, ir_crop_events = crop_objects(ir_objects, crop_box, output_width, output_height, args.min_area)
            for event in rgb_crop_events:
                add_anomaly(anomalies, split, pair.key, "rgb_crop_event", event)
            for event in ir_crop_events:
                add_anomaly(anomalies, split, pair.key, "ir_crop_event", event)

            if pair.rgb_label is not None and pair.ir_label is not None and pair.rgb_label.resolve() == pair.ir_label.resolve():
                fused_objects = rgb_objects
                fusion_stats = {"matched": 0, "rgb_only": len(rgb_objects), "ir_only": 0, "class_conflict": 0}
                fusion_events: list[str] = []
                fusion_strategy = "shared_label"
            else:
                fused_objects, fusion_stats, fusion_events = fuse_label_sets(
                    rgb_objects, ir_objects, args.fusion_iou_thresh
                )
                fusion_strategy = "fused" if (rgb_objects and ir_objects) else "single_source"
            for event in fusion_events:
                add_anomaly(anomalies, split, pair.key, "fusion_event", event)
            for name, value in fusion_stats.items():
                split_summaries[split][f"fusion_{name}"] += value

            if not fused_objects and not args.allow_empty_labels:
                add_anomaly(
                    anomalies,
                    split,
                    pair.key,
                    "empty_after_fusion",
                    "No labels remain after crop/fusion and allow-empty-labels is disabled.",
                )
                split_summaries[split]["skipped_pairs"] += 1
                continue

            rgb_output_path = rgb_output_dir / f"{pair.key}{pair.rgb_image.suffix.lower()}"
            ir_output_path = ir_output_dir / f"{pair.key}{pair.ir_image.suffix.lower()}"
            obb_label_path = obb_output_dir / f"{pair.key}.txt"
            rgb_label_path = rgb_label_output_dir / f"{pair.key}.txt"

            save_image(rgb_cropped, rgb_output_path)
            save_image(ir_cropped, ir_output_path)
            write_label_file(fused_objects, obb_label_path, output_width, output_height)
            copy_text_file(obb_label_path, rgb_label_path)

            split_summaries[split]["processed_pairs"] += 1
            split_summaries[split]["processed_objects"] += len(fused_objects)
            if pair.rgb_label is None or pair.ir_label is None:
                split_summaries[split]["single_source_labels"] += 1
            if not fused_objects:
                split_summaries[split]["empty_labels"] += 1

            pair_indexes[split].append(
                {
                    "id": pair.key,
                    "split": split,
                    "rgb_image": relative_posix(rgb_output_path, args.output_root),
                    "ir_image": relative_posix(ir_output_path, args.output_root),
                    "label_obb": relative_posix(obb_label_path, args.output_root),
                    "label_rgb_compat": relative_posix(rgb_label_path, args.output_root),
                    "source_rgb_image": str(pair.rgb_image),
                    "source_ir_image": str(pair.ir_image),
                    "source_rgb_label": str(pair.rgb_label) if pair.rgb_label else None,
                    "source_ir_label": str(pair.ir_label) if pair.ir_label else None,
                    "original_size": [image_width, image_height],
                    "processed_size": [output_width, output_height],
                    "crop_box_xyxy": list(crop_box),
                    "objects": len(fused_objects),
                    "fusion_strategy": fusion_strategy,
                }
            )

    temporal_indexes: dict[str, list[dict[str, Any]]] = {}
    for split, entries in pair_indexes.items():
        ordered_entries = sorted(entries, key=lambda item: natural_sort_key(item["id"]))
        temporal_indexes[split] = []
        previous_id: str | None = None
        for entry in ordered_entries:
            temporal_indexes[split].append(
                {
                    "id": entry["id"],
                    "split": split,
                    "current_pair": entry["id"],
                    "previous_pair": previous_id,
                    "rgb_image": entry["rgb_image"],
                    "ir_image": entry["ir_image"],
                    "label_rgb_compat": entry["label_rgb_compat"],
                    "label_obb": entry["label_obb"],
                }
            )
            previous_id = entry["id"]

    class_mapping_path = output_dirs["reports"] / "class_mapping.json"
    write_json(class_mapping_path, class_mapper.export())
    for split in ("train", "val"):
        write_json(output_dirs["index"] / f"{split}_pairs.json", sorted(pair_indexes.get(split, []), key=lambda item: natural_sort_key(item["id"])))
        write_json(output_dirs["index"] / f"{split}_temporal.json", temporal_indexes.get(split, []))

    anomaly_counter = Counter(item["type"] for item in anomalies)
    expected_input_size = infer_single_size(observed_input_sizes)
    expected_output_size = infer_single_size(observed_output_sizes)
    expected_fixed_crop_box = None
    if args.crop_mode == "fixed" and expected_input_size is not None:
        expected_fixed_crop_box = list(
            compute_fixed_crop_box(
                image_width=expected_input_size[0],
                image_height=expected_input_size[1],
                crop_left=args.crop_left,
                crop_top=args.crop_top,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
            )
        )
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "split_mode": args.split_mode,
        "crop_mode": args.crop_mode,
        "fixed_crop": {
            "left": args.crop_left,
            "top": args.crop_top,
            "right": args.crop_right,
            "bottom": args.crop_bottom,
        },
        "expected_input_size": expected_input_size,
        "expected_output_size": expected_output_size,
        "fixed_crop_box_rule": {
            "format": "xyxy",
            "x0": "crop_left",
            "y0": "crop_top",
            "x1": "image_width - crop_right",
            "y1": "image_height - crop_bottom",
            "expected_crop_box_xyxy": expected_fixed_crop_box,
        },
        "data_yaml": relative_posix(create_data_yaml(args.output_root, class_mapper.names), args.output_root),
        "splits": {split: dict(counter) for split, counter in split_summaries.items()},
        "class_names": class_mapper.names,
        "anomaly_counts": dict(anomaly_counter),
        "anomaly_total": len(anomalies),
        "reports": {
            "class_mapping": relative_posix(class_mapping_path, args.output_root),
            "anomalies": "reports/anomalies.json",
        },
    }
    write_json(output_dirs["reports"] / "preprocess_summary.json", summary)
    write_json(output_dirs["reports"] / "anomalies.json", anomalies)


if __name__ == "__main__":
    main()
