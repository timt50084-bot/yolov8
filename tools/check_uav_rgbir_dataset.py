from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_preprocess_uav import (
    load_image,
    natural_sort_key,
    polygon_area,
    stems_from_directory,
    validate_polygon_line,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a prepared UAV RGB/IR OBB dataset for structure, label validity, and current Ultralytics compatibility."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Prepared dataset root to inspect.")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional data YAML. Defaults to dataset-root/data/uav_rgb_obb.yaml.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional report path. Defaults to dataset-root/reports/check_summary.json.",
    )
    parser.add_argument(
        "--skip-dataset-scan",
        action="store_true",
        help="Skip the YOLODataset OBB compatibility scan.",
    )
    return parser.parse_args()


def add_issue(issues: list[dict[str, Any]], split: str, item_id: str, issue_type: str, detail: str, **extra: Any) -> None:
    issues.append({"split": split, "id": item_id, "type": issue_type, "detail": detail, **extra})


def load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_scan(dataset_root: Path, split: str, class_names: list[str]) -> tuple[bool, str]:
    """Build a current YOLODataset(task='obb') scan on the processed RGB training view."""
    from ultralytics.data.dataset import YOLODataset

    hyp = SimpleNamespace(
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
    )
    image_root = dataset_root / "images" / "rgb" / split
    try:
        dataset = YOLODataset(
            img_path=str(image_root),
            imgsz=640,
            batch_size=1,
            augment=False,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix=f"{split}: ",
            task="obb",
            classes=None,
            data={"names": class_names, "channels": 3},
            fraction=1.0,
        )
        return True, f"dataset_scan_ok:{len(dataset.labels)}"
    except Exception as error:
        return False, str(error)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    data_yaml_path = args.data_yaml or dataset_root / "data" / "uav_rgb_obb.yaml"
    report_path = args.report_path or dataset_root / "reports" / "check_summary.json"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")

    data_yaml = {}
    if data_yaml_path.exists():
        import yaml

        data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}

    class_mapping = load_json(dataset_root / "reports" / "class_mapping.json") or {}
    class_names = data_yaml.get("names") or class_mapping.get("names") or []
    num_classes = len(class_names) if class_names else None

    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"dataset_root": str(dataset_root), "splits": {}}

    required_dirs = [
        dataset_root / "images" / "rgb" / "train",
        dataset_root / "images" / "rgb" / "val",
        dataset_root / "images" / "ir" / "train",
        dataset_root / "images" / "ir" / "val",
        dataset_root / "labels" / "obb" / "train",
        dataset_root / "labels" / "obb" / "val",
        dataset_root / "labels" / "rgb" / "train",
        dataset_root / "labels" / "rgb" / "val",
        dataset_root / "index",
        dataset_root / "reports",
    ]
    for directory in required_dirs:
        if not directory.exists():
            add_issue(issues, "global", directory.name, "missing_directory", f"Required directory '{directory}' is missing.")

    pair_indexes = {
        split: load_json(dataset_root / "index" / f"{split}_pairs.json") or [] for split in ("train", "val")
    }
    temporal_indexes = {
        split: load_json(dataset_root / "index" / f"{split}_temporal.json") or [] for split in ("train", "val")
    }

    for split in ("train", "val"):
        split_counter: Counter[str] = Counter()
        rgb_dir = dataset_root / "images" / "rgb" / split
        ir_dir = dataset_root / "images" / "ir" / split
        obb_label_dir = dataset_root / "labels" / "obb" / split
        rgb_label_dir = dataset_root / "labels" / "rgb" / split

        rgb_map = stems_from_directory(rgb_dir)
        ir_map = stems_from_directory(ir_dir)
        obb_label_map = stems_from_directory(obb_label_dir)
        rgb_label_map = stems_from_directory(rgb_label_dir)
        all_keys = sorted(set(rgb_map) | set(ir_map) | set(obb_label_map) | set(rgb_label_map), key=natural_sort_key)

        for key in all_keys:
            split_counter["samples"] += 1
            rgb_image = rgb_map.get(key)
            ir_image = ir_map.get(key)
            obb_label = obb_label_map.get(key)
            rgb_label = rgb_label_map.get(key)

            if rgb_image is None or ir_image is None:
                add_issue(
                    issues,
                    split,
                    key,
                    "pair_mismatch",
                    "RGB and IR files are not one-to-one.",
                    rgb_image=str(rgb_image) if rgb_image else None,
                    ir_image=str(ir_image) if ir_image else None,
                )
            for image_path, image_type in ((rgb_image, "rgb"), (ir_image, "ir")):
                if image_path is None:
                    continue
                try:
                    load_image(image_path)
                except Exception as error:
                    add_issue(issues, split, key, "image_read_error", str(error), image_type=image_type, image=str(image_path))
            if obb_label is None:
                add_issue(issues, split, key, "missing_obb_label", "Canonical OBB label is missing.")
            if rgb_label is None:
                add_issue(issues, split, key, "missing_rgb_label", "Current-training-compatible RGB label is missing.")
            elif obb_label is not None and rgb_label.read_text(encoding="utf-8") != obb_label.read_text(encoding="utf-8"):
                add_issue(issues, split, key, "label_mirror_mismatch", "labels/obb and labels/rgb contents differ.")

            if rgb_label is not None:
                lines = [line for line in rgb_label.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not lines:
                    split_counter["empty_labels"] += 1
                    add_issue(issues, split, key, "empty_label", "Label file is empty.")
                for line_number, line in enumerate(lines, start=1):
                    valid, error = validate_polygon_line(line, num_classes=num_classes)
                    if not valid:
                        add_issue(issues, split, key, "invalid_polygon", f"line {line_number}: {error}")
                        continue
                    coords = [float(value) for value in line.split()[1:]]
                    polygon = np.asarray(coords, dtype=float).reshape(4, 2)
                    if polygon_area(polygon) <= 1e-6:
                        add_issue(issues, split, key, "degenerate_polygon", f"line {line_number}: polygon area is zero.")

        pair_index_ids = [entry.get("id") for entry in pair_indexes[split]]
        if pair_index_ids != sorted(pair_index_ids, key=natural_sort_key):
            add_issue(issues, split, split, "pair_index_order", "Pair index is not in stable natural order.")
        for entry in pair_indexes[split]:
            entry_id = entry.get("id")
            if entry_id not in rgb_map or entry_id not in ir_map:
                add_issue(issues, split, entry_id or "unknown", "pair_index_missing_file", "Pair index references a missing RGB/IR file.")

        seen_temporal_ids: set[str] = set()
        for position, entry in enumerate(temporal_indexes[split]):
            current_id = entry.get("id")
            previous_id = entry.get("previous_pair")
            if current_id in seen_temporal_ids:
                add_issue(issues, split, current_id, "temporal_duplicate", "Temporal index contains duplicate IDs.")
            seen_temporal_ids.add(current_id)
            if current_id not in rgb_map:
                add_issue(issues, split, current_id or "unknown", "temporal_missing_current", "Temporal index current pair is missing.")
            if previous_id is not None and previous_id not in rgb_map:
                add_issue(issues, split, current_id or "unknown", "temporal_missing_previous", "Temporal index previous pair is missing.")
            if position == 0 and previous_id is not None:
                add_issue(issues, split, current_id or "unknown", "temporal_first_previous", "First temporal entry should not have previous_pair.")

        if not args.skip_dataset_scan:
            compatible, detail = dataset_scan(dataset_root, split, class_names)
            split_counter["dataset_scan_ok" if compatible else "dataset_scan_failed"] += 1
            if not compatible:
                add_issue(issues, split, split, "dataset_scan_failed", detail)

        summary["splits"][split] = dict(split_counter)

    issue_counts = Counter(issue["type"] for issue in issues)
    summary["class_names"] = class_names
    summary["issue_counts"] = dict(issue_counts)
    summary["issue_total"] = len(issues)
    summary["data_yaml"] = str(data_yaml_path) if data_yaml_path.exists() else None
    summary["pair_index_counts"] = {split: len(items) for split, items in pair_indexes.items()}
    summary["temporal_index_counts"] = {split: len(items) for split, items in temporal_indexes.items()}

    write_json(report_path, {"summary": summary, "issues": issues})


if __name__ == "__main__":
    main()
