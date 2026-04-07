# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from ultralytics.utils.instance import Instances

from .rgbir_temporal_transforms import RGBIRTemporalTransform


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset yaml to load into a mapping, but got {type(data).__name__}.")
    return data


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON list at {path}, but got {type(data).__name__}.")
    return data


def _resolve_path(root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else (root / path)


def _coerce_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        def _sort_key(key: Any) -> tuple[int, Any]:
            key_str = str(key)
            return (0, int(key_str)) if key_str.isdigit() else (1, key_str)

        return [str(names[k]) for k in sorted(names, key=_sort_key)]
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(name) for name in names]
    return []


def _load_data_config(data_root: str | Path | None, data: str | Path | dict[str, Any] | None) -> tuple[dict[str, Any], Path]:
    yaml_path = None
    if isinstance(data, dict):
        config = dict(data)
    elif data is not None:
        yaml_path = Path(data)
        config = _load_yaml(yaml_path)
    else:
        config = {}

    if data_root is not None:
        root = Path(data_root)
        if not config:
            default_yaml = root / "data" / "uav_rgb_obb.yaml"
            if default_yaml.exists():
                yaml_path = default_yaml
                config = _load_yaml(default_yaml)
    elif "path" in config:
        root = Path(config["path"])
        if not root.is_absolute() and yaml_path is not None:
            root = (yaml_path.parent / root).resolve()
    elif yaml_path is not None:
        root = yaml_path.parent
    else:
        raise ValueError("Either data_root or data must be provided for RGBIRTemporalOBBDataset.")

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    config.setdefault("path", str(root))
    return config, root


def _segments_to_xyxy(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    min_xy = segments.min(axis=1)
    max_xy = segments.max(axis=1)
    return np.concatenate((min_xy, max_xy), axis=1).astype(np.float32)


class RGBIRTemporalOBBDataset(Dataset):
    """Opt-in RGB + IR + one-step temporal OBB dataset built from Stage 1 outputs.

    The dataset consumes Stage 1 pair and temporal index files rather than guessing paths from directory structure.
    It returns current RGB, current IR, previous RGB, and OBB labels in the tensor layout expected by later phases,
    while remaining fully disconnected from the default baseline dataset builder path.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        data: str | Path | dict[str, Any] | None = None,
        mode: str = "train",
        imgsz: int | tuple[int, int] = 640,
        augment: bool = False,
        hyp: Any | None = None,
        previous_strategy: str = "copy_current",
    ) -> None:
        self.data, self.data_root = _load_data_config(data_root, data)
        self.mode = "train" if mode == "train" else "val"
        self.imgsz = imgsz
        self.augment = augment
        self.previous_strategy = previous_strategy
        self.names = _coerce_names(self.data.get("names", []))
        self.nc = len(self.names)
        self.transform = RGBIRTemporalTransform(imgsz=imgsz, augment=augment, hyp=hyp)
        self.samples = self._build_samples()

    def _index_path(self, kind: str) -> Path:
        key = f"{kind}_{self.mode}"
        default_name = f"{self.mode}_{'pairs' if kind == 'pair_index' else 'temporal'}.json"
        rel = self.data.get(key, f"index/{default_name}")
        path = _resolve_path(self.data_root, rel)
        if not path.exists():
            raise FileNotFoundError(f"Required {kind} file not found for {self.mode}: {path}")
        return path

    def _build_samples(self) -> list[dict[str, Any]]:
        pair_records = _load_json(self._index_path("pair_index"))
        temporal_records = _load_json(self._index_path("temporal_index"))
        pair_by_id = {str(item["id"]): item for item in pair_records}
        samples = []
        for entry in temporal_records:
            sample_id = str(entry.get("id") or entry.get("current_pair"))
            current_id = str(entry.get("current_pair") or sample_id)
            pair = pair_by_id.get(current_id)
            if pair is None:
                raise KeyError(f"Temporal record {sample_id} references missing pair id {current_id}.")

            prev_id_raw = entry.get("previous_pair")
            prev_pair = pair_by_id.get(str(prev_id_raw)) if prev_id_raw is not None else None
            temporal_valid = prev_pair is not None
            if not temporal_valid and self.previous_strategy != "copy_current":
                raise ValueError(
                    f"Sample {sample_id} has no previous frame and previous_strategy={self.previous_strategy!r}."
                )

            prev_rgb = pair["rgb_image"] if prev_pair is None else prev_pair["rgb_image"]
            sample = {
                "sample_id": sample_id,
                "current_pair_id": current_id,
                "previous_pair_id": None if prev_pair is None else str(prev_pair["id"]),
                "seq_id": entry.get("seq_id", pair.get("seq_id", sample_id)),
                "frame_id": entry.get("frame_id", pair.get("frame_id", sample_id)),
                "temporal_valid": temporal_valid,
                "im_file": _resolve_path(self.data_root, pair["rgb_image"]),
                "im_file_ir": _resolve_path(self.data_root, pair["ir_image"]),
                "im_file_prev": _resolve_path(self.data_root, prev_rgb),
                "label_file": _resolve_path(self.data_root, pair["label_obb"]),
            }
            samples.append(sample)

        if not samples:
            raise ValueError(f"No samples were built for {self.mode} from {self.data_root}.")
        return samples

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _load_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Missing OBB label file: {path}")

        classes: list[list[float]] = []
        segments: list[np.ndarray] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 9:
                    raise ValueError(f"Expected 9 columns in {path}:{line_number}, but got {len(parts)}.")
                cls = float(parts[0])
                points = np.asarray(parts[1:], dtype=np.float32).reshape(4, 2)
                if np.any(points < 0.0) or np.any(points > 1.0):
                    raise ValueError(f"Polygon points out of normalized range in {path}:{line_number}.")
                classes.append([cls])
                segments.append(points)

        cls_arr = np.asarray(classes, dtype=np.float32).reshape(-1, 1) if classes else np.zeros((0, 1), dtype=np.float32)
        seg_arr = (
            np.asarray(segments, dtype=np.float32).reshape(-1, 4, 2)
            if segments
            else np.zeros((0, 4, 2), dtype=np.float32)
        )
        return cls_arr, seg_arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        img = self._load_image(sample["im_file"])
        img_ir = self._load_image(sample["im_file_ir"])
        img_prev = self._load_image(sample["im_file_prev"])
        cls, segments = self._load_label(sample["label_file"])
        instances = Instances(bboxes=_segments_to_xyxy(segments), segments=segments, bbox_format="xyxy", normalized=True)
        payload = {
            "img": img,
            "img_ir": img_ir,
            "img_prev": img_prev,
            "cls": cls,
            "instances": instances,
            "sample_id": sample["sample_id"],
            "frame_id": sample["frame_id"],
            "seq_id": sample["seq_id"],
            "temporal_valid": sample["temporal_valid"],
            "im_file": str(sample["im_file"]),
            "im_file_ir": str(sample["im_file_ir"]),
            "im_file_prev": str(sample["im_file_prev"]),
            "label_file": str(sample["label_file"]),
            "ori_shape": img.shape[:2],
            "ori_shape_ir": img_ir.shape[:2],
            "ori_shape_prev": img_prev.shape[:2],
        }
        return self.transform(payload)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate RGB, IR, previous RGB, OBB targets, and metadata into one stable batch."""
        out = {
            "img": torch.stack([sample["img"] for sample in batch], 0),
            "img_ir": torch.stack([sample["img_ir"] for sample in batch], 0),
            "img_prev": torch.stack([sample["img_prev"] for sample in batch], 0),
            "cls": torch.cat([sample["cls"] for sample in batch], 0),
            "bboxes": torch.cat([sample["bboxes"] for sample in batch], 0),
            "segments": torch.cat([sample["segments"] for sample in batch], 0),
            "temporal_valid": torch.stack([sample["temporal_valid"] for sample in batch], 0),
            "sample_id": [sample["sample_id"] for sample in batch],
            "frame_id": [sample["frame_id"] for sample in batch],
            "seq_id": [sample["seq_id"] for sample in batch],
            "im_file": [sample["im_file"] for sample in batch],
            "im_file_ir": [sample["im_file_ir"] for sample in batch],
            "im_file_prev": [sample["im_file_prev"] for sample in batch],
            "label_file": [sample["label_file"] for sample in batch],
            "ori_shape": [sample["ori_shape"] for sample in batch],
            "ori_shape_ir": [sample["ori_shape_ir"] for sample in batch],
            "ori_shape_prev": [sample["ori_shape_prev"] for sample in batch],
            "resized_shape": [sample["resized_shape"] for sample in batch],
            "ratio_pad": [sample["ratio_pad"] for sample in batch],
            "ratio_pad_ir": [sample["ratio_pad_ir"] for sample in batch],
            "ratio_pad_prev": [sample["ratio_pad_prev"] for sample in batch],
        }
        batch_idx = []
        for i, sample in enumerate(batch):
            sample_batch_idx = sample["batch_idx"].clone()
            sample_batch_idx += i
            batch_idx.append(sample_batch_idx)
        out["batch_idx"] = torch.cat(batch_idx, 0) if batch_idx else torch.zeros(0)
        return out
