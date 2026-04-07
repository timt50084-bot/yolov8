from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def polygon_area_norm(points: np.ndarray) -> float:
    """Return the normalized polygon area for one quadrilateral label."""
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def parse_small_object_label_file(label_file: str | Path, area_thr_norm: float) -> dict[str, float]:
    """Parse one Stage 1 polygon OBB label file and summarize its small-object content."""
    label_path = Path(label_file)
    areas: list[float] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                raise ValueError(f"Expected 9 columns in {label_path}:{line_number}, but got {len(parts)}.")
            points = np.asarray(parts[1:], dtype=np.float32).reshape(4, 2)
            area = polygon_area_norm(points)
            areas.append(area)

    total_objects = len(areas)
    if total_objects == 0:
        return {
            "total_objects": 0,
            "small_objects": 0,
            "small_ratio": 0.0,
            "small_severity": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
        }

    areas_np = np.asarray(areas, dtype=np.float32)
    small_mask = areas_np <= float(area_thr_norm)
    small_ratio = float(small_mask.mean())
    # Smaller-than-threshold objects contribute more strongly but remain clipped to [0, 1].
    severity = np.clip(1.0 - (areas_np / max(float(area_thr_norm), 1e-12)), 0.0, 1.0)
    small_severity = float(severity[small_mask].sum()) if small_mask.any() else 0.0
    return {
        "total_objects": int(total_objects),
        "small_objects": int(small_mask.sum()),
        "small_ratio": small_ratio,
        "small_severity": small_severity,
        "min_area": float(areas_np.min()),
        "max_area": float(areas_np.max()),
    }


def compute_small_object_weight(
    *,
    small_severity: float,
    power: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """Convert one image's small-object severity score into a conservative sampling weight."""
    if max_weight < min_weight:
        raise ValueError(f"max_weight ({max_weight}) must be >= min_weight ({min_weight}).")
    score = max(float(small_severity), 0.0) ** max(float(power), 0.0)
    raw_weight = min_weight + (max_weight - min_weight) * (1.0 - math.exp(-score))
    return float(np.clip(raw_weight, min_weight, max_weight))


def build_small_object_sampling_weights(
    dataset: Any,
    *,
    area_thr_norm: float,
    power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 3.0,
) -> tuple[torch.DoubleTensor, dict[str, Any]]:
    """Build per-image sampling weights for a dataset that exposes Stage 2 `samples` records."""
    if not hasattr(dataset, "samples"):
        raise TypeError("Small-object sampling expects a dataset with a `samples` attribute.")

    weights: list[float] = []
    per_image: list[dict[str, Any]] = []
    for sample in dataset.samples:
        label_file = sample.get("label_file")
        stats = parse_small_object_label_file(label_file, area_thr_norm=area_thr_norm)
        weight = compute_small_object_weight(
            small_severity=stats["small_severity"],
            power=power,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        weights.append(weight)
        per_image.append(
            {
                "sample_id": str(sample.get("sample_id", sample.get("current_pair_id", ""))),
                "label_file": str(label_file),
                "weight": weight,
                "small_score": float(stats["small_severity"]),
                **stats,
            }
        )

    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    weight_array = weights_tensor.cpu().numpy()
    ranked = sorted(per_image, key=lambda item: item["weight"], reverse=True)
    summary = {
        "enabled": True,
        "area_thr_norm": float(area_thr_norm),
        "power": float(power),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "num_images": int(len(per_image)),
        "num_images_with_small_objects": int(sum(item["small_objects"] > 0 for item in per_image)),
        "weight_min": float(weight_array.min()) if len(weight_array) else 0.0,
        "weight_mean": float(weight_array.mean()) if len(weight_array) else 0.0,
        "weight_max": float(weight_array.max()) if len(weight_array) else 0.0,
        "top_weighted_samples": ranked[: min(5, len(ranked))],
        "bottom_weighted_samples": sorted(per_image, key=lambda item: item["weight"])[: min(5, len(per_image))],
        "per_image": per_image,
    }
    return weights_tensor, summary


def build_weighted_small_object_sampler(
    dataset: Any,
    *,
    area_thr_norm: float,
    power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 3.0,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    """Return a WeightedRandomSampler and its summary for explicit small-object training."""
    weights, summary = build_small_object_sampling_weights(
        dataset,
        area_thr_norm=area_thr_norm,
        power=power,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler, summary
