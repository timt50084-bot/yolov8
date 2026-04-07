from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils.metrics import OBBMetrics


def normalized_obb_areas(bboxes: torch.Tensor, imgsz: tuple[int, int] | None = None) -> torch.Tensor:
    """Return normalized OBB areas from xywhr boxes.

    If `imgsz` is provided, the boxes are assumed to be in pixels and will be normalized by image area. Otherwise
    the input is assumed to already use normalized xywhr.
    """
    if bboxes.numel() == 0:
        return bboxes.new_zeros((0,))
    areas = (bboxes[:, 2] * bboxes[:, 3]).float()
    if imgsz is None:
        return areas
    image_area = max(float(imgsz[0] * imgsz[1]), 1.0)
    return areas / image_area


def small_object_mask(bboxes: torch.Tensor, area_thr_norm: float, imgsz: tuple[int, int] | None = None) -> torch.Tensor:
    """Return a boolean mask for small oriented boxes using the shared normalized area threshold."""
    return normalized_obb_areas(bboxes, imgsz=imgsz) <= float(area_thr_norm)


def safe_process_obb_metrics(
    metrics: OBBMetrics,
    *,
    save_dir: Path,
    plot: bool = False,
    on_plot=None,
) -> dict[str, np.ndarray]:
    """Process OBB metrics only when stats exist, otherwise leave them at zero."""
    has_stats = any(len(v) for v in metrics.stats.values())
    if not has_stats:
        return {}
    return metrics.process(save_dir=save_dir, plot=plot, on_plot=on_plot)


def small_object_results_dict(metrics: OBBMetrics) -> dict[str, float]:
    """Return explicit small-object metric keys without affecting the overall fitness key."""
    mp, mr, map50, map5095 = metrics.mean_results()
    return {
        "metrics/precision_small(B)": float(mp),
        "metrics/recall_small(B)": float(mr),
        "metrics/mAP50_small(B)": float(map50),
        "metrics/mAP50-95_small(B)": float(map5095),
    }


def update_small_obb_metrics(
    metrics: OBBMetrics,
    *,
    process_batch_fn,
    predn: dict[str, torch.Tensor],
    pbatch: dict[str, Any],
    area_thr_norm: float,
) -> dict[str, int]:
    """Update a dedicated OBBMetrics instance using only small-object ground truth for one image."""
    small_mask = small_object_mask(pbatch["bboxes"], area_thr_norm=area_thr_norm, imgsz=pbatch["imgsz"])
    if small_mask.numel() == 0 or not small_mask.any():
        return {"small_images": 0, "small_instances": 0}

    cls_small = pbatch["cls"][small_mask]
    bboxes_small = pbatch["bboxes"][small_mask]
    no_pred = predn["cls"].shape[0] == 0
    metrics.update_stats(
        {
            **process_batch_fn(predn, {**pbatch, "cls": cls_small, "bboxes": bboxes_small}),
            "target_cls": cls_small.cpu().numpy(),
            "target_img": np.unique(cls_small.cpu().numpy()),
            "conf": np.zeros(0, dtype=np.float32) if no_pred else predn["conf"].cpu().numpy(),
            "pred_cls": np.zeros(0, dtype=np.float32) if no_pred else predn["cls"].cpu().numpy(),
        }
    )
    return {"small_images": 1, "small_instances": int(cls_small.shape[0])}
