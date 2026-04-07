from __future__ import annotations

from typing import Any

import torch


LOSS_INDEX = {"box": 0, "cls": 1, "dfl": 2, "angle": 3}


def parse_small_object_loss_on(raw: Any) -> tuple[str, ...]:
    """Normalize configured loss component names."""
    if raw is None:
        return ("box", "cls", "dfl", "angle")
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    names = []
    for name in raw:
        key = str(name).strip().lower()
        if key not in LOSS_INDEX:
            raise ValueError(f"Unsupported small-object loss target {name!r}. Expected subset of {tuple(LOSS_INDEX)}.")
        if key not in names:
            names.append(key)
    return tuple(names)


def compute_small_object_batch_scale(
    batch: dict[str, torch.Tensor],
    *,
    area_thr_norm: float,
    gain: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a conservative batch-level scale factor from normalized OBB areas.

    This intentionally does not rewrite the native v8 OBB criterion. Instead it derives one small-object emphasis
    factor from the current batch and scales selected aggregated loss components.
    """
    bboxes = batch.get("bboxes")
    if bboxes is None or not isinstance(bboxes, torch.Tensor) or bboxes.numel() == 0:
        scale = torch.ones((), device=batch["img"].device, dtype=batch["img"].dtype)
        return scale, {"small_count": 0.0, "target_count": 0.0, "small_ratio": 0.0, "severity_mean": 0.0, "scale": 1.0}

    areas = (bboxes[:, 2] * bboxes[:, 3]).float()
    small_mask = areas <= float(area_thr_norm)
    severity = torch.clamp(1.0 - (areas / max(float(area_thr_norm), 1e-12)), min=0.0, max=1.0)
    severity_mean = severity[small_mask].mean() if small_mask.any() else areas.new_zeros(())
    scale = 1.0 + float(gain) * severity_mean
    return scale.to(device=batch["img"].device, dtype=batch["img"].dtype), {
        "small_count": float(small_mask.sum().item()),
        "target_count": float(areas.numel()),
        "small_ratio": float(small_mask.float().mean().item()),
        "severity_mean": float(severity_mean.item()),
        "scale": float(scale.item()),
    }


def apply_small_object_loss_weighting(
    loss_items: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    enabled: bool,
    area_thr_norm: float,
    gain: float,
    loss_on: tuple[str, ...] | list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Scale selected aggregated OBB loss components for small-object-heavy batches."""
    if not enabled:
        return loss_items, {"small_count": 0.0, "target_count": 0.0, "small_ratio": 0.0, "severity_mean": 0.0, "scale": 1.0}

    scale, stats = compute_small_object_batch_scale(batch, area_thr_norm=area_thr_norm, gain=gain)
    if float(scale.item()) == 1.0:
        return loss_items, stats

    weighted = loss_items.clone()
    for name in parse_small_object_loss_on(loss_on):
        weighted[LOSS_INDEX[name]] = weighted[LOSS_INDEX[name]] * scale
    return weighted, stats
