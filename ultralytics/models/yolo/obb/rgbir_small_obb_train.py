from __future__ import annotations

from copy import deepcopy

import torch

from ultralytics.utils.loss_small_object import apply_small_object_loss_weighting, parse_small_object_loss_on

from .rgbir_obb_train import RGBIRTrainAssistOBBModel


class RGBIRSmallObjectOBBModel(RGBIRTrainAssistOBBModel):
    """Stage 4 small-object extension over the explicit Stage 3 RGB-IR train-assist OBB model.

    The deployable path stays RGB-only exactly as in Stage 3. Stage 4 adds an optional, conservative batch-level loss
    weighting pass for small objects during training only. `img_prev` remains ignored on purpose and is still reserved
    for the later temporal stage.
    """

    def __init__(self, cfg="yolov8-rgbir-obb-small.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True) -> None:
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else cfg
        self.use_small_object_loss_weighting = bool(cfg.get("use_small_object_loss_weighting", False))
        self.small_object_area_thr_norm = float(cfg.get("small_object_area_thr_norm", 0.005))
        self.small_object_loss_gain = float(cfg.get("small_object_loss_gain", 0.25))
        self.small_object_loss_on = parse_small_object_loss_on(cfg.get("small_object_loss_on", ("box", "cls", "dfl", "angle")))
        self.last_small_object_scale = 1.0
        self.last_small_object_stats = {
            "small_count": 0.0,
            "target_count": 0.0,
            "small_ratio": 0.0,
            "severity_mean": 0.0,
            "scale": 1.0,
        }
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch: dict[str, torch.Tensor], preds=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the native Stage 3 OBB loss first, then optionally up-weight small-object-heavy batches."""
        loss_items, detached_items = super().loss(batch, preds)
        if self.training and self.use_small_object_loss_weighting:
            weighted_loss_items, stats = apply_small_object_loss_weighting(
                loss_items,
                batch,
                enabled=True,
                area_thr_norm=self.small_object_area_thr_norm,
                gain=self.small_object_loss_gain,
                loss_on=self.small_object_loss_on,
            )
            self.last_small_object_scale = stats["scale"]
            self.last_small_object_stats = stats
            weighted_detached = detached_items.clone()
            weighted_detached[:4] = detached_items[:4] * float(stats["scale"])
            return weighted_loss_items, weighted_detached

        self.last_small_object_scale = 1.0
        self.last_small_object_stats = {
            "small_count": 0.0,
            "target_count": 0.0,
            "small_ratio": 0.0,
            "severity_mean": 0.0,
            "scale": 1.0,
        }
        return loss_items, detached_items
