from __future__ import annotations

import math
import random
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .rgbir_obb_train import RGBIRTrainAssistOBBModel


class RGBIROBBTrainer(OBBTrainer):
    """Explicit opt-in trainer for Stage 3 train-time RGB-IR cooperative OBB training.

    Train:
    - uses the Stage 2 RGBIRTemporal dataset to obtain `img`, `img_ir`, labels, and ignored `img_prev`.

    Val / final eval:
    - stays RGB-only through the native OBB validator and native RGB image paths.

    This keeps the default OBB trainer untouched while providing a dedicated branch for Stage 3 experiments.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: list[Any] | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.rgbir_settings = {}
        for key in (
            "use_rgbir_train_assist",
            "fusion_type",
            "ir_branch_width",
            "ir_feature_stages",
            "rgbir_aux_loss_weight",
            "rgbir_residual_scale",
        ):
            if key in overrides:
                self.rgbir_settings[key] = overrides.pop(key)
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> RGBIRTrainAssistOBBModel:
        """Return the explicit Stage 3 RGB-IR train-assist model."""
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        for key in (
            "use_rgbir_train_assist",
            "fusion_type",
            "ir_branch_width",
            "ir_feature_stages",
            "rgbir_aux_loss_weight",
            "rgbir_residual_scale",
        ):
            value = self.rgbir_settings.get(key, None)
            if value is not None:
                model_cfg[key] = value
        model = RGBIRTrainAssistOBBModel(
            model_cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        if RANK == -1:
            model.log_assist_configuration()
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Use the Phase 2 RGBIRTemporal dataset only for explicit Stage 3 training; keep val RGB-only."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32) if not isinstance(self.model, (str, Path)) else 32
        if mode == "train":
            from ultralytics.data.build import build_rgbir_temporal_obb_dataset

            return build_rgbir_temporal_obb_dataset(
                cfg=self.args,
                data=self.data,
                data_root=self.data.get("path"),
                mode=mode,
                imgsz=self.args.imgsz,
                augment=True,
                hyp=self.args,
            )
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize RGB, IR, and previous RGB consistently while leaving labels and meta untouched."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        image_keys = [key for key in ("img", "img_ir", "img_prev") if key in batch and isinstance(batch[key], torch.Tensor)]
        for key in image_keys:
            batch[key] = batch[key].float() / 255

        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                for key in image_keys:
                    batch[key] = nn.functional.interpolate(batch[key], size=ns, mode="bilinear", align_corners=False)
        return batch

    def get_validator(self):
        """Return the native OBB validator while exposing the extra RGB-IR auxiliary loss item."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss", "rgbir_aux_loss"
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_labels(self) -> None:
        """Skip label-cache plotting when the explicit Stage 3 train dataset does not expose YOLODataset.labels."""
        if hasattr(self.train_loader.dataset, "labels"):
            return super().plot_training_labels()
        LOGGER.info("Skipping plot_training_labels for RGBIRTemporalOBBDataset because it does not expose YOLODataset.labels.")
