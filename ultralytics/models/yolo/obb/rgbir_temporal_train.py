from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from typing import Any

from ultralytics.data.build import build_rgbir_temporal_obb_dataset
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

from .rgbir_small_train import RGBIRSmallObjectOBBTrainer
from .rgbir_temporal_obb_train import RGBIRTemporalOBBModel
from .rgbir_temporal_val import TemporalOBBValidator


class RGBIRTemporalOBBTrainer(RGBIRSmallObjectOBBTrainer):
    """Explicit Stage 5 trainer with lightweight two-frame temporal refine on top of Stage 4."""

    TEMPORAL_KEYS = (
        "use_temporal",
        "temporal_mode",
        "temporal_fusion_type",
        "temporal_feature_stages",
        "temporal_branch_width",
        "temporal_loss_weight",
        "temporal_residual_scale",
    )

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: list[Any] | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.temporal_settings = self._temporal_settings_defaults(overrides.get("model"))
        for key in self.TEMPORAL_KEYS:
            if key in overrides:
                self.temporal_settings[key] = overrides.pop(key)
        super().__init__(cfg, overrides, _callbacks)

    @staticmethod
    def _temporal_settings_defaults(model_cfg: str | dict | None) -> dict[str, Any]:
        defaults = {
            "use_temporal": False,
            "temporal_mode": "off",
            "temporal_fusion_type": "diff_gate",
            "temporal_feature_stages": [6, 9],
            "temporal_branch_width": 0.25,
            "temporal_loss_weight": 0.02,
            "temporal_residual_scale": 0.10,
        }
        if isinstance(model_cfg, dict):
            defaults.update({k: model_cfg[k] for k in defaults if k in model_cfg})
            return defaults
        if isinstance(model_cfg, (str, Path)):
            path = Path(model_cfg)
            if path.suffix in {".yaml", ".yml"} and path.exists():
                cfg = yaml_model_load(path)
                defaults.update({k: cfg[k] for k in defaults if k in cfg})
        return defaults

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> RGBIRTemporalOBBModel:
        """Return the explicit Stage 5 temporal model while preserving Stage 3/4 settings."""
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        for source in (self.rgbir_settings, self.small_object_settings, self.temporal_settings):
            for key, value in source.items():
                if value is not None:
                    model_cfg[key] = value
        model = RGBIRTemporalOBBModel(
            model_cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        if RANK == -1:
            model.log_assist_configuration()
            LOGGER.info(
                "Stage 4 small-object settings: "
                f"sampling={self.small_object_settings['use_small_object_sampling']}, "
                f"loss_weighting={self.small_object_settings['use_small_object_loss_weighting']}, "
                f"small_metrics={self.small_object_settings['enable_small_object_metrics']}, "
                f"area_thr_norm={self.small_object_settings['small_object_area_thr_norm']}"
            )
            model.log_temporal_configuration()
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Keep Stage 2 dataset for train, and opt into it for val only when temporal refine is enabled."""
        if mode == "val" and self.temporal_settings["use_temporal"]:
            return build_rgbir_temporal_obb_dataset(
                cfg=self.args,
                data=self.data,
                data_root=self.data.get("path"),
                mode=mode,
                imgsz=self.args.imgsz,
                augment=False,
                hyp=self.args,
            )
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_validator(self):
        """Return the Stage 5 validator with optional temporal val and preserved small-object metrics."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss", "rgbir_aux_loss", "temporal_aux_loss"
        return TemporalOBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            use_temporal=bool(self.temporal_settings["use_temporal"]),
            enable_small_object_metrics=bool(self.small_object_settings["enable_small_object_metrics"]),
            small_object_area_thr_norm=float(self.small_object_settings["small_object_area_thr_norm"]),
        )
