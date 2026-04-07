from __future__ import annotations

import os
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import InfiniteDataLoader, build_dataloader, seed_worker
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

from ultralytics.utils.small_object_sampler import build_weighted_small_object_sampler

from .rgbir_small_obb_train import RGBIRSmallObjectOBBModel
from .rgbir_small_val import SmallObjectOBBValidator
from .rgbir_train import RGBIROBBTrainer


class RGBIRSmallObjectOBBTrainer(RGBIROBBTrainer):
    """Explicit Stage 4 trainer that extends the Stage 3 RGB-IR path with opt-in small-object optimization."""

    SMALL_KEYS = (
        "use_small_object_sampling",
        "small_object_area_thr_norm",
        "small_object_sampling_power",
        "small_object_sampling_min_weight",
        "small_object_sampling_max_weight",
        "use_small_object_loss_weighting",
        "small_object_loss_gain",
        "small_object_loss_on",
        "enable_small_object_metrics",
    )

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: list[Any] | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.small_object_settings = self._small_settings_defaults(overrides.get("model"))
        for key in self.SMALL_KEYS:
            if key in overrides:
                self.small_object_settings[key] = overrides.pop(key)
        self.sampler_summary: dict[str, Any] | None = None
        super().__init__(cfg, overrides, _callbacks)

    @staticmethod
    def _small_settings_defaults(model_cfg: str | dict | None) -> dict[str, Any]:
        defaults = {
            "use_small_object_sampling": False,
            "small_object_area_thr_norm": 0.005,
            "small_object_sampling_power": 1.0,
            "small_object_sampling_min_weight": 1.0,
            "small_object_sampling_max_weight": 3.0,
            "use_small_object_loss_weighting": False,
            "small_object_loss_gain": 0.25,
            "small_object_loss_on": ("box", "cls", "dfl", "angle"),
            "enable_small_object_metrics": False,
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
    ) -> RGBIRSmallObjectOBBModel:
        """Return the Stage 4 RGB-IR + small-object opt-in model."""
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        for key, value in self.rgbir_settings.items():
            if value is not None:
                model_cfg[key] = value
        for key, value in self.small_object_settings.items():
            if value is not None:
                model_cfg[key] = value
        model = RGBIRSmallObjectOBBModel(
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
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Use weighted small-object sampling only for the explicit Stage 4 training branch when enabled."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode != "train" or not self.small_object_settings["use_small_object_sampling"]:
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=mode == "train",
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        if rank != -1:
            LOGGER.warning("Stage 4 small-object weighted sampling is currently single-process only; falling back to standard sampling.")
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers,
                shuffle=True,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        sampler, summary = build_weighted_small_object_sampler(
            dataset,
            area_thr_norm=float(self.small_object_settings["small_object_area_thr_norm"]),
            power=float(self.small_object_settings["small_object_sampling_power"]),
            min_weight=float(self.small_object_settings["small_object_sampling_min_weight"]),
            max_weight=float(self.small_object_settings["small_object_sampling_max_weight"]),
        )
        self.sampler_summary = summary
        top = summary["top_weighted_samples"][0] if summary["top_weighted_samples"] else None
        if top is not None:
            LOGGER.info(
                "Stage 4 weighted sampling active: "
                f"weight_mean={summary['weight_mean']:.3f}, "
                f"weight_max={summary['weight_max']:.3f}, "
                f"top_sample={top['sample_id']} (small={top['small_objects']}, total={top['total_objects']})"
            )

        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
        )

    def get_validator(self):
        """Return the Stage 4 validator with optional small-object-only metrics."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss", "rgbir_aux_loss"
        return SmallObjectOBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            enable_small_object_metrics=bool(self.small_object_settings["enable_small_object_metrics"]),
            small_object_area_thr_norm=float(self.small_object_settings["small_object_area_thr_norm"]),
        )
