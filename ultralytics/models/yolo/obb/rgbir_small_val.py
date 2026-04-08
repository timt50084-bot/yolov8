from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import OBBMetrics
from ultralytics.utils.metrics_small_object import (
    safe_process_obb_metrics,
    small_object_results_dict,
    update_small_obb_metrics,
)


class SmallObjectOBBValidator(OBBValidator):
    """Explicit Stage 4 OBB validator with additional small-object-only metrics.

    Overall metrics continue to use the native OBB evaluation path. When enabled, a second dedicated OBBMetrics
    instance is updated on the subset of ground-truth objects whose normalized area is below the configured threshold.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        *,
        enable_small_object_metrics: bool = False,
        small_object_area_thr_norm: float = 0.005,
    ) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.enable_small_object_metrics = bool(enable_small_object_metrics)
        self.small_object_area_thr_norm = float(small_object_area_thr_norm)
        self.small_metrics = OBBMetrics()
        self.small_seen = 0
        self.small_instances = 0

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize both the native overall metrics and the explicit small-object metrics."""
        super().init_metrics(model)
        self.small_metrics = OBBMetrics(names=model.names)
        self.small_metrics.names = model.names
        self.small_seen = 0
        self.small_instances = 0

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]) -> None:
        """Update overall metrics exactly as before, plus the optional small-object subset metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0, dtype=np.float32) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0, dtype=np.float32) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            if self.enable_small_object_metrics:
                info = update_small_obb_metrics(
                    self.small_metrics,
                    process_batch_fn=self._process_batch,
                    predn=predn,
                    pbatch=pbatch,
                    area_thr_norm=self.small_object_area_thr_norm,
                )
                self.small_seen += info["small_images"]
                self.small_instances += info["small_instances"]

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Carry over speed/save-dir metadata for both overall and small-object metrics."""
        super().finalize_metrics()
        self.small_metrics.speed = self.speed
        self.small_metrics.save_dir = self.save_dir

    def gather_stats(self) -> None:
        """Gather overall and small-object stats across ranks."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in self.metrics.stats.keys()}
            for stats_dict in gathered_stats:
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            self.metrics.stats = merged_stats

            gathered_small = [None] * dist.get_world_size()
            dist.gather_object(self.small_metrics.stats, gathered_small, dst=0)
            merged_small = {key: [] for key in self.small_metrics.stats.keys()}
            for stats_dict in gathered_small:
                for key in merged_small:
                    merged_small[key].extend(stats_dict[key])
            self.small_metrics.stats = merged_small

            gathered_counts = [None] * dist.get_world_size()
            dist.gather_object({"small_seen": self.small_seen, "small_instances": self.small_instances}, gathered_counts, dst=0)
            self.small_seen = sum(item["small_seen"] for item in gathered_counts)
            self.small_instances = sum(item["small_instances"] for item in gathered_counts)
            self.seen = len(self.dataloader.dataset)
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            dist.gather_object(self.small_metrics.stats, None, dst=0)
            dist.gather_object({"small_seen": self.small_seen, "small_instances": self.small_instances}, None, dst=0)
            self.metrics.clear_stats()
            self.small_metrics.clear_stats()

    def get_stats(self) -> dict[str, float]:
        """Return native overall metrics plus explicit small-object-only metrics."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        overall = dict(self.metrics.results_dict)
        self.metrics.clear_stats()

        if self.enable_small_object_metrics:
            safe_process_obb_metrics(self.small_metrics, save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
            overall.update(small_object_results_dict(self.small_metrics))
            overall["metrics/images_small(B)"] = float(self.small_seen)
            overall["metrics/instances_small(B)"] = float(self.small_instances)
            self.small_metrics.clear_stats()
        return overall

    def print_results(self) -> None:
        """Print the standard overall metrics and keep small-object details opt-in for terminal clarity."""
        super().print_results()
        if self.enable_small_object_metrics and os.getenv("ULTRALYTICS_SHOW_SMALL_OBJECT_TERMINAL_METRICS") == "1":
            LOGGER.info(
                "small-object subset: "
                f"images={self.small_seen}, "
                f"instances={self.small_instances}, "
                f"P={self.small_metrics.mean_results()[0]:.3g}, "
                f"R={self.small_metrics.mean_results()[1]:.3g}, "
                f"mAP50={self.small_metrics.mean_results()[2]:.3g}, "
                f"mAP50-95={self.small_metrics.mean_results()[3]:.3g}"
            )
