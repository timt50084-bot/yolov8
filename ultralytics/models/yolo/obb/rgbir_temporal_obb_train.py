from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from ultralytics.nn.modules.temporal_refine import TemporalGatedRefine, TemporalStageAdapter, temporal_alignment_loss
from ultralytics.nn.modules.rgbir_fusion import rgbir_alignment_loss
from ultralytics.utils.loss_small_object import apply_small_object_loss_weighting
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import feature_visualization

from .rgbir_obb_train import _parse_stage_list
from .rgbir_small_obb_train import RGBIRSmallObjectOBBModel


class RGBIRTemporalOBBModel(RGBIRSmallObjectOBBModel):
    """Stage 5 explicit RGB-IR + small-object + lightweight two-frame temporal OBB model.

    Current-frame RGB remains the deployable main path. When explicitly enabled, Stage 5 consumes `img_prev` through a
    lightweight high-level temporal refine branch at selected stages. This is intentionally limited to one previous
    frame so the model can still:
    - train with `img_prev`,
    - validate and predict in RGB-only mode when temporal context is unavailable,
    - avoid long-sequence memory or tracking semantics, which remain reserved for later stages.
    """

    def __init__(self, cfg="yolov8-rgbir-temporal-obb.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True) -> None:
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else cfg
        self.use_temporal = bool(cfg.get("use_temporal", False))
        self.temporal_mode = str(cfg.get("temporal_mode", "off"))
        self.temporal_fusion_type = str(cfg.get("temporal_fusion_type", "diff_gate"))
        self.temporal_branch_width = float(cfg.get("temporal_branch_width", 0.25))
        self.temporal_feature_stages = _parse_stage_list(cfg.get("temporal_feature_stages", [6, 9]))
        self.temporal_loss_weight = float(cfg.get("temporal_loss_weight", 0.02))
        self.temporal_residual_scale = float(cfg.get("temporal_residual_scale", 0.1))
        self.last_temporal_used = False
        self.last_temporal_stage_ids: tuple[int, ...] = ()
        self._last_temporal_aux_loss: torch.Tensor | None = None
        self._temporal_prev_cache: torch.Tensor | None = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self._temporal_stage_channels = self._infer_stage_channels(self.temporal_feature_stages) if self.temporal_feature_stages else {}
        self.temporal_stage_adapters = torch.nn.ModuleDict(
            {
                str(stage): TemporalStageAdapter(out_channels=channels, width_mult=self.temporal_branch_width)
                for stage, channels in self._temporal_stage_channels.items()
            }
        )
        self.temporal_refiners = torch.nn.ModuleDict(
            {
                str(stage): TemporalGatedRefine(
                    channels=channels,
                    fusion_type=self.temporal_fusion_type,
                    residual_scale=self.temporal_residual_scale,
                )
                for stage, channels in self._temporal_stage_channels.items()
            }
        )

    @staticmethod
    def _coerce_temporal_valid(x: torch.Tensor, temporal_valid: torch.Tensor | None) -> torch.Tensor:
        """Normalize temporal_valid to a batch-aligned bool tensor."""
        if temporal_valid is None:
            return torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        if not isinstance(temporal_valid, torch.Tensor):
            temporal_valid = torch.as_tensor(temporal_valid, device=x.device)
        temporal_valid = temporal_valid.to(device=x.device).bool().view(-1)
        if temporal_valid.numel() == 1 and x.shape[0] > 1:
            temporal_valid = temporal_valid.expand(x.shape[0])
        return temporal_valid

    def _temporal_requested(self, img_prev: torch.Tensor | None, temporal_valid: torch.Tensor | None) -> bool:
        """Return True when Stage 5 temporal refine should run for this forward."""
        if (
            not self.use_temporal
            or self.temporal_mode in {"off", "none"}
            or img_prev is None
            or len(self.temporal_feature_stages) == 0
            or (self.temporal_fusion_type == "none" and self.temporal_loss_weight <= 0.0)
        ):
            return False
        if temporal_valid is None:
            return True
        if not isinstance(temporal_valid, torch.Tensor):
            temporal_valid = torch.as_tensor(temporal_valid)
        return bool(temporal_valid.bool().any().item())

    def _set_temporal_state(self, loss: torch.Tensor, used: bool, stages: tuple[int, ...]) -> None:
        """Track the latest temporal usage for smoke checks and debug logging."""
        self._last_temporal_aux_loss = loss
        self.last_temporal_used = used
        self.last_temporal_stage_ids = stages

    def reset_temporal_cache(self) -> None:
        """Reset the lightweight one-step inference cache used by the Stage 5 demo script."""
        self._temporal_prev_cache = None

    def predict_with_prev_cache(
        self,
        x: torch.Tensor,
        *,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
    ) -> Any:
        """Run one-step temporal inference using an internal previous-frame cache.

        The first frame falls back to single-frame RGB-only prediction. Each successful call caches the current RGB
        tensor as the next frame's previous context.
        """
        img_prev = self._temporal_prev_cache
        temporal_valid = None if img_prev is None else x.new_ones((x.shape[0],), dtype=torch.bool)
        output = self.predict(
            x,
            profile=profile,
            visualize=visualize,
            augment=augment,
            embed=embed,
            img_prev=img_prev,
            temporal_valid=temporal_valid,
        )
        self._temporal_prev_cache = x.detach()
        return output

    def predict(
        self,
        x: torch.Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
        img_ir: torch.Tensor | None = None,
        img_prev: torch.Tensor | None = None,
        temporal_valid: torch.Tensor | None = None,
    ) -> Any:
        """Run RGB-only prediction by default, with optional Stage 3 IR assist and Stage 5 temporal refine."""
        if augment:
            return self._predict_augment(x)

        zero_ref = x
        use_ir = self._assist_requested(img_ir)
        use_temporal = self._temporal_requested(img_prev, temporal_valid)
        if not use_ir and not use_temporal:
            self._set_aux_state(zero_ref.new_zeros(()), False, ())
            self._set_temporal_state(zero_ref.new_zeros(()), False, ())
            return super().predict(x, profile=profile, visualize=visualize, augment=False, embed=embed, img_ir=img_ir)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        rgbir_aux_terms: list[torch.Tensor] = []
        temporal_aux_terms: list[torch.Tensor] = []
        used_ir_stages: list[int] = []
        used_temporal_stages: list[int] = []

        temporal_valid_tensor = self._coerce_temporal_valid(x, temporal_valid) if use_temporal else None
        if use_ir:
            img_ir = img_ir.to(device=x.device, dtype=x.dtype)
        if use_temporal:
            img_prev = img_prev.to(device=x.device, dtype=x.dtype)

        for m in self.model:
            if m.f != -1:
                x = self._from_saved_outputs(y, x, m.f)
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if use_ir and m.i in self.ir_feature_stages and isinstance(x, torch.Tensor):
                stage_key = str(m.i)
                ir_stage = self.ir_stage_adapters[stage_key](img_ir, target_size=x.shape[-2:])
                if ir_stage is None:
                    raise RuntimeError(f"RGB-IR assist stage {m.i} could not build an IR feature.")
                rgbir_aux_terms.append(rgbir_alignment_loss(x, ir_stage))
                x = self.rgbir_fusions[stage_key](
                    x,
                    ir_stage,
                    enabled=self.fusion_type not in {"none", "align_only"},
                )
                used_ir_stages.append(m.i)

            if use_temporal and m.i in self.temporal_feature_stages and isinstance(x, torch.Tensor):
                stage_key = str(m.i)
                prev_stage = self.temporal_stage_adapters[stage_key](img_prev, target_size=x.shape[-2:])
                if prev_stage is None:
                    raise RuntimeError(f"Temporal refine stage {m.i} could not build a previous-frame feature.")
                if self.temporal_loss_weight > 0.0:
                    temporal_aux_terms.append(temporal_alignment_loss(x, prev_stage, temporal_valid=temporal_valid_tensor))
                x = self.temporal_refiners[stage_key](
                    x,
                    prev_stage,
                    temporal_valid=temporal_valid_tensor,
                    enabled=self.temporal_fusion_type not in {"none", "align_only"},
                )
                used_temporal_stages.append(m.i)

            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    self._set_aux_state(
                        torch.stack(rgbir_aux_terms).mean() if rgbir_aux_terms else x.new_zeros(()),
                        bool(used_ir_stages),
                        tuple(used_ir_stages),
                    )
                    self._set_temporal_state(
                        torch.stack(temporal_aux_terms).mean() if temporal_aux_terms else x.new_zeros(()),
                        bool(used_temporal_stages),
                        tuple(used_temporal_stages),
                    )
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        self._set_aux_state(
            torch.stack(rgbir_aux_terms).mean() if rgbir_aux_terms else zero_ref.new_zeros(()),
            bool(used_ir_stages),
            tuple(used_ir_stages),
        )
        self._set_temporal_state(
            torch.stack(temporal_aux_terms).mean() if temporal_aux_terms else zero_ref.new_zeros(()),
            bool(used_temporal_stages),
            tuple(used_temporal_stages),
        )
        return x

    def loss(self, batch: dict[str, torch.Tensor], preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Stage 4 loss items plus one optional low-weight temporal consistency term."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.predict(
                batch["img"],
                img_ir=batch.get("img_ir"),
                img_prev=batch.get("img_prev"),
                temporal_valid=batch.get("temporal_valid"),
            )

        base_loss, base_items = self.criterion(preds, batch)
        if self.training and self.use_small_object_loss_weighting:
            weighted_base_loss, stats = apply_small_object_loss_weighting(
                base_loss,
                batch,
                enabled=True,
                area_thr_norm=self.small_object_area_thr_norm,
                gain=self.small_object_loss_gain,
                loss_on=self.small_object_loss_on,
            )
            self.last_small_object_scale = stats["scale"]
            self.last_small_object_stats = stats
            weighted_base_items = base_items.clone()
            weighted_base_items[:4] = base_items[:4] * float(stats["scale"])
        else:
            weighted_base_loss = base_loss
            weighted_base_items = base_items
            self.last_small_object_scale = 1.0
            self.last_small_object_stats = {
                "small_count": 0.0,
                "target_count": 0.0,
                "small_ratio": 0.0,
                "severity_mean": 0.0,
                "scale": 1.0,
            }

        weighted_rgbir = (
            self._last_rgbir_aux_loss * self.rgbir_aux_loss_weight
            if self.training
            and self.use_rgbir_train_assist
            and batch.get("img_ir") is not None
            and self._last_rgbir_aux_loss is not None
            else weighted_base_loss[:1] * 0.0
        )
        weighted_temporal = (
            self._last_temporal_aux_loss * self.temporal_loss_weight
            if self.training
            and self.use_temporal
            and batch.get("img_prev") is not None
            and self._last_temporal_aux_loss is not None
            else weighted_base_loss[:1] * 0.0
        )
        return (
            torch.cat((weighted_base_loss, weighted_rgbir.view(1), weighted_temporal.view(1))),
            torch.cat((weighted_base_items, weighted_rgbir.detach().view(1), weighted_temporal.detach().view(1))),
        )

    def log_temporal_configuration(self) -> None:
        """Emit one concise line describing the active Stage 5 temporal settings."""
        LOGGER.info(
            "Stage 5 temporal refine: "
            f"enabled={self.use_temporal}, "
            f"mode={self.temporal_mode}, "
            f"fusion_type={self.temporal_fusion_type}, "
            f"stages={list(self.temporal_feature_stages)}, "
            f"loss_weight={self.temporal_loss_weight}"
        )
