from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.modules.rgbir_fusion import RGBIRGatedFusion, RGBIRStageAdapter, rgbir_alignment_loss
from ultralytics.nn.tasks import OBBModel, yaml_model_load
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import feature_visualization


def _parse_stage_list(raw: Any) -> tuple[int, ...]:
    """Normalize configured fusion stage ids to a sorted tuple of ints."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    stages = sorted({int(x) for x in raw})
    return tuple(stages)


class RGBIRTrainAssistOBBModel(OBBModel):
    """Training-time RGB-IR cooperative OBB model with RGB-only deployment behavior.

    Stage 3 keeps the native RGB OBB path as the only inference/deployment path. During training, selected RGB feature
    stages can receive lightweight IR assistance and a small alignment loss. During validation and prediction, the IR
    branch is bypassed automatically so only `img` is required. `img_prev` is intentionally ignored in Stage 3 and
    remains reserved for the later temporal stage.
    """

    def __init__(self, cfg="yolov8-rgbir-obb.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True) -> None:
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        self.use_rgbir_train_assist = bool(cfg.get("use_rgbir_train_assist", False))
        self.ir_train_only = bool(cfg.get("ir_train_only", True))
        self.fusion_type = str(cfg.get("fusion_type", "gated_add"))
        self.ir_branch_width = float(cfg.get("ir_branch_width", 0.25))
        self.rgbir_aux_loss_weight = float(cfg.get("rgbir_aux_loss_weight", 0.05))
        self.rgbir_residual_scale = float(cfg.get("rgbir_residual_scale", 0.1))
        self.ir_feature_stages = _parse_stage_list(cfg.get("ir_feature_stages", [9]))
        self.ignore_img_prev = True
        self.last_assist_used = False
        self.last_assist_stage_ids: tuple[int, ...] = ()
        self._last_rgbir_aux_loss: torch.Tensor | None = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self._stage_channels = self._infer_stage_channels(self.ir_feature_stages) if self.ir_feature_stages else {}
        self.ir_stage_adapters = nn.ModuleDict(
            {
                str(stage): RGBIRStageAdapter(out_channels=channels, width_mult=self.ir_branch_width)
                for stage, channels in self._stage_channels.items()
            }
        )
        self.rgbir_fusions = nn.ModuleDict(
            {
                str(stage): RGBIRGatedFusion(
                    channels=channels,
                    fusion_type=self.fusion_type,
                    residual_scale=self.rgbir_residual_scale,
                )
                for stage, channels in self._stage_channels.items()
            }
        )

    def _infer_stage_channels(self, stage_ids: tuple[int, ...]) -> dict[int, int]:
        """Run one dummy RGB forward pass to infer selected stage channel counts."""
        captured: dict[int, int] = {}
        hooks = []
        was_training = self.training

        def _make_hook(stage_id: int):
            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"RGB-IR assist stage {stage_id} must output a tensor, but got {type(output).__name__}.")
                captured[stage_id] = int(output.shape[1])

            return _hook

        try:
            for stage_id in stage_ids:
                if stage_id < 0 or stage_id >= len(self.model):
                    raise ValueError(f"RGB-IR assist stage index {stage_id} is out of range for model length {len(self.model)}.")
                hooks.append(self.model[stage_id].register_forward_hook(_make_hook(stage_id)))
            self.eval()
            dummy = torch.zeros(1, self.yaml.get("channels", 3), 256, 256)
            with torch.no_grad():
                super().predict(dummy)
        finally:
            for hook in hooks:
                hook.remove()
            if was_training:
                self.train()

        missing = [stage_id for stage_id in stage_ids if stage_id not in captured]
        if missing:
            raise ValueError(f"Failed to infer RGB-IR assist stage channels for {missing}.")
        return captured

    def _assist_requested(self, img_ir: torch.Tensor | None) -> bool:
        """Return True when Stage 3 train-time RGB-IR assist should run."""
        return (
            self.training
            and self.use_rgbir_train_assist
            and img_ir is not None
            and len(self.ir_feature_stages) > 0
            and (self.fusion_type != "none" or self.rgbir_aux_loss_weight > 0.0)
        )

    @staticmethod
    def _from_saved_outputs(y: list[Any], x: Any, f: int | list[int]) -> Any:
        """Resolve a parse_model `from` reference the same way as the native predictor loop."""
        if isinstance(f, int):
            return y[f] if f != -1 else x
        return [x if j == -1 else y[j] for j in f]

    def _set_aux_state(self, loss: torch.Tensor, used: bool, stages: tuple[int, ...]) -> None:
        """Track the latest assist usage for debugging and smoke checks."""
        self._last_rgbir_aux_loss = loss
        self.last_assist_used = used
        self.last_assist_stage_ids = stages

    def predict(
        self,
        x: torch.Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
        img_ir: torch.Tensor | None = None,
    ) -> Any:
        """Run RGB-only inference by default, with optional training-time IR assistance when explicitly enabled."""
        if augment:
            return self._predict_augment(x)

        if not self._assist_requested(img_ir):
            self._set_aux_state(x.new_zeros(()), False, ())
            return super().predict(x, profile=profile, visualize=visualize, augment=False, embed=embed)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        aux_terms: list[torch.Tensor] = []
        used_stages: list[int] = []
        img_ir = img_ir.to(device=x.device, dtype=x.dtype)

        for m in self.model:
            if m.f != -1:
                x = self._from_saved_outputs(y, x, m.f)
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if m.i in self.ir_feature_stages and isinstance(x, torch.Tensor):
                stage_key = str(m.i)
                ir_stage = self.ir_stage_adapters[stage_key](img_ir, target_size=x.shape[-2:])
                if ir_stage is None:
                    raise RuntimeError(f"RGB-IR assist stage {m.i} could not build an IR feature.")
                aux_terms.append(rgbir_alignment_loss(x, ir_stage))
                x = self.rgbir_fusions[stage_key](
                    x,
                    ir_stage,
                    enabled=self.fusion_type not in {"none", "align_only"},
                )
                used_stages.append(m.i)

            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    self._set_aux_state(
                        torch.stack(aux_terms).mean() if aux_terms else x.new_zeros(()),
                        bool(used_stages),
                        tuple(used_stages),
                    )
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        self._set_aux_state(torch.stack(aux_terms).mean() if aux_terms else x.new_zeros(()), bool(used_stages), tuple(used_stages))
        return x

    def loss(self, batch: dict[str, torch.Tensor], preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute standard OBB loss plus a small optional RGB-IR train-time auxiliary alignment term."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.predict(batch["img"], img_ir=batch.get("img_ir"))
        else:
            self._set_aux_state(batch["img"].new_zeros(()), False, ())

        base_loss, base_items = self.criterion(preds, batch)
        if self.training and self.use_rgbir_train_assist and batch.get("img_ir") is not None and self._last_rgbir_aux_loss is not None:
            weighted_aux = self._last_rgbir_aux_loss * self.rgbir_aux_loss_weight
        else:
            weighted_aux = base_loss[:1] * 0.0
        total_items = torch.cat((base_loss, weighted_aux.view(1)))
        return total_items, torch.cat((base_items, weighted_aux.detach().view(1)))

    def log_assist_configuration(self) -> None:
        """Emit one concise line describing the active Stage 3 RGB-IR assist settings."""
        LOGGER.info(
            "RGB-IR train assist: "
            f"enabled={self.use_rgbir_train_assist}, "
            f"fusion_type={self.fusion_type}, "
            f"stages={list(self.ir_feature_stages)}, "
            f"aux_weight={self.rgbir_aux_loss_weight}"
        )
