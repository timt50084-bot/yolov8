from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _broadcast_temporal_valid(
    temporal_valid: torch.Tensor | None,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert temporal_valid to both [B] bool and [B,1,1,1] float masks."""
    if temporal_valid is None:
        valid_bool = torch.ones(reference.shape[0], device=reference.device, dtype=torch.bool)
    else:
        if not isinstance(temporal_valid, torch.Tensor):
            temporal_valid = torch.as_tensor(temporal_valid, device=reference.device)
        valid_bool = temporal_valid.to(device=reference.device).bool().view(-1)
        if valid_bool.numel() == 1 and reference.shape[0] > 1:
            valid_bool = valid_bool.expand(reference.shape[0])
    valid_mask = valid_bool.to(dtype=reference.dtype).view(-1, 1, 1, 1)
    return valid_bool, valid_mask


class TemporalStageAdapter(nn.Module):
    """Project the previous RGB frame into a lightweight stage-aligned feature map.

    Stage 5 intentionally stays in a two-frame regime: current-frame RGB is the main path and the previous RGB frame is
    compressed into a shallow context feature at selected stages. This avoids the complexity of long-sequence memory or
    video-transformer style models while still letting the detector consume `img_prev` in a real feature path.
    """

    def __init__(self, out_channels: int, width_mult: float = 0.25) -> None:
        super().__init__()
        hidden = max(16, int(out_channels * width_mult))
        self.proj = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, img_prev: torch.Tensor | None, target_size: tuple[int, int]) -> torch.Tensor | None:
        """Return a lightweight previous-frame feature aligned to one selected stage size."""
        if img_prev is None:
            return None
        img_prev = F.adaptive_avg_pool2d(img_prev, target_size)
        return self.proj(img_prev)


class TemporalGatedRefine(nn.Module):
    """Lightweight two-frame feature refine module for Stage 5.

    Supported modes:
    - `diff_gate`: gate(abs(curr - prev)) controls a residual pull toward previous-frame context
    - `gated_add`: gate([curr, prev]) injects previous-frame context additively
    - `align_only` / `none`: bypass current feature unchanged

    The module is intentionally limited to two frames and a small residual scale so that:
    - temporal logic is easy to disable,
    - boundary frames with `temporal_valid=False` can cleanly bypass,
    - deployment can still fall back to single-frame RGB-only inference.
    """

    def __init__(self, channels: int, fusion_type: str = "diff_gate", residual_scale: float = 0.1) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.residual_scale = float(residual_scale)
        self.diff_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.cat_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor | None = None,
        *,
        temporal_valid: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        """Refine current-frame features with lightweight previous-frame context when enabled and valid."""
        if previous is None or not enabled or self.fusion_type in {"none", "align_only"}:
            return current

        _, valid_mask = _broadcast_temporal_valid(temporal_valid, current)
        if self.fusion_type == "gated_add":
            gate = self.cat_gate(torch.cat((current, previous), dim=1))
            refined = current + self.residual_scale * gate * previous
        else:
            aligned = self.align(previous)
            diff = torch.abs(current - aligned)
            gate = self.diff_gate(diff)
            refined = current + self.residual_scale * gate * (aligned - current)

        return current + valid_mask * (refined - current)


def temporal_alignment_loss(
    current: torch.Tensor | None,
    previous: torch.Tensor | None,
    temporal_valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Small train-time cosine consistency loss between current and previous-frame stage descriptors."""
    if current is None or previous is None:
        raise ValueError("temporal_alignment_loss expects both current and previous features.")
    valid_bool, _ = _broadcast_temporal_valid(temporal_valid, current)
    if not valid_bool.any():
        return current.new_zeros(())
    current_vec = F.normalize(current.mean(dim=(2, 3)), dim=1, eps=eps)
    previous_vec = F.normalize(previous.mean(dim=(2, 3)), dim=1, eps=eps)
    loss = 1.0 - (current_vec * previous_vec).sum(dim=1)
    return loss[valid_bool].mean() if valid_bool.any() else current.new_zeros(())
