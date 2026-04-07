# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBIRStageAdapter(nn.Module):
    """Project the raw IR image into a lightweight stage-aligned feature map.

    The adapter intentionally stays small: it pools IR to the target spatial size and applies a shallow projection so
    Stage 3 can consume `img_ir` during training without rewriting the RGB deployment backbone.
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

    def forward(self, ir: torch.Tensor | None, target_size: tuple[int, int]) -> torch.Tensor | None:
        """Return a lightweight IR feature aligned to the requested stage size."""
        if ir is None:
            return None
        ir = F.adaptive_avg_pool2d(ir, target_size)
        return self.proj(ir)


class RGBIRGatedFusion(nn.Module):
    """Lightweight gated residual fusion for training-time RGB-IR cooperation.

    Supported modes:
    - `gated_add`: RGB + sigmoid(gate([RGB, IR])) * IR
    - `weighted_sum`: RGB + sigmoid(weight) * IR
    - `align_only` / `none`: bypass RGB unchanged
    """

    def __init__(self, channels: int, fusion_type: str = "gated_add", residual_scale: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.fusion_type = fusion_type
        self.residual_scale = float(residual_scale)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor | None = None, enabled: bool = True) -> torch.Tensor:
        """Fuse aligned IR features into RGB in a conservative residual manner."""
        if ir is None or not enabled or self.fusion_type in {"none", "align_only"}:
            return rgb
        if self.fusion_type == "weighted_sum":
            return rgb + self.residual_scale * torch.sigmoid(self.weight) * ir
        gate = self.gate(torch.cat((rgb, ir), dim=1))
        return rgb + self.residual_scale * gate * ir


def rgbir_alignment_loss(rgb: torch.Tensor | None, ir: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    """Cosine alignment loss between global RGB and IR stage descriptors.

    This is intentionally small and train-only. It constrains the RGB representation to stay compatible with IR cues
    without making IR a deployment dependency.
    """
    if rgb is None or ir is None:
        raise ValueError("rgbir_alignment_loss expects both rgb and ir features.")
    rgb_vec = F.normalize(rgb.mean(dim=(2, 3)), dim=1, eps=eps)
    ir_vec = F.normalize(ir.mean(dim=(2, 3)), dim=1, eps=eps)
    return (1.0 - (rgb_vec * ir_vec).sum(dim=1)).mean()
