"""Utilities for lightweight OBB-aware association in the explicit Stage 6 UAV tracker.

This module is intentionally independent from the native ByteTrack/BoT-SORT matching utilities so Stage 6 can remain
an opt-in path. The matcher prioritizes oriented-box overlap, then refines candidates with normalized center distance
and an optional lightweight appearance term.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from ultralytics.utils.metrics import batch_probiou

LARGE_COST = 1e6

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except Exception:  # pragma: no cover - scipy may be unavailable in some lightweight environments
    scipy_linear_sum_assignment = None


def _as_numpy_float(item: Any) -> np.ndarray:
    """Convert a tensor-like item to a float32 numpy array without copying more than necessary."""
    if isinstance(item, np.ndarray):
        return item.astype(np.float32, copy=False)
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(item, dtype=np.float32)


def _extract_obb(item: Any) -> np.ndarray:
    """Extract an `xywhr` vector from a detection/track-like object."""
    if hasattr(item, "predicted_obb"):
        obb = getattr(item, "predicted_obb")
    elif hasattr(item, "obb"):
        obb = getattr(item, "obb")
    elif isinstance(item, dict):
        obb = item.get("predicted_obb", item.get("obb"))
    else:
        obb = item
    arr = _as_numpy_float(obb).reshape(-1)
    if arr.size < 5:
        raise ValueError(f"Expected an xywhr-like vector with at least 5 values, but got shape {arr.shape}.")
    arr = arr[:5].copy()
    arr[2:4] = np.maximum(arr[2:4], 1e-3)
    return arr


def _extract_cls(item: Any) -> int:
    """Extract an integer class id from a detection/track-like object when available."""
    if hasattr(item, "cls"):
        return int(getattr(item, "cls"))
    if isinstance(item, dict) and "cls" in item:
        return int(item["cls"])
    arr = _as_numpy_float(item).reshape(-1)
    return int(arr[-1]) if arr.size >= 7 else -1


def _extract_embedding(item: Any) -> np.ndarray | None:
    """Extract a lightweight appearance embedding when present."""
    emb = None
    if hasattr(item, "smooth_embedding"):
        emb = getattr(item, "smooth_embedding")
    elif hasattr(item, "embedding"):
        emb = getattr(item, "embedding")
    elif isinstance(item, dict):
        emb = item.get("embedding")
    if emb is None:
        return None
    emb = _as_numpy_float(emb).reshape(-1)
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 1e-8 else emb


def obb_iou_matrix(atracks: Sequence[Any], btracks: Sequence[Any]) -> np.ndarray:
    """Compute a pairwise OBB overlap matrix using Ultralytics' probabilistic OBB IoU helper."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    a = torch.as_tensor(np.stack([_extract_obb(item) for item in atracks]), dtype=torch.float32)
    b = torch.as_tensor(np.stack([_extract_obb(item) for item in btracks]), dtype=torch.float32)
    return batch_probiou(a, b).cpu().numpy().astype(np.float32, copy=False)


def center_distance_matrix(
    atracks: Sequence[Any],
    btracks: Sequence[Any],
    max_center_distance: float = 3.0,
) -> np.ndarray:
    """Compute a normalized center-distance cost matrix between OBBs."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    a = np.stack([_extract_obb(item) for item in atracks]).astype(np.float32, copy=False)
    b = np.stack([_extract_obb(item) for item in btracks]).astype(np.float32, copy=False)
    a_centers, b_centers = a[:, :2], b[:, :2]
    a_diag = np.sqrt((a[:, 2] ** 2) + (a[:, 3] ** 2))[:, None]
    b_diag = np.sqrt((b[:, 2] ** 2) + (b[:, 3] ** 2))[None, :]
    norm = np.maximum((a_diag + b_diag) * 0.5, 1e-6)
    dists = np.linalg.norm(a_centers[:, None, :] - b_centers[None, :, :], axis=-1) / norm
    if max_center_distance > 0:
        dists = np.clip(dists / max_center_distance, 0.0, 1.0)
    return dists.astype(np.float32, copy=False)


def appearance_distance_matrix(atracks: Sequence[Any], btracks: Sequence[Any]) -> np.ndarray:
    """Compute a conservative cosine-distance matrix for lightweight appearance embeddings."""
    cost = np.full((len(atracks), len(btracks)), 0.5, dtype=np.float32)  # neutral default when embeddings are absent
    if len(atracks) == 0 or len(btracks) == 0:
        return cost

    a_embeddings = [_extract_embedding(item) for item in atracks]
    b_embeddings = [_extract_embedding(item) for item in btracks]
    if not any(emb is not None for emb in a_embeddings) or not any(emb is not None for emb in b_embeddings):
        return cost

    for i, a_emb in enumerate(a_embeddings):
        if a_emb is None:
            continue
        for j, b_emb in enumerate(b_embeddings):
            if b_emb is None:
                continue
            sim = float(np.clip(np.dot(a_emb, b_emb), -1.0, 1.0))
            cost[i, j] = 0.5 * (1.0 - sim)
    return cost


def class_penalty_matrix(atracks: Sequence[Any], btracks: Sequence[Any], penalty: float = 0.75) -> np.ndarray:
    """Return an additive cost penalty when classes mismatch."""
    if len(atracks) == 0 or len(btracks) == 0 or penalty <= 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    a_cls = np.asarray([_extract_cls(item) for item in atracks], dtype=np.int64)
    b_cls = np.asarray([_extract_cls(item) for item in btracks], dtype=np.int64)
    return (a_cls[:, None] != b_cls[None, :]).astype(np.float32) * float(penalty)


def build_obb_cost_matrix(
    atracks: Sequence[Any],
    btracks: Sequence[Any],
    *,
    iou_weight: float = 0.7,
    center_weight: float = 0.3,
    appearance_weight: float = 0.0,
    class_mismatch_penalty: float = 0.75,
    match_iou_thresh: float = 0.15,
    max_center_distance: float = 3.0,
    use_appearance: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a conservative association cost matrix and expose its components for debugging/tests."""
    if len(atracks) == 0 or len(btracks) == 0:
        empty = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
        return empty, {
            "cost": empty,
            "iou": empty,
            "center": empty,
            "appearance": empty,
            "class_penalty": empty,
        }

    iou = obb_iou_matrix(atracks, btracks)
    iou_cost = 1.0 - iou
    center_cost = center_distance_matrix(atracks, btracks, max_center_distance=max_center_distance)
    appearance_cost = (
        appearance_distance_matrix(atracks, btracks) if use_appearance and appearance_weight > 0 else np.zeros_like(iou)
    )
    class_cost = class_penalty_matrix(atracks, btracks, penalty=class_mismatch_penalty)

    weight_sum = max(float(iou_weight) + float(center_weight) + (float(appearance_weight) if use_appearance else 0.0), 1e-6)
    cost = (
        float(iou_weight) * iou_cost
        + float(center_weight) * center_cost
        + (float(appearance_weight) * appearance_cost if use_appearance else 0.0)
    ) / weight_sum
    cost = cost + class_cost

    if match_iou_thresh > 0:
        cost = np.where(iou >= float(match_iou_thresh), cost, LARGE_COST)

    return cost.astype(np.float32, copy=False), {
        "cost": cost.astype(np.float32, copy=False),
        "iou": iou.astype(np.float32, copy=False),
        "center": center_cost.astype(np.float32, copy=False),
        "appearance": appearance_cost.astype(np.float32, copy=False),
        "class_penalty": class_cost.astype(np.float32, copy=False),
    }


def linear_assignment(cost_matrix: np.ndarray, cost_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a linear assignment with a scipy path and a greedy fallback."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(cost_matrix.shape[0], dtype=np.int64),
            np.arange(cost_matrix.shape[1], dtype=np.int64),
        )

    if scipy_linear_sum_assignment is not None:
        row_ind, col_ind = scipy_linear_sum_assignment(cost_matrix)
        matches = []
        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_matrix[r, c] <= cost_thresh and cost_matrix[r, c] < LARGE_COST:
                matches.append((r, c))
                assigned_rows.add(r)
                assigned_cols.add(c)
        unmatched_rows = np.asarray([i for i in range(cost_matrix.shape[0]) if i not in assigned_rows], dtype=np.int64)
        unmatched_cols = np.asarray([i for i in range(cost_matrix.shape[1]) if i not in assigned_cols], dtype=np.int64)
        return np.asarray(matches, dtype=np.int64), unmatched_rows, unmatched_cols

    candidates = np.argwhere(cost_matrix <= cost_thresh)
    if candidates.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(cost_matrix.shape[0], dtype=np.int64),
            np.arange(cost_matrix.shape[1], dtype=np.int64),
        )

    order = np.argsort(cost_matrix[candidates[:, 0], candidates[:, 1]])
    matches: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for idx in order.tolist():
        r, c = candidates[idx].tolist()
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    unmatched_rows = np.asarray([i for i in range(cost_matrix.shape[0]) if i not in used_rows], dtype=np.int64)
    unmatched_cols = np.asarray([i for i in range(cost_matrix.shape[1]) if i not in used_cols], dtype=np.int64)
    return np.asarray(matches, dtype=np.int64), unmatched_rows, unmatched_cols


def match_obb_tracks(
    atracks: Sequence[Any],
    btracks: Sequence[Any],
    *,
    match_thresh: float = 0.85,
    iou_weight: float = 0.7,
    center_weight: float = 0.3,
    appearance_weight: float = 0.0,
    class_mismatch_penalty: float = 0.75,
    match_iou_thresh: float = 0.15,
    max_center_distance: float = 3.0,
    use_appearance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """High-level helper that builds the cost matrix and solves the assignment."""
    cost_matrix, diagnostics = build_obb_cost_matrix(
        atracks,
        btracks,
        iou_weight=iou_weight,
        center_weight=center_weight,
        appearance_weight=appearance_weight,
        class_mismatch_penalty=class_mismatch_penalty,
        match_iou_thresh=match_iou_thresh,
        max_center_distance=max_center_distance,
        use_appearance=use_appearance,
    )
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, cost_thresh=match_thresh)
    diagnostics = {**diagnostics, "matches": matches}
    return matches, unmatched_a, unmatched_b, diagnostics


def obb_iou_distance(atracks: Sequence[Any], btracks: Sequence[Any]) -> np.ndarray:
    """Compatibility helper returning `1 - OBB IoU`, mirroring the native tracker utility naming style."""
    return 1.0 - obb_iou_matrix(atracks, btracks)
