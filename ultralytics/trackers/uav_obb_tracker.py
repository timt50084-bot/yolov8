"""Explicit Stage 6 UAV OBB tracking-by-detection implementation.

This tracker is intentionally kept outside the native `TRACKER_MAP` so baseline `yolo track` remains untouched.
Stage 6 focuses on a lightweight RGB-only deployment path:
- detections come from a single-frame or temporal detector,
- association is OBB-aware,
- state management supports tentative/tracked/lost/removed,
- short-term reactivation is supported without adding a heavy standalone ReID network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.ops import xywhr2xyxyxyxy

from .utils.obb_matching import match_obb_tracks


class UAVTrackState(str, Enum):
    """Discrete states used by the explicit Stage 6 tracker."""

    TENTATIVE = "tentative"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class UAVDetection:
    """Container for one detection observation consumed by the tracker."""

    obb: np.ndarray
    score: float
    cls: int
    det_idx: int
    embedding: np.ndarray | None = None


@dataclass
class UAVTrack:
    """Internal track state for the explicit Stage 6 tracker."""

    track_id: int
    obb: np.ndarray
    score: float
    cls: int
    frame_id: int
    start_frame: int
    state: UAVTrackState = UAVTrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    embedding: np.ndarray | None = None
    smooth_embedding: np.ndarray | None = None
    last_detection_idx: int = -1
    history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=2))

    def __post_init__(self) -> None:
        self.obb = np.asarray(self.obb, dtype=np.float32).reshape(5)
        self.obb[2:4] = np.maximum(self.obb[2:4], 1e-3)
        self.history.append(self.obb.copy())
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
            self.smooth_embedding = _normalize_embedding(self.embedding)

    @property
    def predicted_obb(self) -> np.ndarray:
        """Return the light motion-projected OBB used for association."""
        pred = self.obb.copy()
        if self.state in {UAVTrackState.TRACKED, UAVTrackState.LOST}:
            pred[:2] += self.velocity
        return pred

    def predict(self) -> None:
        """Advance the track age and carry its short-term motion prior forward."""
        self.age += 1
        self.time_since_update += 1

    def _update_embedding(self, embedding: np.ndarray | None, momentum: float) -> None:
        """Update the optional lightweight appearance descriptor with EMA smoothing."""
        if embedding is None:
            return
        embedding = _normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))
        self.embedding = embedding
        if self.smooth_embedding is None:
            self.smooth_embedding = embedding
        else:
            smooth = (float(momentum) * self.smooth_embedding) + ((1.0 - float(momentum)) * embedding)
            self.smooth_embedding = _normalize_embedding(smooth)

    def update(self, detection: UAVDetection, frame_id: int, *, appearance_momentum: float, min_confirm_frames: int) -> None:
        """Update an existing track with a matched detection."""
        prev_center = self.obb[:2].copy()
        self.obb = np.asarray(detection.obb, dtype=np.float32).reshape(5)
        self.obb[2:4] = np.maximum(self.obb[2:4], 1e-3)
        new_velocity = self.obb[:2] - prev_center
        self.velocity = (0.7 * self.velocity) + (0.3 * new_velocity)
        self.score = float(detection.score)
        self.cls = int(detection.cls)
        self.frame_id = int(frame_id)
        self.time_since_update = 0
        self.hits += 1
        self.last_detection_idx = int(detection.det_idx)
        self.history.append(self.obb.copy())
        self._update_embedding(detection.embedding, momentum=appearance_momentum)
        self.state = UAVTrackState.TRACKED if self.hits >= int(min_confirm_frames) else UAVTrackState.TENTATIVE

    def reactivate(self, detection: UAVDetection, frame_id: int, *, appearance_momentum: float, min_confirm_frames: int) -> None:
        """Reactivate a lost track with a newly matched detection without changing its ID."""
        self.update(
            detection,
            frame_id,
            appearance_momentum=appearance_momentum,
            min_confirm_frames=min_confirm_frames,
        )
        self.state = UAVTrackState.TRACKED

    def mark_lost(self) -> None:
        """Mark the track as temporarily lost."""
        self.state = UAVTrackState.LOST

    def mark_removed(self) -> None:
        """Mark the track as removed and no longer eligible for matching."""
        self.state = UAVTrackState.REMOVED

    def to_array(self) -> np.ndarray:
        """Return the track in Ultralytics OBB tracking tensor layout `[xywhr, track_id, score, cls]`."""
        return np.asarray([*self.obb.tolist(), float(self.track_id), float(self.score), float(self.cls)], dtype=np.float32)

    def to_dict(self, frame_id: int) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the explicit Stage 6 script."""
        return {
            "frame_id": int(frame_id),
            "track_id": int(self.track_id),
            "class_id": int(self.cls),
            "score": float(self.score),
            "obb": [float(x) for x in self.obb.tolist()],
            "state": self.state.value,
            "age": int(self.age),
            "hits": int(self.hits),
            "time_since_update": int(self.time_since_update),
        }


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding while preserving zero vectors."""
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 1e-8 else embedding


def _obb_to_xyxy(obb: np.ndarray) -> tuple[int, int, int, int]:
    """Convert one `xywhr` OBB into a clipped axis-aligned crop box."""
    points = xywhr2xyxyxyxy(torch.as_tensor(obb[None, :], dtype=torch.float32)).view(4, 2).cpu().numpy()
    x1, y1 = np.floor(points.min(axis=0)).astype(np.int32)
    x2, y2 = np.ceil(points.max(axis=0)).astype(np.int32)
    return int(x1), int(y1), int(x2), int(y2)


def extract_lightweight_appearance_embeddings(
    image_rgb: np.ndarray,
    obbs: np.ndarray,
    *,
    bins: int = 8,
) -> np.ndarray:
    """Build conservative RGB crop descriptors without introducing a standalone ReID network."""
    obbs = np.asarray(obbs, dtype=np.float32)
    if obbs.ndim == 1:
        obbs = obbs[None, :]
    embeddings = np.zeros((len(obbs), (bins * 3) + 6), dtype=np.float32)
    h, w = image_rgb.shape[:2]
    for i, obb in enumerate(obbs):
        x1, y1, x2, y2 = _obb_to_xyxy(obb[:5])
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = crop.reshape(-1, 3)
        parts = []
        for c in range(3):
            hist, _ = np.histogram(crop[:, c], bins=bins, range=(0, 255), density=True)
            parts.append(hist.astype(np.float32))
        mean = crop.mean(axis=0).astype(np.float32) / 255.0
        std = crop.std(axis=0).astype(np.float32) / 255.0
        emb = np.concatenate([*parts, mean, std], axis=0).astype(np.float32, copy=False)
        embeddings[i] = _normalize_embedding(emb)
    return embeddings


class UAVOBBTracker:
    """Lightweight explicit UAV OBB tracker with OBB-aware matching and short-term reactivation.

    The tracker is designed for Stage 6 only. It consumes RGB-only detector outputs, optionally uses conservative
    crop-based appearance descriptors, and keeps all state inside the tracker object so sequence boundaries can be
    controlled explicitly via `reset()`.
    """

    def __init__(self, args: Any | None = None, frame_rate: int = 30) -> None:
        if isinstance(args, (str, bytes)):
            args = IterableSimpleNamespace(**YAML.load(check_yaml(args)))
        elif isinstance(args, dict):
            args = IterableSimpleNamespace(**args)
        self.args = args or IterableSimpleNamespace()
        self.frame_rate = int(frame_rate)
        self.tracker_type = str(getattr(self.args, "tracker_type", "uav_obb_tracker"))
        self.use_obb_matching = bool(getattr(self.args, "use_obb_matching", True))
        self.use_appearance = bool(getattr(self.args, "use_appearance", False))
        self.track_high_thresh = float(getattr(self.args, "track_high_thresh", 0.25))
        self.track_low_thresh = float(getattr(self.args, "track_low_thresh", 0.05))
        self.new_track_thresh = float(getattr(self.args, "new_track_thresh", 0.25))
        self.track_buffer = int(getattr(self.args, "track_buffer", 30))
        self.min_confirm_frames = int(getattr(self.args, "min_confirm_frames", 2))
        self.match_thresh = float(getattr(self.args, "match_thresh", 0.85))
        self.match_iou_thresh = float(getattr(self.args, "match_iou_thresh", 0.15))
        self.iou_weight = float(getattr(self.args, "iou_weight", 0.7))
        self.center_weight = float(getattr(self.args, "center_weight", 0.2))
        self.appearance_weight = float(getattr(self.args, "appearance_weight", 0.1))
        self.class_mismatch_penalty = float(getattr(self.args, "class_mismatch_penalty", 0.75))
        self.max_center_distance = float(getattr(self.args, "max_center_distance", 3.0))
        self.reactivation = bool(getattr(self.args, "reactivation", True))
        self.temporal_hint = bool(getattr(self.args, "temporal_hint", False))
        self.appearance_momentum = float(getattr(self.args, "appearance_momentum", 0.85))
        self.appearance_bins = int(getattr(self.args, "appearance_bins", 8))
        self.last_matches = np.empty((0, 2), dtype=np.int64)
        self.last_diagnostics: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Clear all tracker state so sequence boundaries stay explicit and controllable."""
        self.frame_id = 0
        self.next_track_id = 1
        self.tracked_tracks: list[UAVTrack] = []
        self.lost_tracks: list[UAVTrack] = []
        self.removed_tracks: list[UAVTrack] = []
        self.last_output: list[dict[str, Any]] = []
        self.last_matches = np.empty((0, 2), dtype=np.int64)
        self.last_diagnostics = {}

    def _new_id(self) -> int:
        """Return the next track id local to this tracker instance."""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _create_track(self, detection: UAVDetection, frame_id: int) -> UAVTrack:
        """Instantiate a new track from a matched/unmatched detection."""
        track = UAVTrack(
            track_id=self._new_id(),
            obb=np.asarray(detection.obb, dtype=np.float32).reshape(5),
            score=float(detection.score),
            cls=int(detection.cls),
            frame_id=int(frame_id),
            start_frame=int(frame_id),
            state=UAVTrackState.TRACKED if self.min_confirm_frames <= 1 else UAVTrackState.TENTATIVE,
            hits=1,
            age=1,
            time_since_update=0,
            embedding=detection.embedding,
            last_detection_idx=int(detection.det_idx),
        )
        return track

    def _prepare_detections(
        self,
        detections: Sequence[Any] | np.ndarray,
        *,
        image_rgb: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> list[UAVDetection]:
        """Normalize raw detection inputs into the lightweight tracker detection structure."""
        if detections is None:
            return []
        if isinstance(detections, np.ndarray):
            det_arr = detections.astype(np.float32, copy=False)
            if det_arr.ndim == 1:
                det_arr = det_arr[None, :]
            if det_arr.shape[1] < 7:
                raise ValueError("Expected detections as [x, y, w, h, angle, score, cls].")
            det_arr = det_arr[det_arr[:, 5] >= self.track_low_thresh]
            if det_arr.size == 0:
                return []
            if embeddings is None and image_rgb is not None and self.use_appearance:
                embeddings = extract_lightweight_appearance_embeddings(
                    image_rgb,
                    det_arr[:, :5],
                    bins=self.appearance_bins,
                )
            prepared = []
            for idx, det in enumerate(det_arr):
                emb = None if embeddings is None else embeddings[idx]
                prepared.append(
                    UAVDetection(
                        obb=np.asarray(det[:5], dtype=np.float32),
                        score=float(det[5]),
                        cls=int(det[6]),
                        det_idx=int(idx),
                        embedding=emb,
                    )
                )
            return prepared

        prepared = []
        for idx, det in enumerate(detections):
            if isinstance(det, UAVDetection):
                if det.score >= self.track_low_thresh:
                    prepared.append(det)
                continue
            if not isinstance(det, dict):
                raise TypeError("Detections must be a numpy array, UAVDetection, or dict-like structure.")
            score = float(det.get("score", 0.0))
            if score < self.track_low_thresh:
                continue
            prepared.append(
                UAVDetection(
                    obb=np.asarray(det["obb"], dtype=np.float32).reshape(5),
                    score=score,
                    cls=int(det["cls"]),
                    det_idx=int(det.get("det_idx", idx)),
                    embedding=None if det.get("embedding") is None else np.asarray(det["embedding"], dtype=np.float32),
                )
            )
        return prepared

    @staticmethod
    def _deduplicate_tracks(tracks: Iterable[UAVTrack]) -> list[UAVTrack]:
        """Keep the latest reference per track id while preserving order."""
        dedup: dict[int, UAVTrack] = {}
        for track in tracks:
            dedup[int(track.track_id)] = track
        return list(dedup.values())

    def get_active_tracks(self) -> list[dict[str, Any]]:
        """Return the last emitted visible tracks as JSON-friendly dictionaries."""
        return [track.to_dict(frame_id=self.frame_id) for track in self.tracked_tracks if track.frame_id == self.frame_id]

    def update(
        self,
        detections: Sequence[Any] | np.ndarray,
        image_rgb: np.ndarray | None = None,
        feats: Any | None = None,
        *,
        frame_id: int | None = None,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update tracker state with one frame of detector outputs.

        Args:
            detections: Raw detections in `[x, y, w, h, angle, score, cls]` format or a list of detection dicts.
            image_rgb: Optional RGB frame used to derive lightweight crop descriptors when appearance is enabled.
            feats: Reserved compatibility argument to mirror native tracker signatures. Not used in Stage 6.
            frame_id: Optional explicit frame id. When omitted, the tracker increments its internal counter.
            embeddings: Optional per-detection appearance descriptors.

        Returns:
            np.ndarray: Visible tracks for the current frame in `[x, y, w, h, angle, track_id, score, cls]` format.
        """
        del feats  # Stage 6 keeps detector/tracker coupling minimal.
        self.frame_id = int(frame_id) if frame_id is not None else self.frame_id + 1
        prepared = self._prepare_detections(detections, image_rgb=image_rgb, embeddings=embeddings)

        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        association_pool = list(self.tracked_tracks)
        if self.reactivation:
            association_pool += list(self.lost_tracks)

        if self.use_obb_matching:
            matches, unmatched_track_idx, unmatched_det_idx, diagnostics = match_obb_tracks(
                association_pool,
                prepared,
                match_thresh=self.match_thresh,
                iou_weight=self.iou_weight,
                center_weight=self.center_weight,
                appearance_weight=self.appearance_weight,
                class_mismatch_penalty=self.class_mismatch_penalty,
                match_iou_thresh=self.match_iou_thresh,
                max_center_distance=self.max_center_distance,
                use_appearance=self.use_appearance,
            )
        else:
            matches = np.empty((0, 2), dtype=np.int64)
            unmatched_track_idx = np.arange(len(association_pool), dtype=np.int64)
            unmatched_det_idx = np.arange(len(prepared), dtype=np.int64)
            diagnostics = {"matches": matches}
        self.last_matches = matches
        self.last_diagnostics = diagnostics

        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()
        current_visible: list[UAVTrack] = []
        reactivated: list[UAVTrack] = []

        for track_idx, det_idx in matches.tolist():
            track = association_pool[int(track_idx)]
            det = prepared[int(det_idx)]
            matched_track_ids.add(track.track_id)
            matched_det_ids.add(int(det_idx))
            if track.state == UAVTrackState.LOST:
                track.reactivate(
                    det,
                    self.frame_id,
                    appearance_momentum=self.appearance_momentum,
                    min_confirm_frames=self.min_confirm_frames,
                )
                reactivated.append(track)
            else:
                track.update(
                    det,
                    self.frame_id,
                    appearance_momentum=self.appearance_momentum,
                    min_confirm_frames=self.min_confirm_frames,
                )
                current_visible.append(track)

        new_lost: list[UAVTrack] = []
        removed_now: list[UAVTrack] = []
        for track in self.tracked_tracks:
            if track.track_id in matched_track_ids:
                continue
            if track.state == UAVTrackState.TENTATIVE:
                track.mark_removed()
                removed_now.append(track)
            else:
                track.mark_lost()
                new_lost.append(track)

        retained_lost: list[UAVTrack] = []
        for track in self.lost_tracks:
            if track.track_id in matched_track_ids:
                continue
            if (self.frame_id - track.frame_id) <= self.track_buffer:
                retained_lost.append(track)
            else:
                track.mark_removed()
                removed_now.append(track)

        new_tracks: list[UAVTrack] = []
        for det_idx in unmatched_det_idx.tolist():
            det = prepared[int(det_idx)]
            if det.score < self.new_track_thresh:
                continue
            new_track = self._create_track(det, self.frame_id)
            new_tracks.append(new_track)

        current_visible.extend(reactivated)
        current_visible.extend(new_tracks)

        self.tracked_tracks = self._deduplicate_tracks(
            track
            for track in current_visible
            if track.state in {UAVTrackState.TENTATIVE, UAVTrackState.TRACKED}
        )
        tracked_ids = {track.track_id for track in self.tracked_tracks}
        self.lost_tracks = self._deduplicate_tracks(
            track for track in [*retained_lost, *new_lost] if track.track_id not in tracked_ids and track.state == UAVTrackState.LOST
        )
        self.removed_tracks.extend(removed_now)
        self.last_output = self.get_active_tracks()

        if not self.tracked_tracks:
            return np.zeros((0, 8), dtype=np.float32)
        visible_rows = [track.to_array() for track in self.tracked_tracks if track.frame_id == self.frame_id]
        return np.stack(visible_rows, axis=0).astype(np.float32) if visible_rows else np.zeros((0, 8), dtype=np.float32)
