from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import cv2
import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _segments_to_xyxy(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    min_xy = segments.min(axis=1)
    max_xy = segments.max(axis=1)
    return np.concatenate((min_xy, max_xy), axis=1).astype(np.float32)


def _polygon_areas(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = segments[..., 0]
    y = segments[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))


def _clip_segments(segments: np.ndarray, w: int, h: int) -> np.ndarray:
    clipped = segments.astype(np.float32, copy=True)
    clipped[..., 0] = clipped[..., 0].clip(0, w - 1)
    clipped[..., 1] = clipped[..., 1].clip(0, h - 1)
    return clipped


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-6)


def _odd_kernel(size: float, minimum: int = 3) -> int:
    kernel = max(minimum, int(round(size)))
    return kernel if kernel % 2 == 1 else kernel + 1


def _as_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _resize_map(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(arr, (w, h), interpolation=interpolation)
    if resized.ndim == 2:
        return resized.astype(np.float32)
    return resized.astype(np.float32)


def _parse_pc_mwa_types(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ("fog", "rain", "low_light")
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    valid = []
    for item in raw:
        name = str(item).strip().lower().replace("-", "_")
        if name in {"fog", "rain", "low_light"} and name not in valid:
            valid.append(name)
    return tuple(valid or ("fog", "rain", "low_light"))


@dataclass(frozen=True)
class UAVTrainAugmentConfig:
    enable_cmcp: bool = False
    cmcp_prob: float = 0.15
    cmcp_max_pastes: int = 3
    cmcp_small_area_thr: float = 1024.0
    cmcp_num_trials: int = 15

    enable_mrre: bool = False
    mrre_prob: float = 0.20
    mrre_radius_ratio: float = 1.5
    mrre_num_regions: int = 2
    mrre_strength: float = 0.35

    enable_pc_mwa: bool = False
    pc_mwa_prob: float = 0.20
    pc_mwa_types: tuple[str, ...] = ("fog", "rain", "low_light")
    pc_mwa_severity_min: float = 0.20
    pc_mwa_severity_max: float = 0.60
    pc_mwa_shared_severity: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.enable_cmcp or self.enable_mrre or self.enable_pc_mwa)

    @classmethod
    def from_hyp(cls, hyp: Any | None) -> "UAVTrainAugmentConfig":
        if hyp is None:
            return cls()
        severity_min = float(getattr(hyp, "pc_mwa_severity_min", 0.20))
        severity_max = float(getattr(hyp, "pc_mwa_severity_max", 0.60))
        severity_min, severity_max = sorted((_clamp01(severity_min), _clamp01(severity_max)))
        return cls(
            enable_cmcp=bool(getattr(hyp, "enable_cmcp", False)),
            cmcp_prob=_clamp01(getattr(hyp, "cmcp_prob", 0.15)),
            cmcp_max_pastes=max(0, int(getattr(hyp, "cmcp_max_pastes", 3))),
            cmcp_small_area_thr=max(0.0, float(getattr(hyp, "cmcp_small_area_thr", 1024.0))),
            cmcp_num_trials=max(1, int(getattr(hyp, "cmcp_num_trials", 15))),
            enable_mrre=bool(getattr(hyp, "enable_mrre", False)),
            mrre_prob=_clamp01(getattr(hyp, "mrre_prob", 0.20)),
            mrre_radius_ratio=max(0.1, float(getattr(hyp, "mrre_radius_ratio", 1.5))),
            mrre_num_regions=max(0, int(getattr(hyp, "mrre_num_regions", 2))),
            mrre_strength=_clamp01(getattr(hyp, "mrre_strength", 0.35)),
            enable_pc_mwa=bool(getattr(hyp, "enable_pc_mwa", False)),
            pc_mwa_prob=_clamp01(getattr(hyp, "pc_mwa_prob", 0.20)),
            pc_mwa_types=_parse_pc_mwa_types(getattr(hyp, "pc_mwa_types", None)),
            pc_mwa_severity_min=severity_min,
            pc_mwa_severity_max=severity_max,
            pc_mwa_shared_severity=bool(getattr(hyp, "pc_mwa_shared_severity", True)),
        )


def _extract_masked_patch(img: np.ndarray, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    x1 = max(0, int(np.floor(segment[:, 0].min())))
    y1 = max(0, int(np.floor(segment[:, 1].min())))
    x2 = min(img.shape[1], int(np.ceil(segment[:, 0].max())))
    y2 = min(img.shape[0], int(np.ceil(segment[:, 1].max())))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = img[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return None
    local = np.round(segment - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    if int(mask.sum()) < 16 * 255:
        return None
    return patch, mask, (x1, y1, x2, y2)


def _apply_cmcp(
    img: np.ndarray,
    img_ir: np.ndarray | None,
    segments: np.ndarray,
    cls: np.ndarray,
    cfg: UAVTrainAugmentConfig,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    if len(segments) == 0 or cfg.cmcp_max_pastes <= 0:
        return img, img_ir, segments, cls
    if img_ir is not None and img_ir.shape[:2] != img.shape[:2]:
        return img, img_ir, segments, cls

    areas = _polygon_areas(segments)
    candidates = [idx for idx, area in enumerate(areas.tolist()) if 1.0 < area <= cfg.cmcp_small_area_thr]
    if not candidates:
        return img, img_ir, segments, cls

    h, w = img.shape[:2]
    out_img = img.copy()
    out_ir = None if img_ir is None else img_ir.copy()
    out_segments = segments.astype(np.float32, copy=True)
    out_cls = cls.astype(np.float32, copy=True)
    existing_boxes = _segments_to_xyxy(out_segments)
    pasted = 0

    for _ in range(cfg.cmcp_max_pastes):
        idx = random.choice(candidates)
        patch_info = _extract_masked_patch(out_img, out_segments[idx])
        if patch_info is None:
            continue
        patch, mask, (src_x1, src_y1, src_x2, src_y2) = patch_info
        if out_ir is not None:
            patch_ir = out_ir[src_y1:src_y2, src_x1:src_x2].copy()
            if patch_ir.shape[:2] != patch.shape[:2]:
                continue
        else:
            patch_ir = None

        patch_h, patch_w = patch.shape[:2]
        if patch_h < 2 or patch_w < 2:
            continue
        base_segment = out_segments[idx]
        accepted = False
        for _trial in range(cfg.cmcp_num_trials):
            dst_x1 = random.randint(0, max(0, w - patch_w))
            dst_y1 = random.randint(0, max(0, h - patch_h))
            translated = base_segment + np.array([dst_x1 - src_x1, dst_y1 - src_y1], dtype=np.float32)
            translated = _clip_segments(translated[None], w, h)[0]
            translated_box = _segments_to_xyxy(translated[None])[0]
            if translated_box[2] - translated_box[0] < 2 or translated_box[3] - translated_box[1] < 2:
                continue
            overlaps = _box_iou(translated_box, existing_boxes)
            if overlaps.size and float(overlaps.max()) > 0.25:
                continue

            dst_y2 = dst_y1 + patch_h
            dst_x2 = dst_x1 + patch_w
            mask_bool = mask.astype(bool)
            out_img[dst_y1:dst_y2, dst_x1:dst_x2][mask_bool] = patch[mask_bool]
            if out_ir is not None and patch_ir is not None:
                out_ir[dst_y1:dst_y2, dst_x1:dst_x2][mask_bool] = patch_ir[mask_bool]
            out_segments = np.concatenate((out_segments, translated[None]), axis=0)
            out_cls = np.concatenate((out_cls, out_cls[idx : idx + 1]), axis=0)
            existing_boxes = np.concatenate((existing_boxes, translated_box[None]), axis=0)
            pasted += 1
            accepted = True
            break
        if not accepted:
            continue

    if pasted == 0:
        return img, img_ir, segments, cls
    return out_img, out_ir, out_segments, out_cls


def _perturb_region(region: np.ndarray, mask: np.ndarray, strength: float, params: dict[str, Any]) -> np.ndarray:
    if region.size == 0 or not np.any(mask):
        return region
    alpha = _clamp01(0.25 + strength * 0.75)
    region_f = region.astype(np.float32)
    mode = params["mode"]
    if mode == "roll":
        shifted = np.roll(region_f, shift=(params["shift_y"], params["shift_x"]), axis=(0, 1))
        perturbed = cv2.addWeighted(region_f, 1.0 - alpha, shifted, alpha, 0.0)
    elif mode == "blur":
        kernel = _odd_kernel(params["kernel"], minimum=3)
        blurred = cv2.GaussianBlur(region_f, (kernel, kernel), 0)
        perturbed = cv2.addWeighted(region_f, 1.0 - alpha, blurred, alpha, 0.0)
    else:
        noise = params["noise"]
        perturbed = region_f + noise * (12.0 + 48.0 * strength)

    out = region_f.copy()
    out[mask] = perturbed[mask]
    return _as_uint8(out)


def _apply_mrre(
    img: np.ndarray,
    img_ir: np.ndarray | None,
    segments: np.ndarray,
    cfg: UAVTrainAugmentConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(segments) == 0 or cfg.mrre_num_regions <= 0:
        return img, img_ir
    if img_ir is not None and img_ir.shape[:2] != img.shape[:2]:
        return img, img_ir

    h, w = img.shape[:2]
    out_img = img.copy()
    out_ir = None if img_ir is None else img_ir.copy()
    candidate_indices = list(range(len(segments)))
    random.shuffle(candidate_indices)

    for idx in candidate_indices[: cfg.mrre_num_regions]:
        segment = segments[idx]
        box = _segments_to_xyxy(segment[None])[0]
        radius = int(round(max(box[2] - box[0], box[3] - box[1]) * cfg.mrre_radius_ratio))
        radius = max(radius, 4)
        x1 = max(0, int(np.floor(box[0] - radius)))
        y1 = max(0, int(np.floor(box[1] - radius)))
        x2 = min(w, int(np.ceil(box[2] + radius)))
        y2 = min(h, int(np.ceil(box[3] + radius)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        local_segment = np.round(segment - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask, [local_segment], 255)
        bg_mask = mask == 0
        if not np.any(bg_mask):
            continue

        mode = random.choice(("roll", "blur", "noise"))
        params: dict[str, Any] = {"mode": mode}
        if mode == "roll":
            params["shift_x"] = random.randint(-radius, radius)
            params["shift_y"] = random.randint(-radius, radius)
        elif mode == "blur":
            params["kernel"] = 3 + cfg.mrre_strength * 8.0 + radius * 0.05
        else:
            params["noise"] = np.random.normal(0.0, 1.0, size=(y2 - y1, x2 - x1, 1)).astype(np.float32)

        out_img[y1:y2, x1:x2] = _perturb_region(out_img[y1:y2, x1:x2], bg_mask, cfg.mrre_strength, params)
        if out_ir is not None:
            out_ir[y1:y2, x1:x2] = _perturb_region(out_ir[y1:y2, x1:x2], bg_mask, cfg.mrre_strength, params)

    return out_img, out_ir


def _build_weather_event(weather_type: str, severity: float) -> dict[str, Any]:
    if weather_type == "fog":
        base = np.random.rand(96, 96).astype(np.float32)
        base = cv2.GaussianBlur(base, (_odd_kernel(21), _odd_kernel(21)), 0)
        base = cv2.normalize(base, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return {"type": weather_type, "severity": severity, "map": base}

    if weather_type == "rain":
        rain = np.zeros((128, 128), dtype=np.float32)
        num_streaks = max(32, int(180 * severity))
        length = max(8, int(14 + 30 * severity))
        for _ in range(num_streaks):
            x = random.randint(0, rain.shape[1] - 1)
            y = random.randint(0, rain.shape[0] - 1)
            dx = random.randint(-4, 4)
            dy = length
            cv2.line(rain, (x, y), (max(0, min(rain.shape[1] - 1, x + dx)), max(0, min(rain.shape[0] - 1, y + dy))), 1.0, 1)
        rain = cv2.GaussianBlur(rain, (_odd_kernel(3 + severity * 2), _odd_kernel(9 + severity * 6)), 0)
        rain = cv2.normalize(rain, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return {"type": weather_type, "severity": severity, "map": rain}

    noise = np.random.normal(0.0, 1.0, size=(96, 96, 1)).astype(np.float32)
    return {"type": "low_light", "severity": severity, "noise": noise}


def _apply_fog(img: np.ndarray, severity: float, modal: str, base_map: np.ndarray) -> np.ndarray:
    alpha = _resize_map(base_map, img.shape[:2])
    alpha = severity * (0.25 + 0.75 * alpha)
    img_f = img.astype(np.float32)
    if modal == "ir":
        veil = np.full_like(img_f, 192.0)
        out = img_f * (1.0 - alpha[..., None] * 0.45) + veil * (alpha[..., None] * 0.20)
    else:
        veil = np.full_like(img_f, 255.0)
        out = img_f * (1.0 - alpha[..., None] * 0.70) + veil * (alpha[..., None] * 0.60)
    blur = cv2.GaussianBlur(out, (_odd_kernel(3 + severity * 6), _odd_kernel(3 + severity * 6)), 0)
    blend = 0.15 + 0.35 * severity
    return _as_uint8(cv2.addWeighted(out, 1.0 - blend, blur, blend, 0.0))


def _apply_rain(img: np.ndarray, severity: float, modal: str, streak_map: np.ndarray) -> np.ndarray:
    streaks = _resize_map(streak_map, img.shape[:2])
    streaks = streaks[..., None]
    img_f = img.astype(np.float32)
    if modal == "ir":
        out = img_f * (1.0 - 0.06 * severity) + 180.0 * streaks * (0.10 + 0.12 * severity)
        out = cv2.GaussianBlur(out, (_odd_kernel(3 + severity * 2), _odd_kernel(3 + severity * 2)), 0)
    else:
        out = img_f * (1.0 - 0.12 * severity) + 255.0 * streaks * (0.18 + 0.22 * severity)
        out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (_odd_kernel(3), _odd_kernel(3)), 0), 0.10 + 0.15 * severity, 0.0)
    return _as_uint8(out)


def _apply_low_light(img: np.ndarray, severity: float, modal: str, noise_seed: np.ndarray) -> np.ndarray:
    noise = _resize_map(noise_seed, img.shape[:2])
    if noise.ndim == 2:
        noise = noise[..., None]
    img_f = img.astype(np.float32)
    if modal == "ir":
        mean = img_f.mean(axis=(0, 1), keepdims=True)
        out = mean + (img_f - mean) * (1.0 - 0.35 * severity)
        out *= 1.0 - 0.20 * severity
        out += noise * (8.0 + 14.0 * severity)
    else:
        gamma = 1.0 + 2.2 * severity
        out = ((img_f / 255.0).clip(0.0, 1.0) ** gamma) * 255.0
        out *= 1.0 - 0.08 * severity
        out += noise * (6.0 + 12.0 * severity)
    return _as_uint8(out)


def _apply_weather(img: np.ndarray | None, event: dict[str, Any], modal: str, severity: float) -> np.ndarray | None:
    if img is None:
        return None
    if event["type"] == "fog":
        return _apply_fog(img, severity, modal, event["map"])
    if event["type"] == "rain":
        return _apply_rain(img, severity, modal, event["map"])
    return _apply_low_light(img, severity, modal, event["noise"])


def _apply_pc_mwa(
    img: np.ndarray,
    img_ir: np.ndarray | None,
    img_prev: np.ndarray | None,
    cfg: UAVTrainAugmentConfig,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    weather_type = random.choice(cfg.pc_mwa_types)
    severity = random.uniform(cfg.pc_mwa_severity_min, cfg.pc_mwa_severity_max)
    event = _build_weather_event(weather_type, severity)

    def _maybe_resample() -> float:
        if cfg.pc_mwa_shared_severity:
            return severity
        return random.uniform(cfg.pc_mwa_severity_min, cfg.pc_mwa_severity_max)

    img = _apply_weather(img, event, modal="rgb", severity=_maybe_resample())
    if img_ir is not None:
        img_ir = _apply_weather(img_ir, event, modal="ir", severity=_maybe_resample())
    if img_prev is not None:
        # Temporal context uses the same weather event so the two-frame sample remains physically coherent.
        img_prev = _apply_weather(img_prev, event, modal="rgb", severity=_maybe_resample())
    return img, img_ir, img_prev


def apply_uav_train_augments(sample: dict[str, Any], cfg: UAVTrainAugmentConfig) -> dict[str, Any]:
    """Apply optional train-only UAV augmentations to current-frame RGB/IR data.

    CMCP and MRRE operate on the current frame only because they depend on current-frame annotations. PC-MWA is also
    applied to `img_prev` when present so temporal samples keep one shared weather event and severity.
    """
    if not cfg.enabled:
        return sample

    img = sample["img"]
    img_ir = sample.get("img_ir")
    img_prev = sample.get("img_prev")
    instances = sample["instances"]
    cls = sample["cls"]
    segments = np.asarray(instances.segments, dtype=np.float32)

    if cfg.enable_cmcp and random.random() < cfg.cmcp_prob:
        img, img_ir, segments, cls = _apply_cmcp(img, img_ir, segments, cls, cfg)

    if cfg.enable_mrre and random.random() < cfg.mrre_prob:
        img, img_ir = _apply_mrre(img, img_ir, segments, cfg)

    if cfg.enable_pc_mwa and random.random() < cfg.pc_mwa_prob:
        img, img_ir, img_prev = _apply_pc_mwa(img, img_ir, img_prev, cfg)

    instances.update(bboxes=_segments_to_xyxy(segments), segments=segments)
    sample["img"] = img
    sample["img_ir"] = img_ir
    sample["img_prev"] = img_prev
    sample["instances"] = instances
    sample["cls"] = cls.astype(np.float32, copy=False)
    return sample
