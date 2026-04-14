from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch

from .uav_train_augment import UAVTrainAugmentConfig, apply_uav_train_augments
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import xyxyxyxy2xywhr


def _ensure_hw(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize imgsz input to (height, width)."""
    if isinstance(imgsz, int):
        return imgsz, imgsz
    if len(imgsz) != 2:
        raise ValueError(f"Expected imgsz to be int or (h, w), but got {imgsz!r}.")
    return int(imgsz[0]), int(imgsz[1])


def _format_image_tensor(img: np.ndarray) -> torch.Tensor:
    """Match Ultralytics image tensor formatting without changing pixel scale."""
    if img.ndim == 2:
        img = img[..., None]
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(img)


def _polygon_area(segments: np.ndarray) -> np.ndarray:
    """Compute polygon areas for quadrilateral OBB segments."""
    if len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = segments[..., 0]
    y = segments[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))


class RGBIRTemporalTransform:
    """Apply conservative synchronized geometry to RGB, IR, previous RGB, and OBB labels.

    Current-frame RGB and IR share exact letterbox parameters so their spatial alignment is preserved. The previous RGB
    frame is resized to the same output shape and receives the same flip decisions. Optional CMCP, MRRE, and PC-MWA
    run only during training before letterbox so the existing OBB/temporal data structure stays unchanged.
    """

    def __init__(
        self,
        imgsz: int | tuple[int, int] = 640,
        augment: bool = False,
        hyp: Any | None = None,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        pad_value: int = 114,
    ) -> None:
        self.new_shape = _ensure_hw(imgsz)
        self.augment = augment
        self.fliplr = float(getattr(hyp, "fliplr", fliplr) if hyp is not None else fliplr)
        self.flipud = float(getattr(hyp, "flipud", flipud) if hyp is not None else flipud)
        self.pad_value = int(pad_value)
        self.train_augment_config = UAVTrainAugmentConfig.from_hyp(hyp)

    def _letterbox(
        self,
        img: np.ndarray,
        *,
        ratio: tuple[float, float] | None = None,
        pad: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        """Resize and pad one image to the configured output shape."""
        shape = img.shape[:2]  # h, w
        new_h, new_w = self.new_shape
        if ratio is None or pad is None:
            gain = min(new_h / shape[0], new_w / shape[1])
            resize_h = int(round(shape[0] * gain))
            resize_w = int(round(shape[1] * gain))
            top = int(round((new_h - resize_h) / 2 - 0.1))
            left = int(round((new_w - resize_w) / 2 - 0.1))
            ratio = (resize_w / shape[1], resize_h / shape[0])
            pad = (left, top)
        else:
            resize_w = int(round(shape[1] * ratio[0]))
            resize_h = int(round(shape[0] * ratio[1]))
            left, top = pad

        if (resize_h, resize_w) != shape:
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        right = new_w - resize_w - left
        bottom = new_h - resize_h - top
        border_value = (self.pad_value, self.pad_value, self.pad_value)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)
        return img, ratio, pad

    @staticmethod
    def _filter_instances(instances: Instances, cls: np.ndarray) -> tuple[Instances, np.ndarray]:
        """Drop degenerate polygons after geometry changes."""
        if len(instances) == 0:
            return instances, cls
        keep = _polygon_area(instances.segments) > 1.0
        if keep.all():
            return instances, cls
        return instances[keep], cls[keep]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Synchronize conservative geometry across current RGB/IR, previous RGB, and OBB targets."""
        img = sample["img"]
        img_ir = sample["img_ir"]
        img_prev = sample["img_prev"]
        instances = sample["instances"]
        cls = sample["cls"]

        if img.shape[:2] != img_ir.shape[:2]:
            raise ValueError(
                f"Current RGB/IR shape mismatch for {sample['sample_id']}: {img.shape[:2]} vs {img_ir.shape[:2]}."
            )

        h, w = img.shape[:2]
        instances.denormalize(w, h)
        if self.augment and self.train_augment_config.enabled:
            sample["img"] = img
            sample["img_ir"] = img_ir
            sample["img_prev"] = img_prev
            sample["instances"] = instances
            sample["cls"] = cls
            sample = apply_uav_train_augments(sample, self.train_augment_config)
            img = sample["img"]
            img_ir = sample["img_ir"]
            img_prev = sample["img_prev"]
            instances = sample["instances"]
            cls = sample["cls"]

        img, ratio, pad = self._letterbox(img)
        img_ir, _, _ = self._letterbox(img_ir, ratio=ratio, pad=pad)
        img_prev, prev_ratio, prev_pad = self._letterbox(img_prev)

        instances.scale(*ratio)
        instances.add_padding(*pad)

        do_fliplr = self.augment and self.fliplr > 0 and random.random() < self.fliplr
        do_flipud = self.augment and self.flipud > 0 and random.random() < self.flipud
        if do_fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            img_ir = np.ascontiguousarray(np.fliplr(img_ir))
            img_prev = np.ascontiguousarray(np.fliplr(img_prev))
            instances.fliplr(self.new_shape[1])
        if do_flipud:
            img = np.ascontiguousarray(np.flipud(img))
            img_ir = np.ascontiguousarray(np.flipud(img_ir))
            img_prev = np.ascontiguousarray(np.flipud(img_prev))
            instances.flipud(self.new_shape[0])

        instances.clip(self.new_shape[1], self.new_shape[0])
        instances, cls = self._filter_instances(instances, cls)
        instances.normalize(self.new_shape[1], self.new_shape[0])

        nl = len(instances)
        segments = (
            torch.from_numpy(np.ascontiguousarray(instances.segments.astype(np.float32)))
            if nl
            else torch.zeros((0, 4, 2), dtype=torch.float32)
        )
        bboxes = xyxyxyxy2xywhr(segments) if nl else torch.zeros((0, 5), dtype=torch.float32)

        formatted = {
            "img": _format_image_tensor(img),
            "img_ir": _format_image_tensor(img_ir),
            "img_prev": _format_image_tensor(img_prev),
            "cls": torch.from_numpy(cls.astype(np.float32)) if nl else torch.zeros((0, 1), dtype=torch.float32),
            "bboxes": bboxes.float(),
            "segments": segments,
            "batch_idx": torch.zeros(nl),
            "sample_id": sample["sample_id"],
            "frame_id": sample["frame_id"],
            "seq_id": sample["seq_id"],
            "temporal_valid": torch.tensor(bool(sample["temporal_valid"]), dtype=torch.bool),
            "im_file": sample["im_file"],
            "im_file_ir": sample["im_file_ir"],
            "im_file_prev": sample["im_file_prev"],
            "label_file": sample["label_file"],
            "ori_shape": sample["ori_shape"],
            "ori_shape_ir": sample["ori_shape_ir"],
            "ori_shape_prev": sample["ori_shape_prev"],
            "resized_shape": self.new_shape,
            "ratio_pad": {"ratio": ratio, "pad": pad},
            "ratio_pad_ir": {"ratio": ratio, "pad": pad},
            "ratio_pad_prev": {"ratio": prev_ratio, "pad": prev_pad},
        }
        return formatted
