from __future__ import annotations

from copy import copy

import torch
from torch import distributed as dist

from ultralytics.utils import RANK, TQDM
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model

from .rgbir_small_val import SmallObjectOBBValidator


class TemporalOBBValidator(SmallObjectOBBValidator):
    """Stage 5 validator that can explicitly pass `img_prev` during training-time validation.

    This validator stays isolated from the baseline validator path. When temporal mode is disabled, or when validation
    happens through a backend that only supports single-frame RGB inference, it falls back to the inherited RGB-only
    behavior. When temporal mode is enabled during training, it feeds `img_prev` and `temporal_valid` to the model so
    the temporal path is actually exercised.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        *,
        use_temporal: bool = False,
        enable_small_object_metrics: bool = False,
        small_object_area_thr_norm: float = 0.005,
    ) -> None:
        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks,
            enable_small_object_metrics=enable_small_object_metrics,
            small_object_area_thr_norm=small_object_area_thr_norm,
        )
        self.use_temporal = bool(use_temporal)

    def preprocess(self, batch: dict[str, object]) -> dict[str, object]:
        """Normalize current and previous RGB frames for the Stage 5 temporal validation path."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        if "img_prev" in batch and isinstance(batch["img_prev"], torch.Tensor):
            batch["img_prev"] = (batch["img_prev"].half() if self.args.half else batch["img_prev"].float()) / 255
        if "img_ir" in batch and isinstance(batch["img_ir"], torch.Tensor):
            batch["img_ir"] = (batch["img_ir"].half() if self.args.half else batch["img_ir"].float()) / 255
        return batch

    def _prepare_batch(self, si: int, batch: dict[str, object]) -> dict[str, object]:
        """Convert the Stage 2/5 dict-style ratio_pad metadata to the tuple format expected by OBB scaling utils."""
        prepared = super()._prepare_batch(si, batch)
        ratio_pad = prepared["ratio_pad"]
        if isinstance(ratio_pad, dict):
            prepared["ratio_pad"] = (tuple(ratio_pad["ratio"]), tuple(ratio_pad["pad"]))
        return prepared

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run temporal-aware validation during training, otherwise fall back to the inherited RGB-only behavior."""
        if trainer is None or not self.use_temporal:
            return super().__call__(trainer=trainer, model=model)

        self.training = True
        augment = False
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != "cpu" and trainer.amp
        model = trainer.ema.ema or trainer.model
        if trainer.args.compile and hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model = model.half() if self.args.half else model.float()
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)

            with dt[1]:
                preds = model(
                    batch["img"],
                    augment=augment,
                    img_prev=batch.get("img_prev"),
                    temporal_valid=batch.get("temporal_valid"),
                )

            with dt[2]:
                self.loss += model.loss(batch, preds)[1]

            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        model.float()
        loss = self.loss.clone().detach()
        if trainer.world_size > 1:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
        if RANK > 0:
            return
        results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
        return {k: round(float(v), 5) for k, v in results.items()}
