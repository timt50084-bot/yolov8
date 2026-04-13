import hashlib
import torch
from pathlib import Path

paths = [
    Path(r"D:\project\ultralytics-main\outputs\uav_pipeline\rgbir\train\train_20260409_1452352\weights\best.pt"),
    Path(r"D:\project\ultralytics-main\outputs\uav_pipeline\rgbir\train\train_20260409_1452352\weights\last.pt"),
]

def model_sha(ckpt):
    model = ckpt.get("model", None)
    if model is None:
        model = ckpt.get("ema", None)
    if model is None:
        return None
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous().numpy()
        h.update(k.encode("utf-8"))
        h.update(t.tobytes())
    return h.hexdigest()

for p in paths:
    ckpt = torch.load(p, map_location="cpu")
    print("=" * 80)
    print(p)
    print("epoch:", ckpt.get("epoch"))
    print("best_fitness:", ckpt.get("best_fitness"))
    print("train_metrics:", ckpt.get("train_metrics"))
    print("sha256:", model_sha(ckpt))