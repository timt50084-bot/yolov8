from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "uav_pipeline"


@dataclass(frozen=True)
class ModeSpec:
    """Stable routing metadata for one explicit Stage 7 pipeline mode."""

    name: str
    description: str
    default_model: Path
    train_script: Path | None
    supports_temporal_predict: bool
    supports_stage6_track: bool
    use_rgbir: bool
    use_small: bool
    use_temporal: bool
    use_tracking: bool
    default_tracker: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the mode spec."""
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


MODE_REGISTRY: dict[str, ModeSpec] = {
    "baseline": ModeSpec(
        name="baseline",
        description="Native Ultralytics OBB path with no RGB-IR, small-object, temporal, or custom tracking additions.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb.yaml",
        train_script=None,
        supports_temporal_predict=False,
        supports_stage6_track=False,
        use_rgbir=False,
        use_small=False,
        use_temporal=False,
        use_tracking=False,
        default_tracker=REPO_ROOT / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml",
    ),
    "rgbir": ModeSpec(
        name="rgbir",
        description="Stage 3 training-time RGB-IR assist with RGB-only validation and deployment.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_obb.py",
        supports_temporal_predict=False,
        supports_stage6_track=False,
        use_rgbir=True,
        use_small=False,
        use_temporal=False,
        use_tracking=False,
    ),
    "rgbir-small": ModeSpec(
        name="rgbir-small",
        description="Stage 4 RGB-IR path with explicit small-object sampling, loss weighting, and small metrics.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-obb-small.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_obb_small.py",
        supports_temporal_predict=False,
        supports_stage6_track=False,
        use_rgbir=True,
        use_small=True,
        use_temporal=False,
        use_tracking=False,
    ),
    "small": ModeSpec(
        name="small",
        description="Single-modal Stage 4 path with explicit small-object sampling, weighting, and metrics but no RGB-IR assist.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb-small.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_obb_small.py",
        supports_temporal_predict=False,
        supports_stage6_track=False,
        use_rgbir=False,
        use_small=True,
        use_temporal=False,
        use_tracking=False,
    ),
    "rgbir-small-temporal": ModeSpec(
        name="rgbir-small-temporal",
        description="Stage 5 RGB-IR + small-object + temporal path combining P2 small-object structure with lightweight two-frame temporal detection.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-small-temporal-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_temporal_obb.py",
        supports_temporal_predict=True,
        supports_stage6_track=False,
        use_rgbir=True,
        use_small=True,
        use_temporal=True,
        use_tracking=False,
    ),
    "temporal_small": ModeSpec(
        name="temporal_small",
        description="Single-modal Stage 5 path combining Stage 4 small-object structure with lightweight two-frame temporal detection.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-small-temporal-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_temporal_obb.py",
        supports_temporal_predict=True,
        supports_stage6_track=False,
        use_rgbir=False,
        use_small=True,
        use_temporal=True,
        use_tracking=False,
    ),
    "rgbir-temporal": ModeSpec(
        name="rgbir-temporal",
        description="Stage 5 RGB-IR path with explicit lightweight two-frame temporal detection enabled.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_temporal_obb.py",
        supports_temporal_predict=True,
        supports_stage6_track=False,
        use_rgbir=True,
        use_small=False,
        use_temporal=True,
        use_tracking=False,
    ),
    "temporal": ModeSpec(
        name="temporal",
        description="Single-modal Stage 5 path with explicit lightweight two-frame temporal detection enabled.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-temporal-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_temporal_obb.py",
        supports_temporal_predict=True,
        supports_stage6_track=False,
        use_rgbir=False,
        use_small=False,
        use_temporal=True,
        use_tracking=False,
    ),
    "rgbir-temporal-track": ModeSpec(
        name="rgbir-temporal-track",
        description="Stage 6 pipeline mode: Stage 5 detector route plus explicit UAV OBB tracking-by-detection.",
        default_model=REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-rgbir-temporal-obb.yaml",
        train_script=REPO_ROOT / "tools" / "train_uav_rgbir_temporal_obb.py",
        supports_temporal_predict=True,
        supports_stage6_track=True,
        use_rgbir=True,
        use_small=False,
        use_temporal=True,
        use_tracking=True,
        default_tracker=REPO_ROOT / "ultralytics" / "cfg" / "trackers" / "uav_obb_tracker.yaml",
    ),
}


def resolve_mode(mode: str) -> ModeSpec:
    """Return the stable mode specification or raise a clear error."""
    if mode not in MODE_REGISTRY:
        raise KeyError(f"Unsupported mode '{mode}'. Available modes: {', '.join(MODE_REGISTRY)}")
    return MODE_REGISTRY[mode]


def normalize_exp_tag(exp_tag: str) -> str:
    """Return a filesystem-friendly experiment tag while preserving readability."""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", exp_tag.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "exp"


def make_default_run_name(subtask: str, exp_tag: str | None = None) -> str:
    """Return a stable timestamp-based run name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_tag:
        return f"{subtask}_{normalize_exp_tag(exp_tag)}_{timestamp}"
    return f"{subtask}_{timestamp}"


def resolve_output_dir(
    *,
    mode: str,
    subtask: str,
    project: str | None,
    name: str | None,
    exp_tag: str | None,
    save_dir: str | None,
    exist_ok: bool = False,
) -> tuple[Path, str, Path]:
    """Resolve `(project_dir, run_name, run_dir)` with Stage 7 default directory conventions."""
    normalized_tag = normalize_exp_tag(exp_tag) if exp_tag else None
    if name and normalized_tag:
        run_name = name if name.endswith(normalized_tag) else f"{name}_{normalized_tag}"
    else:
        run_name = name or make_default_run_name(subtask, normalized_tag)
    if save_dir:
        run_dir = Path(save_dir).expanduser().resolve()
        return run_dir.parent, run_dir.name, run_dir
    project_dir = Path(project).expanduser().resolve() if project else (DEFAULT_OUTPUT_ROOT / mode / subtask)
    run_dir = project_dir / run_name
    if not exist_ok and run_dir.exists():
        index = 2
        while (project_dir / f"{run_name}{index}").exists():
            index += 1
        run_name = f"{run_name}{index}"
        run_dir = project_dir / run_name
    return project_dir, run_name, run_dir


def route_manifest(
    *,
    subtask: str,
    mode_spec: ModeSpec,
    project_dir: Path,
    run_name: str,
    run_dir: Path,
    route_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-friendly route manifest for dry-run and execution logging."""
    return {
        "subtask": subtask,
        "mode": mode_spec.name,
        "mode_spec": mode_spec.to_dict(),
        "route_type": route_type,
        "project_dir": str(project_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "payload": payload,
    }
