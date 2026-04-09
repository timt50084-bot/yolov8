from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from tools.utils_uav_pipeline import ModeSpec, resolve_mode, resolve_output_dir, route_manifest


def extra_args(items: list[str]) -> list[str]:
    """Normalize passthrough arguments collected after `--`."""
    if items and items[0] == "--":
        return items[1:]
    return items


def bool_override(default: bool, enable: bool, disable: bool) -> bool:
    """Resolve a default boolean with optional explicit overrides."""
    if enable and disable:
        raise ValueError("Received both enable and disable flags for the same option.")
    if enable:
        return True
    if disable:
        return False
    return default


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist the resolved route manifest for traceability."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "route.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def print_manifest(manifest: dict[str, Any]) -> None:
    """Print a readable route summary for dry-run and debugging."""
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def execute_subprocess(
    cmd: list[str],
    *,
    run_dir: Path,
    dry_run: bool,
    print_route: bool,
    manifest: dict[str, Any],
) -> int:
    """Run a subprocess route or print it for dry-run."""
    if print_route or dry_run:
        print_manifest(manifest)
    if dry_run:
        return 0
    save_manifest(run_dir, manifest)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def execute_yolo_call(
    *,
    run_dir: Path,
    dry_run: bool,
    print_route: bool,
    manifest: dict[str, Any],
    model_source: str,
    action: str,
    kwargs: dict[str, Any],
) -> Any:
    """Run a direct Ultralytics API call or print the route for dry-run."""
    if print_route or dry_run:
        print_manifest(manifest)
    if dry_run:
        return None
    save_manifest(run_dir, manifest)
    model = YOLO(model_source)
    method = getattr(model, action)
    return method(**kwargs)


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None, type=str, help="Optional project/output root override.")
    parser.add_argument("--name", default=None, type=str, help="Optional run name override.")
    parser.add_argument("--save-dir", default=None, type=str, help="Optional full run directory override.")
    parser.add_argument("--exp-tag", default=None, type=str, help="Optional experiment tag appended to the run name.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved route without executing it.")
    parser.add_argument("--print-route", action="store_true", help="Print the resolved route before execution.")


def add_common_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "rgbir", "rgbir-small", "rgbir-small-temporal", "rgbir-temporal", "rgbir-temporal-track"],
        help="Stage mode to route.",
    )
    parser.add_argument("--model", default=None, type=str, help="Optional model yaml override.")
    parser.add_argument("--weights", default=None, type=str, help="Optional checkpoint override.")
    parser.add_argument("--data", default=None, type=str, help="Dataset yaml.")
    parser.add_argument("--device", default="cpu", type=str, help="Device, e.g. cpu or 0.")
    add_common_output_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 7 explicit UAV pipeline launcher.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Run the Stage 1 preprocess pipeline.")
    preprocess.add_argument("--input-root", required=True, type=str, help="Raw dataset root.")
    preprocess.add_argument("--output-root", default=None, type=str, help="Prepared dataset output root.")
    preprocess.add_argument("--split-mode", default="auto", choices=["auto", "existing", "random"])
    preprocess.add_argument("--val-ratio", default=0.2, type=float)
    preprocess.add_argument("--seed", default=0, type=int)
    preprocess.add_argument("--overwrite", action="store_true")
    add_common_output_args(preprocess)
    preprocess.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to the Stage 1 script.")

    check_data = subparsers.add_parser("check-data", help="Run the Stage 1 dataset checker.")
    check_data.add_argument("--dataset-root", required=True, type=str, help="Prepared dataset root.")
    check_data.add_argument("--data-yaml", default=None, type=str, help="Optional dataset yaml override.")
    check_data.add_argument("--report-path", default=None, type=str, help="Optional report path override.")
    check_data.add_argument("--skip-dataset-scan", action="store_true")
    add_common_output_args(check_data)
    check_data.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to the Stage 1 checker.")

    train = subparsers.add_parser("train", help="Run training through the baseline or staged training paths.")
    add_common_mode_args(train)
    train.add_argument("--epochs", default=1, type=int)
    train.add_argument("--batch", default=2, type=int)
    train.add_argument("--imgsz", default=256, type=int)
    train.add_argument("--workers", default=0, type=int)
    train.add_argument("--close-mosaic", default=10, type=int)
    train.add_argument("--exist-ok", action="store_true")
    train.add_argument("--plots", action="store_true")
    train.add_argument("--val", action="store_true")
    train.add_argument(
        "--console-epoch-summary-only",
        dest="console_epoch_summary_only",
        action="store_true",
        help="Disable batch-level progress bars and keep only one compact epoch summary on the console.",
    )
    train.add_argument(
        "--disable-console-epoch-summary-only",
        dest="console_epoch_summary_only",
        action="store_false",
        help="Explicitly keep the default batch-level training progress bar enabled.",
    )
    train.set_defaults(console_epoch_summary_only=False)
    train.add_argument("--enable-rgbir", action="store_true")
    train.add_argument("--disable-rgbir", action="store_true")
    train.add_argument("--enable-small-object", action="store_true")
    train.add_argument("--disable-small-object", action="store_true")
    train.add_argument("--enable-temporal", action="store_true")
    train.add_argument("--disable-temporal", action="store_true")
    train.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to the staged train script.")

    val = subparsers.add_parser("val", help="Run RGB-only validation for the selected mode.")
    add_common_mode_args(val)
    val.add_argument("--batch", default=2, type=int)
    val.add_argument("--imgsz", default=256, type=int)
    val.add_argument("--workers", default=0, type=int)
    val.add_argument("--conf", default=0.001, type=float)
    val.add_argument("--iou", default=0.6, type=float)
    val.add_argument("--max-det", default=300, type=int)
    val.add_argument("--plots", action="store_true")
    val.add_argument("--exist-ok", action="store_true")
    val.add_argument(
        "--console-epoch-summary-only",
        dest="console_epoch_summary_only",
        action="store_true",
        help="Disable batch-level validation progress bars and keep only the final metric print.",
    )
    val.add_argument(
        "--disable-console-epoch-summary-only",
        dest="console_epoch_summary_only",
        action="store_false",
        help="Explicitly keep the default batch-level validation progress bar enabled.",
    )
    val.set_defaults(console_epoch_summary_only=False)

    predict = subparsers.add_parser("predict", help="Run single-frame RGB-only prediction for the selected mode.")
    add_common_mode_args(predict)
    predict.add_argument("--source", required=True, type=str, help="Image or directory source.")
    predict.add_argument("--imgsz", default=256, type=int)
    predict.add_argument("--conf", default=0.25, type=float)
    predict.add_argument("--iou", default=0.45, type=float)
    predict.add_argument("--max-det", default=300, type=int)
    predict.add_argument("--exist-ok", action="store_true")

    temporal_predict = subparsers.add_parser(
        "temporal-predict",
        help="Run the explicit Stage 5 temporal sequence predictor for supported modes.",
    )
    add_common_mode_args(temporal_predict)
    temporal_predict.add_argument("--source", required=True, type=str, help="Sequential image directory or single image.")
    temporal_predict.add_argument("--imgsz", default=256, type=int)
    temporal_predict.add_argument("--max-frames", default=0, type=int)
    temporal_predict.add_argument("--enable-temporal", action="store_true")
    temporal_predict.add_argument("--disable-temporal", action="store_true")
    temporal_predict.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to the temporal predict script.")

    track = subparsers.add_parser("track", help="Run baseline tracking or the explicit Stage 6 UAV tracker.")
    add_common_mode_args(track)
    track.add_argument("--source", required=True, type=str, help="Image directory or video source.")
    track.add_argument("--tracker", default=None, type=str, help="Optional tracker yaml override.")
    track.add_argument("--imgsz", default=256, type=int)
    track.add_argument("--conf", default=0.10, type=float)
    track.add_argument("--iou", default=0.45, type=float)
    track.add_argument("--max-det", default=100, type=int)
    track.add_argument("--max-frames", default=0, type=int)
    track.add_argument("--save-json", default=None, type=str, help="Optional tracking JSON path override.")
    track.add_argument("--enable-temporal-detector", action="store_true")
    track.add_argument("--disable-temporal-detector", action="store_true")
    track.add_argument("--exist-ok", action="store_true")
    track.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to the Stage 6 tracking script.")

    return parser


def preprocess_route(args: argparse.Namespace) -> int:
    project_dir, run_name, run_dir = resolve_output_dir(
        mode="data",
        subtask="preprocess",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir or args.output_root,
        exist_ok=False,
    )
    output_root = args.output_root or str(run_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "prepare_uav_rgbir_obb_dataset.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        output_root,
        "--split-mode",
        args.split_mode,
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    cmd.extend(extra_args(args.extra_args))
    manifest = route_manifest(
        subtask="preprocess",
        mode_spec=ModeSpec(
            name="data",
            description="Stage 1 preprocessing route.",
            default_model=Path(""),
            train_script=None,
            supports_temporal_predict=False,
            supports_stage6_track=False,
            use_rgbir=False,
            use_small=False,
            use_temporal=False,
            use_tracking=False,
        ),
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={"command": cmd, "output_root": output_root},
    )
    rc = execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)
    if not args.dry_run:
        save_manifest(run_dir, manifest)
    return rc


def check_data_route(args: argparse.Namespace) -> int:
    project_dir, run_name, run_dir = resolve_output_dir(
        mode="data",
        subtask="check-data",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=False,
    )
    report_path = args.report_path or str(run_dir / "check_summary.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "check_uav_rgbir_dataset.py"),
        "--dataset-root",
        args.dataset_root,
        "--report-path",
        report_path,
    ]
    if args.data_yaml:
        cmd.extend(["--data-yaml", args.data_yaml])
    if args.skip_dataset_scan:
        cmd.append("--skip-dataset-scan")
    cmd.extend(extra_args(args.extra_args))
    manifest = route_manifest(
        subtask="check-data",
        mode_spec=ModeSpec(
            name="data",
            description="Stage 1 dataset-check route.",
            default_model=Path(""),
            train_script=None,
            supports_temporal_predict=False,
            supports_stage6_track=False,
            use_rgbir=False,
            use_small=False,
            use_temporal=False,
            use_tracking=False,
        ),
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={"command": cmd, "report_path": report_path},
    )
    return execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)


def train_route(args: argparse.Namespace) -> Any:
    spec = resolve_mode(args.mode)
    if not args.data:
        raise ValueError("--data is required for train.")
    project_dir, run_name, run_dir = resolve_output_dir(
        mode=spec.name,
        subtask="train",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=args.exist_ok,
    )
    model_path = str(Path(args.model).resolve()) if args.model else str(spec.default_model)
    use_rgbir = bool_override(spec.use_rgbir, args.enable_rgbir, args.disable_rgbir)
    use_small = bool_override(spec.use_small, args.enable_small_object, args.disable_small_object)
    use_temporal = bool_override(spec.use_temporal, args.enable_temporal, args.disable_temporal)

    if spec.name == "baseline":
        model_source = str(Path(args.weights).resolve()) if args.weights else model_path
        kwargs = {
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": True,
            "close_mosaic": args.close_mosaic,
            "plots": args.plots,
            "val": args.val,
            "console_epoch_summary_only": args.console_epoch_summary_only,
        }
        if args.weights and not args.model:
            # Training from a checkpoint uses the checkpoint as the model source directly.
            pass
        elif args.weights:
            kwargs["pretrained"] = str(Path(args.weights).resolve())
        manifest = route_manifest(
            subtask="train",
            mode_spec=spec,
            project_dir=project_dir,
            run_name=run_name,
            run_dir=run_dir,
            route_type="ultralytics_api",
            payload={"action": "train", "model_source": model_source, "kwargs": kwargs},
        )
        return execute_yolo_call(
            run_dir=run_dir,
            dry_run=args.dry_run,
            print_route=args.print_route,
            manifest=manifest,
            model_source=model_source,
            action="train",
            kwargs=kwargs,
        )

    if spec.train_script is None:
        raise ValueError(f"Mode '{spec.name}' does not provide a staged training script.")
    cmd = [
        sys.executable,
        str(spec.train_script),
        "--data",
        args.data,
        "--model",
        model_path,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--project",
        str(project_dir),
        "--name",
        run_name,
        "--close-mosaic",
        str(args.close_mosaic),
    ]
    if args.weights:
        cmd.extend(["--pretrained", str(Path(args.weights).resolve())])
    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.plots:
        cmd.append("--plots")
    if args.val:
        cmd.append("--val")

    cmd.append("--use-rgbir-train-assist" if use_rgbir else "--disable-rgbir-train-assist")
    if spec.name in {"rgbir-small", "rgbir-small-temporal", "rgbir-temporal", "rgbir-temporal-track"}:
        cmd.append("--use-small-object-sampling" if use_small else "--disable-small-object-sampling")
        if spec.name == "rgbir-small" and use_small:
            cmd.append("--disable-small-object-loss-weighting")
        else:
            cmd.append("--use-small-object-loss-weighting" if use_small else "--disable-small-object-loss-weighting")
        cmd.append("--enable-small-object-metrics" if use_small else "--disable-small-object-metrics")
    elif use_small:
        raise ValueError(f"Mode '{spec.name}' does not support small-object toggles.")

    if spec.name in {"rgbir-small-temporal", "rgbir-temporal", "rgbir-temporal-track"}:
        cmd.append("--use-temporal" if use_temporal else "--disable-temporal")
    elif use_temporal:
        raise ValueError(f"Mode '{spec.name}' does not support temporal training.")

    cmd.extend(extra_args(args.extra_args))
    manifest = route_manifest(
        subtask="train",
        mode_spec=spec,
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={
            "command": cmd,
            "resolved_features": {
                "use_rgbir": use_rgbir,
                "use_small": use_small,
                "use_temporal": use_temporal,
            },
        },
    )
    return execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)


def validate_model_source_for_inference(spec: ModeSpec, weights: str | None, model: str | None, subtask: str) -> str:
    """Return the correct model source, requiring checkpoints for staged inference routes."""
    if spec.name == "baseline":
        return str(Path(weights).resolve()) if weights else str(Path(model).resolve() if model else spec.default_model)
    if weights is None:
        raise ValueError(f"--weights is required for '{subtask}' in mode '{spec.name}' to preserve the staged path.")
    return str(Path(weights).resolve())


def val_route(args: argparse.Namespace) -> Any:
    spec = resolve_mode(args.mode)
    if not args.data:
        raise ValueError("--data is required for val.")
    project_dir, run_name, run_dir = resolve_output_dir(
        mode=spec.name,
        subtask="val",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=args.exist_ok,
    )
    if spec.name == "baseline":
        model_source = validate_model_source_for_inference(spec, args.weights, args.model, "val")
        kwargs = {
            "data": args.data,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "plots": args.plots,
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": True,
            "console_epoch_summary_only": args.console_epoch_summary_only,
        }
        manifest = route_manifest(
            subtask="val",
            mode_spec=spec,
            project_dir=project_dir,
            run_name=run_name,
            run_dir=run_dir,
            route_type="ultralytics_api",
            payload={"action": "val", "model_source": model_source, "kwargs": kwargs},
        )
        return execute_yolo_call(
            run_dir=run_dir,
            dry_run=args.dry_run,
            print_route=args.print_route,
            manifest=manifest,
            model_source=model_source,
            action="val",
            kwargs=kwargs,
        )

    if args.weights is None:
        raise ValueError(f"--weights is required for val in mode '{spec.name}'.")
    model_path = str(Path(args.model).resolve()) if args.model else str(spec.default_model)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "val_uav_rgbir_obb.py"),
        "--mode",
        spec.name,
        "--data",
        args.data,
        "--weights",
        str(Path(args.weights).resolve()),
        "--model",
        model_path,
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--project",
        str(project_dir),
        "--name",
        run_name,
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.plots:
        cmd.append("--plots")
    if spec.use_small:
        cmd.append("--enable-small-object-metrics")
    else:
        cmd.append("--disable-small-object-metrics")
    manifest = route_manifest(
        subtask="val",
        mode_spec=spec,
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={
            "command": cmd,
            "resolved_features": {
                "use_small_metrics": spec.use_small,
                "use_temporal_val": False,
            },
        },
    )
    return execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)


def predict_route(args: argparse.Namespace) -> Any:
    spec = resolve_mode(args.mode)
    project_dir, run_name, run_dir = resolve_output_dir(
        mode=spec.name,
        subtask="predict",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=args.exist_ok,
    )
    model_source = validate_model_source_for_inference(spec, args.weights, args.model, "predict")
    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save": True,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
    }
    manifest = route_manifest(
        subtask="predict",
        mode_spec=spec,
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="ultralytics_api",
        payload={"action": "predict", "model_source": model_source, "kwargs": kwargs},
    )
    return execute_yolo_call(
        run_dir=run_dir,
        dry_run=args.dry_run,
        print_route=args.print_route,
        manifest=manifest,
        model_source=model_source,
        action="predict",
        kwargs=kwargs,
    )


def temporal_predict_route(args: argparse.Namespace) -> int:
    spec = resolve_mode(args.mode)
    if not spec.supports_temporal_predict:
        raise ValueError(
            f"Mode '{spec.name}' does not support temporal-predict. "
            "Use 'rgbir-small-temporal', 'rgbir-temporal', or 'rgbir-temporal-track'."
        )
    project_dir, run_name, run_dir = resolve_output_dir(
        mode=spec.name,
        subtask="temporal_predict",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=False,
    )
    model_path = str(Path(args.model).resolve()) if args.model else str(spec.default_model)
    use_temporal = bool_override(spec.use_temporal, args.enable_temporal, args.disable_temporal)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "predict_uav_temporal_obb.py"),
        "--source",
        args.source,
        "--model",
        model_path,
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--sequence",
    ]
    if args.weights:
        cmd.extend(["--weights", str(Path(args.weights).resolve())])
    if args.data:
        cmd.extend(["--data", args.data])
    if args.max_frames > 0:
        cmd.extend(["--max-frames", str(args.max_frames)])
    cmd.append("--use-temporal" if use_temporal else "--disable-temporal")
    cmd.extend(extra_args(args.extra_args))
    manifest = route_manifest(
        subtask="temporal_predict",
        mode_spec=spec,
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={"command": cmd, "resolved_features": {"use_temporal": use_temporal}},
    )
    return execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)


def track_route(args: argparse.Namespace) -> Any:
    spec = resolve_mode(args.mode)
    project_dir, run_name, run_dir = resolve_output_dir(
        mode=spec.name,
        subtask="track",
        project=args.project,
        name=args.name,
        exp_tag=args.exp_tag,
        save_dir=args.save_dir,
        exist_ok=args.exist_ok,
    )

    if spec.name == "baseline":
        model_source = validate_model_source_for_inference(spec, args.weights, args.model, "track")
        tracker_path = str(Path(args.tracker).resolve()) if args.tracker else str(spec.default_tracker)
        kwargs = {
            "source": args.source,
            "tracker": tracker_path,
            "imgsz": args.imgsz,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "save": True,
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": True,
        }
        manifest = route_manifest(
            subtask="track",
            mode_spec=spec,
            project_dir=project_dir,
            run_name=run_name,
            run_dir=run_dir,
            route_type="ultralytics_api",
            payload={"action": "track", "model_source": model_source, "kwargs": kwargs},
        )
        return execute_yolo_call(
            run_dir=run_dir,
            dry_run=args.dry_run,
            print_route=args.print_route,
            manifest=manifest,
            model_source=model_source,
            action="track",
            kwargs=kwargs,
        )

    if not spec.supports_stage6_track:
        raise ValueError("Stage 6 tracking is exposed only through mode 'rgbir-temporal-track'.")

    model_path = str(Path(args.model).resolve()) if args.model else str(spec.default_model)
    tracker_path = str(Path(args.tracker).resolve()) if args.tracker else str(spec.default_tracker)
    use_temporal_detector = bool_override(spec.use_temporal, args.enable_temporal_detector, args.disable_temporal_detector)
    save_json = args.save_json or str(run_dir / "tracks.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "track_uav_obb.py"),
        "--source",
        args.source,
        "--model",
        model_path,
        "--tracker",
        tracker_path,
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--save-json",
        save_json,
    ]
    if args.weights:
        cmd.extend(["--weights", str(Path(args.weights).resolve())])
    if args.data:
        cmd.extend(["--data", args.data])
    if args.max_frames > 0:
        cmd.extend(["--max-frames", str(args.max_frames)])
    cmd.append("--use-temporal-detector" if use_temporal_detector else "--disable-temporal-detector")
    cmd.extend(extra_args(args.extra_args))
    manifest = route_manifest(
        subtask="track",
        mode_spec=spec,
        project_dir=project_dir,
        run_name=run_name,
        run_dir=run_dir,
        route_type="subprocess",
        payload={"command": cmd, "save_json": save_json, "resolved_features": {"use_temporal_detector": use_temporal_detector}},
    )
    return execute_subprocess(cmd, run_dir=run_dir, dry_run=args.dry_run, print_route=args.print_route, manifest=manifest)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "preprocess":
        preprocess_route(args)
    elif args.subcommand == "check-data":
        check_data_route(args)
    elif args.subcommand == "train":
        train_route(args)
    elif args.subcommand == "val":
        val_route(args)
    elif args.subcommand == "predict":
        predict_route(args)
    elif args.subcommand == "temporal-predict":
        temporal_predict_route(args)
    elif args.subcommand == "track":
        track_route(args)
    else:
        raise ValueError(f"Unsupported subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
