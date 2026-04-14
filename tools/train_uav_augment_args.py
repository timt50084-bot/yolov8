from __future__ import annotations

import argparse
from typing import Any


def parse_name_list(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize comma-separated or sequence input into a clean list of names."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def add_train_augment_args(parser: argparse.ArgumentParser) -> None:
    """Register conservative Stage 7 train-only augmentation arguments."""
    parser.add_argument("--enable-cmcp", dest="enable_cmcp", action="store_true", help="Enable CMCP copy-paste.")
    parser.add_argument("--disable-cmcp", dest="enable_cmcp", action="store_false", help="Disable CMCP copy-paste.")
    parser.set_defaults(enable_cmcp=False)
    parser.add_argument("--cmcp-prob", default=0.15, type=float, help="CMCP application probability.")
    parser.add_argument("--cmcp-max-pastes", default=3, type=int, help="Maximum pasted instances per image.")
    parser.add_argument("--cmcp-small-area-thr", default=1024.0, type=float, help="Maximum object area in pixels.")
    parser.add_argument("--cmcp-num-trials", default=15, type=int, help="Placement retries per pasted instance.")

    parser.add_argument("--enable-mrre", dest="enable_mrre", action="store_true", help="Enable MRRE local perturbation.")
    parser.add_argument("--disable-mrre", dest="enable_mrre", action="store_false", help="Disable MRRE local perturbation.")
    parser.set_defaults(enable_mrre=False)
    parser.add_argument("--mrre-prob", default=0.20, type=float, help="MRRE application probability.")
    parser.add_argument("--mrre-radius-ratio", default=1.5, type=float, help="MRRE neighborhood radius ratio.")
    parser.add_argument("--mrre-num-regions", default=2, type=int, help="Maximum perturbed regions per image.")
    parser.add_argument("--mrre-strength", default=0.35, type=float, help="MRRE perturbation strength.")

    parser.add_argument("--enable-pc-mwa", dest="enable_pc_mwa", action="store_true", help="Enable PC-MWA weather augmentation.")
    parser.add_argument("--disable-pc-mwa", dest="enable_pc_mwa", action="store_false", help="Disable PC-MWA weather augmentation.")
    parser.set_defaults(enable_pc_mwa=False)
    parser.add_argument("--pc-mwa-prob", default=0.20, type=float, help="PC-MWA application probability.")
    parser.add_argument("--pc-mwa-types", default="fog,rain,low_light", type=str, help="Comma-separated weather types.")
    parser.add_argument("--pc-mwa-severity-min", default=0.20, type=float, help="Minimum PC-MWA severity.")
    parser.add_argument("--pc-mwa-severity-max", default=0.60, type=float, help="Maximum PC-MWA severity.")
    parser.add_argument(
        "--pc-mwa-shared-severity",
        dest="pc_mwa_shared_severity",
        action="store_true",
        help="Share one weather severity across RGB, IR, and temporal context.",
    )
    parser.add_argument(
        "--disable-pc-mwa-shared-severity",
        dest="pc_mwa_shared_severity",
        action="store_false",
        help="Allow independent weather severity per modality.",
    )
    parser.set_defaults(pc_mwa_shared_severity=True)


def augment_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build trainer overrides for the Stage 7 train-only augmentation pipeline."""
    return {
        "enable_cmcp": args.enable_cmcp,
        "cmcp_prob": args.cmcp_prob,
        "cmcp_max_pastes": args.cmcp_max_pastes,
        "cmcp_small_area_thr": args.cmcp_small_area_thr,
        "cmcp_num_trials": args.cmcp_num_trials,
        "enable_mrre": args.enable_mrre,
        "mrre_prob": args.mrre_prob,
        "mrre_radius_ratio": args.mrre_radius_ratio,
        "mrre_num_regions": args.mrre_num_regions,
        "mrre_strength": args.mrre_strength,
        "enable_pc_mwa": args.enable_pc_mwa,
        "pc_mwa_prob": args.pc_mwa_prob,
        "pc_mwa_types": parse_name_list(args.pc_mwa_types),
        "pc_mwa_severity_min": args.pc_mwa_severity_min,
        "pc_mwa_severity_max": args.pc_mwa_severity_max,
        "pc_mwa_shared_severity": args.pc_mwa_shared_severity,
    }
