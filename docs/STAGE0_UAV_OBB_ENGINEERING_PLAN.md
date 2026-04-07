# Stage 0 UAV OBB Engineering Plan

## Current Repository Baseline

This repository already contains two critical baseline capabilities that Stage 0 must preserve:

1. Native YOLOv8 OBB model definitions and task flow for `train`, `val`, and `predict`.
2. Native tracking entry points and tracker configs for `mode=track`.

Stage 0 does not change those baseline paths. The purpose of this phase is to prepare safe extension points for later work.

## Future Phases

The intended roadmap after Stage 0 is:

1. RGB-IR preprocessing and dataset tooling.
2. RGB-IR training-time cooperative learning.
3. Lightweight temporal refinement for sequential frames.
4. OBB small-object optimization for UAV imagery.
5. Tracker extensions for UAV-oriented OBB tracking.

These capabilities are intentionally not implemented in Stage 0.

## Why Stage 0 Stops at Skeletons

Directly introducing RGB-IR fusion, temporal reasoning, tracking rewrites, or large loss/head changes would create
avoidable regression risk for the current OBB and tracking baseline. Stage 0 therefore only adds:

- documentation,
- config entry points,
- placeholder modules,
- explicit safety boundaries.

This keeps the change set minimal, mergeable, and low-risk while still making future work more structured.

## Module Boundaries for Later Phases

### Data Preprocessing

Future preprocessing utilities will be responsible for:

- RGB and IR pair alignment checks,
- adjacent-frame sampling metadata,
- OBB label consistency checks,
- train-only auxiliary data packaging.

### RGB-IR Dataset

The future dataset layer will be responsible for loading:

- current RGB image,
- current IR image,
- neighboring RGB frames,
- OBB annotations.

The Stage 0 dataset skeleton documents this contract but is not wired into the default dataset builder.

### Training-Time Cooperative Model

Future RGB-IR model work will support train-time auxiliary fusion only. IR must not become a deployment dependency.

Stage 0 keeps all RGB-IR fields disabled by default and does not alter the runtime RGB inference path.

### Lightweight Temporal Module

Future temporal work will refine detector features with neighboring RGB context. Stage 0 only reserves config keys and
placeholder module files.

### Tracker Extension

Future tracking work may extend:

- OBB-aware matching,
- appearance features,
- track reactivation,
- temporal memory.

Stage 0 adds a tracker config skeleton and placeholder files without replacing the native tracker registry.

### Small-Object Optimization

Future small-object work may involve targeted feature refinement, assigner/loss tuning, or resolution-aware handling
for UAV data. Stage 0 only reserves configuration switches for this area.

## Hard Constraints

The following constraints are mandatory for all later phases:

1. IR is allowed only for training-time cooperation.
2. Single-image and video inference must depend on RGB only.
3. New features must be controlled by explicit configuration switches.
4. All new switches must default to the disabled or no-op state.
5. The existing OBB and tracking baseline must remain intact.
6. Validation, metric computation, and deployment paths must not be polluted by train-only auxiliary branches.

## Stage 0 Deliverables

Stage 0 adds the following engineering skeleton:

- model YAML entry points for RGB-IR and RGB-IR-temporal OBB variants,
- a tracker YAML skeleton for future UAV OBB tracking work,
- dataset and module placeholder files,
- tracker placeholder files,
- an acceptance checklist for future phases.

None of these files are enabled by default in baseline commands.
