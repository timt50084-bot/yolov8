# Stage 0 Acceptance Checklist

## Engineering Safety

- Baseline OBB behavior remains unchanged.
- Baseline tracking behavior remains unchanged.
- New functionality is disabled by default.
- Placeholder files do not take over the default train, val, predict, or track paths.

## Required Checks for Every Later Phase

Each future phase must complete at least a minimal train smoke test and val smoke test when usable data is available.

## Training Stability Checks

Each modification must check whether:

- loss is finite,
- no loss term is NaN or inf,
- no critical loss term is stuck at zero unexpectedly,
- no loss term becomes obviously invalid or negative when it should be non-negative,
- no early loss explosion appears relative to the baseline,
- no abnormal loss stagnation appears,
- no sustained abnormal oscillation appears.

## Precision and Recall Checks

Each modification must check whether:

- Precision and Recall are both produced normally,
- Precision/Recall balance remains plausible,
- there is no obvious false-positive-heavy precision collapse,
- there is no obvious false-negative-heavy recall collapse,
- metric behavior is not significantly worse than the baseline under the same smoke setting.

## mAP Checks

Each modification must check whether:

- mAP50 and mAP50-95 are produced normally,
- mAP is finite and numerically plausible,
- early smoke-test mAP is not abnormally degraded relative to the baseline,
- there is no sign that matching or evaluation paths are broken.

## Path Integrity Checks

Each modification must check whether:

- label formatting changes can affect evaluation,
- OBB output formatting can affect matching and metric aggregation,
- config placeholders accidentally route validation or prediction into unfinished code,
- original baseline command paths still work.

## Decision Rule

The following are not sufficient by themselves:

- "the program did not crash",
- "metrics printed once",
- "there was no NaN or inf".

Any obvious loss instability, training stagnation, precision-recall imbalance, metric collapse, or unexplained
degradation relative to baseline should be treated as an abnormal risk and reported explicitly.
