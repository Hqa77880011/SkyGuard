# SkyGuard

Implementation of **SkyGuard: Mission-Aware Risk-Calibrated Predictive Handover**. A common temporal predictor feeds rate lower bounds and outage upper margins into a risk-feasible handover rule with hysteresis, time-to-trigger and minimum dwell.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

`prepare` generates radio/link trajectories from `simulate` in `skyguard/cli.py`; these are numerical scenarios. `configs/default.json` fixes partition-disjoint seeds, slots and stress conditions.

```bash
skyguard prepare --config configs/default.json --output data/prepared
```

## Training and inference

```bash
skyguard train --data data/prepared --output runs/train --seed 42
```

## Evaluation

```bash
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method strongest --output runs/strongest
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method hysteresis --output runs/hysteresis
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method predictive --output runs/predictive
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method dropout --output runs/dropout
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method static --output runs/static
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --method oracle --output runs/oracle
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --no-group --output runs/no_group
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --no-update --output runs/no_update
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --no-dwell --output runs/no_dwell
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --no-risk --output runs/no_risk
```

## Outputs

Training writes `model.pt` and `history.json`. Evaluation writes per-trace JSON decisions, `metrics.json` and a `service.png` trace. Each trace reports packet deadline misses, rolling-horizon mission misses, radio outages, handovers/minute, ping-pong and handover-failure rates, mean and fifth-percentile throughput, rate-interval coverage and width, rolling risk-budget violations, and measured decision latency. All ratios are fractions. Coverage tests the full access-point/horizon rate forecast at each origin. Latency includes forecasting, calibration and control on the current device.

Risk-budget violation uses the empirical frequency of any deadline miss within a horizon, over rolling windows of `--violation-window` such events. It is distinct from counting individual missed packets. Zero handovers produce zero ping-pong and handover-failure rates. Insufficient scenario duration gives `null` for a missing rolling-window statistic. An unbounded interval is represented by `null` width, while coverage still evaluates its full interval.
