# SkyGuard

Implementation of **SkyGuard: Mission-Aware Risk-Calibrated Predictive Handover**. A common temporal predictor feeds rate lower bounds and outage upper margins into a risk-feasible handover rule with hysteresis, time-to-trigger and minimum dwell.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Validation environment: Python 3.13.5, CPU with one PyTorch thread. Run `python -m unittest discover -s tests -v`: two tests exercise generated NPZ data, one training epoch (seed 42), finite loss, nonzero gradients and parameter updates, exact same-input evaluation after state restoration, all seven controllers, four ablations, JSON aggregation and plots. Quantiles and constant-service metrics are checked against hand-computed values. The tiny generated scenarios are functional checks, not the paper's experimental results. Full training and the reported three-seed study have not been rerun.

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

Training writes `model.pt` and `history.json`. Evaluation writes per-trace JSON decisions, `metrics.json`, per-condition cross-seed mean/sample-standard-deviation `summary.json`, and a `service.png` trace. Each trace reports packet deadline misses, rolling-horizon mission misses, radio outages, handovers/minute, ping-pong and handover-failure rates, mean and fifth-percentile throughput, rate-interval coverage and width, rolling risk-budget violations, and measured decision latency. All ratios are fractions. Coverage tests the full access-point/horizon rate forecast at each origin. Latency includes forecasting, calibration and control on the current device.

Risk-budget violation uses the empirical frequency of any deadline miss within a horizon, over rolling windows of `--violation-window` such events. It is distinct from counting individual missed packets. Zero handovers produce zero ping-pong and handover-failure rates. Insufficient scenario duration gives `null` for a missing rolling-window statistic. An unbounded interval is represented by `null` width, while coverage still evaluates its full interval.

## Paper-to-code scope

| Paper component | Entry point | Command/configuration |
| --- | --- | --- |
| Multi-step temporal rate/risk prediction | `Predictor`, `train` | `skyguard train`, L=20, H=10, Adam 1e-3 |
| Mondrian/sliding conformal calibration | `bucket`, `calibrate`, `initial_scores` | speed 14 m/s, altitude 90 m, alpha .05, W=200 |
| Risk union bound, utility, hysteresis/TTT/dwell | `control_trace` | `skyguard evaluate`, delta .05, TTT=3, dwell=12 |
| All reported metric families and plots | `control_trace`, `evaluate` | `metrics.json`, `summary.json`, `service.png` |

This repository contains an implementation; no original training checkpoint or raw experiment log was available for comparison. The procedural generator uses a log-distance link model and Markov blockage, rather than a verified reconstruction of the paper's building/sector/3GPP simulation. The risk head is trained on `rate < required_rate` as a service-outage proxy; backhaul-delay and switching-interruption deadline failures are included in final metrics but not forecast by this head. Consequently its risk bound is not a certificate for the full measured mission event. Rate calibration uses a conservative maximum over candidate APs and all sampled origins per calibration trajectory; online updates use completed horizons only. The GRU width 48, dropout .15, Huber plus 5×binary-cross-entropy loss, physical feature scales and control weights are explicit implementation choices where a frozen original setting was unavailable. Training/dev/cal/test seeds are disjoint; evaluation outcomes update only the causal online buffer, never predictor weights or tuned thresholds.
