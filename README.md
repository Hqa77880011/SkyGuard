# SkyGuard

Implementation of **SkyGuard: Mission-Aware Risk-Calibrated Predictive Handover**. A common temporal predictor feeds rate lower bounds and outage upper margins into a risk-feasible handover rule with hysteresis, time-to-trigger and minimum dwell.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Run checks with `python -m unittest discover -s tests -v`.

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

## Modules and metrics

| Component | Entry point | Command/configuration |
| --- | --- | --- |
| Multi-step temporal rate/risk prediction | `Predictor`, `train` | `skyguard train`, L=20, H=10, Adam 1e-3 |
| Mondrian/sliding conformal calibration | `bucket`, `calibrate`, `initial_scores` | speed 14 m/s, altitude 90 m, alpha .05, W=200 |
| Risk union bound, utility, hysteresis/TTT/dwell | `control_trace` | `skyguard evaluate`, delta .05, TTT=3, dwell=12 |
| All reported metric families and plots | `control_trace`, `evaluate` | `metrics.json`, `summary.json`, `service.png` |

The predictor trains and calibrates the complete per-horizon critical-packet failure event: required-rate shortfall, backhaul delay, radio outage and first-step switching interruption. Prediction targets conservatively charge the first-step interruption to every candidate; the oracle uses the actual stay/switch mask. Every saved checkpoint binds the event thresholds. Evaluation restores those thresholds. Future realized outcomes enter the sliding buffer only when their full horizons complete. Main-method dwell exceeds the default horizon; the no-dwell ablation changes that controller contract explicitly.

`radio.py` builds street-separated 3-D buildings, performs line-segment/building intersections, and evaluates three site sectors, shadowing, load, noise and inter-site interference. The antenna-element pattern follows [TR 38.901 Table 7.3-1](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/14.02.00_60/tr_138901v140200p.pdf). The configured LOS/NLOS coefficients, 10-degree downtilt and physical scales define procedural sub-6-GHz scenarios. Import measured trajectories through the same data contract.

## Development selection and measured inputs

All seven controllers use the same margin/TTT/dwell grid, development traces and trained predictor. The objective is development horizon-miss ratio plus .001 times handovers/minute. No test labels enter checkpoint or policy selection.

```bash
python -m skyguard.pipeline tune --data data/prepared --checkpoint runs/train/model.pt --input configs/tuning.json --output runs/policy.json
python -m skyguard.pipeline suite --data data/prepared --checkpoint runs/train/model.pt --input runs/policy.json --output runs/comparison
python -m skyguard.pipeline import --input data/measured_manifest.json --output data/measured
```

Measured manifest entries contain `path`, `seed` (independent trajectory identity), `split` (train/dev/cal/test) and `condition`. NPZ files contain `state[T,5]` and `link[T,AP,5]` in the feature order stated above. Imports enforce finite arrays, shape agreement and split-disjoint identities. The suite writes every method's full metric families, per-condition mean/sample-standard-deviation summaries and a mission-risk comparison plot. Existing CLI flags implement no-group/no-update/no-dwell/no-risk ablations on the identical prepared inputs.

Actual extended CPU checks include full deadline target cases, exact geometry occlusion, one-epoch loss/backward/all-parameter updates, same-input checkpoint equality, development-grid selection for seven methods, measured-schema import, all seven controllers and all four ablations. These checks use small generated traces; full training, full radio-field calibration and hardware decision-latency validation have not been run.
