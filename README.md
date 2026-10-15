# SkyGuard

Implementation of **SkyGuard: Mission-Aware Risk-Calibrated Predictive Handover**. A common temporal predictor feeds rate lower bounds and outage upper margins into a risk-feasible handover rule with hysteresis, time-to-trigger and minimum dwell.

## Install and run

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
skyguard prepare --config configs/default.json --output data/prepared
skyguard train --data data/prepared --output runs/train
skyguard evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

The data command generates complete trajectories, not precomputed experiment results. `configs/default.json` declares independent train, development, calibration and test seeds, trajectory duration, and test conditions. It covers in-distribution, changed blockage density, increased speed, abrupt blockage, load shift and position noise. Each condition uses the same test seeds and access-point placement. The simulator uses five access points, 100 ms slots, altitude levels of 60/90/120 m and nominal speeds of 8/14/22 m/s in a 1200 m square.

Every archive contains `state[T,5]` for position, altitude, speed and heading, and `link[T,5,5]` for SINR, rate, line-of-sight score, load and backhaul delay. Real traces can use the same schema. Units are metres, metres/second, radians, dB, Mbps and milliseconds. Keep complete flights in separate partitions.

## Controls and ablations

```bash
skyguard evaluate --method strongest --output runs/strongest
skyguard evaluate --method hysteresis --output runs/hysteresis
skyguard evaluate --method predictive --output runs/predictive
skyguard evaluate --method dropout --output runs/dropout
skyguard evaluate --method static --output runs/static
skyguard evaluate --method oracle --output runs/oracle
skyguard evaluate --no-group --output runs/no_group
skyguard evaluate --no-update --output runs/no_update
skyguard evaluate --no-dwell --output runs/no_dwell
skyguard evaluate --no-risk --output runs/no_risk
```

All predictive policies load the same checkpoint. Strongest uses current SINR; hysteresis adds temporal safeguards. Predictive uses point forecasts. Dropout takes stochastic passes through the same predictor. Static calibration uses one fixed global buffer. The oracle accesses future link outcomes and provides a reference under the same switching constraints.

Training parameters include `--length` (20 history slots), `--horizon` (10 future slots), `--epochs`, `--batch-size`, `--lr`, `--seed` and `--required-rate` (8 Mbps). `--device cuda` enables a compatible GPU installation. The rate forecast uses a GRU and the outage head is trained on failure to reach the required rate.

Evaluation parameters:

| Flag | Default | Use |
| --- | --- | --- |
| `--alpha` | 0.05 | Calibration error level |
| `--delta` | 0.05 | Horizon risk budget |
| `--window` | 200 | Maximum recent completed-horizon scores per bucket |
| `--dwell` | 12 | Minimum slots between switches |
| `--trigger` | 3 | Consecutive eligible slots before switching |
| `--margin` | 2 | Required switching-utility advantage |
| `--switch-cost` | 5 | Switching penalty in forecast utility |
| `--risk-weight` | 20 | Outage penalty in utility |
| `--discount` | 0.95 | Per-step utility discount |
| `--interruption` | 0.3 | Fraction of a slot lost during a switch |
| `--packet-deadline-ms` | 80 | Backhaul plus interruption deadline |
| `--violation-window` | 50 | Horizon-event observations per violation window |
| `--ping-window` | 30 | Return-handover window in slots |
| `--outage-sinr` | 0 | Radio-outage threshold in dB |
| `--dropout-passes` | 20 | Stochastic passes for the dropout control |

If no candidate satisfies the risk budget, the controller keeps its current association. A feasible candidate must still satisfy the utility, dwell and trigger rules. Freeze operating thresholds before test comparisons.

## Metrics and files

Training writes `model.pt` and `history.json`. Evaluation writes per-trace JSON decisions, `metrics.json` and a `service.png` trace. Each trace reports packet deadline misses, rolling-horizon mission misses, radio outages, handovers/minute, ping-pong and handover-failure rates, mean and fifth-percentile throughput, rate-interval coverage and width, rolling risk-budget violations, and measured decision latency. All ratios are fractions. Coverage tests the full access-point/horizon rate forecast at each origin. Latency includes forecasting, calibration and control on the current device.

Risk-budget violation uses the empirical frequency of any deadline miss within a horizon, over rolling windows of `--violation-window` such events. It is distinct from counting individual missed packets. Zero handovers produce zero ping-pong and handover-failure rates. Insufficient scenario duration gives `null` for a missing rolling-window statistic. An unbounded interval is represented by `null` width, while coverage still evaluates its full interval.

## Implementation choices

Initial calibration uses one maximum residual per independent trajectory and speed/altitude bucket. Rate scores maximize absolute error over access points and horizon steps. Risk scores maximize observed binary failure minus predicted failure probability; the smallest additive margin satisfying the corrected empirical exceedance constraint is chosen. This makes the manuscript's threshold-controlled risk set concrete. Upper bounds are clipped to one and horizon feasibility uses their union bound. Speed 14 m/s and altitude 90 m define bucket boundaries.

Online scores enter only after the entire forecast horizon has completed. Sliding buffers adapt empirical margins; their overlapping online windows do not imply exchangeable-flight coverage under arbitrary drift. The procedural radio generator uses log-distance attenuation, correlated shadowing, Markov blockage and load-dependent Shannon service. It is a packet/service simulation rather than a complete cellular channel implementation. Building density is represented through blockage probability. One critical service demand occurs per slot; insufficient service or excessive backhaul/interruption delay creates a deadline miss. These conventions define the measurable quantities used by the controller and comparisons.
