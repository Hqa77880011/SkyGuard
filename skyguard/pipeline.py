"""Frozen development tuning and common-realization comparison commands."""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .cli import control_trace, evaluate, initial_scores, paths
from .method import Predictor

METHODS = ("full", "strongest", "hysteresis", "predictive", "dropout", "static", "oracle")


def options(data, checkpoint, method, output, **values):
    defaults = dict(data=data, checkpoint=checkpoint, method=method, output=output,
                    device="cpu", window=200, dwell=12, trigger=3, ping_window=30,
                    violation_window=50, dropout_passes=20, alpha=.05, delta=.05,
                    discount=.95, risk_weight=20, switch_cost=5, margin=2,
                    outage_sinr=0, interruption=.3, packet_deadline_ms=80,
                    no_group=False, no_update=False, no_dwell=False, no_risk=False)
    return SimpleNamespace(**(defaults | values))


def tune(data, checkpoint, grid, output, handover_penalty=.001):
    """Choose on dev only, using the same declared grid for every controller."""
    if set(grid) - {"margin", "trigger", "dwell"}:
        raise ValueError("Tuning grid supports margin, trigger and dwell")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved.get("failure_target") != "deadline_rate_sinr_switch_v1":
        raise ValueError("Tuning requires the current full packet-event checkpoint")
    net = Predictor(saved["horizon"])
    net.load_state_dict(saved["model"])
    net.eval()
    keys = sorted(grid)
    candidates = [dict(zip(keys, row)) for row in itertools.product(*(grid[k] for k in keys))]
    if not candidates:
        raise ValueError("Empty development grid")
    chosen, scores = {}, {}
    for method in METHODS:
        args = options(data, checkpoint, method, output)
        for key in ("outage_sinr", "interruption", "packet_deadline_ms"):
            setattr(args, key, saved[key])
        buffers = initial_scores(net, args, saved, method == "full")
        rows = []
        for candidate in candidates:
            for key, value in candidate.items():
                setattr(args, key, value)
            measures = []
            for file in paths(data, "dev"):
                with np.load(file) as trace:
                    metrics, _ = control_trace(net, trace, args, saved, buffers)
                measures.append(metrics["mission_horizon_miss_ratio"]
                                + handover_penalty * metrics["handovers_per_minute"])
            rows.append({"parameters": candidate, "dev_objective": float(np.mean(measures))})
        best = min(rows, key=lambda row: row["dev_objective"])
        chosen[method] = best["parameters"]
        scores[method] = rows
    artifact = {"selection_split": "dev", "calibration_split": "cal", "grid": grid,
                "objective": {"mission_horizon_miss": 1, "handovers_per_minute": handover_penalty},
                "methods": chosen, "scores": scores}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def import_traces(manifest, output):
    """Import measured state[T,5]/link[T,AP,5] traces with explicit split IDs."""
    manifest = Path(manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    identities = {}
    for row in rows:
        split, seed = row["split"], int(row["seed"])
        if split not in {"train", "dev", "cal", "test"}:
            raise ValueError("Unknown trajectory split")
        if seed in identities and identities[seed] != split:
            raise ValueError("One trajectory identity occurs across partitions")
        identities[seed] = split
        source = manifest.parent / row["path"]
        with np.load(source, allow_pickle=False) as trace:
            state, link = trace["state"], trace["link"]
        if (state.ndim != 2 or state.shape[1] != 5 or link.ndim != 3
                or link.shape[0] != len(state) or link.shape[2] != 5
                or not np.isfinite(state).all() or not np.isfinite(link).all()):
            raise ValueError(f"Invalid measured trace schema: {source}")
        condition = row.get("condition", "measured")
        if not condition.replace("_", "").isalnum():
            raise ValueError("Condition names must be alphanumeric with underscores")
        target = Path(output) / split / f"{seed}_{condition}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, state=state.astype(np.float32),
                            link=link.astype(np.float32), seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["import", "tune", "suite"])
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--checkpoint", default="runs/train/model.pt")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "import":
        import_traces(args.input, args.output)
    elif args.command == "tune":
        tune(args.data, args.checkpoint, json.loads(Path(args.input).read_text()), args.output)
    else:
        policy = json.loads(Path(args.input).read_text())
        if policy["selection_split"] != "dev" or set(policy["methods"]) != set(METHODS):
            raise ValueError("Expected a frozen development policy for all seven controllers")
        for method in METHODS:
            evaluate(options(args.data, args.checkpoint, method, Path(args.output) / method,
                             **policy["methods"][method]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        for method in METHODS:
            values = json.loads((Path(args.output) / method / "metrics.json").read_text())
            ax.scatter([method] * len(values), [v["mission_horizon_miss_ratio"] for v in values])
        ax.set_ylabel("Mission horizon miss ratio")
        fig.tight_layout()
        fig.savefig(Path(args.output) / "mission.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
