import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .method import Predictor, bucket, calibrate, features


def simulate(seed, slots, condition):
    rng = np.random.default_rng(seed)
    ap = rng.uniform(0, 1200, (5, 2))
    state, links = [], []
    position = rng.uniform(100, 1100, 2)
    heading = rng.uniform(-np.pi, np.pi)
    speed = float(rng.choice([8, 14, 22])) * (1.6 if condition == "cross_speed" else 1)
    altitude = float(rng.choice([60, 90, 120]))
    blockage = np.zeros(5, bool)
    shadow = np.zeros(5)
    load = rng.uniform(0.1, 0.6, 5)
    for t in range(slots):
        heading += rng.normal(0, 0.03)
        position = position + 0.1 * speed * np.array([np.cos(heading), np.sin(heading)])
        if np.any((position < 0) | (position > 1200)):
            position = np.clip(position, 0, 1200)
            heading += np.pi
        density = 0.04 if condition == "cross_map" else 0.02
        enter, leave = rng.random(5) < density, rng.random(5) < 0.12
        blockage = (blockage | enter) & ~leave
        if condition == "blockage" and slots // 3 <= t < slots // 3 + 40:
            blockage[0] = True
        shadow = 0.97 * shadow + rng.normal(0, 1.1, 5)
        load = np.clip(0.98 * load + 0.02 * rng.uniform(0.1, 0.7, 5) + rng.normal(0, 0.01, 5), 0.05, 0.95)
        if condition == "load_shift" and t >= slots // 2:
            load[0] = 0.95
        distance = np.sqrt(((ap - position) ** 2).sum(1) + altitude ** 2)
        sinr = 38 - 22 * np.log10(distance / 30) + shadow - 23 * blockage
        rate = 20 * np.log2(1 + 10 ** (sinr / 10)) * (1 - load)
        los = np.clip(1 - distance / 1800 - 0.5 * blockage, 0.05, 0.95)
        delay = 5 + 70 * load
        measured = position + (rng.normal(0, 25, 2) if condition == "position_noise" else 0)
        state.append([*measured, altitude, speed, (heading + np.pi) % (2 * np.pi) - np.pi])
        links.append(np.stack([sinr, rate, los, load, delay], -1))
    return np.array(state, np.float32), np.array(links, np.float32)


def prepare(args):
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    used = set()
    for split, seeds in config["seeds"].items():
        if used & set(seeds):
            raise ValueError("Trajectory seeds overlap partitions")
        used.update(seeds)
        conditions = config["conditions"] if split == "test" else ["id"]
        target = out / split
        target.mkdir(exist_ok=True)
        for seed in seeds:
            for condition in conditions:
                state, link = simulate(seed, config["slots"], condition)
                np.savez_compressed(target / f"{seed}_{condition}.npz", state=state, link=link, seed=seed)


def paths(data, split):
    files = sorted((Path(data) / split).glob("*.npz"))
    if not files:
        raise ValueError(f"Missing traces: {split}")
    return files


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = Predictor(args.horizon).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_paths, dev_paths = paths(args.data, "train"), paths(args.data, "dev")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        scores = {}
        for split, files in [("train", train_paths), ("dev", dev_paths)]:
            net.train(split == "train")
            losses = []
            for file in files:
                trace = np.load(file)
                indices = np.arange(args.length - 1, len(trace["state"]) - args.horizon)
                if not len(indices):
                    raise ValueError(f"Trace {file} needs at least length + horizon slots")
                if split == "train":
                    indices = rng.permutation(indices)
                for start in range(0, len(indices), args.batch_size):
                    times = indices[start:start + args.batch_size]
                    x = np.concatenate([features(trace, t, args.length) for t in times])
                    y = np.concatenate([trace["link"][t + 1:t + args.horizon + 1, :, 1].T for t in times])
                    rate = torch.as_tensor(y, device=args.device)
                    with torch.set_grad_enabled(split == "train"):
                        forecast, logits = net(torch.as_tensor(x, device=args.device))
                        loss = F.smooth_l1_loss(forecast, rate) + 5 * F.binary_cross_entropy_with_logits(logits, (rate < args.required_rate).float())
                        if split == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    losses.append(float(loss.detach()))
            scores[split] = float(np.mean(losses))
        history.append({"epoch": epoch + 1, **scores})
        if scores["dev"] < best:
            best = scores["dev"]
            torch.save({"model": net.state_dict(), "horizon": args.horizon, "length": args.length,
                        "required_rate": args.required_rate}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False))


def forecast(net, trace, t, saved, device):
    rate, logits = net(torch.as_tensor(features(trace, t, saved["length"]), device=device))
    return rate.cpu().numpy(), logits.sigmoid().cpu().numpy()


def initial_scores(net, args, saved, grouped):
    buffers = {g: [] for g in range(4)}
    with torch.no_grad():
        for path in paths(args.data, "cal"):
            trace = np.load(path)
            per_group = {}
            for t in range(saved["length"] - 1, len(trace["state"]) - saved["horizon"], saved["horizon"]):
                rate, probability = forecast(net, trace, t, saved, args.device)
                truth = trace["link"][t + 1:t + saved["horizon"] + 1, :, 1].T
                g = bucket(trace["state"][t]) if grouped else 0
                pair = (float(abs(rate - truth).max()), float(((truth < saved["required_rate"]) - probability).max()))
                per_group.setdefault(g, []).append(pair)
            for g, values in per_group.items():
                buffers[g].append(tuple(np.max(values, axis=0)))
    return buffers


def control_trace(net, trace, args, saved, original):
    grouped = args.method == "full" and not args.no_group
    buffers = {g: list(values) for g, values in original.items()}
    pending, rows = [], []
    serving, previous, last_switch, target, counter = 0, -1, -args.dwell, -1, 0
    switches, failures, pingpong = 0, 0, 0
    H, L = saved["horizon"], saved["length"]
    if len(trace["state"]) < L + 2 * H - 1:
        raise ValueError("Evaluation needs length + 2 * horizon - 1 slots for horizon metrics")
    for t in range(L - 1, len(trace["state"]) - H):
        begin = perf_counter()
        while pending and pending[0][0] <= t:
            end, g, r, p, start = pending.pop(0)
            truth = trace["link"][start + 1:end + 1, :, 1].T
            pair = (float(abs(r - truth).max()), float(((truth < saved["required_rate"]) - p).max()))
            if args.method == "full" and not args.no_update:
                buffers[g].append(pair)
                buffers[g] = buffers[g][-args.window:]
        g = bucket(trace["state"][t]) if grouped else 0
        with torch.no_grad():
            rate, probability = forecast(net, trace, t, saved, args.device)
            if args.method == "dropout":
                net.train()
                draws = [forecast(net, trace, t, saved, args.device) for _ in range(args.dropout_passes)]
                net.eval()
                dr, dp = map(np.asarray, zip(*draws))
                lower = np.maximum(dr.mean(0) - 2 * dr.std(0), 0)
                upper = np.minimum(dp.mean(0) + 2 * dp.std(0), 1)
            elif args.method in {"full", "static"}:
                values = buffers[g]
                q, margin = calibrate([v[0] for v in values], [v[1] for v in values], args.alpha)
                lower = np.maximum(rate - q, 0)
                upper = np.minimum(probability + margin, 1)
            else:
                q = 0
                lower, upper = rate, probability
        pending.append((t + H, g, rate, probability, t))
        if args.method == "oracle":
            lower = trace["link"][t + 1:t + H + 1, :, 1].T
            upper = (lower < saved["required_rate"]).astype(float)
        risk = upper.sum(1).clip(0, 1)
        discount = args.discount ** np.arange(H)
        utility = lower @ discount - args.risk_weight * (upper @ discount)
        utility -= args.switch_cost * (np.arange(len(utility)) != serving)
        feasible = risk <= args.delta
        if args.no_risk:
            feasible[:] = True
        best = int(np.argmax(np.where(feasible, utility, -np.inf))) if feasible.any() else serving
        if args.method in {"strongest", "hysteresis"}:
            utility = trace["link"][t, :, 0]
            best = int(utility.argmax())
        dwell = 0 if args.no_dwell or args.method == "strongest" else args.dwell
        trigger = 1 if args.method == "strongest" else args.trigger
        margin = 0 if args.method == "strongest" else args.margin
        eligible = best != serving and utility[best] - utility[serving] >= margin and t - last_switch >= dwell
        counter = counter + 1 if eligible and best == target else int(eligible)
        target = best
        switched = eligible and counter >= trigger
        if switched:
            pingpong += int(best == previous and t - last_switch <= args.ping_window)
            previous, serving, last_switch = serving, best, t
            switches += 1
            failures += int(trace["link"][t + 1, serving, 0] < args.outage_sinr)
            counter = 0
        interruption = args.interruption if switched else 0
        actual_rate = float(trace["link"][t + 1, serving, 1]) * (1 - interruption)
        deadline_miss = bool(actual_rate < saved["required_rate"] or trace["link"][t + 1, serving, 4] + 100 * interruption > args.packet_deadline_ms)
        truth = trace["link"][t + 1:t + H + 1, :, 1].T
        covered = bool(np.all(np.abs(truth - rate) <= q)) if args.method in {"full", "static"} else None
        rows.append({"slot": t, "serving": serving, "rate_mbps": actual_rate, "miss": deadline_miss,
                     "outage": bool(trace["link"][t + 1, serving, 0] < args.outage_sinr),
                     "handover": bool(switched), "risk_bound": float(risk[serving]), "covered": covered,
                     "interval_width": float(2 * q) if args.method in {"full", "static"} and np.isfinite(q) else None,
                     "decision_ms": 1000 * (perf_counter() - begin)})
    miss = np.array([r["miss"] for r in rows])
    horizon_miss = np.array([miss[t:t + H].any() for t in range(len(miss) - H + 1)])
    violation = [horizon_miss[t:t + args.violation_window].mean() > args.delta for t in range(max(0, len(horizon_miss) - args.violation_window + 1))]
    rates = np.array([r["rate_mbps"] for r in rows])
    valid_width = [r["interval_width"] for r in rows if r["interval_width"] is not None]
    metrics = {"packet_deadline_miss_ratio": float(miss.mean()), "mission_horizon_miss_ratio": float(horizon_miss.mean()),
               "outage_probability": float(np.mean([r["outage"] for r in rows])),
               "handovers_per_minute": switches / (len(rows) * 0.1 / 60),
               "handover_failure_rate": failures / switches if switches else 0,
               "pingpong_rate": pingpong / switches if switches else 0,
               "average_mbps": float(rates.mean()), "p5_mbps": float(np.quantile(rates, 0.05)),
               "risk_budget_violation": float(np.mean(violation)) if violation else None,
               "coverage": float(np.mean([r["covered"] for r in rows])) if args.method in {"full", "static"} else None,
               "mean_interval_width_mbps": float(np.mean(valid_width)) if valid_width else None,
               "mean_decision_ms": float(np.mean([r["decision_ms"] for r in rows]))}
    return metrics, rows


def evaluate(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    net = Predictor(saved["horizon"]).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    grouped = args.method == "full" and not args.no_group
    buffers = initial_scores(net, args, saved, grouped)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in paths(args.data, "test"):
        metrics, rows = control_trace(net, np.load(path), args, saved, buffers)
        summaries.append({"trace": path.stem, **metrics})
        (out / f"{path.stem}.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    aggregate = {}
    for condition in sorted({r["trace"].split("_", 1)[1] for r in summaries}):
        group = [r for r in summaries if r["trace"].split("_", 1)[1] == condition]
        aggregate[condition] = {}
        for key in group[0]:
            if key == "trace":
                continue
            values = [r[key] for r in group if r[key] is not None]
            aggregate[condition][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "seeds_with_value": len(values),
            }
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([r["slot"] * 0.1 for r in rows], [r["rate_mbps"] for r in rows])
    ax.set(xlabel="Time (s)", ylabel="Delivered service (Mbps)")
    fig.tight_layout()
    fig.savefig(out / "service.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--output", default="data/prepared")
    p.set_defaults(function=prepare)
    for name in ["train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--device", default="cpu")
        if name == "train":
            for key, value in [("epochs", 30), ("length", 20), ("horizon", 10), ("batch-size", 32), ("seed", 42)]:
                p.add_argument("--" + key, type=int, default=value)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--required-rate", type=float, default=8)
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--method", choices=["full", "strongest", "hysteresis", "predictive", "dropout", "static", "oracle"], default="full")
            for key, value in [("window", 200), ("dwell", 12), ("trigger", 3), ("ping-window", 30), ("violation-window", 50), ("dropout-passes", 20)]:
                p.add_argument("--" + key, type=int, default=value)
            for key, value in [("alpha", 0.05), ("delta", 0.05), ("discount", 0.95), ("risk-weight", 20), ("switch-cost", 5), ("margin", 2), ("outage-sinr", 0), ("interruption", 0.3), ("packet-deadline-ms", 80)]:
                p.add_argument("--" + key, type=float, default=value)
            for flag in ["no-group", "no-update", "no-dwell", "no-risk"]:
                p.add_argument("--" + flag, action="store_true")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
