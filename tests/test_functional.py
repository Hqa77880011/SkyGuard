"""Small CPU checks; generated trajectories are not paper performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from skyguard.cli import control_trace, evaluate, prepare, train
from skyguard.method import Predictor, calibrate, features, quantile


class FunctionalTest(unittest.TestCase):
    def test_training_checkpoint_and_controls(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"slots": 28, "seeds": {
                "train": [0], "dev": [100], "cal": [200, 201], "test": [400]},
                "conditions": ["id", "load_shift"]}))
            data = root / "data"
            prepare(SimpleNamespace(config=config, output=data))
            train(SimpleNamespace(seed=42, horizon=3, length=4, device="cpu",
                                  lr=0.001, data=data, output=root / "train",
                                  epochs=1, batch_size=16, required_rate=8))
            saved = torch.load(root / "train/model.pt", weights_only=True)
            self.assertTrue(all(np.isfinite(row[k]) for row in json.loads(
                (root / "train/history.json").read_text()) for k in ("train", "dev")))
            net = Predictor(3)
            initial = {k: v.clone() for k, v in net.state_dict().items()}
            with np.load(data / "test/400_id.npz") as source:
                trace = {key: source[key] for key in source.files}
            x = torch.as_tensor(features(trace, 4, 4))
            rate, logits = net(x)
            loss = rate.square().mean() + logits.square().mean()
            loss.backward()
            for name, parameter in net.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            torch.optim.SGD(net.parameters(), lr=0.01).step()
            self.assertTrue(any(not torch.equal(initial[k], v)
                                for k, v in net.state_dict().items()))
            net.load_state_dict(saved["model"])
            net.eval()
            restored = Predictor(3)
            restored.load_state_dict(saved["model"])
            restored.eval()
            with torch.no_grad():
                for a, b in zip(net(x), restored(x)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            options = dict(data=data, checkpoint=root / "train/model.pt", device="cpu",
                           window=20, dwell=12, trigger=3, ping_window=30,
                           violation_window=5, dropout_passes=2, alpha=0.5,
                           delta=0.05, discount=0.95, risk_weight=20, switch_cost=5,
                           margin=2, outage_sinr=0, interruption=0.3,
                           packet_deadline_ms=80, no_group=False, no_update=False,
                           no_dwell=False, no_risk=False)
            for method in ("full", "strongest", "hysteresis", "predictive",
                           "dropout", "static", "oracle"):
                evaluate(SimpleNamespace(**options, method=method, output=root / method))
                self.assertTrue((root / method / "service.png").is_file())
                summary = json.loads((root / method / "summary.json").read_text())
                self.assertEqual(summary["id"]["average_mbps"]["seeds_with_value"], 1)
            for flag in ("no_group", "no_update", "no_dwell", "no_risk"):
                evaluate(SimpleNamespace(**{**options, flag: True}, method="full",
                                         output=root / flag))

    def test_quantiles_and_hand_computed_metrics(self):
        self.assertEqual(quantile([1, 2, 3], 0.5), 2)
        self.assertTrue(np.isinf(quantile([1, 2], 0.05)))
        self.assertEqual(calibrate([1, 2, 3], [0.1, 0.2, 0.3], 0.5), (2, 0.2))
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x), 2), 10.0), torch.full((len(x), 2), -10.0)
        trace = {"state": np.zeros((10, 5), np.float32),
                 "link": np.zeros((10, 5, 5), np.float32)}
        trace["link"][:, :, 1] = 10
        args = SimpleNamespace(method="predictive", no_group=True, no_update=True,
                               no_dwell=True, no_risk=True, dwell=0, window=5,
                               discount=1, risk_weight=0, switch_cost=0, delta=0.05,
                               trigger=1, margin=0, ping_window=3, interruption=0.3,
                               outage_sinr=0, packet_deadline_ms=80,
                               violation_window=2, device="cpu")
        metrics, rows = control_trace(Constant(), trace, args,
                                     {"horizon": 2, "length": 2, "required_rate": 8},
                                     {i: [] for i in range(4)})
        self.assertEqual(len(rows), 7)
        self.assertEqual(metrics["average_mbps"], 10)
        for key in ("packet_deadline_miss_ratio", "mission_horizon_miss_ratio",
                    "outage_probability", "handovers_per_minute", "pingpong_rate",
                    "handover_failure_rate", "risk_budget_violation"):
            self.assertEqual(metrics[key], 0, key)


if __name__ == "__main__":
    unittest.main()
