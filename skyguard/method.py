import math

import numpy as np
import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, horizon, width=48):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(10, width, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(width, horizon * 2)

    def forward(self, x):
        _, hidden = self.encoder(x)
        output = self.head(self.dropout(hidden[-1])).reshape(-1, self.horizon, 2)
        return torch.nn.functional.softplus(output[..., 0]), output[..., 1]


def quantile(values, alpha):
    values = np.sort(np.asarray(values, dtype=float))
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(values[rank - 1]) if rank <= len(values) else float("inf")


def bucket(state):
    return int(state[3] >= 14) * 2 + int(state[2] >= 90)


def features(trace, t, length):
    state = trace["state"][t - length + 1:t + 1]
    link = trace["link"][t - length + 1:t + 1]
    states = np.broadcast_to(state[:, None], (*link.shape[:2], 5))
    x = np.concatenate([states, link], -1).transpose(1, 0, 2).copy()
    x /= np.array([1200, 1200, 120, 22, np.pi, 30, 50, 1, 1, 100], dtype=np.float32)
    return x.astype(np.float32)


def calibrate(rate_scores, risk_scores, alpha):
    q = quantile(rate_scores, alpha)
    values = np.sort(np.maximum(risk_scores, 0))
    if len(values) + 1 < 1 / alpha:
        return q, 1.0
    candidates = np.unique(np.r_[0, values, 1])
    margin = 1.0
    for candidate in candidates:
        loss = np.sum(np.asarray(risk_scores) > candidate)
        if (loss + 1) / (len(values) + 1) <= alpha:
            margin = float(candidate)
            break
    return q, margin
