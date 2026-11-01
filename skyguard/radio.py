"""Procedural geometry and declared sub-6 GHz urban link equations.

The antenna element pattern follows TR 38.901 Table 7.3-1. Log-distance
path-loss coefficients and height ranges are configured scenario inputs.
"""
import numpy as np


def city(rng, density=0.35, area=1200.0):
    """Return rectangular 3-D buildings separated by a street grid."""
    buildings = []
    for x in np.arange(60, area - 60, 100):
        for y in np.arange(60, area - 60, 100):
            if rng.random() < density:
                width, depth = rng.uniform(25, 65, 2)
                buildings.append([[x, y, 0], [x + width, y + depth, rng.uniform(12, 100)]])
    return np.asarray(buildings, dtype=float).reshape(-1, 2, 3)


def visible(origin, targets, buildings):
    """Exact segment/AABB slab intersection, including building height."""
    result = np.ones(len(targets), dtype=bool)
    for i, target in enumerate(targets):
        delta = np.asarray(target) - origin
        for lower, upper in buildings:
            near, far = 0.0, 1.0
            intersects = True
            for axis in range(3):
                if abs(delta[axis]) < 1e-12:
                    if not lower[axis] <= origin[axis] <= upper[axis]:
                        intersects = False
                        break
                else:
                    a, b = (lower[axis] - origin[axis]) / delta[axis], (upper[axis] - origin[axis]) / delta[axis]
                    near, far = max(near, min(a, b)), min(far, max(a, b))
            if intersects and near <= far and far > 0 and near < 1:
                result[i] = False
                break
    return result


def link_state(position, sites, buildings, load, shadow, blocked,
               frequency_ghz=3.5, bandwidth_mhz=20, tx_dbm=30):
    """Compute site-sector gain, interference-aware SINR, load and delay."""
    offset = np.asarray(position)[None] - sites
    distance = np.maximum(np.linalg.norm(offset, axis=1), 1)
    azimuth = np.degrees(np.arctan2(offset[:, 1], offset[:, 0]))
    elevation = np.degrees(np.arctan2(offset[:, 2], np.linalg.norm(offset[:, :2], axis=1)))
    differences = (azimuth[:, None] - np.array([0, 120, 240]) + 180) % 360 - 180
    horizontal = np.minimum(12 * (differences / 65) ** 2, 30).min(1)
    vertical = np.minimum(12 * ((elevation + 10) / 65) ** 2, 30)
    gain = 8 - np.minimum(horizontal + vertical, 30)
    los = visible(np.asarray(position), sites, buildings) & ~blocked
    loss_los = 28 + 22 * np.log10(distance) + 20 * np.log10(frequency_ghz)
    loss_nlos = 13.54 + 39.08 * np.log10(distance) + 20 * np.log10(frequency_ghz)
    path_loss = np.where(los, loss_los, np.maximum(loss_los, loss_nlos))
    received = 10 ** ((tx_dbm + gain - path_loss + shadow) / 10)
    noise = 10 ** ((-174 + 10 * np.log10(bandwidth_mhz * 1e6) + 7) / 10)
    interference = (received * load).sum() - received * load
    sinr_linear = received / (noise + interference)
    sinr = 10 * np.log10(sinr_linear)
    rate = bandwidth_mhz * np.log2(1 + sinr_linear) * (1 - load)
    return np.stack([sinr, rate, los.astype(float), load, 5 + 70 * load], -1)
