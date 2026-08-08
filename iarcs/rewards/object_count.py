"""Object count reward.

Encourages realistic object counts by comparing to the empirical training
distribution. Three modes: KL divergence (recommended, prevents mode
collapse), NLL (per-scene), or Gaussian (legacy).
"""

import torch


# Empirical distribution from 3,722 training scenes.
TRAINING_PROBS = torch.tensor([
    0.0001,   # 0 objects (smoothing)
    0.0001,   # 1
    0.0001,   # 2
    0.1405,   # 3: 14.05%
    0.2536,   # 4: 25.36%
    0.3052,   # 5: 30.52%
    0.1580,   # 6: 15.80%
    0.0930,   # 7: 9.30%
    0.0272,   # 8
    0.0124,   # 9
    0.0062,   # 10
    0.0031,   # 11
    0.0006,   # 12
])
TRAINING_PROBS = TRAINING_PROBS / TRAINING_PROBS.sum()


def compute_object_count_reward(parsed_scene, mode="kl", target_count=5.21,
                                std_dev=1, **kwargs):
    """Returns (B,) object count rewards."""
    is_empty = parsed_scene["is_empty"]
    device = is_empty.device
    B = is_empty.shape[0]

    object_counts = (~is_empty).sum(dim=1).long()
    training_probs = TRAINING_PROBS.to(device)

    if mode == "kl":
        batch_histogram = torch.zeros(13, device=device)
        clamped = object_counts.clamp(0, 12)
        for c in range(13):
            batch_histogram[c] = (clamped == c).sum().float()
        batch_probs = batch_histogram / B

        eps = 1e-10
        bp = (batch_probs + eps) / (batch_probs + eps).sum()
        tp = (training_probs + eps) / (training_probs + eps).sum()
        kl = (bp * (torch.log(bp) - torch.log(tp))).sum()
        return (-kl).expand(B)

    elif mode == "nll":
        log_probs = torch.log(training_probs + 1e-10)
        return log_probs[object_counts.clamp(0, 12)]

    else:  # gaussian
        deviation = torch.abs(object_counts.float() - target_count)
        return -(deviation / std_dev) ** 2
