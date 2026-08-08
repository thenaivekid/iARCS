"""Valid-size reward.

Penalizes size channels that drift outside the normalized [-1, 1] range,
preventing the policy from exploiting degenerate geometries that trivially
satisfy penetration and boundary constraints.
"""

from typing import Dict

import torch


def compute_valid_size_reward(
    parsed_scenes: Dict[str, torch.Tensor],
    sigma_overflow: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    """Returns (B,) tensor in (0, 1]. Exactly 1.0 when all sizes are in range."""
    sizes_norm = parsed_scenes.get("sizes_normalized")
    is_empty = parsed_scenes["is_empty"]
    device = parsed_scenes["device"]
    B = is_empty.shape[0]

    if sizes_norm is None:
        return torch.ones(B, device=device)

    overflow = torch.relu(sizes_norm.abs() - 1.0).sum(dim=-1)
    valid = (~is_empty).to(overflow.dtype)
    n_valid = valid.sum(dim=-1).clamp(min=1.0)
    mean_overflow = (overflow * valid).sum(dim=-1) / n_valid

    return torch.exp(-mean_overflow / sigma_overflow)
