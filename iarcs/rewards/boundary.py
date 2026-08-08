"""Boundary violation reward.

Checks whether objects stay within the floor polygon using oriented bounding
box intersection. Objects outside the room boundary receive a penalty
proportional to the out-of-bound area.
"""

from typing import Dict

import numpy as np
import torch
from shapely.geometry import Polygon


def compute_boundary_violation_reward(
    parsed_scene: Dict[str, torch.Tensor],
    floor_polygons,
    area_tol: float = 1e-5,
    erosion: float = 0.05,
    use_area_ratio: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Returns (B,) boundary rewards: +2 if no violations, else -penalty."""
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    device = parsed_scene["device"]
    B, N = positions.shape[:2]
    rewards = torch.zeros(B, device=device)

    if floor_polygons is None:
        raise ValueError("floor_polygons required for boundary reward.")

    def _bbox_corners(x, z, hx, hz, cos_t, sin_t):
        hx = max(float(hx) - erosion, 0.0)
        hz = max(float(hz) - erosion, 0.0)
        local = np.array([[-hx, -hz], [-hx, hz], [hx, hz], [hx, -hz]], dtype=np.float64)
        rot = np.array([[float(cos_t), float(sin_t)],
                        [-float(sin_t), float(cos_t)]], dtype=np.float64)
        return local @ rot.T + np.array([float(x), float(z)], dtype=np.float64)

    for b in range(B):
        floor_poly_np = np.asarray(floor_polygons[b], dtype=np.float64)
        if floor_poly_np.ndim != 2 or floor_poly_np.shape[0] < 3:
            rewards[b] = -10.0
            continue

        floor_poly = Polygon(floor_poly_np.tolist())
        if not floor_poly.is_valid or floor_poly.area <= 1e-12:
            rewards[b] = -10.0
            continue

        bad_count = 0
        soft_penalty = 0.0

        for n in range(N):
            if is_empty[b, n]:
                continue
            corners = _bbox_corners(
                positions[b, n, 0].item(), positions[b, n, 2].item(),
                sizes[b, n, 0].item(), sizes[b, n, 2].item(),
                orientations[b, n, 0].item(), orientations[b, n, 1].item(),
            )
            obj_poly = Polygon(corners.tolist())
            if not obj_poly.is_valid or obj_poly.area <= 1e-12:
                continue

            inter_area = floor_poly.intersection(obj_poly).area
            oob_area = max(obj_poly.area - inter_area, 0.0)
            if oob_area > area_tol:
                bad_count += 1
            if use_area_ratio:
                soft_penalty += oob_area / max(obj_poly.area, 1e-12)
            else:
                soft_penalty += oob_area

        rewards[b] = 2.0 if bad_count == 0 else -soft_penalty

    return rewards
