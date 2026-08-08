"""Non-penetration reward using oriented bounding boxes (OBB).

Computes pairwise 3D intersection volumes between objects in the xz plane
with per-object rotation, then converts to a bounded reward in (0, 1] via
r = exp(-violation / sigma). Includes a seating/table tuck-under exception
and near-contact erosion tolerance.
"""

from typing import Dict, List

import numpy as np
import torch
from shapely.geometry import Polygon

BEDROOM_LABELS = {
    0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
    4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
    8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
    11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
    15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
    19: "tv_stand", 20: "wardrobe", 21: "empty",
}

LIVINGROOM_LABELS = {
    0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
    4: "chaise_longue_sofa", 5: "chinese_chair", 6: "coffee_table",
    7: "console_table", 8: "corner_side_table", 9: "desk",
    10: "dining_chair", 11: "dining_table", 12: "l_shaped_sofa",
    13: "lazy_sofa", 14: "lounge_chair", 15: "loveseat_sofa",
    16: "multi_seat_sofa", 17: "pendant_lamp", 18: "round_end_table",
    19: "shelf", 20: "stool", 21: "tv_stand", 22: "wardrobe",
    23: "wine_cabinet", 24: "empty",
}

CEILING_CLASSES = {"ceiling_lamp", "pendant_lamp"}
SEATING_CLASSES = {"chair", "dressing_chair", "stool", "dining_chair",
                   "chinese_chair", "lounge_chair"}
TABLES_WITH_UNDERSPACE = {"coffee_table", "desk", "dressing_table", "table",
                          "dining_table", "console_table"}
TABLE_TOP_SLAB_M = 0.10


def _labels_for_room(room_type: str) -> Dict[int, str]:
    if room_type and "living" in room_type.lower():
        return LIVINGROOM_LABELS
    return BEDROOM_LABELS


def _obb_corners_xz(cx, cz, hx, hz, cos_t, sin_t):
    local = np.array([[-hx, -hz], [-hx, hz], [hx, hz], [hx, -hz]], dtype=np.float64)
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cz], dtype=np.float64)


def _obb_intersection_area(corners_a, corners_b):
    pa, pb = Polygon(corners_a.tolist()), Polygon(corners_b.tolist())
    if not pa.is_valid or not pb.is_valid or pa.area <= 1e-12 or pb.area <= 1e-12:
        return 0.0
    return float(pa.intersection(pb).area)


def _classify_object(label):
    if label in CEILING_CLASSES:
        return "ceiling"
    if label in SEATING_CLASSES:
        return "seating"
    if label in TABLES_WITH_UNDERSPACE:
        return "table"
    return "other"


def compute_non_penetration_reward(
    parsed_scenes: Dict[str, torch.Tensor],
    sigma_volume: float = 0.05,
    erosion: float = 0.05,
    volume_tol: float = 1e-4,
    exempt_seating_table: bool = True,
    **kwargs,
) -> torch.Tensor:
    """OBB-aware non-penetration reward. Returns (B,) tensor in (0, 1]."""
    room_type = kwargs.get("room_type", "bedroom")
    labels_map = _labels_for_room(room_type)

    positions = parsed_scenes["positions"]
    sizes = parsed_scenes["sizes"]
    orientations = parsed_scenes["orientations"]
    object_indices = parsed_scenes["object_indices"]
    is_empty = parsed_scenes["is_empty"]
    device = parsed_scenes["device"]

    B, N = positions.shape[:2]
    rewards = torch.zeros(B, device=device)

    pos_np = positions.detach().cpu().numpy()
    siz_np = sizes.detach().cpu().numpy()
    ori_np = orientations.detach().cpu().numpy()
    cls_np = object_indices.detach().cpu().numpy()
    emp_np = is_empty.detach().cpu().numpy()

    for b in range(B):
        kinds = []
        for n in range(N):
            if emp_np[b, n]:
                kinds.append("empty")
            else:
                kinds.append(_classify_object(labels_map.get(int(cls_np[b, n]), "")))

        active = [n for n in range(N) if kinds[n] not in ("empty", "ceiling")]
        corners = {}
        for n in active:
            corners[n] = _obb_corners_xz(
                pos_np[b, n, 0], pos_np[b, n, 2],
                max(float(siz_np[b, n, 0]) - erosion, 0.0),
                max(float(siz_np[b, n, 2]) - erosion, 0.0),
                ori_np[b, n, 0], ori_np[b, n, 1],
            )

        total_volume = 0.0
        for i_idx, i in enumerate(active):
            for j in active[i_idx + 1:]:
                ki, kj = kinds[i], kinds[j]
                if exempt_seating_table and {ki, kj} == {"seating", "table"}:
                    continue

                area = _obb_intersection_area(corners[i], corners[j])
                if area <= 1e-9:
                    continue

                top_i = pos_np[b, i, 1] + siz_np[b, i, 1]
                bot_i = pos_np[b, i, 1] - siz_np[b, i, 1]
                top_j = pos_np[b, j, 1] + siz_np[b, j, 1]
                bot_j = pos_np[b, j, 1] - siz_np[b, j, 1]

                yov = max(0.0, min(top_i, top_j) - max(bot_i, bot_j))
                vol = area * yov
                if vol <= volume_tol:
                    continue
                total_volume += vol

        violation = total_volume / max(1, len(active))
        rewards[b] = float(np.exp(-violation / max(sigma_volume, 1e-6)))

    return rewards
