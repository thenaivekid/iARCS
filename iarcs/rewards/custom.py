"""Dynamic loading of custom (task-specific) reward functions."""

import importlib

_CACHE = {}

BEDROOM_LABELS = {
    0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
    4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
    8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
    11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
    15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
    19: "tv_stand", 20: "wardrobe",
}


def load_custom_reward_fn(name):
    """Load compute_reward from iarcs.rewards.custom_rewards.<name>."""
    if name in _CACHE:
        return _CACHE[name]

    module = importlib.import_module(f"iarcs.rewards.custom_rewards.{name}")

    if hasattr(module, "compute_reward") and callable(module.compute_reward):
        fn = module.compute_reward
    else:
        candidates = [
            getattr(module, a) for a in dir(module)
            if a.startswith("compute_") and callable(getattr(module, a))
        ]
        if len(candidates) == 1:
            fn = candidates[0]
        else:
            raise ValueError(
                f"No valid compute_reward in iarcs.rewards.custom_rewards.{name}"
            )

    _CACHE[name] = fn
    return fn


def custom_reward_kwargs(room_type, floor_geometry=None, floor_polygons=None,
                         extra=None):
    """Build kwargs dict for custom reward functions."""
    kwargs = {"room_type": room_type, "idx_to_labels": BEDROOM_LABELS}
    if floor_polygons is not None:
        kwargs["floor_polygons"] = floor_polygons
    if floor_geometry is not None:
        kwargs["floor_plan_vertices"] = [e["floor_plan_vertices"] for e in floor_geometry]
        kwargs["floor_plan_faces"] = [e["floor_plan_faces"] for e in floor_geometry]
        kwargs["floor_plan_centroid"] = [e["floor_plan_centroid"] for e in floor_geometry]
    if extra:
        kwargs.update(dict(extra))
    return kwargs
