"""Selection module: computes rewards, normalizes advantages, selects samples."""

import importlib
import json
import os
import pickle

import numpy as np
import torch

from iarcs.rewards import (
    compute_non_penetration_reward,
    compute_boundary_violation_reward,
    compute_object_count_reward,
    compute_valid_size_reward,
    load_custom_reward_fn,
)
from iarcs.rewards.custom import custom_reward_kwargs
from iarcs.sampling import parse_and_descale_scenes
from iarcs.utils import seed_everything

_FLOOR_POLYGONS_CACHE = {}
_FLOOR_GEOMETRY_CACHE = {}
_ROOM_FEATURES_INDEX = {}
_VERIFY_FN_CACHE = {}



def _normalize_names(value):
    """Normalize a config value to a deduplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(dict.fromkeys(str(v).strip() for v in value))


def _get_floor_polygons(config):
    path = config.midiffusion.floor_polygons_path
    if path not in _FLOOR_POLYGONS_CACHE:
        with open(path) as f:
            _FLOOR_POLYGONS_CACHE[path] = json.load(f)
    return _FLOOR_POLYGONS_CACHE[path]


def _get_floor_geometry(config):
    path = getattr(config.midiffusion, "floor_geometry_path", None)
    if not path:
        return None
    if path not in _FLOOR_GEOMETRY_CACHE:
        with open(path) as f:
            _FLOOR_GEOMETRY_CACHE[path] = json.load(f)
    return _FLOOR_GEOMETRY_CACHE[path]


def _fpbpn_key(fpbpn, scale=1e6):
    arr = np.asarray(fpbpn, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr * scale).astype(np.int64).tolist())


def _resolve_fpbpn_indices(fpbpn_list, config):
    if not fpbpn_list:
        return []
    first = fpbpn_list[0]
    if isinstance(first, (int, float)):
        return [int(i) for i in fpbpn_list]
    if isinstance(first, list) and first and isinstance(first[0], (int, float)):
        return [int(i) for batch in fpbpn_list for i in batch]
    # Full fpbpn vectors — match to index via room_features
    path = config.midiffusion.floor_conditions
    key = (path, 1e6)
    if key not in _ROOM_FEATURES_INDEX:
        with open(path) as f:
            features = json.load(f)
        _ROOM_FEATURES_INDEX[key] = {_fpbpn_key(fp): idx for idx, fp in enumerate(features)}
    index_map = _ROOM_FEATURES_INDEX[key]
    return [index_map[_fpbpn_key(fp)] for fp in fpbpn_list]


def _compute_reward_components(parsed, config, room_type, floor_polygons=None,
                               floor_geometry=None):
    """Compute all enabled reward components. Returns (total, components_dict, mode_name)."""
    use_universal = getattr(config, "universal_rewards", False)
    custom_names = _normalize_names(getattr(config, "custom_reward", None))

    components = {}

    if not use_universal and custom_names:
        for name in custom_names:
            fn = load_custom_reward_fn(name)
            components[f"custom_{name}"] = fn(
                parsed,
                **custom_reward_kwargs(room_type, floor_geometry, floor_polygons,
                                       getattr(config, "custom_reward_kwargs", None)),
            )
        return sum(components.values()), components, "custom"

    if use_universal:
        allowed = {"penetration", "boundary", "object_count", "valid_size"}
        requested = _normalize_names(getattr(config, "universal_reward_components", None))
        active = requested or ["penetration", "boundary"]

        if "penetration" in active:
            components["penetration"] = compute_non_penetration_reward(parsed, room_type=room_type)
        if "boundary" in active:
            components["boundary"] = compute_boundary_violation_reward(parsed, floor_polygons=floor_polygons)
        if "object_count" in active:
            mode = getattr(config, "object_count_mode", "nll")
            components["object_count"] = compute_object_count_reward(parsed, mode=mode)
        if "valid_size" in active:
            components["valid_size"] = compute_valid_size_reward(parsed)

        for name in custom_names:
            fn = load_custom_reward_fn(name)
            components[f"custom_{name}"] = fn(
                parsed,
                **custom_reward_kwargs(room_type, floor_geometry, floor_polygons,
                                       getattr(config, "custom_reward_kwargs", None)),
            )
        return sum(components.values()), components, "universal"

    # Legacy single-reward fallback
    components["penetration"] = compute_non_penetration_reward(parsed, room_type=room_type)
    return components["penetration"], components, "penetration"


def _compute_verify_mean(parsed, config, floor_polygons=None):
    name = getattr(config, "verify_module", None)
    if not name:
        return None
    fn = _VERIFY_FN_CACHE.get(name, False)
    if fn is False:
        try:
            module = importlib.import_module(f"iarcs.rewards.custom_rewards.{name}")
            fn = getattr(module, "verify", None)
        except Exception:
            fn = None
        _VERIFY_FN_CACHE[name] = fn
    if fn is None:
        return None
    try:
        room_type = getattr(config.midiffusion, "room_type", "bedroom")
        from iarcs.rewards.custom import BEDROOM_LABELS
        kwargs = {"idx_to_labels": BEDROOM_LABELS, "room_type": room_type}
        if floor_polygons is not None:
            kwargs["floor_polygons"] = floor_polygons
        with torch.no_grad():
            out = fn(parsed, **kwargs)
        return float(out.float().mean().item())
    except Exception:
        return None


def score_scenes(x0_raw, save_dir, config, indices=None, floor_polygons=None,
                 floor_geometry=None, only_raw_scores=True):
    """Compute rewards for a batch of denoised 3D scenes.

    Returns: (eval_scores, metrics, raw_rewards)
    """
    room_type = getattr(config.midiffusion, "room_type", "bedroom")
    num_classes = getattr(config.midiffusion, "num_classes", 22)

    with torch.no_grad():
        parsed = parse_and_descale_scenes(x0_raw, num_classes=num_classes, room_type=room_type)
        R, reward_components, mode = _compute_reward_components(
            parsed, config, room_type, floor_polygons, floor_geometry,
        )
        R = R.cpu()

    if only_raw_scores:
        return R, {}, R

    os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
    verify_mean = _compute_verify_mean(parsed, config, floor_polygons)

    # History-based z-score normalization
    unique_id = config.exp_name
    history_file = os.path.join(config.save_path, unique_id, f"history_{mode}_scores.pkl")
    history_data = []
    if config.eval.history_cnt > 0 and os.path.exists(history_file):
        with open(history_file, "rb") as f:
            history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]

    combined = torch.cat(history_data + [R]) if history_data else R
    global_mean = combined.mean().item()
    global_std = combined.std().item()

    history_data.append(R)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    with open(history_file, "wb") as f:
        pickle.dump(history_data, f)

    eval_scores = (R - global_mean) / (global_std + 1e-8)

    metrics = {
        "raw_scores_mean": float(R.mean()),
        "raw_scores_std": float(R.std()),
        "normalized_scores_mean": float(eval_scores.mean()),
        "normalized_scores_std": float(eval_scores.std()),
        "component/total_raw_mean": float(R.mean()),
    }
    for cname, cvals in reward_components.items():
        cv = cvals.detach().cpu()
        metrics[f"component/{cname}_mean"] = float(cv.mean())
        metrics[f"component/{cname}_std"] = float(cv.std())
    if verify_mean is not None:
        metrics["verify_success_mean"] = float(verify_mean)
    return eval_scores, metrics, R


def run_selection(config, stage_idx, logger, wandb_run=None):
    """Compute rewards and select training samples. Returns (save_dir, metrics)."""
    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)

    save_dir = os.path.join(config.save_path, config.exp_name, f"stage{stage_idx}")

    with open(os.path.join(save_dir, "sample.pkl"), "rb") as f:
        samples = pickle.load(f)
    with open(os.path.join(save_dir, "fpbpn_list.json")) as f:
        ground = json.load(f)

    device = f"cuda:{config.dev_id}" if torch.cuda.is_available() else "cpu"
    x0_raw = samples["next_scenes"][:, -1].to(device)
    indices = _resolve_fpbpn_indices(ground, config)

    floor_polygons_all = _get_floor_polygons(config)
    floor_polygons = [floor_polygons_all[int(i)] for i in indices]
    floor_geometry = _get_floor_geometry(config)

    eval_scores, metrics, R = score_scenes(
        x0_raw, save_dir, config, indices=indices,
        floor_polygons=floor_polygons, floor_geometry=floor_geometry,
        only_raw_scores=False,
    )

    total = x0_raw.shape[0]
    selected_mask = (eval_scores > config.eval.pos_threshold) | (eval_scores < config.eval.neg_threshold)

    num_selected = int(selected_mask.sum().item())
    num_positive = int((eval_scores > config.eval.pos_threshold).sum().item())
    num_negative = int((eval_scores < config.eval.neg_threshold).sum().item())

    selected_samples = {k: v[selected_mask] for k, v in samples.items()}
    selected_samples["eval_scores"] = eval_scores[selected_mask]

    with open(os.path.join(save_dir, "sample_stage.pkl"), "wb") as f:
        pickle.dump(selected_samples, f)

    metrics.update({
        "num_selected": num_selected,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_generated": total,
        "num_rejected": total - num_selected,
        "mean_reward": float(R.mean()),
        "std_reward": float(R.std()),
    })

    # Prefix component metrics for WandB
    prefixed = {}
    for k, v in metrics.items():
        if k.startswith("component/") or k.startswith("verify"):
            prefixed[f"score/{k}"] = v
        else:
            prefixed[k] = v

    logger.info(f"Selection: {num_selected}/{total} selected, "
                f"reward={float(R.mean()):.4f}±{float(R.std()):.4f}")
    return save_dir, prefixed
