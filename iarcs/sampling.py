"""Sampling module: generates trajectories via DDIM for RL training."""

import contextlib
import json
import os
import pickle
import random

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler
from tqdm import tqdm

from iarcs.diffusion.ddim import ddim_step_with_logprob
from iarcs.utils import seed_everything


def descale_to_origin(x, minimum, maximum):
    x = (x + 1) / 2
    return x * (maximum - minimum)[None, None, :] + minimum[None, None, :]


# Per-room-type normalization statistics from the training data.
_POS_STATS = {
    "bedroom": (
        [-2.7625005, 0.045, -2.75275],
        [2.7784417, 3.6248395, 2.8185427],
    ),
    "livingroom": (
        [-5.672918693230125, 0.0375, -5.716401580065309],
        [5.09667921844729, 3.3577405149437496, 5.4048500000000015],
    ),
    "diningroom": (
        [-5.089109999999999, 0.0375, -5.716401580065309],
        [5.09667921844729, 3.384158412231445, 5.4048500000000015],
    ),
}
_POS_STATS["livingdining"] = _POS_STATS["livingroom"]

_SIZE_STATS = {
    "bedroom": (
        [0.03998289, 0.02000002, 0.012772],
        [2.8682, 1.770065, 1.698315],
    ),
    "livingroom": (
        [0.03998999999999997, 0.020000020334800084, 0.0328434999999998],
        [2.3802699999999994, 1.7700649999999998, 1.3224289999999996],
    ),
    "diningroom": (
        [0.032552000000000025, 0.020000020334800084, 0.009999999999999884],
        [2.3802699999999994, 1.7700649999999998, 1.4137885],
    ),
}
_SIZE_STATS["livingdining"] = _SIZE_STATS["diningroom"]


def descale_pos(positions, device="cuda", room_type="bedroom"):
    mn, mx = _POS_STATS[room_type]
    return descale_to_origin(
        positions,
        torch.tensor(mn, device=device),
        torch.tensor(mx, device=device),
    )


def descale_size(sizes, device="cuda", room_type="bedroom"):
    mn, mx = _SIZE_STATS[room_type]
    if os.environ.get("B2_CLAMP_SIZES", "1") != "0":
        sizes = sizes.clamp(-1.0, 1.0)
    return descale_to_origin(
        sizes,
        torch.tensor(mn, device=device),
        torch.tensor(mx, device=device),
    )


def parse_and_descale_scenes(scenes, num_classes=22, room_type="bedroom"):
    """Parse (B, N, 30) scene tensor into world-space geometry."""
    device = scenes.device
    positions_norm = scenes[:, :, 0:3]
    sizes_norm = scenes[:, :, 3:6]
    orientations = scenes[:, :, 6:8]
    one_hot = scenes[:, :, 8:8 + num_classes]

    positions = descale_pos(positions_norm, device=device, room_type=room_type)
    sizes = descale_size(sizes_norm, device=device, room_type=room_type)
    object_indices = torch.argmax(one_hot, dim=-1)
    is_empty = object_indices == (num_classes - 1)

    return {
        "one_hot": one_hot,
        "positions": positions,
        "sizes": sizes,
        "sizes_normalized": sizes_norm,
        "orientations": orientations,
        "object_indices": object_indices,
        "is_empty": is_empty,
        "device": device,
    }


def _load_checkpoint_if_needed(config, stage_idx, resume_from_ckpt, pipeline,
                               accelerator, logger=None):
    """Load LoRA or full checkpoint from previous stage."""
    if not resume_from_ckpt:
        # Bootstrap from universal checkpoint at stage 0 if configured.
        if (getattr(config, "continue_from_universal", False)
                and stage_idx == 0
                and int(getattr(config.pipeline, "continue_from_stage", 0)) == 0):
            path = str(config.path_to_universal_lora)
            if config.use_lora:
                lora_path = path if path.endswith(".pt") else os.path.join(path, "lora_weights.pt")
                state = torch.load(lora_path, map_location=accelerator.device)
                pipeline.load_state_dict(state, strict=False)
            else:
                model_path = path if path.endswith(".pt") else os.path.join(path, "model.pt")
                state = torch.load(model_path, map_location=accelerator.device)
                pipeline.model.load_state_dict(state)
            if logger:
                logger.info(f"Bootstrapped from universal checkpoint: {path}")
        return

    prev_stage = stage_idx - 1
    ckpt_num = config.train.num_epochs // config.train.save_interval
    ckpt_path = os.path.join(
        config.save_path, config.exp_name,
        f"stage{prev_stage}", "checkpoints", f"checkpoint_{ckpt_num}",
    )
    ckpt_path = os.path.normpath(os.path.expanduser(ckpt_path))
    if "checkpoint_" not in os.path.basename(ckpt_path):
        candidates = [x for x in os.listdir(ckpt_path) if "checkpoint_" in x]
        ckpt_path = os.path.join(
            ckpt_path,
            sorted(candidates, key=lambda x: int(x.split("_")[-1]))[-1],
        )
    if config.use_lora:
        state = torch.load(
            os.path.join(ckpt_path, "lora_weights.pt"),
            map_location=accelerator.device,
        )
        pipeline.load_state_dict(state, strict=False)
    else:
        state = torch.load(
            os.path.join(ckpt_path, "model.pt"),
            map_location=accelerator.device,
        )
        pipeline.model.load_state_dict(state)
    if logger:
        logger.info(f"Resumed from {ckpt_path}")


def _make_save_load_hooks(config, pipeline):
    def save_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora:
            lora_state = {k: v for k, v in models[0].state_dict().items() if "lora_" in k}
            torch.save(lora_state, os.path.join(output_dir, "lora_weights.pt"))
        else:
            torch.save(models[0].state_dict(), os.path.join(output_dir, "model.pt"))
        weights.pop()

    def load_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora:
            state = torch.load(os.path.join(input_dir, "lora_weights.pt"), map_location="cpu")
            models[0].load_state_dict(state, strict=False)
        else:
            state = torch.load(os.path.join(input_dir, "model.pt"), map_location="cpu")
            models[0].load_state_dict(state)
        models.pop()

    return save_hook, load_hook


def run_sampling(config, stage_idx, logger, wandb_run=None, pipeline=None,
                 trainable_layers=None, resume_from_ckpt=False):
    """Generate trajectories via DDIM. Returns save_dir path."""
    if pipeline is None:
        raise ValueError("Pipeline must be provided.")

    torch.cuda.set_device(config.dev_id)
    save_dir = os.path.join(config.save_path, config.exp_name, f"stage{stage_idx}")
    os.makedirs(save_dir, exist_ok=True)

    acc_config = ProjectConfiguration(
        project_dir=save_dir, automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    accelerator = Accelerator(mixed_precision=config.mixed_precision,
                              project_config=acc_config)

    save_hook, load_hook = _make_save_load_hooks(config, pipeline)
    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    if trainable_layers is not None and not hasattr(trainable_layers, "_hf_hook"):
        trainable_layers = accelerator.prepare(trainable_layers)

    _load_checkpoint_if_needed(
        config, stage_idx, resume_from_ckpt, pipeline, accelerator, logger,
    )

    seed_everything(config.seed)
    autocast = contextlib.nullcontext

    with open(config.midiffusion.floor_conditions) as f:
        floor_cond_np = np.array(json.load(f), dtype=np.float32)
    floor_cnt = len(floor_cond_np)

    num_objects = getattr(config.midiffusion, "num_objects", 12)
    scene_dim = getattr(config.midiffusion, "scene_dim", 30)

    ddim = DDIMScheduler(
        num_train_timesteps=getattr(config.midiffusion, "num_timesteps", 1000),
        beta_start=getattr(config.midiffusion, "beta_start", 1e-4),
        beta_end=getattr(config.midiffusion, "beta_end", 0.02),
        clip_sample=False, prediction_type="epsilon", steps_offset=1,
    )

    pipeline.eval()
    samples = []
    total_fpbpn = []
    total_samples = None

    if os.path.exists(os.path.join(save_dir, "fpbpn_list.json")):
        with open(os.path.join(save_dir, "fpbpn_list.json")) as f:
            total_fpbpn = json.load(f)
    if os.path.exists(os.path.join(save_dir, "sample.pkl")):
        with open(os.path.join(save_dir, "sample.pkl"), "rb") as f:
            total_samples = pickle.load(f)

    split_step = config.split_step
    split_time = getattr(config, "split_time", 1)

    for idx in tqdm(range(config.sample.num_batches_per_epoch), desc="Sampling"):
        batch_indices = [random.randrange(floor_cnt)
                        for _ in range(config.sample.batch_size)]
        fpbpn_batch = torch.tensor(
            floor_cond_np[batch_indices], dtype=torch.float32,
            device=accelerator.device,
        )
        x_noise = torch.randn(
            config.sample.batch_size, num_objects, scene_dim,
            dtype=torch.float32, device=accelerator.device,
        )

        ddim.set_timesteps(config.sample.num_steps, device=accelerator.device)
        ts = ddim.timesteps
        extra_kwargs = {"eta": getattr(config.sample, "eta", 0.0)}

        branch_num = split_time
        noise_list = [x_noise] + [torch.randn_like(x_noise) for _ in range(branch_num - 1)]
        scenes = [[n] for n in noise_list]
        log_probs = [[] for _ in range(branch_num)]

        for i, t in enumerate(ts):
            with autocast():
                with torch.no_grad():
                    for k in range(len(scenes)):
                        x_t = scenes[k][i]
                        t_batch = t.expand(config.sample.batch_size)
                        noise_pred = pipeline.predict_noise(x_t, t_batch, fpbpn_batch)
                        x_prev, log_prob, _ = ddim_step_with_logprob(
                            ddim, noise_pred, t, x_t, **extra_kwargs,
                        )
                        scenes[k].append(x_prev)
                        log_probs[k].append(log_prob)

        sample_num = len(scenes)
        total_fpbpn.extend([batch_indices] * sample_num)

        for k in range(sample_num):
            store_scenes = torch.stack(scenes[k], dim=1)
            store_log_probs = torch.stack(log_probs[k], dim=1)
            timesteps_rep = ts.unsqueeze(0).expand(config.sample.batch_size, -1)
            samples.append({
                "fpbpn": fpbpn_batch.cpu().detach(),
                "timesteps": timesteps_rep.cpu().detach(),
                "log_probs": store_log_probs.cpu().detach(),
                "scenes": store_scenes[:, :-1].cpu().detach(),
                "next_scenes": store_scenes[:, 1:].cpu().detach(),
            })

        if (idx + 1) % config.sample.save_interval == 0 or idx == config.sample.num_batches_per_epoch - 1:
            keys = ["fpbpn", "timesteps", "log_probs", "scenes", "next_scenes"]
            new_tensors = {k: torch.cat([s[k] for s in samples]) for k in keys}
            with open(os.path.join(save_dir, "fpbpn_list.json"), "w") as f:
                json.dump(total_fpbpn, f)
            with open(os.path.join(save_dir, "sample.pkl"), "wb") as f:
                if total_samples is None:
                    pickle.dump(new_tensors, f)
                else:
                    pickle.dump(
                        {k: torch.cat([total_samples[k], new_tensors[k]]) for k in keys},
                        f,
                    )

    logger.info(f"Sampling completed: {len(total_fpbpn)} batches")
    return save_dir
