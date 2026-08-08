"""Training module: PPO-style policy gradient updates on selected trajectories."""

import contextlib
import copy
import json
import os

import torch
import tree
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler
from tqdm import tqdm

from iarcs.diffusion.ddim import ddim_step_with_logprob
from iarcs.utils import load_sample_stage, seed_everything

import logging
logger = logging.getLogger(__name__)


def run_training(config, stage_idx, external_logger, wandb_run=None,
                 pipeline=None, trainable_layers=None, training_timesteps=None):
    """Run PPO-style training on selected samples. Returns save_dir."""
    if pipeline is None or trainable_layers is None:
        raise ValueError("Pipeline and trainable_layers must be provided.")

    torch.cuda.set_device(config.dev_id)
    save_dir = os.path.join(config.save_path, config.exp_name, f"stage{stage_idx}")

    if getattr(config.train, "incremental_training", False) and training_timesteps is not None:
        num_train_ts = len(training_timesteps)
    else:
        num_train_ts = int(config.split_step * config.train.timestep_fraction)

    acc_config = ProjectConfiguration(
        project_dir=save_dir, automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=acc_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_ts,
    )

    # Register save/load hooks
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

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    if getattr(config, "continue_from_universal", False) and stage_idx == 0:
        lora_root = getattr(config, "path_to_universal_lora", "")
        if not lora_root:
            raise ValueError("continue_from_universal requires path_to_universal_lora")
        lora_path = (lora_root if lora_root.endswith(".pt")
                     else os.path.join(lora_root, "lora_weights.pt"))
        state = torch.load(lora_path, map_location=accelerator.device)
        pipeline.load_state_dict(state, strict=False)
        external_logger.info(f"Loaded universal LoRA from {lora_path}")

    seed_everything(config.seed)
    autocast = contextlib.nullcontext

    optimizer = torch.optim.AdamW(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    samples = load_sample_stage(save_dir)
    accelerator.save_state()

    ddim = DDIMScheduler(
        num_train_timesteps=getattr(config.midiffusion, "num_timesteps", 1000),
        beta_start=getattr(config.midiffusion, "beta_start", 1e-4),
        beta_end=getattr(config.midiffusion, "beta_end", 0.02),
        clip_sample=False, prediction_type="epsilon", steps_offset=1,
    )
    ddim.set_timesteps(config.sample.num_steps, device=accelerator.device)

    init_samples = copy.deepcopy(samples)
    es = init_samples.get("eval_scores", torch.tensor([]))
    if isinstance(es, list):
        es = torch.stack(es) if es else torch.tensor([])
        init_samples["eval_scores"] = es
    if es.shape[0] == 0:
        logger.warning("No training samples — skipping training.")
        return save_dir

    for epoch in range(config.train.num_epochs):
        total_bs = init_samples["eval_scores"].shape[0]
        perm = torch.randperm(total_bs)
        samples = {k: v[perm] for k, v in init_samples.items()}

        num_ts = samples["timesteps"].shape[1]
        perms = torch.stack([torch.randperm(num_ts) for _ in range(total_bs)])
        for key in ["scenes", "next_scenes", "log_probs", "timesteps"]:
            samples[key] = samples[key][torch.arange(total_bs)[:, None], perms]

        pipeline.model.train()

        for idx in tqdm(range(0, total_bs // 2 * 2, config.train.batch_size),
                        desc="Update", leave=False):
            sample = tree.map_structure(
                lambda v: v[idx:idx + config.train.batch_size].to(accelerator.device),
                samples,
            )
            fpbpn = sample["fpbpn"]

            if getattr(config.train, "incremental_training", False) and training_timesteps is not None:
                t_indices = training_timesteps
            else:
                t_indices = range(sample["timesteps"].shape[1])

            for t in t_indices:
                eval_score = sample["eval_scores"][:]

                with accelerator.accumulate(pipeline.model):
                    with autocast():
                        noise_pred = pipeline.predict_noise(
                            sample["scenes"][:, t],
                            sample["timesteps"][:, t],
                            fpbpn,
                        )
                        _, total_prob, _ = ddim_step_with_logprob(
                            ddim, noise_pred,
                            sample["timesteps"][:, t],
                            sample["scenes"][:, t],
                            eta=getattr(config.sample, "eta", 0.0),
                            prev_sample=sample["next_scenes"][:, t],
                        )

                        total_ref_prob = sample["log_probs"][:, t]
                        ratio = torch.exp(total_prob - total_ref_prob)

                        beta1 = torch.ones_like(eval_score) * config.train.beta1
                        beta2 = torch.ones_like(eval_score) * config.train.beta2
                        weight = torch.where(eval_score > 0, beta1, beta2)
                        advantages = torch.clamp(
                            eval_score, -config.train.adv_clip_max, config.train.adv_clip_max,
                        ) * weight

                        unclipped = -advantages * ratio
                        clipped = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.eps, 1.0 + config.train.eps,
                        )
                        loss = torch.mean(torch.maximum(unclipped, clipped))

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_layers.parameters(), config.train.max_grad_norm,
                        )

                    if wandb_run and config.wandb.enabled:
                        wandb_run.log({"train/loss": loss.item(), "train/epoch": epoch})

                    optimizer.step()
                    optimizer.zero_grad()

        if (epoch + 1) % config.train.save_interval == 0:
            accelerator.save_state()

    external_logger.info(f"Training completed for stage {stage_idx}")
    return save_dir
