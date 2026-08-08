"""Training pipeline orchestrator.

Manages the multi-stage RL loop: sample -> select -> train -> (reflect).
Loads the model once and reuses it across all stages.
"""

import logging
import os
import sys
import time

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from iarcs.sampling import run_sampling
from iarcs.selection import run_selection
from iarcs.training import run_training

logger = logging.getLogger(__name__)


def _login_wandb():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        os.environ.setdefault("WANDB_SILENT", "true")
    else:
        raise RuntimeError("WANDB_API_KEY not found in .env")


class TrainingPipeline:
    def __init__(self, config: DictConfig):
        self.config = config
        self.start_time = time.time()
        self.stage_times = []
        self.wandb_run = None
        self.pipeline = None
        self.trainable_layers = None
        self._inc_training_timesteps = set()
        self._last_metrics = {}

        # Reward reflection (iARCS agentic loop)
        self.reflector = None
        refl = getattr(config, "reflection", None)
        if refl and getattr(refl, "enabled", False):
            from iarcs.reflection import RewardReflector
            hist_dir = os.path.join("logs", f"{config.exp_name}_reflection")
            self.reflector = RewardReflector(
                task_name=getattr(refl, "task_name", None) or config.exp_name,
                task_prompt=getattr(refl, "task_prompt", ""),
                history_dir=hist_dir,
                model=getattr(refl, "model", "gemini-2.5-flash"),
            )
            if int(getattr(config.pipeline, "continue_from_stage", 0)) > 0:
                self.reflector.sync_iteration_from_disk()

        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(os.path.join(config.save_path, config.exp_name), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        if config.wandb.enabled:
            self._init_wandb()
        else:
            wandb.init(mode="disabled")

        self._load_model()

    def _init_wandb(self):
        _login_wandb()
        entity = self.config.wandb.entity or os.getenv("WANDB_ENTITY", "")
        group = getattr(self.config.wandb, "group", None) or None
        if self.config.resume_id:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=entity or None,
                id=self.config.resume_id, resume="must",
                config=OmegaConf.to_container(self.config, resolve=True),
            )
        else:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=entity or None,
                name=self.config.exp_name, group=group,
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=["pipeline", self.config.exp_name],
            )

    def _load_model(self):
        """Load MiDiffusion model once for all stages."""
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        import importlib

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        midiff_root = os.path.join(project_root, "3d_layout_generation", "MiDiffusion")
        if midiff_root not in sys.path:
            sys.path.insert(0, midiff_root)

        sd_module = getattr(getattr(self.config, "midiffusion", None),
                           "scene_diffuser_module", None) or "midiffusion.ashok_midiffusion"
        build_scene_diffuser = importlib.import_module(sd_module).build_scene_diffuser
        from midiffusion.datasets.threed_front_encoding import get_encoded_dataset

        device = torch.device(f"cuda:{self.config.dev_id}")
        torch.cuda.set_device(self.config.dev_id)

        cfg_path = self.config.midiffusion.config_path
        if cfg_path and not os.path.isabs(cfg_path):
            cfg_path = os.path.join(project_root, cfg_path)
        with open(cfg_path) as f:
            midiff_cfg = yaml.load(f, Loader=Loader)

        data_cfg = dict(midiff_cfg["data"])
        if not os.path.isabs(data_cfg["dataset_directory"]):
            data_cfg["dataset_directory"] = os.path.join(
                project_root, "3d_layout_generation/ThreedFront/output/3d_front_processed/",
                data_cfg["dataset_directory"],
            )
        if not os.path.isabs(data_cfg["annotation_file"]):
            data_cfg["annotation_file"] = os.path.join(
                project_root, "3d_layout_generation/ThreedFront/dataset_files/",
                data_cfg["annotation_file"],
            )

        encoded_dataset = get_encoded_dataset(
            data_cfg, path_to_bounds=None,
            augmentations=data_cfg.get("augmentations"),
            split=["train", "val"],
            max_length=midiff_cfg["network"]["sample_num_points"],
            include_room_mask=(
                midiff_cfg["network"].get("room_mask_condition", False)
                and midiff_cfg.get("feature_extractor", {}).get("name") == "resnet18"
            ),
        )
        class_dim = int(encoded_dataset.n_object_types) + 1
        network = build_scene_diffuser(midiff_cfg, class_dim)

        ckpt_path = self.config.midiffusion.checkpoint_path
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            network.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {ckpt_path}")

        network.requires_grad_(False)

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=getattr(self.config, "lora_rank", 16),
                lora_alpha=getattr(self.config, "lora_alpha", 16),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=getattr(self.config, "lora_dropout", 0.0),
                bias="none",
            )
            network = get_peft_model(network, lora_config)
            network.print_trainable_parameters()
            self.trainable_layers = network
        else:
            network.floor_encoder.requires_grad_(False)
            network.model.requires_grad_(True)
            self.trainable_layers = network.model

        network.to(device)
        self.pipeline = network
        logger.info("Model loaded.")

    def _split_step(self, stage_idx):
        interval = self.config.pipeline.split_step_right - self.config.pipeline.split_step_left + 1
        level = (stage_idx * interval) // self.config.pipeline.stage_cnt
        return level + self.config.pipeline.split_step_left

    def _stage_config(self, stage_idx):
        cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        cfg.split_step = self._split_step(stage_idx)
        cfg.seed = self.config.seed + stage_idx
        return cfg

    def _cleanup(self, save_dir):
        import shutil
        for f in ["sample.pkl", "sample_stage.pkl"]:
            p = os.path.join(save_dir, f)
            if os.path.exists(p):
                os.remove(p)
        img_dir = os.path.join(save_dir, "images")
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir, ignore_errors=True)

    def _run_stage(self, stage_idx, resume_from_ckpt=False):
        t0 = time.time()
        cfg = self._stage_config(stage_idx)

        # Incremental training timesteps
        training_ts = None
        if getattr(self.config.train, "incremental_training", False):
            import numpy as np
            total = int(self.config.sample.num_steps)
            candidates = np.arange(total)
            target = self.config.train.only_train_steps if self.config.train.only_train_steps > 0 else total
            if target < total:
                if not self._inc_training_timesteps:
                    indices = np.linspace(0, len(candidates) - 1, target).round().astype(int)
                    self._inc_training_timesteps.update(candidates[indices].tolist())
                else:
                    remaining = sorted(set(candidates.tolist()) - self._inc_training_timesteps)
                    to_add = max(0, target - len(self._inc_training_timesteps))
                    if to_add > 0 and remaining:
                        indices = np.linspace(0, len(remaining) - 1, to_add).round().astype(int)
                        self._inc_training_timesteps.update(remaining[i] for i in indices)
            else:
                self._inc_training_timesteps = set(candidates.tolist())
            training_ts = sorted(self._inc_training_timesteps)

        # Sample
        expected_pkl = os.path.join(cfg.save_path, cfg.exp_name, f"stage{stage_idx}", "sample.pkl")
        if os.path.exists(expected_pkl):
            save_dir = os.path.dirname(expected_pkl)
        else:
            save_dir = run_sampling(
                cfg, stage_idx, logger, self.wandb_run,
                self.pipeline, self.trainable_layers, resume_from_ckpt,
            )

        # Select
        save_dir, metrics = run_selection(cfg, stage_idx, logger, self.wandb_run)

        # Train
        save_dir = run_training(
            cfg, stage_idx, logger, self.wandb_run,
            self.pipeline, self.trainable_layers, training_ts,
        )

        self._last_metrics = metrics
        self._cleanup(save_dir)

        dt = time.time() - t0
        self.stage_times.append(dt)

        if self.config.wandb.enabled and self.wandb_run:
            log_dict = {
                "progression/mean_reward": metrics.get("mean_reward", 0),
                "progression/std_reward": metrics.get("std_reward", 0),
                "progression/num_selected": metrics.get("num_selected", 0),
                "progression/num_generated": metrics.get("num_generated", 0),
                "pipeline/stage": stage_idx,
                "pipeline/elapsed_hours": (time.time() - self.start_time) / 3600,
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k.startswith("score/component/"):
                    comp_key = k.replace("score/component/", "")
                    log_dict[f"progression/reward_components/{comp_key}"] = float(v)
            self.wandb_run.log(log_dict)

        logger.info(
            f"Stage {stage_idx}: {dt:.0f}s, "
            f"reward={metrics.get('mean_reward', 0):.4f}+/-{metrics.get('std_reward', 0):.4f}, "
            f"selected={metrics.get('num_selected', 0)}/{metrics.get('num_generated', 0)}"
        )

    def _maybe_reflect(self, stage_idx):
        if self.reflector is None:
            return
        interval = int(getattr(self.config.reflection, "interval", 10))
        if (stage_idx + 1) % interval != 0:
            return
        cur_name = self.config.custom_reward
        if not isinstance(cur_name, str):
            return
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cur_path = os.path.join(project_root, "iarcs", "rewards", "custom_rewards", f"{cur_name}.py")
        try:
            cur_src = open(cur_path).read()
        except Exception:
            return
        m = self._last_metrics
        stats = {
            "reward_mean": m.get("mean_reward"),
            "reward_std": m.get("std_reward"),
        }
        stats = {k: round(v, 4) for k, v in stats.items() if v is not None}
        new_name, accepted = self.reflector.reflect(cur_src, stats)
        if accepted and new_name:
            self.config.custom_reward = new_name
            logger.info(f"Reflection: switched reward {cur_name} -> {new_name}")

    def run(self):
        logger.info(f"Starting iARCS pipeline: {self.config.exp_name}")
        logger.info(f"Stages: {self.config.pipeline.stage_cnt}, "
                     f"split=[{self.config.pipeline.split_step_left}, "
                     f"{self.config.pipeline.split_step_right}]")

        if self.reflector and not self.config.custom_reward:
            name = self.reflector.generate_initial()
            self.config.custom_reward = name
            logger.info(f"Generated initial reward: {name}")

        start = int(self.config.pipeline.continue_from_stage)
        if start > 0:
            self._run_stage(start, resume_from_ckpt=True)
            self._maybe_reflect(start)
            start += 1

        for stage_idx in range(start, self.config.pipeline.stage_cnt):
            self._run_stage(stage_idx)
            self._maybe_reflect(stage_idx)
            if stage_idx < self.config.pipeline.stage_cnt - 1:
                time.sleep(self.config.pipeline.sleep_between_stages)

        total = time.time() - self.start_time
        logger.info(f"Pipeline completed in {total / 3600:.2f}h "
                     f"({len(self.stage_times)} stages)")

        if self.config.wandb.enabled and self.wandb_run:
            self.wandb_run.finish()
