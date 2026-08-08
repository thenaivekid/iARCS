# iARCS: Iterative Agentic RL for Controllable 3D Scene Generation

This is the official implementation of [iARCS: Iterative Agentic RL for Controllable 3D Scene Generation](https://arxiv.org/abs/2608.06161).

iARCS fine-tunes pretrained 3D scene layout diffusion models using reinforcement learning with modular reward functions. It features a two-stage strategy: (1) universal-reward pretraining for physical plausibility, and (2) task-specific fine-tuning using LLM-generated reward programs that are iteratively refined from training feedback.

## Setup

```bash
conda create -n iarcs python=3.10
conda activate iarcs
pip install -e .
pip install gdown
```

### Download pretrained model and data

We provide the pretrained **MiDiffusion** checkpoint and preprocessed floor-plan data for bedrooms [here](https://drive.google.com/file/d/1vgnTcYXdXc3IgKmmdok1mU69vxTrUsNK/view?usp=drive_link). Download them and place them under `3d_layout_generation/MiDiffusion/`.

```bash
# Download pretrained model and data (~170MB)
gdown 1vgnTcYXdXc3IgKmmdok1mU69vxTrUsNK -O ckpt.zip
unzip ckpt.zip -d 3d_layout_generation/MiDiffusion/
mkdir -p 3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn
mv 3d_layout_generation/MiDiffusion/best_model.pt \
   3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/
```

This gives you everything needed to start training:
```
3d_layout_generation/MiDiffusion/
├── output/log/pretrained_3d_layout_custom_attn/best_model.pt
├── config.yaml
├── room_features.json
├── floor_polygons_test.json
└── floor_geometry_test.json
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

- `WANDB_API_KEY`: for logging (get from [wandb.ai/authorize](https://wandb.ai/authorize))
- `WANDB_ENTITY`: your wandb team/user name
- `GEMINI_API_KEY`: required only if using reward reflection (`reflection.enabled=true`)

## Usage

### Universal reward training (penetration + boundary)

```bash
python train.py \
    exp_name="universal_training" \
    universal_rewards=true \
    universal_reward_components=[penetration,boundary] \
    pipeline.stage_cnt=100
```

### Task-specific training with iARCS reflection

```bash
python train.py \
    exp_name="iarcs_walkability" \
    universal_rewards=true \
    universal_reward_components=[penetration,boundary] \
    custom_reward=walkability \
    reflection.enabled=true \
    reflection.task_name=walkability \
    reflection.task_prompt="Make sure to keep a tv_stand in the room, I don't care where it is placed."
```

### Key configuration options

| Option | Description |
|--------|-------------|
| `universal_rewards` | Enable universal constraint rewards |
| `universal_reward_components` | List: `penetration`, `boundary`, `object_count`, `valid_size` |
| `custom_reward` | Task-specific reward module name |
| `reflection.enabled` | Enable LLM-based reward reflection loop |
| `use_lora` | LoRA fine-tuning (recommended) |
| `pipeline.stage_cnt` | Number of RL stages |

All options can be overridden from the command line via Hydra syntax. See `configs/config.yaml` for the full configuration.

### Continuing from a universal checkpoint

```bash
python train.py \
    exp_name="iarcs_tv_bed" \
    continue_from_universal=true \
    path_to_universal_lora="checkpoints/universal_training/stage99/checkpoints/checkpoint_1" \
    universal_rewards=true \
    universal_reward_components=[penetration] \
    reflection.enabled=true \
    reflection.task_name=tv_bed \
    reflection.task_prompt="Place the TV stand facing the bed"
```

### Generating scenes from a trained checkpoint

```bash
python scripts/generate.py \
    --lora checkpoints/<exp>/stage<N>/checkpoints/checkpoint_1/lora_weights.pt \
    --result_tag my_results \
    --n_syn_scenes 1000
```

Results are saved to `3d_layout_generation/MiDiffusion/output/predicted_results/<result_tag>/results.pkl`.

### Working with generated scenes

Inspect and visualize generated layouts:

```bash
# Print scene summaries
python scripts/using_synthetic_data.py --results path/to/results.pkl

# Generate 2D top-down views
python scripts/using_synthetic_data.py --results path/to/results.pkl --plot --out views.png

# Quick demo with dummy data (no results file needed)
python scripts/using_synthetic_data.py --plot
```

Each scene layout is a dict with `class_labels`, `translations`, `sizes`, and `angles` — see the script for details.

For full 3D rendering with textured meshes, see [ThreedFront](https://github.com/MIT-SPARK/ThreedFront).

## Generated Scene Datasets

We release pre-generated scene layouts (4000 scenes each) as `ThreedFrontResults` pkl files. Download and inspect them directly:

| Room type | Scenes | Download |
|-----------|--------|----------|
| Bedroom | 4000 | `gdown <BEDROOM_FILE_ID> -O bedroom_results.pkl` |
| Dining room | 4000 | `gdown <DININGROOM_FILE_ID> -O diningroom_results.pkl` |
| Living room | 4000 | `gdown <LIVINGROOM_FILE_ID> -O livingroom_results.pkl` |

Load and visualize:

```python
from scripts.using_synthetic_data import load_results, plot_scenes

scene_indices, layouts = load_results("bedroom_results.pkl")
print(f"{len(layouts)} scenes loaded")

# Each layout has: class_labels (N,C), translations (N,3), sizes (N,3), angles (N,1)
layout = layouts[0]
print(layout["translations"].shape)  # (N, 3) object positions

# 2D top-down visualization
plot_scenes(layouts, scene_indices, "bedroom_views.png", max_scenes=16)
```

## Project Structure

```
├── train.py                  # Entry point
├── configs/config.yaml       # Hydra configuration
├── iarcs/
│   ├── pipeline.py           # Multi-stage RL orchestrator
│   ├── sampling.py           # DDIM trajectory generation
│   ├── selection.py          # Reward computation and sample selection
│   ├── training.py           # PPO-style policy gradient updates
│   ├── reflection.py         # LLM-based reward refinement (iARCS)
│   ├── diffusion/ddim.py     # DDIM step with log-probability
│   └── rewards/              # Modular reward functions
│       ├── penetration.py    # OBB collision detection
│       ├── boundary.py       # Floor polygon boundary check
│       ├── object_count.py   # Object count distribution matching
│       ├── valid_size.py     # Size channel validity
│       └── custom.py         # Dynamic custom reward loader
└── scripts/                  # Example run scripts
```

## Citation

```bibtex
@misc{adhikari2026iarcsiterativeagenticrl,
      title={iARCS: Iterative Agentic RL for Controllable 3D Scene Generation}, 
      author={Saugat Adhikari and Ashok Prasad Neupane and Pramish Paudel and Ajad Chhatkuli and Danda Pani Paudel},
      year={2026},
      eprint={2608.06161},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.06161}, 
}
```
