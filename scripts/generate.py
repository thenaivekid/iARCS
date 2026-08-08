"""Generate 3D bedroom layouts using a trained iARCS LoRA checkpoint.

Wraps the MiDiffusion generation pipeline, loading the pretrained model
and optionally applying a LoRA adapter from iARCS training.

Usage:
    python scripts/generate.py \
        --lora checkpoints/<exp>/stage<N>/checkpoints/checkpoint_1/lora_weights.pt \
        --result_tag my_results \
        --n_syn_scenes 100
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenes with iARCS-trained LoRA checkpoint"
    )
    parser.add_argument(
        "--lora", default=None,
        help="Path to LoRA weights (.pt). If omitted, uses pretrained model only.",
    )
    parser.add_argument(
        "--weight_file", default=None,
        help="Pretrained model weights. Default: MiDiffusion pretrained checkpoint.",
    )
    parser.add_argument("--result_tag", default="iarcs_generated")
    parser.add_argument("--n_syn_scenes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_denoising_steps", type=int, default=20)
    parser.add_argument("--fix_bed_flip", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midiff_root = os.path.join(project_root, "3d_layout_generation", "MiDiffusion")
    if midiff_root not in sys.path:
        sys.path.insert(0, midiff_root)

    weight_file = args.weight_file or os.path.join(
        midiff_root, "output", "log",
        "pretrained_3d_layout_custom_attn", "best_model.pt",
    )

    gen_args = [
        weight_file,
        "--result_tag", args.result_tag,
        "--n_syn_scenes", str(args.n_syn_scenes),
        "--batch_size", str(args.batch_size),
        "--gpu", str(args.gpu),
        "--num_denoising_steps", str(args.num_denoising_steps),
    ]
    if args.lora:
        gen_args.extend(["--lora", args.lora])
    if args.fix_bed_flip:
        gen_args.append("--fix_bed_flip")

    from scripts.ashok_generate_results import main as generate_main
    generate_main(gen_args)


if __name__ == "__main__":
    main()
