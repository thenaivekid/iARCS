"""Top-down 2D visualization of generated 3D scenes as oriented bounding boxes."""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iarcs.sampling import parse_and_descale_scenes

BEDROOM_LABELS = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair",
    "children_cabinet", "coffee_table", "desk", "double_bed",
    "dressing_chair", "dressing_table", "kids_bed", "nightstand",
    "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table",
    "tv_stand", "wardrobe",
]

COLORS = {
    "double_bed": "#E74C3C", "single_bed": "#E74C3C", "kids_bed": "#E74C3C",
    "nightstand": "#3498DB", "wardrobe": "#2ECC71", "desk": "#F39C12",
    "chair": "#9B59B6", "dressing_chair": "#9B59B6", "armchair": "#8E44AD",
    "sofa": "#1ABC9C", "tv_stand": "#E67E22", "bookshelf": "#34495E",
    "cabinet": "#7F8C8D", "children_cabinet": "#95A5A6", "coffee_table": "#D35400",
    "dressing_table": "#C0392B", "shelf": "#2C3E50", "stool": "#16A085",
    "table": "#F1C40F", "ceiling_lamp": "#BDC3C7", "pendant_lamp": "#BDC3C7",
}


def draw_scene(ax, positions, sizes, orientations, obj_indices, is_empty,
               floor_polygon=None, title=""):
    ax.set_aspect("equal")
    ax.set_facecolor("#F8F9FA")

    if floor_polygon is not None:
        fp = np.array(floor_polygon)
        poly = plt.Polygon(fp, closed=True, fill=True, facecolor="#FFFFFF",
                           edgecolor="#2C3E50", linewidth=2)
        ax.add_patch(poly)

    for i in range(positions.shape[0]):
        if is_empty[i]:
            continue
        cls = int(obj_indices[i])
        if cls >= len(BEDROOM_LABELS):
            continue
        label = BEDROOM_LABELS[cls]
        x, z = float(positions[i, 0]), float(positions[i, 2])
        hx, hz = float(sizes[i, 0]), float(sizes[i, 2])
        cos_a, sin_a = float(orientations[i, 0]), float(orientations[i, 1])
        angle = np.degrees(np.arctan2(sin_a, cos_a))

        color = COLORS.get(label, "#95A5A6")
        rect = patches.FancyBboxPatch(
            (-hx, -hz), 2 * hx, 2 * hz,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.7,
        )
        t = matplotlib.transforms.Affine2D().rotate_deg(angle).translate(x, z) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        ax.text(x, z, label[:6], ha="center", va="center", fontsize=5,
                fontweight="bold", color="white",
                transform=ax.transData)

    ax.set_title(title, fontsize=9)
    ax.autoscale()
    margin = 0.5
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax.set_ylim(ylim[0] - margin, ylim[1] + margin)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_pkl", help="Path to sample.pkl from a training stage")
    parser.add_argument("--num", type=int, default=16, help="Number of scenes to render")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--floor_polygons", default=None, help="Floor polygons JSON")
    parser.add_argument("--fpbpn_json", default=None, help="fpbpn_list.json from same stage")
    parser.add_argument("--num_classes", type=int, default=22)
    parser.add_argument("--room_type", default="bedroom")
    args = parser.parse_args()

    with open(args.sample_pkl, "rb") as f:
        samples = pickle.load(f)

    x0 = samples["next_scenes"][:, -1]
    n = min(args.num, x0.shape[0])

    parsed = parse_and_descale_scenes(x0[:n], num_classes=args.num_classes,
                                       room_type=args.room_type)

    floor_polygons = None
    if args.floor_polygons and args.fpbpn_json:
        with open(args.floor_polygons) as f:
            all_fp = json.load(f)
        with open(args.fpbpn_json) as f:
            fpbpn = json.load(f)
        indices = fpbpn[:n] if isinstance(fpbpn[0], int) else list(range(n))
        floor_polygons = [all_fp[int(i)] for i in indices]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(n):
        fp = floor_polygons[i] if floor_polygons else None
        draw_scene(
            axes[i],
            parsed["positions"][i].cpu().numpy(),
            parsed["sizes"][i].cpu().numpy(),
            parsed["orientations"][i].cpu().numpy(),
            parsed["object_indices"][i].cpu().numpy(),
            parsed["is_empty"][i].cpu().numpy(),
            floor_polygon=fp,
            title=f"Scene {i}",
        )

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    out = args.output or os.path.join(os.path.dirname(args.sample_pkl), "scenes_topdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {n} scenes to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
