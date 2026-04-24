#!/usr/bin/env python3
"""
viz_topk_heatmaps.py

Save PNGs of top-k activating images with SAE heatmap overlays.

This version supports:
1. Single latent via --latent_id
2. Batch range via --batch_start / --batch_count
3. All SAE features via --all_features

Examples
--------
All features, top 5:
python viz_topk_heatmaps.py \
  --top_k_images_dir "$DATA_DIR/input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_128_24161611/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  --dataset ILSVRC/imagenet-1k \
  --split test \
  --top_k 5 \
  --all_features \
  --save_dir outputs/viz_top5_all

Single latent:
python viz_topk_heatmaps.py \
  --top_k_images_dir "$DATA_DIR/input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_128_24161611/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  --dataset ILSVRC/imagenet-1k \
  --split test \
  --latent_id 6 \
  --top_k 5 \
  --save_dir outputs/viz_topk

Batch mode:
python viz_topk_heatmaps.py \
  --top_k_images_dir "/home/ubuntu/tokenWorkViT/HF-SAE-main/input_features_explainer/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_128_19813003/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  --dataset ILSVRC/imagenet-1k \
  --split test \
  --batch_start 0 \
  --batch_count 20 \
  --top_k 5 \
  --save_dir outputs/viz_topk
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

REPO_DIR = os.environ.get("REPO_DIR")
if REPO_DIR is None:
    REPO_DIR = str(Path(__file__).resolve().parent)
    os.environ["REPO_DIR"] = REPO_DIR

sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "src"))

from src.steering_utils import read_top_k_images  # noqa: E402
from dictionary_learning.utils import load_dataset_from_yaml  # noqa: E402
from packages.overcomplete.visualization.top_concepts import overlay_heatmaps_to_images  # noqa: E402
from utils.utils import numpy_to_pil, list_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--top_k_images_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k")
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Must match the split used when creating top_k_images_dir.",
    )
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--random_sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all_features", action="store_true", help="Save PNGs for all SAE features.")
    p.add_argument("--latent_id", type=int, default=None, help="Visualize one latent.")
    p.add_argument("--batch_start", type=int, default=None, help="Start index for batch mode.")
    p.add_argument("--batch_count", type=int, default=20)
    p.add_argument("--save_dir", type=str, default="outputs/viz_topk_heatmaps")
    return p.parse_args()


def ensure_rgb(images: List[Image.Image]) -> List[Image.Image]:
    out = []
    for im in images:
        if isinstance(im, Image.Image):
            out.append(im.convert("RGB"))
        else:
            out.append(Image.fromarray(np.asarray(im)).convert("RGB"))
    return out


def get_latent_names(top_k_images_dir: Path) -> List[str]:
    latents = sorted(list_features(top_k_images_dir))
    return [latent for latent in latents if latent.startswith("latent_")]


def load_top_k_metadata(top_k_images_dir: Path) -> dict:
    metadata_path = top_k_images_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r") as f:
        return json.load(f)


def load_overlayed_images(
    ds,
    top_k_images_dir: str,
    latent_id: int,
    top_k: int,
    alpha: float,
    random_sample: bool,
) -> List[Image.Image]:
    output = read_top_k_images(
        ds=ds,
        latent_id=str(latent_id),
        top_k_images_dir=top_k_images_dir,
        k=top_k,
        top_k_random_sample=random_sample,
        partition=None,
    )

    images = ensure_rgb(output["images"])
    heatmaps = output["heatmaps"]

    if not heatmaps:
        return images

    valid_pairs = [(image, heatmap) for image, heatmap in zip(images, heatmaps) if heatmap is not None]
    if not valid_pairs:
        return images

    images, heatmaps = zip(*valid_pairs)
    overlayed = overlay_heatmaps_to_images(images, heatmaps, alpha=alpha)
    overlayed = [numpy_to_pil(x).convert("RGB") for x in overlayed if x is not None]
    return overlayed if overlayed else list(images)


def feature_grid_dims(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def render_single_latent(
    overlayed_images: List[Image.Image],
    latent_id: int,
    top_k: int,
    output_path: Path,
    image_size: int,
) -> None:
    n = min(top_k, len(overlayed_images))
    if n == 0:
        print(f"Skipping latent {latent_id}: no images found")
        return

    rows, cols = feature_grid_dims(n)
    fig = plt.figure(figsize=(3.2 * cols, 3.4 * rows))
    fig.suptitle(f"latent {latent_id}", fontsize=14)

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = overlayed_images[i].resize((image_size, image_size))
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()

    top_k_images_dir = Path(args.top_k_images_dir)
    if not top_k_images_dir.exists():
        raise FileNotFoundError(f"top_k_images_dir does not exist: {top_k_images_dir}")
    if not top_k_images_dir.is_dir():
        raise NotADirectoryError(f"top_k_images_dir is not a directory: {top_k_images_dir}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_top_k_metadata(top_k_images_dir)

    dataset_obj, _ = load_dataset_from_yaml(args.dataset)
    if args.split not in dataset_obj:
        raise KeyError(
            f"Split '{args.split}' not found for dataset '{args.dataset}'. "
            f"Available splits: {list(dataset_obj.keys())}"
        )
    ds = dataset_obj[args.split]

    latent_names = get_latent_names(top_k_images_dir)
    if not latent_names:
        raise ValueError(f"No latent_* entries found in {top_k_images_dir}")

    if args.all_features:
        latent_ids = [int(x.replace("latent_", "")) for x in latent_names]
    elif args.latent_id is not None:
        latent_ids = [args.latent_id]
    elif args.batch_start is not None:
        subset = latent_names[args.batch_start : args.batch_start + args.batch_count]
        if not subset:
            raise ValueError(
                f"No latents found for batch_start={args.batch_start}, batch_count={args.batch_count}"
            )
        latent_ids = [int(x.replace("latent_", "")) for x in subset]
    else:
        raise ValueError("Provide one of: --all_features, --latent_id, or --batch_start")

    if metadata.get("uses_spatial_heatmaps") is False:
        print(
            "No spatial heatmaps are available for this top-k directory; "
            "saving plain top-k images instead."
        )

    for latent_id in latent_ids:
        overlayed_images = load_overlayed_images(
            ds=ds,
            top_k_images_dir=str(top_k_images_dir),
            latent_id=latent_id,
            top_k=args.top_k,
            alpha=args.alpha,
            random_sample=args.random_sample,
        )
        output_path = save_dir / f"latent_{latent_id}_top{args.top_k}.png"
        render_single_latent(
            overlayed_images=overlayed_images,
            latent_id=latent_id,
            top_k=args.top_k,
            output_path=output_path,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
