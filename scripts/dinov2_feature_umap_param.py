#!/usr/bin/env python3
"""
Explanation-based SAE feature UMAP with CLI parameters.

Example:
python dinov2_feature_umap_param.py \
  --repo_dir /lambda/nfs/tokenWorkViT/HF-SAE-main \
  --explanations_base /lambda/nfs/tokenWorkViT/HF-SAE-main/outputs/validation/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_128_24161611 \
  --topk_images_dir /lambda/nfs/tokenWorkViT/HF-SAE-main/input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_128_24161611/top_k_images/ILSVRC_imagenet-1k_mean_validation_5_50000 \
  --output_dir /lambda/nfs/tokenWorkViT/HF-SAE-main/outputs/umap_results \
  --embedding_model sentence-transformers/all-mpnet-base-v2 \
  --lm_model_name google/gemma-3-4b-it \
  --n_clusters 15 \
  --n_umap_neighbors 15 \
  --umap_min_dist 0.1 \
  --n_samples_label 10 \
  --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_dir", type=str, default=None)
    parser.add_argument("--explanations_base", type=str, required=True)
    parser.add_argument("--topk_images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--view_name",
        type=str,
        default=None,
        help="Human-readable label for this UMAP view.",
    )
    parser.add_argument(
        "--source_sae_path",
        type=str,
        default=None,
        help="Optional SAE directory used to generate this view.",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument(
        "--lm_model_name",
        type=str,
        default="google/gemma-3-4b-it",
    )

    parser.add_argument("--n_clusters", type=int, default=15)
    parser.add_argument("--n_umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--n_samples_label", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="UMAP metric",
    )
    parser.add_argument(
        "--skip_cluster_labeling",
        action="store_true",
        help="Skip LLM-based cluster naming",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_png_name",
        type=str,
        default="datamapplot_sae_features.png",
    )

    return parser.parse_args()


def find_latest_method_dir(explanations_base: Path) -> Path:
    if not explanations_base.exists():
        raise FileNotFoundError(f"Explanation base does not exist: {explanations_base}")

    method_dirs = [d for d in explanations_base.iterdir() if d.is_dir()]
    if not method_dirs:
        raise FileNotFoundError(
            f"No explanation directories found under {explanations_base}"
        )

    return sorted(method_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def load_explanations(explanations_dir: Path) -> tuple[list[int], list[str]]:
    latent_ids: list[int] = []
    explanations: list[str] = []

    latent_dirs = sorted(
        explanations_dir.glob("latent_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )

    for latent_dir in tqdm(latent_dirs, desc="Reading explanations"):
        exp_path = latent_dir / "explanations" / "explanation.json"
        if not exp_path.exists():
            continue

        with open(exp_path, "r") as f:
            data = json.load(f)

        latent_id = int(str(data.get("feature_name", latent_dir.name.split("_")[-1])))
        explanation = data.get("explanation", "")
        if isinstance(explanation, list):
            explanation = explanation[0] if explanation else ""

        explanation = str(explanation).strip()
        if not explanation:
            continue

        latent_ids.append(latent_id)
        explanations.append(explanation)

    if not explanations:
        raise ValueError(f"No explanations found in {explanations_dir}")

    return latent_ids, explanations


def label_clusters_with_gemma(
    explanations: list[str],
    cluster_labels: np.ndarray,
    lm_model_name: str,
    device: str,
    n_samples_label: int,
) -> dict[int, str]:
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    print(f"Loading {lm_model_name} for cluster labeling...")
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = Gemma3ForConditionalGeneration.from_pretrained(
        lm_model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device,
    )
    lm_model.eval()

    def label_cluster(sample_explanations: list[str]) -> str:
        prompt = (
            f"You are given {len(sample_explanations)} explanations of visual or semantic "
            "elements detected by a neural network feature. Summarize in 1 to 4 words "
            "the key common category from these explanations. Try to make the category "
            "include all the explanations. Answer ONLY with the summary, nothing else.\n\n"
            "Explanations:\n" + "\n".join(sample_explanations)
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)

        with torch.no_grad():
            output = lm_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        decoded = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return decoded

    unique_clusters = np.unique(cluster_labels)
    cluster_name_map: dict[int, str] = {}

    print("Labeling clusters with Gemma...")
    for cluster_id in tqdm(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        rng = np.random.default_rng(
            int(hashlib.md5(str(cluster_id).encode()).hexdigest(), 16) % (2**32)
        )
        sample_idx = rng.choice(
            indices,
            size=min(n_samples_label, len(indices)),
            replace=False,
        )
        sample_explanations = [explanations[i] for i in sample_idx]
        label = label_cluster(sample_explanations)
        cluster_name_map[int(cluster_id)] = label
        print(f"Cluster {cluster_id} ({len(indices)} latents): {label}")

    return cluster_name_map


def load_thumbnails(topk_images_dir: Path, latent_ids: list[int]) -> list[np.ndarray]:
    from PIL import Image

    thumbnails: list[np.ndarray] = []
    thumb_size = (64, 64)

    print("Loading thumbnail images...")
    for latent_id in latent_ids:
        pt_path = topk_images_dir / f"latent_{latent_id}.pt"
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            heatmaps = data.get("heatmaps", [])
            if heatmaps and heatmaps[0] is not None:
                hm = heatmaps[0]
                if isinstance(hm, np.ndarray):
                    hm_norm = (
                        (hm - hm.min()) / (hm.max() - hm.min() + 1e-8) * 255
                    ).astype(np.uint8)
                    img = Image.fromarray(hm_norm).convert("RGB").resize(thumb_size)
                else:
                    img = Image.new("RGB", thumb_size, color=(128, 128, 128))
            else:
                img = Image.new("RGB", thumb_size, color=(128, 128, 128))
        except Exception:
            img = Image.new("RGB", thumb_size, color=(128, 128, 128))

        thumbnails.append(np.array(img))

    return thumbnails


def main() -> None:
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else None
    if repo_dir is not None and str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    explanations_base = Path(args.explanations_base).resolve()
    topk_images_dir = Path(args.topk_images_dir).resolve()

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import umap
    import datamapplot
    import matplotlib.pyplot as plt

    explanations_dir = find_latest_method_dir(explanations_base)
    print(f"Using explanations from: {explanations_dir}")

    view_name = args.view_name or output_dir.name.replace("_", " ").title()
    plot_title = args.plot_title or f"{view_name} UMAP"
    print(f"View label: {view_name}")

    latent_ids, explanations = load_explanations(explanations_dir)
    print(f"Loaded {len(explanations)} explanations")

    print(f"Embedding with {args.embedding_model}...")
    embed_model = SentenceTransformer(args.embedding_model)
    embeddings_np = embed_model.encode(
        explanations,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {embeddings_np.shape}")

    n_clusters = min(args.n_clusters, max(1, len(embeddings_np) // 3))
    if n_clusters < 1:
        raise ValueError("Not enough explanations to cluster.")

    print(f"Clustering into {n_clusters} clusters with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")

    print("Computing UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.metric,
        random_state=42,
    )
    umap_embeddings = reducer.fit_transform(embeddings_np)

    if args.skip_cluster_labeling:
        cluster_name_map = {int(c): f"Cluster {int(c)}" for c in np.unique(cluster_labels)}
    else:
        cluster_name_map = label_clusters_with_gemma(
            explanations=explanations,
            cluster_labels=cluster_labels,
            lm_model_name=args.lm_model_name,
            device=args.device,
            n_samples_label=args.n_samples_label,
        )

    full_labels = [cluster_name_map[int(c)] for c in cluster_labels]

    np.save(output_dir / "umap_embeddings.npy", umap_embeddings)
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    with open(output_dir / "cluster_names.json", "w") as f:
        json.dump({str(k): v for k, v in cluster_name_map.items()}, f, indent=2)

    with open(output_dir / "latent_ids.json", "w") as f:
        json.dump(latent_ids, f, indent=2)

    with open(output_dir / "explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    metadata = {
        "view_name": view_name,
        "plot_title": plot_title,
        "output_dir": str(output_dir),
        "output_png": str(output_dir / args.output_png_name),
        "source_sae_path": str(Path(args.source_sae_path).resolve()) if args.source_sae_path else None,
        "topk_images_dir": str(topk_images_dir),
        "explanations_base": str(explanations_base),
        "explanations_dir": str(explanations_dir),
        "embedding_model": args.embedding_model,
        "lm_model_name": args.lm_model_name,
        "n_clusters": int(n_clusters),
        "n_umap_neighbors": int(args.n_umap_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "metric": args.metric,
    }
    with open(output_dir / "view_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _ = load_thumbnails(topk_images_dir, latent_ids)

    print("Creating datamapplot figure...")
    fig, ax = datamapplot.create_plot(
        umap_embeddings,
        full_labels,
        label_over_points=True,
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=0.5,
        min_font_size=10,
    )

    ax.set_title(plot_title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    output_png = output_dir / args.output_png_name
    fig.savefig(output_png, dpi=200, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved UMAP embeddings to {output_dir / 'umap_embeddings.npy'}")
    print(f"Saved cluster labels to {output_dir / 'cluster_labels.npy'}")
    print(f"Saved cluster names to {output_dir / 'cluster_names.json'}")
    print(f"Saved view metadata to {output_dir / 'view_metadata.json'}")
    print(f"Saved plot to {output_png}")


if __name__ == "__main__":
    main()
