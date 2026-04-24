# Register Work UMAP Artifacts

This repository contains the labeled UMAP outputs for the four comparison views:

- `base_model_full`
- `outlier_only`
- `register_only`
- `register_model_full`

The final outputs live under `outputs/umap_views/`. Each view directory contains:

- the renamed UMAP PNG
- `umap_embeddings.npy`
- `cluster_labels.npy`
- `cluster_names.json`
- `latent_ids.json`
- `explanations.json`
- `view_metadata.json`

## Layout

- `outputs/umap_views/`
  - final labeled UMAP outputs
  - provenance for each view
- `scripts/dinov2_feature_umap_param.py`
  - the UMAP-generation script used for these artifacts
- `requirements.txt`
  - minimal Python dependencies for the UMAP step

## Reproducibility Scope

This repo intentionally includes only the files needed to:

1. inspect the final UMAP outputs
2. trace each view back to its source SAE, top-k image directory, and explanation directory
3. rerun the UMAP-generation step when those source inputs are available

It intentionally excludes the larger training runs, raw datasets, full model checkpoints, and the broader research codebase.

## View Mapping

See `outputs/umap_views/README.md` for the exact source directories behind each of the four labeled views.

## Minimal Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
