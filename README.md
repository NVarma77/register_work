# Register Work UMAP Artifacts

This repository is the paper-submission subset for the UMAP figure outputs. It contains:

- the final labeled UMAP artifacts under `outputs/umap_views/`
- the minimal script needed to regenerate those UMAPs from existing explanation and top-k-image directories
- a manifest and runner script encoding the exact four views used in the comparison

The four views are:

- `base_model_full`
- `outlier_only`
- `register_only`
- `register_model_full`

## Included Files

- `outputs/umap_views/`
  - final labeled UMAP outputs and per-view provenance
- `scripts/dinov2_feature_umap_param.py`
  - the UMAP-generation script
- `scripts/run_paper_umaps.sh`
  - reruns the four paper UMAPs from existing source inputs
- `paper_umap_sources.json`
  - exact SAE, explanation, and top-k directories for each view
- `requirements.txt`
  - minimal dependencies for the UMAP-generation step

## Reproducibility Scope

This repo intentionally excludes training code, checkpoints beyond the referenced source paths, datasets, and unrelated analysis code. It only includes what is needed to:

1. inspect the final UMAP outputs
2. identify the exact source directories used for each paper view
3. regenerate the UMAP outputs when those source directories are available locally

## Output Contents

Each directory in `outputs/umap_views/` contains:

- the renamed UMAP PNG
- `umap_embeddings.npy`
- `cluster_labels.npy`
- `cluster_names.json`
- `latent_ids.json`
- `explanations.json`
- `view_metadata.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Regenerating The Paper UMAPs

The source inputs themselves are not duplicated in this repo. They are referenced in `paper_umap_sources.json`.

If those directories exist on your machine, run:

```bash
export SOURCE_ROOT=/path/to/original/artifact/root
bash scripts/run_paper_umaps.sh
```

`SOURCE_ROOT` should be the directory that contains the referenced `saes/`, `outputs/`, and `input_features_explainer/` trees.

By default the script writes regenerated outputs under `reproduced_umap_views/`.

## Source Mapping

See:

- `paper_umap_sources.json`
- `outputs/umap_views/README.md`
