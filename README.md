# Register Work UMAP Artifacts

This repository is the paper-submission subset for the UMAP figure outputs. It contains:

- the final labeled UMAP artifacts under `outputs/umap_views/`
- the minimal code path needed to regenerate those UMAPs
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
- `scripts/viz_topk_heatmaps.py`
  - the top-k visualization script used before explanation/UMAP generation
- `scripts/run_paper_umaps.sh`
  - reruns the four paper UMAPs from existing source inputs
- `paper_umap_sources.json`
  - exact SAE, explanation, and top-k directories for each view
- `src/`
  - minimal pipeline entrypoints: SAE training/loading config, top-k extraction, explanation generation, and helpers
- `utils/`
  - utility code imported by the included pipeline scripts
- `dictionary_learning/`
  - the required SAE package subset used by `demo.py` and the feature-extraction steps
  - trimmed to the `top_k` SAE path used for the paper artifacts
- `packages/overcomplete/`
  - only the visualization subset needed for heatmap overlays
- `config/`
  - model, dataset, and SAE path configuration used by the included scripts
- `requirements.txt`
  - minimal dependencies for the included paper pipeline code

## Reproducibility Scope

This repo intentionally excludes notebooks, logs, archives, tests, unrelated evaluation code, raw datasets, and checkpoints beyond the referenced source paths. It only includes what is needed to:

1. inspect the final UMAP outputs
2. identify the exact source directories used for each paper view
3. rerun the script chain used to get to those UMAPs when the source artifacts are available locally

This subset is intentionally restricted to the DINOv2 TopK SAE workflow used for the paper figure. Other SAE trainer families from the larger research repo are not included here.

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

## Included Paper Pipeline Stages

The included code covers the pipeline stages relevant to the figure:

1. `src/demo.py`
2. `src/get_max_activating_vision.py`
3. `scripts/viz_topk_heatmaps.py`
4. `src/get_steering_explanations.py`
5. `scripts/dinov2_feature_umap_param.py`

The repository does not include the rest of the research repo outside the dependency closure of those steps, and the copied SAE support code has been reduced to the TopK path used in this figure.

## Source Mapping

See:

- `paper_umap_sources.json`
- `outputs/umap_views/README.md`
