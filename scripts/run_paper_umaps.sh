#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/reproduced_umap_views}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_ROOT"

run_view() {
  local slug="$1"
  local view_name="$2"
  local source_sae_path="$3"
  local explanations_base="$4"
  local topk_images_dir="$5"
  local output_png_name="$6"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/dinov2_feature_umap_param.py" \
    --view_name "$view_name" \
    --plot_title "$view_name" \
    --source_sae_path "$SOURCE_ROOT/$source_sae_path" \
    --explanations_base "$SOURCE_ROOT/$explanations_base" \
    --topk_images_dir "$SOURCE_ROOT/$topk_images_dir" \
    --output_dir "$OUTPUT_ROOT/$slug" \
    --output_png_name "$output_png_name" \
    --lm_model_name google/gemma-3-4b-it \
    --n_clusters 15 \
    --n_umap_neighbors 15 \
    --umap_min_dist 0.1 \
    --device cuda
}

run_view \
  "base_model_full" \
  "Base Model Full" \
  "saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_2048_6_1.0_21215741/trainer_0" \
  "outputs/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_21215741" \
  "input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_21215741/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  "base_model_full_umap.png"

run_view \
  "outlier_only" \
  "Outlier Only" \
  "saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_2048_6_1.0_10758825_outlier_patches_p0.0237/trainer_0" \
  "outputs/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_10758825" \
  "input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_10758825/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  "outlier_only_umap.png"

run_view \
  "register_only" \
  "Register Only" \
  "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_22192860_registers_only/trainer_0" \
  "outputs/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_22192860" \
  "input_features_explainer/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_22192860/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  "register_only_umap.png"

run_view \
  "register_model_full" \
  "Register Model Full" \
  "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_24096850/trainer_0" \
  "outputs/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_24096850" \
  "input_features_explainer/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_24096850/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000" \
  "register_model_full_umap.png"

echo "Reproduced UMAP outputs written to: $OUTPUT_ROOT"
