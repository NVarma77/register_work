# UMAP Views

This directory contains the renamed existing UMAP outputs, grouped by the four labels used in the comparison.

## Views

- `base_model_full`
  - UMAP image: `base_model_full/base_model_full_umap.png`
  - SAE: `saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_2048_6_1.0_21215741/trainer_0`
  - Top-k inputs: `input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_21215741/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000`
  - Explanations: `outputs/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_21215741/HF_easy3_masks_TOPK-5_MODEL-google_gemma-3-4b-it_RNDTOPK-False_SIZE-100000`

- `outlier_only`
  - UMAP image: `outlier_only/outlier_only_umap.png`
  - SAE: `saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_2048_6_1.0_10758825_outlier_patches_p0.0237/trainer_0`
  - Top-k inputs: `input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_10758825/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000`
  - Explanations: `outputs/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_10758825/HF_easy3_masks_TOPK-5_MODEL-google_gemma-3-4b-it_RNDTOPK-False_SIZE-100000`

- `register_only`
  - UMAP image: `register_only/register_only_umap.png`
  - SAE: `saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_22192860_registers_only/trainer_0`
  - Top-k inputs: `input_features_explainer/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_22192860/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000`
  - Explanations: `outputs/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_22192860/HF_easy3_images_TOPK-5_MODEL-google_gemma-3-4b-it_RNDTOPK-False_SIZE-100000`

- `register_model_full`
  - UMAP image: `register_model_full/register_model_full_umap.png`
  - SAE: `saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_24096850/trainer_0`
  - Top-k inputs: `input_features_explainer/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_24096850/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000`
  - Explanations: `outputs/facebook_dinov2-with-registers-small_AutoEncoderTopK/enc_res_out_layer_8_2048_24096850/HF_easy3_masks_TOPK-5_MODEL-google_gemma-3-4b-it_RNDTOPK-False_SIZE-100000`

Each view directory also includes a `view_metadata.json` file with the same provenance in machine-readable form.
