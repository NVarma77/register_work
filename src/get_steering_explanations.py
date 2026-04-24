# %%
import sys
import os
from pathlib import Path
from dotenv import dotenv_values
from dotenv import load_dotenv
load_dotenv()

REPO_DIR = os.environ.get('REPO_DIR')

# Check if REPO_DIR environment variable is set
if REPO_DIR is None:
    REPO_DIR = str(Path(__file__).resolve().parent.parent)
    os.environ['REPO_DIR'] = REPO_DIR
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

from datetime import datetime
import torch
from torch import Tensor
from typing import List, Tuple, Callable, Union
from dataclasses import dataclass
import argparse
import json
import re
from tqdm import tqdm
 
from utils.utils import load_model, resolve_attr
from dictionary_learning.utils import (
    hf_dataset_to_generator,
    load_dictionary,
    load_dataset_from_yaml,
)
from dictionary_learning.buffer import tokenized_batch
from datasets import load_from_disk
import demo_config
from demo_config import get_activation_dim
from PIL import Image
import numpy as np
from utils.hf_hook_utils import get_activation_addition_output_post_hook, get_activation_addition_output_post_hook_v2
from transformers import GenerationConfig

from utils.hf_hook_utils import add_hooks
from utils.hf_hook_utils import generate_completions
from utils.utils import save_json_data, create_output_path, load_vision_model
from utils.utils import get_paths
from utils.utils import list_features, get_or_create_attribution_images
from utils.utils import get_sae_run_id
from utils.hf_hook_utils import generate_completions_internvl
from src.steering_utils import read_top_k_images

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
# Set logging level to ERROR to suppress WARNING messages
import logging
logging.getLogger().setLevel(logging.ERROR)

DATA_DIR = os.environ.get('DATA_DIR')
# TODO: change by reading from env vars
SAES_DIR = f"{DATA_DIR}/saes"
DATASET_NAME = "ILSVRC/imagenet-1k"
DATASET_SPLIT = "test"
TOP_K_IMAGES = 5
TEMPERATURES = {
    'google/gemma-3-4b-it': 0.7,
    'google/gemma-3-12b-it': 0.7,
    'google/gemma-3-27b-it': 0.7,
    'OpenGVLab/InternVL3-14B': 0.5
}

# Temporal!!
kwargs_top_k_images = {'top_k': TOP_K_IMAGES}


# List of strings to replace in the prompt completions
to_replace_list = ["The highlighted element in the image is a ",
                    "Here's a description of the highlighted element in the image:",
                    "Here's a description of the highlighted element across all the images:",
                    "Here's a description of the highlighted element across all images:",
                    "*",
                    "Summary:"
                    ]

def summarize_explanations(model, processor, data_batch, cfg) -> str:

    input_len = data_batch["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**data_batch, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return [decoded]

def get_top_k_images_path(subject_model_name, layer, subset="test", dataset_size=50000, top_k_images_dir=None):
    # Given a subject model name and a layer, return the path to the top k images
    if top_k_images_dir is not None:
        return str(top_k_images_dir)
    top_k_images_path = get_paths(subject_model_name, layer, path_to_get="top_k_images_id")
    top_k_images_path = top_k_images_path.replace('test', subset)
    top_k_images_path = top_k_images_path.replace('50000', str(dataset_size))
    return top_k_images_path

def get_baseline_image_steering(prompt, steering_type, base_image_type, tokenizer, processor, cfg, batch_size):
    assert steering_type in ['raw', 'steering_aided_top_k', 'none']
    
    # Default prompt for the baseline inputs
    baseline_image = Image.new("RGB", (224, 224), color="white")

    raw_inputs = [{'text': [prompt], 'image': [baseline_image]}]*batch_size
    data_batch = tokenized_batch(raw_inputs, tokenizer, cfg, processor)
    
    if base_image_type == 'blank':
        data_batch['pixel_values'] /= data_batch['pixel_values']
    elif base_image_type == 'random':
        data_batch['pixel_values'] = torch.randn_like(data_batch['pixel_values'])
    elif base_image_type == 'black':
        data_batch['pixel_values'] *= 0.0001

    return data_batch, None

def load_prompt(model_name, steering_type, base_image_type, mask_flag=True, heatmaps_flag=False, short_explanation=False):
    if 'paligemma' in model_name.lower():
        prompt = "Describe the image in great detail."
    else:
        if steering_type == 'raw':
            if base_image_type == 'blank' or base_image_type == 'none':
                prompt = 'You are given an image highlighting and repeating a visual or semantic element. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Please in your description ignore the white background and the repetition of the element across the image. Start the highlighted element description with "The highlighted element in the image is a"'
            elif base_image_type == 'black':
                prompt = "You are given a black image where a specific visual/semantic element has been highlighted. Your task is to provide a unique concise explanation of 3-5 words of the highlighted element. Do not get distracted by the black background. Answer directly with the explanation."
        elif steering_type == 'steering_aided_top_k' or steering_type == 'none':
            if mask_flag:
                prompt = 'You are given set of images highlighting a visual or semantic element. The patches of the images not showing the element are masked out, giving the impression of a pixelated image. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Provide a single description for the highlighted element appearing in all images, and please ignore the pixelated effect of the mask when describing the element. Start the highlighted element description with "The highlighted element in the image is a".'
            elif heatmaps_flag:
                prompt = 'You are given set of images highlighting a visual or semantic element. The patches of the images showing the element are highlighted with a green heatmap. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Provide a single description for the highlighted element appearing in all images, and please ignore the overlayed green heatmap when describing the element. Start the highlighted element description with "The highlighted element in the image is a".'
            else:
                prompt = 'You are given a set of images that strongly activate the same visual feature. Your task is to describe the shared visual or semantic concept that appears across these images in a single, clear sentence. The concept may range from a low-level visual pattern to a high-level abstract object or scene property. Favor a more general interpretation. Start the highlighted element description with "The highlighted element in the image is a".'
    if short_explanation:
        prompt = prompt.replace("in a single, clear sentence", "in 3-5 words")
    return prompt

@dataclass
class ExperimentConfig:
    engine: str = 'HF'
    model_name: str = 'google/gemma-3-4b-it'
    lm_model_name: str = 'google/gemma-3-4b-it'
    dataset: str = "ILSVRC/imagenet-1k"
    dataset_type: str = 'raw'
    device: str = 'cuda:0'
    model_type: str = demo_config.LLM_CONFIG[model_name].model_type
    model_path: str = demo_config.LLM_CONFIG[model_name].model_path
    submodel: str = 'enc'
    site: str = 'res'
    io: str = 'out'
    activation_dim: int = get_activation_dim(model_name, submodel)
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True
    ctx_len: int = 128
    get_full_model: bool = True
    sae_path: str = None
    coeff: float = None
    layer: int = None
    max_new_tokens: int = 100
    base_image_type: str = 'blank'
    n_generations: int = 5
    batch_size: int = 8
    save: bool = False
    temperature: float = 1
    sample: bool = False
    steering_position: str = 'input'
    run_type: str = 'validation'
    steering_type: str = 'raw'
    dataset_size: int = 50000
    intervention_type: str = 'addition'
    top_k_random_sample: bool = False
    type_mask: str = 'masks'
    short_explanation: bool = False
    top_k_images_dir: str = None
    stream_dataset: bool = False
    max_dataset_examples: int = None

def main(cfg):
    device = cfg.device
    save = cfg.save
    sae_path = cfg.sae_path
    coeff = cfg.coeff
    max_new_tokens = cfg.max_new_tokens
    base_image_type = cfg.base_image_type
    lm_model_name = cfg.lm_model_name
    batch_size = cfg.batch_size
    steering_position = cfg.steering_position
    lm_model_path = demo_config.LLM_CONFIG[lm_model_name].model_path
    top_k_random_sample = cfg.top_k_random_sample

    print(f'Top k random sample: {top_k_random_sample}')
    print(f'Type mask: {cfg.type_mask}')

    if cfg.type_mask == 'masks':
        mask_flag = True
        heatmaps_flag = False
    elif cfg.type_mask == 'heatmaps':
        heatmaps_flag = True
        mask_flag = False
    elif cfg.type_mask == 'none':
        heatmaps_flag = False
        mask_flag = False
    else:
        raise ValueError(f"Invalid type_mask: {cfg.type_mask}")

    print()

    if cfg.sample is False:
        cfg.temperature = 0.0001
        cfg.n_generations = 1
        print("No sampling, temperature set to 0.0 and n_generations set to 1")
    else:
        cfg.temperature = TEMPERATURES[cfg.lm_model_name]
        print(f"Sampling, temperature set to {cfg.temperature}")
    sample = cfg.sample

    if cfg.steering_type == 'raw':
        num_images = 1
    else:
        num_images = TOP_K_IMAGES

    print(f"num images: {num_images}")
    print(f"Loaded SAE path: {sae_path}")

    # Get the random number id from the sae path
    rnd_num_id = get_sae_run_id(sae_path)

    # Load SAE
    sae, sae_config = load_dictionary(sae_path, device)
    model_name = cfg.model_name
    model_type = demo_config.LLM_CONFIG[model_name].model_type
    
    # Load SAE config metadata
    dict_size = sae_config['trainer']['dict_size']
    submodule_name = sae_config['trainer']['submodule_name']
    submodule_name_list = submodule_name.split('_')
    submodel = submodule_name_list[0].replace('-', '_')
    site = submodule_name_list[1]
    io = submodule_name_list[2]
    layer_num = int(submodule_name_list[4])
    activation_dim = get_activation_dim(model_name, submodel)
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    print(f"Loaded dict_size: {dict_size}")
    print(f"Loaded submodule_name: {submodule_name}")

    # Set the config parameters
    cfg.model_type = model_type
    cfg.model_type = demo_config.LLM_CONFIG[lm_model_name].model_type
    cfg.activation_dim = activation_dim
    cfg.submodel = submodel
    cfg.site = site
    cfg.io = io
    cfg.dtype = dtype

    sae_architecture = sae_config['trainer']['dict_class']
    sae.to(device)
    sae.to(dtype)

    model, tokenizer, processor = load_model(lm_model_path, cfg, device=device)

    if 'gemma-3' in model_name.lower() or 'paligemma' in model_name.lower():
        submodule = resolve_attr(model.vision_tower.vision_model, f"encoder.layers[{layer_num}]")
    elif 'internvl3' in model_name.lower():
        submodule = resolve_attr(model.vision_model, f"encoder.layers[{layer_num}]")
    else:
        submodule = resolve_attr(model.vision_tower.vision_model, f"encoder.layers[{layer_num}]")

    def get_baseline_batch_inputs(prompt: str, batch_latent_ids: List[int], base_image_type: str, steering_type: str, **top_k_kwargs):
        batch_size = len(batch_latent_ids)

        if cfg.steering_type == 'raw':
            return get_baseline_image_steering(prompt, steering_type, base_image_type, tokenizer, processor, cfg, batch_size)

        elif cfg.steering_type == 'steering_aided_top_k' or cfg.steering_type == 'none':
            batch_top_k_data = [
                read_top_k_images(
                    top_k_kwargs['ds'],
                    latent_id,
                    top_k_kwargs['top_k_images_dir'],
                    k=top_k_kwargs['num_images'],
                    top_k_random_sample=top_k_kwargs['top_k_random_sample']
                )
                for latent_id in batch_latent_ids[::top_k_kwargs['num_images']]
            ]

            batch_top_masked_images = []
            positions_heatmaps = []
            for top_k_data in batch_top_k_data:
                images_latent = top_k_data['images']
                heatmaps_latent = top_k_data.get('heatmaps', [])

                if not mask_flag and not heatmaps_flag:
                    batch_top_masked_images.append(images_latent)
                    continue

                image_arrays = []
                feature_heatmaps = []
                for image_data, heatmap_data in zip(images_latent, heatmaps_latent):
                    img_array = np.array(image_data) if isinstance(image_data, Image.Image) else image_data
                    image_arrays.append(img_array)
                    feature_heatmaps.append(heatmap_data)
                    positions_heatmap = np.where(heatmap_data.flatten() > 0)[0]
                    positions_heatmaps.append(positions_heatmap)

                batch_top_images = get_or_create_attribution_images(
                    image_arrays,
                    feature_heatmaps,
                    heatmaps_flag=top_k_kwargs['heatmaps_flag'],
                    masks_flag=top_k_kwargs['mask_flag']
                )
                batch_top_masked_images.append(batch_top_images)

            raw_inputs = [{'text': [prompt], 'image': batch_top_images} for batch_top_images in batch_top_masked_images]
            data_batch = tokenized_batch(raw_inputs, tokenizer, cfg, processor)
            return data_batch, positions_heatmaps

        raise ValueError(f"Unsupported steering_type: {cfg.steering_type}")

    if cfg.steering_type != 'raw':
        if cfg.stream_dataset:
            ds_iter, _ = hf_dataset_to_generator(
                DATASET_NAME,
                split=DATASET_SPLIT,
                max_examples=cfg.max_dataset_examples,
                streaming=True,
            )
            ds = list(ds_iter)
        else:
            ds = load_dataset_from_yaml(DATASET_NAME, split=DATASET_SPLIT)[0]
    else:
        ds = None

    clean_model_name = model_name.replace("/", "_")
    sae_id = f"{clean_model_name}_{sae_architecture}/{submodule_name}_{dict_size}_{rnd_num_id}"

    if 'gemma-3' in model_name.lower() or 'paligemma' in model_name.lower() or 'llama' in model_name.lower():
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            top_p=0.95 if sample else None,
            top_k=64 if sample else None,
            temperature=cfg.temperature,
            num_return_sequences=cfg.n_generations
        )
        print(f"Generation config: {generation_config.to_dict()}")
        generation_kwargs = {
            'output_scores': True,
            'return_dict_in_generate': True,
            'renormalize_logits': True,
            'num_return_sequences': cfg.n_generations
        }

    elif 'qwen2.5-omni' in model_name.lower():
        generation_config = None
        generation_kwargs = {'use_audio_in_video': True, 'return_audio': False}

    elif 'internvl3' in model_name.lower():
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=cfg.temperature
        )
        generation_kwargs = {}

    if 'generation_config' not in locals():
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=cfg.temperature,
            num_return_sequences=cfg.n_generations
        )
        generation_kwargs = {
            'output_scores': True,
            'return_dict_in_generate': True,
            'renormalize_logits': True,
            'num_return_sequences': cfg.n_generations
        }

    prompt = load_prompt(
        model_name,
        cfg.steering_type,
        cfg.base_image_type,
        mask_flag=mask_flag,
        heatmaps_flag=heatmaps_flag,
        short_explanation=cfg.short_explanation
    )
    print(f"Prompt: {prompt}")

    if cfg.run_type == "validation":
        stop_latent = min(500, dict_size)
        latents_range = list(range(0, stop_latent))
    elif cfg.run_type == "test":
        if dict_size <= 500:
            latents_range = list(range(0, dict_size))
            print(f"dict_size <= 500, so test run will use all latents: 0..{dict_size - 1}")
        else:
            latents_range = list(range(500, dict_size))
    else:
        raise ValueError(f"Invalid run type: {cfg.run_type}")

    top_k_images_dir = get_top_k_images_path(
        cfg.model_name,
        cfg.layer,
        subset=DATASET_SPLIT,
        dataset_size=cfg.dataset_size,
        top_k_images_dir=cfg.top_k_images_dir,
    )
    print(f'Top k images dir: {top_k_images_dir}')

    if steering_position == 'middle':
        context_length = demo_config.LLM_CONFIG[model_name].context_length
        steering_position_in_model = [
            (context_length // 2) - int(context_length * 0.2),
            (context_length // 2) + int(context_length * 0.2)
        ]
    elif steering_position == 'input':
        steering_position_in_model = 'input'
    else:
        raise ValueError(f"Invalid steering_position: {steering_position}")

    all_features = list(list_features(Path(top_k_images_dir)))
    print(f"Num entries in top_k_images_dir: {len(all_features)}")
    print(f"First 10 entries: {all_features[:10]}")

    latent_features = [feature for feature in all_features if str(feature).startswith("latent_")]
    print(f"Num latent_* features found: {len(latent_features)}")

    features = [
        feature for i, feature in enumerate(all_features)
        if i in latents_range and str(feature).startswith("latent_")
    ]

    print(f"Num selected features: {len(features)}")
    print(f"First 10 selected features: {features[:10]}")

    for features_batch_idx in tqdm(range(0, len(features), batch_size), desc="Generating explanations"):
        batch_latent_ids = features[features_batch_idx:features_batch_idx + batch_size] if features_batch_idx + batch_size <= len(features) else features[features_batch_idx:len(features)]
        batch_latent_ids = [int(feature.split("latent_")[-1]) for feature in batch_latent_ids for _ in range(num_images)]

        direction = sae.decoder.weight[:, batch_latent_ids]
        num_sequences_per_prompt = cfg.n_generations
        direction_t = direction.t()
        repeated_direction_t = torch.repeat_interleave(direction_t, num_sequences_per_prompt, dim=0)

        if cfg.steering_type == 'steering_aided_top_k' or cfg.steering_type == 'none':
            top_k_kwargs = {
                'ds': ds,
                'top_k_images_dir': top_k_images_dir,
                'num_images': num_images,
                'heatmaps_flag': heatmaps_flag,
                'mask_flag': mask_flag,
                'top_k_random_sample': top_k_random_sample
            }
            data_batch, positions_heatmaps = get_baseline_batch_inputs(
                prompt,
                batch_latent_ids,
                cfg.base_image_type,
                cfg.steering_type,
                **top_k_kwargs
            )
            if cfg.steering_type == 'steering_aided_top_k':
                steering_fwd_post_hooks = [
                    (
                        submodule,
                        get_activation_addition_output_post_hook_v2(
                            vector=repeated_direction_t,
                            coeff=coeff,
                            position=positions_heatmaps,
                            intervention_type=cfg.intervention_type
                        )
                    )
                ]
            elif cfg.steering_type == 'none':
                steering_fwd_post_hooks = []
        elif cfg.steering_type == 'raw':
            data_batch, _ = get_baseline_batch_inputs(
                prompt,
                batch_latent_ids,
                cfg.base_image_type,
                cfg.steering_type
            )
            steering_fwd_post_hooks = [
                (
                    submodule,
                    get_activation_addition_output_post_hook_v2(
                        vector=repeated_direction_t,
                        coeff=coeff,
                        position=steering_position_in_model,
                        intervention_type=cfg.intervention_type
                    )
                )
            ]
        else:
            raise ValueError(f"Unsupported steering_type: {cfg.steering_type}")

        if cfg.model_name == 'OpenGVLab/InternVL3-14B':
            questions = data_batch['questions'][0][0]
            pixel_values = data_batch['pixel_values']
            num_patches_list = data_batch['num_patches_list']
            pixel_values = torch.repeat_interleave(pixel_values, cfg.n_generations, dim=0)
            num_patches_list = [1] * cfg.n_generations
            questions = [questions] * cfg.n_generations
            steered_completions = generate_completions_internvl(
                model,
                tokenizer,
                pixel_values,
                questions,
                num_patches_list,
                fwd_pre_hooks=[],
                fwd_hooks=steering_fwd_post_hooks,
                generation_config=generation_config,
                **generation_kwargs
            )
            steered_log_probs = None
        else:
            steered_completions, steered_log_probs = generate_completions(
                model,
                tokenizer,
                data_batch,
                fwd_pre_hooks=[],
                fwd_hooks=steering_fwd_post_hooks,
                generation_config=generation_config,
                **generation_kwargs
            )

        for i, _latent_id in zip(range(0, len(steered_completions), num_sequences_per_prompt), batch_latent_ids[::num_images]):
            prompt_completions = steered_completions[i:i + num_sequences_per_prompt]
            for to_replace in to_replace_list:
                prompt_completions = [completion.replace(to_replace, "").strip().capitalize() for completion in prompt_completions]

            log_probs = steered_log_probs[i:i + num_sequences_per_prompt] if steered_log_probs is not None else None
            print(f"Prompt completions: {prompt_completions}", end='\n')

            if save:
                clean_model_name = model_name.replace("/", "_")
                if cfg.steering_type == 'none':
                    if mask_flag:
                        explanation_mode = f'HF_easy3_masks_TOPK-{num_images}_MODEL-{lm_model_name.replace("/", "_")}' + f'_RNDTOPK-{cfg.top_k_random_sample}'
                    elif heatmaps_flag:
                        explanation_mode = f'HF_easy3_heatmaps_TOPK-{num_images}_MODEL-{lm_model_name.replace("/", "_")}' + f'_RNDTOPK-{cfg.top_k_random_sample}'
                    else:
                        explanation_mode = f'HF_easy3_images_TOPK-{num_images}_MODEL-{lm_model_name.replace("/", "_")}' + f'_RNDTOPK-{cfg.top_k_random_sample}'
                    if cfg.short_explanation:
                        explanation_mode = explanation_mode + f"_SHORT"

                else:
                    fixed_steering_suffix = f'{lm_model_name.replace("/", "_")}_sampling-{cfg.sample}' + "_" + cfg.base_image_type + "_input_"
                    if cfg.short_explanation:
                        fixed_steering_suffix = fixed_steering_suffix + f"_SHORT"
                    if cfg.steering_type == 'steering_aided_top_k':
                        if mask_flag:
                            explanation_mode = f"easy_w_steering_masks_" + fixed_steering_suffix + f"_RNDTOPK-{cfg.top_k_random_sample}" + f"_COEFF-{int(cfg.coeff)}"
                        elif heatmaps_flag:
                            explanation_mode = f"easy_w_steering_heatmaps_" + fixed_steering_suffix + f"_RNDTOPK-{cfg.top_k_random_sample}" + f"_COEFF-{int(cfg.coeff)}"
                    elif cfg.steering_type == 'raw':
                        explanation_mode = f"steering_" + fixed_steering_suffix + f"_COEFF-{int(cfg.coeff)}"

                if cfg.dataset_size != 50000:
                    explanation_mode = explanation_mode + f"_SIZE-{cfg.dataset_size}"

                sae_id = f"{clean_model_name}_{sae_architecture}/{submodule_name}_{dict_size}_{rnd_num_id}"
                if cfg.run_type == "validation":
                    output_dir = f"{DATA_DIR}/outputs/validation"
                elif cfg.run_type == "test":
                    output_dir = f"{DATA_DIR}/outputs"
                else:
                    raise ValueError(f"Invalid run type: {cfg.run_type}")

                save_dir = os.path.join(output_dir, sae_id, explanation_mode, f"latent_{_latent_id}", 'explanations')
                print(f"Saving explanations to {save_dir}")
                logging.info(f"Saving explanations to {save_dir}")

                explanation = {
                    "feature_name": _latent_id,
                    "prompt": prompt,
                    "explanation": prompt_completions,
                    "raw_output": prompt_completions,
                    "prompt_log_probs": log_probs,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "model": lm_model_name.replace("/", "_"),
                        "prompt_system": "",
                        "base_image_type": base_image_type,
                        "steering_coeff": str(int(coeff)),
                        "sampling": generation_config.to_dict() if type(generation_config) == GenerationConfig else generation_config,
                        "sae_id": sae_id,
                    }
                }

                latent_filename = create_output_path(Path(save_dir), filename=f"explanation.json")
                save_json_data(explanation, Path(latent_filename))

    print('DONE!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=str, default="mid", choices=["early", "mid", "later"])
    parser.add_argument("--lm_model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--submodel", type=str, default="enc")
    parser.add_argument("--coeff", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--steering_position", type=str, default="input")
    parser.add_argument("--steering_type", type=str, choices=["raw", "steering_aided_top_k", "none"], default="raw")
    parser.add_argument("--base_image_type", type=str, choices=["blank", "random", "black", "none"], default="blank")
    parser.add_argument("--intervention_type", type=str, choices=["addition", "replacement"], default="addition")
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--sample", action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--run_type", type=str, choices=["validation", "test"], default="validation")
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--top_k_random_sample", action='store_true', default=False)
    parser.add_argument("--type_mask", type=str, choices=["heatmaps", "masks", "none"], default="masks")
    parser.add_argument("--short_explanation", action='store_true', default=False)
    parser.add_argument("--top_k_images_dir", type=str, default=None)
    parser.add_argument("--sae_path", type=str, default=None)
    parser.add_argument("--stream_dataset", action="store_true", default=False)
    parser.add_argument("--max_dataset_examples", type=int, default=None)
    args = parser.parse_args()

    assert args.steering_type in ['raw', 'steering_aided_top_k', 'none'], "Steering type must be raw, steering_aided_top_k or none"
    assert args.steering_position != 'middle', "Steering position must be input or top_k_images"

    sae_path = args.sae_path if args.sae_path is not None else get_paths(args.model_name, args.layer)

    cfg = ExperimentConfig(
        model_name=args.model_name,
        lm_model_name=args.lm_model_name,
        sae_path=sae_path,
        coeff=args.coeff,
        max_new_tokens=args.max_new_tokens,
        steering_position=args.steering_position,
        n_generations=args.n_generations,
        save=args.save,
        sample=args.sample,
        temperature=args.temperature,
        run_type=args.run_type,
        steering_type=args.steering_type,
        layer=args.layer,
        dataset_size=args.dataset_size,
        intervention_type=args.intervention_type,
        top_k_random_sample=args.top_k_random_sample,
        type_mask=args.type_mask,
        short_explanation=args.short_explanation,
        top_k_images_dir=args.top_k_images_dir,
        stream_dataset=args.stream_dataset,
        max_dataset_examples=args.max_dataset_examples,
    )

    if args.steering_type == 'steering_aided_top_k':
        cfg.base_image_type = 'top_k_images'

    if cfg.lm_model_name == 'OpenGVLab/InternVL3-14B':
        cfg.batch_size = 1
        print(f'Setting batch size to 1 for {cfg.lm_model_name}')

    main(cfg)


'''
Example:

.venv/bin/python -m src.get_steering_explanations \
  --model_name facebook/dinov2-small \
  --lm_model_name google/gemma-3-4b-it \
  --layer mid \
  --sae_path /lambda/nfs/tokenWorkViT/HF-SAE-main/saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_2048_6_1.0_21215741/trainer_0 \
  --steering_type none \
  --type_mask masks \
  --run_type test \
  --save \
  --n_generations 1 \
  --dataset_size 100000 \
  --top_k_images_dir /lambda/nfs/tokenWorkViT/HF-SAE-main/input_features_explainer/facebook_dinov2-small_AutoEncoderTopK/enc_res_out_layer_8_2048_21215741/top_k_images/ILSVRC_imagenet-1k_mean_test_10_100000
'''
# %%
