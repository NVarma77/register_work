# %%
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
REPO_DIR = os.environ.get('REPO_DIR')
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import List, Tuple, Callable, Union
from dataclasses import dataclass
import gc
import argparse
import json
from tqdm import tqdm
from utils.utils import load_model, numpy_to_pil
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary
from collections import defaultdict
import demo_config
from demo_config import get_activation_dim
from PIL import Image
import numpy as np
from utils.sae_utils import get_patch_sae_codes
from dictionary_learning.buffer import hf_forward
from src.processing import tokenized_batch
from utils.utils import get_paths, load_active_latents, get_sae_run_id

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

device = "cuda:0"
seed = 42
DATA_DIR = os.environ['DATA_DIR']

@dataclass
class ExperimentConfig:
    model_name: str = None
    layer: int = None
    dataset: str = "ILSVRC/imagenet-1k"
    dataset_split: str = 'validation'
    dataset_type: str = 'raw'
    device: str = 'cuda:0'
    model_type: str = None
    model_path: str = None
    submodel: str = None
    io: str = None
    activation_dim: int = None
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True
    get_full_model: bool = False
    partition: int = 4
    token_subset: str = "all"

@dataclass
class TopKImagesConfig(ExperimentConfig):
    top_k: int = 10
    subset_n_batches: int = 16
    batch_size: int = 4
    n_images: Union[str, int] = "all"  # can be a number or "all" # TODO: change to be all when None
    aggregation_function: str = "mean" # ("mean" or "max")
    sae_path: str = None
    ids_selection: str = "top_k"
    model_img_size: int = None
    max_dataset_examples: int | None = None
    stream_dataset: bool = False
    num_register_tokens: int | None = None


def get_token_subset(sae_config: dict) -> str:
    return sae_config.get("run", {}).get("token_subset", "all")


def uses_spatial_heatmaps(cfg: ExperimentConfig) -> bool:
    return cfg.token_subset != "registers_only"


def aggregate_register_latents(
    batch_latent_acts: torch.Tensor,
    cfg: TopKImagesConfig,
) -> torch.Tensor:
    num_register_tokens = cfg.num_register_tokens
    if num_register_tokens is None or num_register_tokens <= 0:
        raise ValueError("num_register_tokens must be set for registers_only top-k extraction")
    if batch_latent_acts.shape[0] % num_register_tokens != 0:
        raise ValueError(
            f"Expected register-token activations to be divisible by num_register_tokens={num_register_tokens}, "
            f"got shape {tuple(batch_latent_acts.shape)}"
        )

    num_images = batch_latent_acts.shape[0] // num_register_tokens
    batch_latent_acts = batch_latent_acts.view(num_images, num_register_tokens, -1)

    if cfg.aggregation_function == "mean":
        return batch_latent_acts.mean(dim=1)
    if cfg.aggregation_function == "max":
        return batch_latent_acts.amax(dim=1)
    raise ValueError(f"Unknown aggregation_function: {cfg.aggregation_function}")

def subset_max_activating_vision(generator, model, tokenizer, processor, cfg):
    """
    Collects activations from a subset of the dataset to find max activating examples.
    
    Parameters
    ----------
    generator : generator
        Generator that yields inputs from the dataset.
    model : torch.nn.Module
        The model to extract activations from.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model.
    processor : transformers.ProcessorMixin
        Processor for the model's inputs.
    cfg : ExperimentConfig
        Configuration object with model parameters.
    Returns
    -------
    tuple
        A tuple containing:
        - layer_out (torch.Tensor): Concatenated activations from all batches.
        - images (list): List of all processed images.
    """

    all_layer_outs = []
    images = []
    for _ in range(cfg.subset_n_batches):
        inputs = [next(generator) for _ in range(cfg.batch_size)]
        data_batch = tokenized_batch(inputs, tokenizer, cfg, processor)
        batch_layer_out = hf_forward(
            model,
            data_batch,
            tokenizer,
            cfg=cfg,
            training=cfg.token_subset != "all",
        ).detach()
        all_layer_outs.append(batch_layer_out)
        images_ = [input['image'] for input in inputs]
        images.extend(images_)

    # Concatenate all batches along dimension 0
    layer_out = torch.cat(all_layer_outs, dim=0)
    return layer_out, images

# Create a generator function to yield image IDs and their corresponding latent IDs
def image_to_latents_generator(image_to_latents, batch_size=16):
    """
    Generator function that yields (image_id, latent_ids) pairs from the top_k_info_sae_latents dictionary.
    """
    
    # Process in batches of batch_size
    items = list(image_to_latents.items())
    for i in range(0, len(items), batch_size):
        output_batch = []  # Reset the output_batch for each new batch
        batch = items[i:i+batch_size]
        for img_id, latents in batch:
            output_batch.append((img_id, latents))
        yield output_batch

def get_top_k(values: torch.Tensor, ids: torch.Tensor, k: int):
    """
    Given a tensor of values and corresponding ids, return the top k ids and values.
    Returns:
        top_ids: list of shape (k,)
        top_values: list of shape (k,)
    """

    sorted_indices = torch.argsort(values, descending=True)
    top_ids = ids[sorted_indices[:k]]
    top_values = values[sorted_indices[:k]]
    return top_ids, top_values


def get_first_k_per_partition(values: torch.Tensor, ids: torch.Tensor, k: int, partition: int = 4):
    """
    Given a tensor of values and corresponding ids, return the first k ids and values from each quantile.
    Returns:
        quantile_ids: list of lists, each of shape (k,)
        quantile_values: list of lists, each of shape (k,)
    """
    n = values.shape[0]
    sorted_indices = torch.argsort(values, descending=True)
    values_sorted = values[sorted_indices]
    ids_sorted = ids[sorted_indices]
    quantile_ids = []
    quantile_values = []
    quantile_size = n // partition
    for i in range(partition):
        
        start = i * quantile_size
        # For the last quantile, include all remaining elements
        end = (i + 1) * quantile_size if i < partition - 1 else n

        k_eff = min(k, end - start)

        # draw k **distinct** indices uniformly at random from the slice [start, end)
        rng = np.random.default_rng(seed=seed)  # reproducible? add seed=…
        idx = rng.choice(np.arange(start, end), size=k_eff, replace=False)

        # pick the same random positions from every parallel array
        q_ids   = ids_sorted[idx].tolist()
        q_vals  = values_sorted[idx].tolist()

        quantile_ids.append(q_ids)
        quantile_values.append(q_vals)

    return quantile_ids, quantile_values

def get_top_ids_values_sae_latents(generator, model, tokenizer, processor, latents_to_consider, sae, cfg):
    """
    Get top IDs and values for SAE latents across multiple batches of images.
    
    This function runs batches of images through the model, encodes the activations
    with an SAE, and tracks the top-k activating images for each latent, updating the top-k values and ids for each latent
    after each batch.
    
    Parameters
    ----------
    generator : generator
        Generator that yields batches of (image_id, latent_ids) pairs.
    model : torch.nn.Module
        The vision model to extract activations from.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model's inputs.
    processor : transformers.ProcessorMixin
        Processor for the model's inputs.
    cfg : ExperimentConfig
        Configuration object with model parameters.
    sae : SparseAutoencoder
        The SAE model to encode the activations.
        
    Returns
    -------
    dict
        Dictionary mapping latent IDs to their top-k activating images and values.
        Format: {latent_id: {'top_values': tensor, 'top_ids': tensor}}
    """
    # We compute the total number of batches to process
    total_batches = cfg.n_images // (cfg.batch_size * cfg.subset_n_batches)
    print(f"total_batches: {total_batches}")

    top_k_info_sae_latents = defaultdict(dict)
    # New: accumulate all activations and indices for each latent
    all_latent_activations = {latent_id: [] for latent_id in latents_to_consider}
    all_latent_indices = {latent_id: [] for latent_id in latents_to_consider}
    #all_latent_codes = {latent_id: [] for latent_id in latents_to_consider}

    processed_images = 0
    for batch_idx in tqdm(range(total_batches)):
        # We get a subset of layer activations and images
        subset_layer_out, _ = subset_max_activating_vision(generator, model, tokenizer, processor, cfg)

        if uses_spatial_heatmaps(cfg):
            num_patches = demo_config.LLM_CONFIG[cfg.model_name].model_img_size // demo_config.LLM_CONFIG[cfg.model_name].model_patch_size
            subset_codes = get_patch_sae_codes(sae, subset_layer_out, num_patches=num_patches)

            if cfg.aggregation_function == "mean":
                batch_latent_acts = subset_codes.mean(dim=(1,2))  # [subset_size, n_latents]
            elif cfg.aggregation_function == "max":
                batch_latent_acts = subset_codes.amax(dim=(1,2))  # [subset_size, n_latents]
            else:
                raise ValueError(f"Unknown aggregation_function: {cfg.aggregation_function}")
            del subset_codes
        else:
            batch_latent_acts = sae.encode(subset_layer_out.to(cfg.dtype)).float()
            if cfg.token_subset == "registers_only":
                batch_latent_acts = aggregate_register_latents(batch_latent_acts, cfg)
        
        
        # batch_latent_acts: [subset_size, n_latents]
        for latent_id in latents_to_consider:
            # Only keep activations > 0
            acts = batch_latent_acts[:, latent_id].detach().cpu()
            idxs = torch.arange(processed_images, processed_images + batch_latent_acts.shape[0])
            mask = acts > 0
            if mask.any():
                all_latent_activations[latent_id].append(acts[mask])
                all_latent_indices[latent_id].append(idxs[mask])
        ############################################################
        processed_images += batch_latent_acts.shape[0]

        del subset_layer_out
        del batch_latent_acts
        torch.cuda.empty_cache()

    if cfg.ids_selection == "partition":
        # After collecting all top_k_info_sae_latents, add quantile selection using the full distribution
        for latent_id in latents_to_consider:
            if len(all_latent_activations[latent_id]) > 0:
                all_values = torch.cat(all_latent_activations[latent_id])
                all_ids = torch.cat(all_latent_indices[latent_id])
                quantile_ids, quantile_values = get_first_k_per_partition(all_values, all_ids, cfg.top_k, partition=cfg.partition)
            else:
                quantile_ids = []
                quantile_values = []

            top_k_info_sae_latents[latent_id]['top_ids'] = quantile_ids
            top_k_info_sae_latents[latent_id]['top_values'] = quantile_values
            if uses_spatial_heatmaps(cfg):
                top_k_info_sae_latents[latent_id]['heatmaps'] = []
                for i in range(len(quantile_ids)):
                    top_k_info_sae_latents[latent_id]['heatmaps'].append([None]*len(quantile_ids[i]))
    
    elif cfg.ids_selection == "top_k":
        for latent_id in latents_to_consider:
            if len(all_latent_activations[latent_id]) > 0:
                all_values = torch.cat(all_latent_activations[latent_id])
                all_ids = torch.cat(all_latent_indices[latent_id])
                top_ids, top_values = get_top_k(all_values, all_ids, cfg.top_k)
                
                top_k_info_sae_latents[latent_id]['top_ids'] = top_ids.tolist()
                top_k_info_sae_latents[latent_id]['top_values'] = top_values.tolist()
                if uses_spatial_heatmaps(cfg):
                    top_k_info_sae_latents[latent_id]['heatmaps'] = [None]*len(top_ids)

    else:
        raise ValueError(f"Unknown ids_selection: {cfg.ids_selection}")

    return top_k_info_sae_latents

def save_latent_data(top_k_info_sae_latents_with_heatmaps, latent_id, save_dir):
    # Create a filename for this latent
    latent_filename = os.path.join(save_dir, f"latent_{latent_id}.pt")
    
    # Extract the data for this latent
    latent_data = top_k_info_sae_latents_with_heatmaps[latent_id]
    
    # Save the data to disk
    torch.save(latent_data, latent_filename)

def main(cfg):
    dataset = cfg.dataset
    print(f"dataset: {dataset}")
    device = cfg.device
    aggregation_function = cfg.aggregation_function
    sae_path = cfg.sae_path

    # Get the random number id from the sae path
    rnd_num_id = get_sae_run_id(sae_path)
    print(f"rnd_num_id: {rnd_num_id}")

    # Load SAE
    sae, sae_config = load_dictionary(sae_path, device)
    model_name = sae_config['trainer']['lm_name']
    print(f"sae_config: {sae_config}")
    
    # Load SAE config metadata
    dict_size = sae_config['trainer']['dict_size']
    layer = sae_config['trainer']['layer']
    submodule_name = sae_config['trainer']['submodule_name'] # f"{submodel}_res_{io}_layer_{layer}"
    submodule_name_list = submodule_name.split('_')
    submodel = submodule_name_list[0]
    io = submodule_name_list[2]
    layer = int(submodule_name_list[4])
    activation_dim = get_activation_dim(model_name, submodel)
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    # Set the config parameters
    cfg.model_name = model_name
    cfg.model_img_size = demo_config.LLM_CONFIG[model_name].model_img_size
    cfg.model_path = demo_config.LLM_CONFIG[model_name].model_path
    cfg.model_type = demo_config.LLM_CONFIG[model_name].model_type
    cfg.activation_dim = activation_dim
    cfg.submodel = submodel
    cfg.io = io
    cfg.layer = layer
    cfg.dtype = dtype
    cfg.context_length = None
    cfg.token_subset = get_token_subset(sae_config)
    cfg.num_register_tokens = sae_config.get("run", {}).get("num_register_tokens")
    sae_architecture = sae_config['trainer']['dict_class']
    sae.to(device)
    sae.to(dtype)
    sae.eval()

    # SAE latent_ids to consider (active latents on validation set)
    latents_to_consider = load_active_latents(sae_path, dict_size)
    print(f"Loaded {len(latents_to_consider)} active latents out of {dict_size}")
    
    # Load model, tokenizer and processor
    model, tokenizer, processor = load_model(model_name, cfg, device=device)

    print(f"Loading {dataset} dataset with split {cfg.dataset_split}")
    generator, len_dataset = hf_dataset_to_generator(
        dataset,
        split=cfg.dataset_split,
        max_examples=cfg.max_dataset_examples,
        streaming=cfg.stream_dataset,
    )
    if cfg.n_images == "all":
        print(f'Loading all {len_dataset} images')
        cfg.n_images = len_dataset
    else:
        cfg.n_images = int(cfg.n_images)

    top_k_info_sae_latents = get_top_ids_values_sae_latents(generator, model, tokenizer, processor, latents_to_consider, sae, cfg)

    gc.collect()
    torch.cuda.empty_cache()

    def get_sae_heatmaps_from_latents(top_k_info_sae_latents, sae, cfg):
        if not uses_spatial_heatmaps(cfg):
            return top_k_info_sae_latents

        # We compute the total number of batches to process
        total_batches = cfg.n_images // (cfg.batch_size * cfg.subset_n_batches)
        print(f"total_batches: {total_batches}")


        # Load image dataset generator
        generator, len_dataset = hf_dataset_to_generator(
            dataset,
            split=cfg.dataset_split,
            max_examples=cfg.max_dataset_examples,
            streaming=cfg.stream_dataset,
        )
        if cfg.n_images == "all":
            print(f'Loading all {len_dataset} images')
            cfg.n_images = len_dataset
        else:
            cfg.n_images = int(cfg.n_images)

        start = 0

        for batch_idx in tqdm(range(total_batches)):
            subset_layer_out, subset_images = subset_max_activating_vision(generator, model, tokenizer, processor, cfg)
            images_ids = list(range(start, start + len(subset_images)))
            subset_images_ids = list(range(len(subset_images)))

            start += len(subset_images_ids)

            assert len(images_ids) == len(subset_images_ids)
            
            # We encode the subset of layers activations with the SAE
            num_patches = demo_config.LLM_CONFIG[cfg.model_name].model_img_size // demo_config.LLM_CONFIG[cfg.model_name].model_patch_size
            # if any(x in cfg.model_name.lower() for x in ('qwen2-vl', 'qwen2.5-vl', 'mimo-vl', 'aloe-vision-7b')):
            #     num_patches *= 2
            subset_codes = get_patch_sae_codes(sae, subset_layer_out, num_patches=num_patches)
            # subset_codes: [subset_size, height, width, n_latents]

            for image_id, subset_image_id in zip(images_ids, subset_images_ids):
                for latent_id in top_k_info_sae_latents.keys():
                    if cfg.ids_selection == "partition":
                        top_ids_latent_quantile = top_k_info_sae_latents[latent_id]['top_ids']
                        for i in range(len(top_ids_latent_quantile)):
                            top_ids_in_quantile = top_ids_latent_quantile[i]
                            if image_id in top_ids_in_quantile:
                                idx_in_quantile_sublist = top_ids_in_quantile.index(image_id)
                                top_k_info_sae_latents[latent_id]['heatmaps'][i][idx_in_quantile_sublist] = subset_codes[subset_image_id, :, :, latent_id].detach().cpu().numpy()
                    elif cfg.ids_selection == "top_k":
                        top_ids_latent_top_k = top_k_info_sae_latents[latent_id]['top_ids']
                        if image_id in top_ids_latent_top_k:
                            i = top_ids_latent_top_k.index(image_id)
                            top_k_info_sae_latents[latent_id]['heatmaps'][i] = subset_codes[subset_image_id, :, :, latent_id].detach().cpu().numpy()

            del subset_layer_out
            del subset_codes
            torch.cuda.empty_cache()

        return top_k_info_sae_latents
    
    top_k_info_sae_latents_with_heatmaps = get_sae_heatmaps_from_latents(top_k_info_sae_latents, sae, cfg)

    print('Saving...')
    clean_model_name = model_name.replace("/", "_")
    save_dir_sae_info = f"input_features_explainer/{clean_model_name}_{sae_architecture}/{submodule_name}_{dict_size}_{rnd_num_id}"
    save_dir = os.path.join(DATA_DIR, save_dir_sae_info)
    save_dir = os.path.join(save_dir, 'top_k_images')
    if cfg.ids_selection == "partition":
        save_dir = os.path.join(save_dir, f"{dataset.replace('/', '_')}_{aggregation_function}_{cfg.dataset_split}_{cfg.top_k}_{cfg.n_images}_partition{cfg.partition}")
    elif cfg.ids_selection == "top_k":
        save_dir = os.path.join(save_dir, f"{dataset.replace('/', '_')}_{aggregation_function}_{cfg.dataset_split}_{cfg.top_k}_{cfg.n_images}")
    else:
        raise ValueError(f"Unknown ids_selection: {cfg.ids_selection}")
    print('Saving in:', save_dir)
    # Save the top_k_info_sae_latents_with_heatmaps dictionary to disk
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving latent information to {save_dir}")

    # Save metadata dictionary with configuration information
    metadata = {
        'n_images': cfg.n_images,
        'model_name': model_name,
        'sae_architecture': sae_architecture,
        'submodule_name': submodule_name,
        'token_subset': cfg.token_subset,
        'uses_spatial_heatmaps': uses_spatial_heatmaps(cfg),
    }
    
    # Save metadata to a file
    metadata_filename = os.path.join(save_dir, "metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f)
    print(f"Saved configuration metadata to {metadata_filename}")

    # Save each latent's information as a separate file
    for latent_id in tqdm(list(top_k_info_sae_latents_with_heatmaps.keys()), desc="Saving latent data"):
        # Create a filename for this latent
        latent_filename = os.path.join(save_dir, f"latent_{latent_id}.pt")
        
        # Extract the data for this latent
        latent_data = top_k_info_sae_latents_with_heatmaps[latent_id]

        
        # Save the data to disk
        torch.save(latent_data, latent_filename)


    print('DONE!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sae_path", type=str, default=None)
    parser.add_argument("--aggregation_function", type=str, default="mean")
    parser.add_argument("--n_images", default="all")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--ids_selection", type=str, default="quantile", choices=["partition", "top_k"])
    parser.add_argument("--partition", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k", help="Dataset name")
    parser.add_argument("--max_dataset_examples", type=int, default=None)
    parser.add_argument("--stream_dataset", action="store_true")
    args = parser.parse_args()

    sae_path = args.sae_path
    print(f"sae_path: {sae_path}")

    cfg = TopKImagesConfig(
        sae_path=sae_path,
        top_k=args.top_k,
        aggregation_function=args.aggregation_function,
        n_images=args.n_images,
        dataset_split=args.dataset_split,
        partition=args.partition,
        ids_selection=args.ids_selection,
        dataset=args.dataset,
        max_dataset_examples=args.max_dataset_examples,
        stream_dataset=args.stream_dataset,
    )

    main(cfg)

"""
Usage:

python -m src.get_max_activating_vision \
        --top_k 10 \
        --ids_selection top_k \
        --sae_path /workspace/data/saes/facebook_dinov2-small/enc_res_out_layer_8_top_k_128_6_0.05_13216850/trainer_0 \
        --n_images 1000

"""
