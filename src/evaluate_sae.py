# %%
import sys
import os
from dotenv import load_dotenv
load_dotenv()
REPO_DIR = os.environ.get('REPO_DIR')
if REPO_DIR is None:
    print("Error: REPO_DIR environment variable is not set.")
    sys.exit(1)
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

import torch
from functools import partial
import argparse
import itertools
from collections import defaultdict
import os
import random
from typing import Union
import json
import torch.multiprocessing as mp
import time
from utils.utils import load_model
import yaml

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
# from dictionary_learning.evaluation import loss_recovered_evaluation
import dictionary_learning
 
from dataclasses import dataclass

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Resolve dataset configuration from YAML when column names are not provided
dataset_config_path = os.path.join(REPO_DIR, "config", "dataset_config.yaml")
if not os.path.isfile(dataset_config_path):
    raise FileNotFoundError(f"Dataset config YAML not found at {dataset_config_path}")

with open(dataset_config_path, "r") as f:
    dataset_cfg = yaml.safe_load(f)

@dataclass
class TestConfig:
    model_name: str = None
    model_path: str = None
    layer: int = None
    dataset: str = None
    device: str = 'cuda:0'
    model_type: str = None
    submodel: str = None
    site: str = None
    io: str = None
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True,
    activation_dim: int = None
    get_full_model: bool = False
    context_length: int = None
    model_img_size: int = None


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument("--sae_path", type=str, help="path to SAE")
    parser.add_argument("--dataset", type=str, help="dataset directory")
    parser.add_argument("--save_results", action="store_true", default=False, help="whether to save evaluation results")
    parser.add_argument("--n_inputs", type=str, default=str(demo_config.eval_num_inputs), 
                        help="number of inputs to evaluate on, can be an integer or 'all'")
    args = parser.parse_args()
    return args

@torch.no_grad()
def eval_saes(
    dataset: str,
    ae_paths: list[str],
    n_inputs: Union[int, str],
    overwrite_prev_results: bool = False,
    save_results: bool = False,
    device: str = 'cuda:0',
) -> dict:
    
    random.seed(demo_config.random_seeds[0])
    torch.manual_seed(demo_config.random_seeds[0])

    for ae_path in ae_paths:
        eval_results = {}
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue
        
        dictionary, config = dictionary_learning.utils.load_dictionary(ae_path, device)
        
        
        # SAE config vars
        model_name = config["trainer"]["lm_name"]
        model_type = demo_config.LLM_CONFIG[model_name].model_type
        model_path = demo_config.LLM_CONFIG[model_name].model_path
        layer = config["trainer"]["layer"]
        submodule_name = config["trainer"]["submodule_name"]
        activation_dim = config["trainer"]["activation_dim"]
        submodel = submodule_name.split('_')[0].replace('-', '_')
        site = submodule_name.split('_')[1]
        io = submodule_name.split('_')[2]

        # These vars depend on the model so we override the config
        dtype = demo_config.LLM_CONFIG[model_name].dtype
        dictionary = dictionary.to(dtype=dtype)
        context_length = demo_config.LLM_CONFIG[model_name].context_length
        llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
        model_img_size = demo_config.LLM_CONFIG[model_name].model_img_size

        sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
        buffer_tokens = demo_config.buffer_tokens if 'gemma-3' not in model_name else 1_500_000

        cfg = TestConfig(
            model_name=model_name,
            model_path=model_path,
            layer=layer,
            dataset=dataset,
            device=device,
            model_type=model_type,
            submodel=submodel,
            site=site,
            io=io,
            dtype=dtype,
            activation_dim=activation_dim,
            context_length=context_length,
            model_img_size=model_img_size
        )

        # Load model, tokenizer and submodule
        model, tokenizer, processor = load_model(model_name, cfg, dtype, device)

        split = dataset_cfg[dataset]['eval_split']
        print(f"Loading {dataset} dataset with split {split}")
        generator, len_dataset = hf_dataset_to_generator(dataset, split=split)
        num_tokens = len_dataset*context_length
        
        if n_inputs == "all":
            n_inputs = num_tokens // context_length
        else:
            n_inputs = int(n_inputs)
            num_tokens = n_inputs * context_length

        # n_inputs: number of inputs (data points)
        n_ctxs = buffer_tokens // context_length # number of contexts in the buffer
        print(f"N_CTXs: {n_ctxs}")
        print(f"N_INPUTS: {n_inputs}")
        assert n_inputs >= n_ctxs

        activation_buffer = ActivationBuffer(
        generator,
        model,
        n_ctxs=n_ctxs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        tokenizer=tokenizer,
        processor=processor,
        max_activation_norm_multiple=demo_config.max_activation_norm_multiple,
        training=False,
        cfg=cfg
        )

        # number of total batches to evaluate on
        n_batches = num_tokens // sae_batch_size
        print(f"N_BATCHES: {n_batches}")

        print('Computing Eval metrics...')
        eval_results, feature_activation_frequency = evaluate(
            dictionary,
            activation_buffer,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
            "model_img_size": model_img_size
        }
        eval_results["hyperparameters"] = hyperparameters
        
        print(eval_results)

        if save_results:
            with open(output_filename, "w") as f:
                json.dump(eval_results, f)
            
            with open(f"{ae_path}/latent_activation_frequency.json", "w") as f:
                json.dump(feature_activation_frequency, f)

if __name__ == "__main__":

    args = get_args()

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start_time = time.time()

    sae_path = args.sae_path

    ae_paths = dictionary_learning.utils.get_nested_folders(sae_path)
    print('ae_paths', ae_paths)
    assert args.dataset is not None, "Dataset is required"

    eval_saes(
        args.dataset,
        ae_paths,
        args.n_inputs,
        overwrite_prev_results=True,
        save_results=args.save_results,
        device=args.device,
    )

    print(f"Total time: {time.time() - start_time}")


#python -m src.evaluate_sae --sae_path /gpfs/projects/bsc70/heka/trained_saes/vision_runs/google_gemma-3-4b-it/enc_res_out_layer_16_top_k_4096_50_1.0_HF_22177746/trainer_0 --dataset ILSVRC/imagenet-1k --n_inputs 2000
# python -m src.evaluate_sae --sae_path /home/bsc/bsc804923/data/vision_runs/Qwen_Qwen2.5-Omni-7B/audio_enc_res_out_layer_16_top_k_4096_50_0.5_HF_30056646/trainer_0 --dataset agkphysics/AudioSet--n_inputs 2000

# python -m src.evaluate_sae --sae_path /gpfs/projects/bsc70/hpai/storage/data/heka/sae/data/saes/google_gemma-3-270m/dec_res_out_layer_14_top_k_8192_25_0.1_25755208/trainer_0 --dataset minipile --n_inputs 2000 --save_results
# python -m src.evaluate_sae --dataset ILSVRC/imagenet-1k --sae_path /gpfs/projects/bsc70/heka/trained_saes_tmp/Aloe-Vision-7B/enc_res_out_layer_18_top_k_8192_25_1.0_12409157/trainer_0 --n_inputs all --save_results
