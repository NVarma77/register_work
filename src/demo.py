# %%
import sys
import os
from dotenv import load_dotenv
load_dotenv()
REPO_DIR = os.environ.get('REPO_DIR')
# Check if REPO_DIR environment variable is set
if REPO_DIR is None:
    print("Error: REPO_DIR environment variable is not set.")
    sys.exit(1)
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

import torch
from functools import partial
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
import os
import random
import torch.multiprocessing as mp
import time
import uuid
import json
import math
import numpy as np
from typing import Optional

from utils.utils import load_model

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer, hf_forward
from dictionary_learning.training import trainSAE
import dictionary_learning

from evaluate_sae import eval_saes
from demo_config import get_activation_dim, get_context_length
from dataclasses import dataclass, field
from src.processing import tokenized_batch

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import logging
logging.getLogger().setLevel(logging.ERROR)

LOG_STEPS = 100  # Log the training on wandb or print to console every log_steps


@dataclass
class TrainingConfig:
    model_name: str = 'google/paligemma2-3b-mix-224'
    layer: int = None
    dataset: str = "ILSVRC/imagenet-1k"
    device: str = 'cuda:0'
    model_type: str = None
    model_path: str = None
    submodel: str = 'enc'
    io: str = 'out'
    activation_dim: int = None
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True
    ratio_of_training_data: float = 0.1
    get_full_model: bool = False
    context_length: int = None
    model_img_size: int = None
    max_dataset_examples: Optional[int] = None
    stream_dataset: bool = False
    sae_batch_size_override: Optional[int] = None

    # New fields. Defaults preserve original behavior.
    token_subset: str = "all"  # all | outlier_patches | registers_only
    outlier_percent: float = 0.0237
    outlier_threshold: float = None
    outlier_stats: dict = field(default_factory=dict)
    num_register_tokens: int = 4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ.get('DATA_DIR'), 'saes'), help="where to store SAEs checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints at different stages of training")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument(
        "--dataset",
        type=str,
        default="txt360",
        help="dataset to use for training"
    )
    parser.add_argument("--test_set", type=str, default=None, help="test set")
    parser.add_argument("--submodel", type=str, default='enc', choices=['enc', 'dec'], help="submodel to use (dec, enc)")
    parser.add_argument("--io", type=str, default='out', choices=['in', 'out'], help="io to use (in, out)")
    parser.add_argument("--ratio_of_training_data", type=float, default=1, help="ratio of training data to use (default: 1)")
    parser.add_argument("--dictionary_width", type=int, default=2048, help="SAE dictionary width")
    parser.add_argument("--max_dataset_examples", type=int, default=None, help="maximum number of dataset examples to use")
    parser.add_argument("--stream_dataset", action="store_true", help="stream dataset examples from HF instead of resolving the full split first")
    parser.add_argument("--skip_eval", action="store_true", help="skip the SAE evaluation pass after training")
    parser.add_argument("--sae_batch_size", type=int, default=None, help="override SAE batch size for small smoke-test runs")

    # New parameters only.
    parser.add_argument(
        "--token_subset",
        type=str,
        default="all",
        choices=["all", "outlier_patches", "registers_only"],
        help="token selection mode. default=all preserves original behavior"
    )
    parser.add_argument(
        "--outlier_percent",
        type=float,
        default=0.0237,
        help="top fraction of patch-token norms to keep when token_subset=outlier_patches"
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=None,
        help="manual patch-token norm threshold to use when token_subset=outlier_patches; skips threshold prepass"
    )
    parser.add_argument(
        "--num_register_tokens",
        type=int,
        default=4,
        help="number of register tokens for with-registers models"
    )

    args = parser.parse_args()
    return args


def _next_input_batch(generator, batch_size):
    batch = []
    for _ in range(batch_size):
        try:
            batch.append(next(generator))
        except StopIteration:
            break
    return batch


def _compute_outlier_threshold(cfg: TrainingConfig, save_dir: str) -> dict:
    """
    Compute the global patch-token norm threshold corresponding to the top
    cfg.outlier_percent fraction of patch tokens.

    This is a prepass only for token_subset=outlier_patches.
    It does not change the original training step schedule.
    """
    if cfg.token_subset != "outlier_patches":
        return {}

    if cfg.model_type != 'vision':
        raise ValueError("token_subset=outlier_patches is only supported for vision models")

    if cfg.submodel != 'enc':
        raise ValueError("token_subset=outlier_patches currently requires submodel='enc'")

    patch_size = demo_config.LLM_CONFIG[cfg.model_name].model_patch_size
    img_size = demo_config.LLM_CONFIG[cfg.model_name].model_img_size
    if patch_size is None or img_size is None:
        raise ValueError("model_patch_size/model_img_size must be available for outlier_patches")

    patch_tokens_per_image = (img_size // patch_size) ** 2

    random.seed(demo_config.random_seeds[0])
    torch.manual_seed(demo_config.random_seeds[0])

    llm_batch_size = demo_config.LLM_CONFIG[cfg.model_name].llm_batch_size
    model, tokenizer, processor = load_model(cfg.model_name, cfg, cfg.dtype, cfg.device)
    generator, len_dataset = hf_dataset_to_generator(
        cfg.dataset,
        ratio_of_training_data=cfg.ratio_of_training_data,
        max_examples=cfg.max_dataset_examples,
        streaming=cfg.stream_dataset,
    )

    total_token_estimate = len_dataset * patch_tokens_per_image
    temp_dir = os.path.join(save_dir, "_tmp_outlier_thresholds")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(
        temp_dir,
        f"{cfg.model_name.replace('/', '_')}_layer{cfg.layer}_p{cfg.outlier_percent:.4f}.mmap"
    )

    norms_mm = np.memmap(temp_path, mode='w+', dtype=np.float32, shape=(total_token_estimate,))
    write_idx = 0

    with tqdm(total=len_dataset, desc=f"Outlier threshold pass layer {cfg.layer}") as pbar:
        while True:
            input_batch = _next_input_batch(generator, llm_batch_size)
            if not input_batch:
                break

            with torch.no_grad():
                data_batch = tokenized_batch(input_batch, tokenizer, cfg, processor)
                # cfg.token_subset=outlier_patches here, but cfg.outlier_threshold is None,
                # so hf_forward returns patch tokens only, without threshold filtering.
                hidden_states = hf_forward(
                    model,
                    data_batch,
                    tokenizer,
                    cfg,
                    remove_high_norm=None,
                    training=True
                )

            if hidden_states.numel() == 0:
                pbar.update(len(input_batch))
                continue

            norms = hidden_states.norm(dim=-1).detach().cpu().to(torch.float32).numpy()
            end_idx = write_idx + len(norms)
            if end_idx > len(norms_mm):
                raise RuntimeError(
                    f"Outlier norm storage overflowed. Needed {end_idx}, allocated {len(norms_mm)}"
                )
            norms_mm[write_idx:end_idx] = norms
            write_idx = end_idx
            pbar.update(len(input_batch))

    if write_idx == 0:
        raise RuntimeError("No patch-token activations collected for outlier threshold")

    threshold = float(np.quantile(norms_mm[:write_idx], 1.0 - cfg.outlier_percent))
    selected_count = int(np.sum(norms_mm[:write_idx] >= threshold))

    del norms_mm
    if os.path.exists(temp_path):
        os.remove(temp_path)

    del model, tokenizer, processor
    torch.cuda.empty_cache()

    stats = {
        "layer": int(cfg.layer),
        "model_name": cfg.model_name,
        "dataset": cfg.dataset,
        "ratio_of_training_data": float(cfg.ratio_of_training_data),
        "outlier_percent": float(cfg.outlier_percent),
        "patch_tokens_per_image": int(patch_tokens_per_image),
        "total_patch_tokens_seen": int(write_idx),
        "selected_patch_tokens": int(selected_count),
        "outlier_threshold": float(threshold),
    }

    print(f"Computed outlier threshold: {threshold:.6f}")
    print(f"Selected patch tokens: {selected_count}/{write_idx}")

    return stats


def _token_subset_save_suffix(cfg: TrainingConfig) -> str:
    if cfg.token_subset == "all":
        return ""
    if cfg.token_subset == "registers_only":
        return "_registers_only"
    if cfg.token_subset == "outlier_patches":
        if cfg.outlier_threshold is not None:
            return f"_outlier_patches_t{cfg.outlier_threshold:g}"
        return f"_outlier_patches_p{cfg.outlier_percent}"
    raise ValueError(f"Unknown token_subset: {cfg.token_subset}")


def _build_run_cfg(cfg: TrainingConfig) -> dict:
    run_cfg = {
        "token_subset": cfg.token_subset,
        "num_register_tokens": int(cfg.num_register_tokens),
    }
    if cfg.token_subset == "outlier_patches":
        run_cfg["outlier_percent"] = float(cfg.outlier_percent)
        run_cfg["outlier_threshold"] = (
            float(cfg.outlier_threshold) if cfg.outlier_threshold is not None else None
        )
    return run_cfg


def _persist_run_metadata(save_dir: str, cfg: TrainingConfig) -> None:
    run_cfg = _build_run_cfg(cfg)
    for trainer_dir in dictionary_learning.utils.get_nested_folders(save_dir):
        config_path = os.path.join(trainer_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        saved_config["run"] = run_cfg
        with open(config_path, "w") as f:
            json.dump(saved_config, f, indent=4)


def _get_short_run_schedule(steps: int) -> tuple[int, int]:
    warmup_steps = min(demo_config.WARMUP_STEPS, max(1, steps // 10))
    sparsity_warmup_steps = min(demo_config.SPARSITY_WARMUP_STEPS, max(1, steps // 2))
    return warmup_steps, sparsity_warmup_steps


def _get_effective_token_count_per_example(cfg: TrainingConfig) -> int:
    if cfg.model_type == 'vision' and cfg.token_subset == "registers_only":
        return max(1, int(cfg.num_register_tokens))
    return cfg.context_length


def _should_normalize_activations(cfg: TrainingConfig, steps: int) -> bool:
    if cfg.max_dataset_examples is not None and cfg.max_dataset_examples <= 5000:
        return False
    if steps < 100:
        return False
    return True


def run_sae_training(
    layer: int,
    save_dir: str,
    architectures: list,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 1_000_000,
    cfg: TrainingConfig = None
):
    random.seed(demo_config.random_seeds[0])
    torch.manual_seed(demo_config.random_seeds[0])

    model_name = cfg.model_name
    dataset = cfg.dataset
    io = cfg.io
    submodel = cfg.submodel
    device = cfg.device
    dtype = cfg.dtype

    if cfg.model_type == 'lm':
        assert submodel == 'dec', "submodel must be dec for language models"

    if cfg.token_subset == "registers_only" and "with-registers" not in cfg.model_name.lower():
        raise ValueError("token_subset=registers_only requires a with-registers model")

    submodule_name = f"{submodel.replace('_', '-')}_res_{io}_layer_{layer}"

    # model and data parameters
    sae_batch_size = cfg.sae_batch_size_override or demo_config.LLM_CONFIG[model_name].sae_batch_size

    print('save_dir', save_dir)
    print('token_subset', cfg.token_subset)

    context_length = _get_effective_token_count_per_example(cfg)
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size

    # Load model, tokenizer and submodule
    model, tokenizer, processor = load_model(model_name, cfg, dtype, device)

    # Get generator from HF dataset (columns resolved from YAML)
    generator, len_dataset = hf_dataset_to_generator(
        dataset,
        ratio_of_training_data=cfg.ratio_of_training_data,
        max_examples=cfg.max_dataset_examples,
        streaming=cfg.stream_dataset,
    )
    n_ctxs = min(max(1, buffer_tokens // context_length), len_dataset)
    print(f"n_ctxs: {n_ctxs}, buffer_size_in_tokens: {buffer_tokens}")

    num_tokens = len_dataset * context_length
    steps = max(1, num_tokens // sae_batch_size)

    print(f"LEN DATASET: {len_dataset}")
    print(f"NUM TOKENS: {num_tokens}")

    warmup_steps, sparsity_warmup_steps = _get_short_run_schedule(steps)
    print(f"Adjusted warmup_steps: {warmup_steps}")
    print(f"Adjusted sparsity_warmup_steps: {sparsity_warmup_steps}")
    normalize_activations = _should_normalize_activations(cfg, steps)
    print(f"normalize_activations: {normalize_activations}")

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
        training=True,
        cfg=cfg
    )

    if save_checkpoints:
        desired_checkpoints = [0.1, 0.25, 0.5, 0.75, 1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        dictionary_widths,
        layer,
        submodule_name,
        steps,
        num_tokens,
        warmup_steps=warmup_steps,
        sparsity_warmup_steps=sparsity_warmup_steps,
        cfg=cfg
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) == 1, "Only one trainer config is supported"
    trainer_config = trainer_configs[0]
    sae_architecture = architectures[0]
    rnd_num_id = str(int(uuid.uuid4()))[:8]
    dict_size = trainer_config['dict_size']
    layer = trainer_config['layer']
    if 'k' in trainer_config:
        k = trainer_config['k']
    else:
        k = trainer_config['target_l0']
    submodule_name = trainer_config['submodule_name']

    suffix = model_name.replace("/", "_")
    save_dir = f"{save_dir}/{suffix}"
    save_dir = f"{save_dir}/{submodule_name}_{sae_architecture}_{dict_size}_{k}_{cfg.ratio_of_training_data}_{rnd_num_id}"
    save_dir = save_dir + _token_subset_save_suffix(cfg)

    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        use_wandb=use_wandb,
        steps=steps,
        save_steps=save_steps,
        save_dir=save_dir,
        log_steps=LOG_STEPS,
        wandb_project=demo_config.wandb_project,
        normalize_activations=normalize_activations,
        verbose=False,
        autocast_dtype=dtype,
    )

    _persist_run_metadata(save_dir, cfg)

    if cfg.token_subset == "outlier_patches" and cfg.outlier_stats:
        stats_path = os.path.join(save_dir, "outlier_threshold_stats.json")
        with open(stats_path, "w") as f:
            json.dump(cfg.outlier_stats, f, indent=2)
        print(f"Saved outlier stats to {stats_path}")

    del model, tokenizer, processor
    return save_dir


if __name__ == "__main__":
    args = get_args()

    # Load config and override with args
    cfg = TrainingConfig(
        model_name=args.model_name,
        model_path=demo_config.LLM_CONFIG[args.model_name].model_path,
        dataset=args.dataset,
        device=args.device,
        submodel=args.submodel,
        io=args.io,
        model_type=demo_config.LLM_CONFIG[args.model_name].model_type,
        activation_dim=get_activation_dim(args.model_name, args.submodel),
        dtype=demo_config.LLM_CONFIG[args.model_name].dtype,
        ratio_of_training_data=args.ratio_of_training_data,
        context_length=get_context_length(args.model_name, args.submodel),
        model_img_size=demo_config.LLM_CONFIG[args.model_name].model_img_size,
        max_dataset_examples=args.max_dataset_examples,
        stream_dataset=args.stream_dataset,
        sae_batch_size_override=args.sae_batch_size,
        token_subset=args.token_subset,
        outlier_percent=args.outlier_percent,
        outlier_threshold=args.outlier_threshold,
        num_register_tokens=args.num_register_tokens,
    )

    if cfg.token_subset == "registers_only" and "with-registers" not in cfg.model_name.lower():
        raise ValueError("token_subset=registers_only requires a DINOv2 with-registers model")
    if cfg.token_subset == "outlier_patches" and "with-registers" in cfg.model_name.lower():
        raise ValueError("token_subset=outlier_patches is intended for base DINOv2 patch tokens, not with-registers models")

    os.environ["WANDB_DIR"] = args.save_dir
    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    start_time = time.time()

    # Gemma-3 models have a lot of context length so we need a bigger buffer not to waste datapoints
    buffer_tokens = demo_config.buffer_tokens if 'gemma-3' not in args.model_name.lower() else 1_500_000

    for layer in args.layers:
        cfg.layer = layer

        # New prepass only for outlier mode.
        if cfg.token_subset == "outlier_patches":
            if cfg.outlier_threshold is None:
                cfg.outlier_stats = _compute_outlier_threshold(cfg, args.save_dir)
                cfg.outlier_threshold = cfg.outlier_stats["outlier_threshold"]
            else:
                cfg.outlier_stats = {
                    "outlier_threshold": float(cfg.outlier_threshold),
                    "outlier_percent": float(cfg.outlier_percent),
                    "ratio_of_training_data": float(cfg.ratio_of_training_data),
                    "manual_threshold": True,
                }
                print(f"Using manual outlier threshold: {cfg.outlier_threshold:.6f}")
        else:
            cfg.outlier_stats = {}
            cfg.outlier_threshold = None

        complete_save_dir = run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            architectures=args.architectures,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=[args.dictionary_width],
            learning_rates=demo_config.learning_rates,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            buffer_tokens=buffer_tokens,
            cfg=cfg
        )

    if not args.skip_eval:
        ae_paths = dictionary_learning.utils.get_nested_folders(complete_save_dir)
        print(f"ae_paths: {ae_paths}")

        if args.test_set is None:
            print("No test set provided, evaluating on default eval set")

        eval_saes(
            demo_config.eval_dataset if args.test_set is None else args.test_set,
            ae_paths,
            n_inputs=2000,
            overwrite_prev_results=True,
            save_results=True,
            device=cfg.device,
        )

    print(f"Total time: {time.time() - start_time}")

# Usage examples:
# Original behavior unchanged:
# python src/demo.py --model_name facebook/dinov2-small --layers 8 --architectures top_k --dataset ILSVRC/imagenet-1k --ratio_of_training_data 1
#
# Outlier patches:
# python src/demo.py --model_name facebook/dinov2-small --layers 8 --architectures top_k --dataset ILSVRC/imagenet-1k --ratio_of_training_data 0.01 --token_subset outlier_patches --outlier_percent 0.0237
#
# Registers only:
# python src/demo.py --model_name facebook/dinov2-with-registers-small --layers 8 --architectures top_k --dataset ILSVRC/imagenet-1k --ratio_of_training_data 0.01 --token_subset registers_only --num_register_tokens 4
