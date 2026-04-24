import sys
import os
sys.path.append(os.environ.get('REPO_DIR'))

from dataclasses import dataclass, asdict
from typing import Optional, Type, Any, Union, Dict
from enum import Enum
import torch as t
import itertools
import yaml

from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK


def get_activation_dim(model_name, submodel):
    config_dim = LLM_CONFIG[model_name].activation_dim
    return config_dim[submodel] if isinstance(config_dim, dict) else config_dim


def get_context_length(model_name, submodel):
    context_length = LLM_CONFIG[model_name].context_length
    if isinstance(context_length, dict):
        return context_length['image'] if submodel == 'enc' else context_length['text']
    return context_length


class TrainerType(Enum):
    TOP_K = "top_k"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: Union[int, Dict]
    sae_batch_size: int
    dtype: t.dtype
    activation_dim: Union[int, Dict]
    model_path: str
    model_type: str
    model_patch_size: Optional[int] = None
    model_img_size: Optional[int] = None
    vision_model: Optional[str] = None
    vision_model_layer: Optional[str] = None
    tokens_to_remove: Optional[list[int]] = None


buffer_tokens = 500_000
eval_num_inputs = 2000
random_seeds = [0]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8

dictionary_widths = [128]
learning_rates = [3e-4]
TARGET_L0s = [6]
max_activation_norm_multiple = 1e6

wandb_project = "sae_training_runs"


def _load_llm_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    dtype_map = {
        'bfloat16': t.bfloat16,
        'bf16': t.bfloat16,
        'float32': t.float32,
        'fp32': t.float32,
        'float16': t.float16,
        'fp16': t.float16,
        'float64': t.float64,
        'fp64': t.float64,
    }
    llm_config = {}
    for model_name, cfg in data.items():
        cfg = dict(cfg)
        dtype_str = cfg.get('dtype', 'bfloat16')
        if isinstance(dtype_str, str):
            if dtype_str not in dtype_map:
                raise ValueError(f"Unsupported dtype '{dtype_str}' in YAML for {model_name}")
            cfg['dtype'] = dtype_map[dtype_str]
        llm_config[model_name] = LLMConfig(**cfg)
    return llm_config


LLM_CONFIG = _load_llm_config_from_yaml(f'{os.environ.get("REPO_DIR")}/config/llm_config.yaml')


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    dict_sizes: list[int],
    layer: str,
    submodule_name: str,
    steps: int,
    num_tokens: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    cfg=None,
) -> list[dict]:
    if architectures != [TrainerType.TOP_K.value] and any(
        arch != TrainerType.TOP_K.value for arch in architectures
    ):
        raise ValueError(
            f"Only {TrainerType.TOP_K.value} is supported in this paper reproducibility subset. "
            f"Got architectures={architectures}"
        )

    decay_start = int(steps * decay_start_fraction)
    model_name = cfg.model_name
    dataset = cfg.dataset
    device = cfg.device
    activation_dim = cfg.activation_dim

    base_config = {
        "lm_name": model_name,
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "submodule_name": submodule_name,
    }

    trainer_configs = []
    for seed, dict_size, learning_rate, k in itertools.product(
        seeds, dict_sizes, learning_rates, TARGET_L0s
    ):
        config = TopKTrainerConfig(
            **base_config,
            trainer=TopKTrainer,
            dict_class=AutoEncoderTopK,
            lr=learning_rate,
            dict_size=dict_size,
            seed=seed,
            k=k,
            wandb_name=f"TopKTrainer-{model_name}-{submodule_name}-{str(num_tokens/1e6)}M-{dataset}",
        )
        trainer_configs.append(asdict(config))

    return trainer_configs
