import torch as t
# HF models
from transformers import AutoModelForCausalLM, ViTForImageClassification
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import gc
from tqdm import tqdm
from functools import partial
from .config import DEBUG
import copy
import time
from PIL import Image
from src.processing import tokenized_batch
from .hook_managers import HookManager


def remove_positions(tokenizer, input_ids, hidden_states, tokens_to_remove=None, remove_bos=True):
    """Remove specified tokens and padding tokens from the activations."""

    # Always remove padding tokens
    mask_to_apply = input_ids != tokenizer.pad_token_id

    if remove_bos:
        # Remove BOS token if specified
        bos_token_mask = input_ids != tokenizer.bos_token_id
        mask_to_apply = bos_token_mask & mask_to_apply

    # Remove additional tokens if specified
    if tokens_to_remove is not None:
        for token_id in tokens_to_remove:
            token_mask = input_ids != token_id
            mask_to_apply = token_mask & mask_to_apply

    return hidden_states[mask_to_apply]


def _get_token_subset(cfg):
    return getattr(cfg, "token_subset", "all")


def _get_num_register_tokens(cfg):
    return int(getattr(cfg, "num_register_tokens", 4))


def _get_outlier_threshold(cfg):
    return getattr(cfg, "outlier_threshold", None)


def _select_vision_tokens(hidden_states, cfg, training=True):
    """
    Preserve the original behavior for token_subset='all'.

    Added modes:
      - registers_only: keep only the register tokens
      - outlier_patches: keep only patch tokens (CLS removed, registers removed if present)

    This follows the repo's existing convention from the original file, where
    with-registers inference removed the last register tokens via [:,1:-4,:].
    """
    if hidden_states.ndim != 3:
        return hidden_states

    has_registers = 'with-registers' in cfg.model_name.lower()
    token_subset = _get_token_subset(cfg)
    num_register_tokens = _get_num_register_tokens(cfg)

    # Preserve original inference behavior exactly.
    if not training:
        if has_registers:
            return hidden_states[:, 1:-num_register_tokens, :]
        return hidden_states[:, 1:, :]

    # Preserve original training behavior exactly for token_subset=all.
    if token_subset == "all":
        return hidden_states[:, 1:, :]

    if token_subset == "registers_only":
        if not has_registers:
            raise ValueError("token_subset='registers_only' requires a with-registers model")
        return hidden_states[:, -num_register_tokens:, :]

    if token_subset == "outlier_patches":
        if has_registers:
            return hidden_states[:, 1:-num_register_tokens, :]
        return hidden_states[:, 1:, :]

    raise ValueError(f"Unknown token_subset: {token_subset}")


def hf_forward(
    model,
    data_batch,
    tokenizer,
    cfg,
    remove_high_norm=None,
    training=True,
    preserve_vision_structure=False,
):
    """Forward pass using HuggingFace transformers to extract activations from the model."""
    if hasattr(model, 'submodule'):
        use_hooks = True
    else:
        use_hooks = False

    model_kwargs = {}
    if use_hooks:
        # Submodule is available, we use it with hooks.
        hook_manager = HookManager()
        hook_manager.attach_and_verify_hook(model.submodule, io=cfg.io)

    else:
        model_kwargs["output_hidden_states"] = True

    if any(x in cfg.model_name.lower() for x in ('qwen2-vl', 'qwen2.5-vl', 'mimo-vl', 'aloe-vision-7b')) and cfg.submodel == 'enc':
        with t.no_grad():
            output = model(data_batch['pixel_values'], grid_thw=data_batch['image_grid_thw'], **model_kwargs)
    elif 'internvl' in cfg.model_name.lower():
        with t.no_grad():
            output = model(data_batch['pixel_values'], **model_kwargs)
    else:
        with t.no_grad():
            output = model(**data_batch, **model_kwargs)

    if not use_hooks:
        if cfg.io == 'out':
            # First hidden state (idx 0) is the output of the embedding layer
            layer = cfg.layer + 1
        else:
            layer = cfg.layer

        hidden_states = output['hidden_states'][layer]

    else:
        hidden_states = t.cat(hook_manager.hooks_saved, dim=0)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Clear previous hook data
        hook_manager.clear_saved_data()
        hook_manager.remove_hooks()

    # When working with vision-language models, remove CLS by indexing
    if cfg.model_type == 'vlm' and cfg.submodel == 'enc' and 'internvl' in cfg.model_name.lower():
        hidden_states = hidden_states[:, 1:, :]

    if cfg.model_type == 'vision':
        hidden_states = _select_vision_tokens(hidden_states, cfg, training=training)

    if training:
        if cfg.model_type == 'vision':
            if hidden_states.ndim == 3:
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

            # New outlier filtering. Only active when token_subset=outlier_patches
            # and an outlier threshold has been computed.
            if _get_token_subset(cfg) == "outlier_patches":
                outlier_threshold = getattr(cfg, "outlier_threshold", None)
                if outlier_threshold is not None:
                    norms = hidden_states.norm(dim=-1)
                    hidden_states = hidden_states[norms >= outlier_threshold]

        elif (cfg.model_type == 'vlm' and cfg.submodel == 'dec') or cfg.model_type == 'lm':
            input_ids = data_batch['input_ids']
            hidden_states = remove_positions(tokenizer, input_ids, hidden_states, cfg.tokens_to_remove, cfg.remove_bos)

    else:
        if cfg.model_type == 'vision' and preserve_vision_structure:
            return hidden_states

        # Reshape to [batch_size*sequence_length, hidden_size]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if cfg.model_type == 'vision' and _get_token_subset(cfg) == "outlier_patches":
            outlier_threshold = _get_outlier_threshold(cfg)
            if outlier_threshold is not None:
                norms = hidden_states.norm(dim=-1)
                hidden_states = hidden_states[norms >= outlier_threshold]

    if training and remove_high_norm is not None:
        # Some models (like Qwen) have random high norm activation sinks
        norms_BL = hidden_states.norm(dim=-1)
        median_norm = norms_BL.median()
        norm_mask = norms_BL > median_norm * remove_high_norm
        if norm_mask.sum() > 0:
            print(f"Removed {norm_mask.sum()} high norm activations")
            print(f"Median norm: {median_norm}, remove_high_norm: {median_norm * remove_high_norm}")
        hidden_states = hidden_states[~norm_mask]

    return hidden_states


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self,
                 data,  # generator which yields text data
                 model,  # LanguageModel from which to extract activations
                 n_ctxs=3e4,  # approximate number of contexts to store in the buffer
                 ctx_len=128,  # length of each context
                 refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192,  # size of batches in which to yield activations
                 tokenizer=None,
                 processor=None,
                 max_activation_norm_multiple=None,
                 training=True,
                 cfg=None
                 ):

        # data vars
        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len

        # tokenizer / processor
        self.tokenizer = tokenizer
        self.processor = processor

        # cfg vars
        self.cfg = cfg
        self.d_submodule = cfg.activation_dim
        self.io = cfg.io
        self.remove_bos = cfg.remove_bos
        self.dtype = cfg.dtype
        self.model_type = cfg.model_type
        self.tokens_to_remove = cfg.tokens_to_remove
        self.submodel = cfg.submodel
        self.device = cfg.device
        self.remove_high_norm = max_activation_norm_multiple

        # buffer vars
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.training = training

        if self.io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        self.activations = t.empty(0, self.d_submodule, device=self.device, dtype=self.dtype)
        self.read = t.zeros(0).bool()

    def __iter__(self):
        return self

    def __next__(self, deterministic=False):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            if deterministic == False:
                idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            else:
                idxs = unreads[:self.out_batch_size]
            self.read[idxs] = True
            return self.activations[idxs]

    def input_batch(self, batch_size=None):

        if batch_size is None:
            batch_size = self.refresh_batch_size
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(self.data))
            except StopIteration:
                break

        if len(batch) == 0:
            raise StopIteration("End of data stream reached")
        return batch

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                # Get input batch
                try:
                    input_batch = self.input_batch()
                except StopIteration:
                    break
                data_batch = tokenized_batch(input_batch, self.tokenizer, self.cfg, self.processor)

                hidden_states = hf_forward(
                    self.model,
                    data_batch,
                    self.tokenizer,
                    self.cfg,
                    remove_high_norm=self.remove_high_norm,
                    training=self.training
                )

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]
            if len(hidden_states) == 0:
                continue
            self.activations[current_idx: current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)

        self.activations = self.activations[:current_idx]
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'n_ctxs': self.n_ctxs,
            'ctx_len': self.ctx_len,
            'refresh_batch_size': self.refresh_batch_size,
            'out_batch_size': self.out_batch_size,
            'device': self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
