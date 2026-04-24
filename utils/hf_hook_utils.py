# Code adapted from https://github.com/andyrdt/refusal_direction/blob/main/pipeline/utils/hook_utils.py
import torch
import contextlib
import functools

from typing import List, Tuple, Callable, Union
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from transformers import GenerationConfig
from dictionary_learning.buffer import tokenized_batch
from dictionary_learning.buffer import hf_forward
import numpy as np


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""], position: Union[str, List[int]] = "all"):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        if activation.shape[1] == 1 and position != "all":
            # We apply the hook only when processing the input, not during generation
            if isinstance(input, tuple):
                return (activation, *input[1:])
            else:
                return activation
        
        # Check if position is a list of integers
        if isinstance(position, list) and all(isinstance(pos, int) for pos in position):
            # Apply the vector only to the specified positions
            vector = vector.to(activation)
            for pos in position:
                if pos < activation.shape[1]:
                    activation[:, pos, :] += coeff * vector
        else:
            # We apply the hook only when processing the input, not during generation
            vector = vector.to(activation)
            activation += coeff * vector
            

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_activation_addition_output_post_hook(vector: Float[Tensor, "batch_size d_model"], coeff: Float[Tensor, ""], position: Union[str, List[int]] = "all", mean: Tensor = None, sae=None):
    def hook_fn(module, input, output):
        nonlocal vector

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        if activation.shape[1] == 1 and position != "all":
            # We apply the hook only when processing the input, not during generation
            if isinstance(output, tuple):
                return (activation, *output[1:])
            else:
                return activation
                    
        # Check if position is a list of integers
        if isinstance(position, list) and all(isinstance(pos, int) for pos in position):
            # Apply the vector only to the specified positions
            vector = vector.to(activation)
            for pos in range(position[0], position[1]):
                if len(activation.shape) == 3:
                    if pos < activation.shape[1]:

                        activation[:, pos, :] += coeff * vector
                else:
                    activation[pos, :] += coeff * vector.squeeze()
                    

        else:
            # We apply the hook only when processing the input, not during generation
            vector = vector.to(activation)
            # We add the position dimension -> [batch_size, 1, d_model]
            if len(activation.shape) == 3:
                vector = vector.unsqueeze(1)
            #activation += coeff * vector
            repeated_vector = torch.repeat_interleave(vector, activation.shape[1], dim=1)
            if mean is not None:
                activation = coeff * repeated_vector + mean
            else:
                activation += coeff * repeated_vector



            # reshaped_vector = vector.unsqueeze(0).unsqueeze(0)
            # reshaped_tensor = reshaped_vector.repeat(1, activation.shape[1], 1) 
            # activation = (coeff * reshaped_tensor)
            

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_activation_addition_output_post_hook_v2(vector: Float[Tensor, "batch_size d_model"], coeff: Float[Tensor, ""], position: Union[str, List[int]] = "all", intervention_type: str = 'addition'):
    def hook_fn(module, input, output):
        nonlocal vector

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        if activation.shape[1] == 1 and position != "all":
            # We apply the hook only when processing the input, not during generation
            if isinstance(output, tuple):
                return (activation, *output[1:])
            else:
                return activation
                    
        # Check if position is a list of integers
        if isinstance(position, list):
            # Apply the vector only to the specified positions
            vector = vector.to(activation)

            # We add the position dimension -> [batch_size, 1, d_model]
            if len(activation.shape) == 3:
                vector = vector.unsqueeze(1)

            for batch_idx, position_element in enumerate(position):
                for pos in position_element:
                    if intervention_type == 'addition':
                        activation[batch_idx, pos, :] += coeff * vector[batch_idx,0]
                    elif intervention_type == 'replacement':
                        activation[batch_idx, pos, :] = coeff * vector[batch_idx,0]
                    
        else:
            # We apply the hook only when processing the input, not during generation
            vector = vector.to(activation)
            # We add the position dimension -> [batch_size, 1, d_model]
            if len(activation.shape) == 3:
                vector = vector.unsqueeze(1)
            repeated_vector = torch.repeat_interleave(vector, activation.shape[1], dim=1)
            if intervention_type == 'addition':
                activation += coeff * repeated_vector
            elif intervention_type == 'replacement':
                activation = coeff * repeated_vector

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def generate_completions(model, tokenizer, tokenized_inputs, fwd_pre_hooks=[], fwd_hooks=[], generation_config=None, **generation_kwargs):
    if generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):

        output = model.generate(
            **tokenized_inputs,
            generation_config=generation_config,
            **generation_kwargs
        )

        if generation_kwargs.get('return_dict_in_generate', False):
            generation_toks_batch = output.sequences
            generation_toks_batch = generation_toks_batch[:, tokenized_inputs.input_ids.shape[-1]:]

            scores = output.scores  # This is a tuple of tensors, one for each generation step

            # Convert scores to log probabilities
            log_probs = []
            for score in scores:
                # Apply softmax to get probabilities and then take log
                probs = torch.nn.functional.softmax(score, dim=-1)
                # torch.log(probs) -> [batch_size, vocab_size]
                #log_probs.append(torch.log(probs))
                log_probs.append(probs)


            # Loop over generations
            generation_mean_log_prob = []
            for i, generation_toks in enumerate(generation_toks_batch):
                chosen_log_probs = []
                for j, token_id in enumerate(generation_toks):
                    if token_id != tokenizer.pad_token_id:
                        chosen_log_probs.append(log_probs[j][i, token_id].item())
                # Convert log probabilities to numpy array
                assert len(chosen_log_probs) > 0, "No log probabilities found for generation"
                generation_mean_log_prob.append(np.array(chosen_log_probs).mean())

        else:
            generation_toks_batch = output
            generation_mean_log_prob = None

        for generation_idx, generation in enumerate(generation_toks_batch):
            completions.append(tokenizer.decode(generation, skip_special_tokens=True).strip())

        return completions, generation_mean_log_prob
            

def generate_completions_internvl(model, tokenizer, pixel_values, question, num_patches_list=None, fwd_pre_hooks=[], fwd_hooks=[], generation_config=None, **generation_kwargs):


    completions = []

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):

        if pixel_values is not None:
            completions = model.batch_chat(tokenizer, pixel_values, question, generation_config,
                                        num_patches_list=num_patches_list, history=None)
        else:
            # Only text
            completions = model.chat(tokenizer, None, question, generation_config, history=None)

        # Check if completions is a string and convert it to a list if needed
        if isinstance(completions, str):
            completions = [completions]
        return completions
            

def forward_pass_with_hooks(model, tokenized_inputs, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8):

        num_inputs = tokenized_inputs.pixel_values.shape[0]
        outputs_list = []
        #for i in tqdm(range(0, num_inputs)):

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):

            output_model = model(**tokenized_inputs)
            output_model = output_model['pooler_output']

            for output_idx, output_model in enumerate(output_model):
                outputs_list.append(output_model.detach())

        return outputs_list

def get_module_output(model, inputs, tokenizer, cfg, processor=None):
    """
    Get the output of a specific module in the model for the given inputs.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to extract activations from.
    inputs : list
        List of input examples to process.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model.
    processor : transformers.ProcessorMixin
        Processor for the model's inputs.
    cfg : ExperimentConfig
        Configuration object with model parameters.
        
    Returns
    -------
    torch.Tensor
        The output activations from the specified module.
    """
    data_batch = tokenized_batch(inputs, tokenizer, cfg, processor)
    batch_layer_out = hf_forward(model, data_batch, tokenizer, cfg=cfg, training=False)
    return batch_layer_out.detach()
