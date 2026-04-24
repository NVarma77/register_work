"""
Utilities for evaluating dictionaries on a model and dataset.
"""

import torch as t
import gc
from collections import defaultdict

from .buffer import ActivationBuffer
from .config import DEBUG
from tqdm import tqdm
import einops

@t.no_grad()
def evaluate(
    dictionary,  # a dictionary
    activations, # a generator of activations; if an ActivationBuffer, also compute loss recovered
    normalize_batch=False, # normalize batch before passing through dictionary
    device="cpu",
    n_batches: int = 1
):
    assert n_batches > 0
    out = defaultdict(float)
    active_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)

    mean_sum_of_squares = []
    mean_act_per_dimension = []
    mean_sum_of_resid_squared = []

    for _ in tqdm(range(n_batches)):
        try:
            x = next(activations)#.to(device)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim ** 0.5)
        except StopIteration:
            # raise StopIteration(
            #     "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            # )
            break
        x_hat, f = dictionary(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        
        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32) # If f is shape (B, L, D), flatten to (B*L, D)
        # Count the number of times each feature activates (non-zero value)
        feature_activations = (features_BF > 0.000001).float().mean(dim=0)  # Shape: [dict_size]
        
        # If this is the first batch, initialize a tensor to store the counts
        if _ == 0:
            feature_activation_counts = feature_activations 
        else:
            # Otherwise, add to the existing counts
            feature_activation_counts += feature_activations
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2

        active_features += features_BF.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        # #compute variance explained
        flattened_sae_input = x
        flattened_sae_output = x_hat
        resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_output).pow(2).sum(dim=-1)
            )
        
        mean_sum_of_squares.append(
            (flattened_sae_input).pow(2).sum(dim=-1).mean(dim=0)  # scalar
        )
        mean_act_per_dimension.append(
            (flattened_sae_input).pow(2).mean(dim=0)  # [d_model]
        )
        mean_sum_of_resid_squared.append(
            resid_sum_of_squares.mean(dim=0)  # scalar
        )
        
        # total_variance = t.var(x, dim=0).sum()
        # residual_variance = t.var(x - x_hat, dim=0).sum()
        # frac_variance_explained = (1 - residual_variance / total_variance)

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        #out["frac_variance_explained"] += frac_variance_explained.item()
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out['relative_reconstruction_bias'] += relative_reconstruction_bias.item()

    out = {key: value / n_batches for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    feature_activation_frequency = feature_activation_counts / n_batches
    #out["feature_activation_frequency"] = feature_activation_frequency.tolist()

    mean_sum_of_squares = t.stack(mean_sum_of_squares).mean(dim=0)
    mean_act_per_dimension = t.cat(mean_act_per_dimension).mean(dim=0)
    total_variance = mean_sum_of_squares - mean_act_per_dimension**2
    residual_variance = t.stack(mean_sum_of_resid_squared).mean(dim=0)
    out["explained_variance"] = (1 - residual_variance / total_variance).item()

    return out, feature_activation_frequency.tolist()


# def loss_recovered_evaluation(generator,
#                               model,
#                               submodule,
#                               dictionary,
#                               tokenizer,
#                               max_len=128,
#                               batch_size=128,
#                               io="out",
#                               normalize_batch=False,
#                               tracer_args={'scan' : False, 'validate' : False},
#                               device="cpu",
#                               n_batches: int = 1,
#                               dataset_type: str = None):
#     assert n_batches > 0
#     out = defaultdict(float)

#     for _ in tqdm(range(n_batches)):
#         if dataset_type == 'acts':
#             acts_batch, input_ids_batch = next(generator)
#             text = tokenizer.decode(input_ids_batch[:max_len])
#         else:
#             text = next(generator)

#         # compute loss recovered
#         loss_original, loss_reconstructed, loss_zero = loss_recovered(
#             text,
#             model,
#             submodule,
#             dictionary,
#             max_len=max_len,
#             normalize_batch=normalize_batch,
#             io=io,
#             tracer_args=tracer_args
#         )
#         frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
        
#         out["loss_original"] += loss_original.item()
#         out["loss_reconstructed"] += loss_reconstructed.item()
#         out["loss_zero"] += loss_zero.item()
#         out["frac_recovered"] += frac_recovered.item()

#         t.cuda.empty_cache()
#         gc.collect()

#     out = {key: value / n_batches for key, value in out.items()}

#     return out
