from functools import partial
from collections import OrderedDict
from typing import Dict, Callable
import torch
import os
import numpy as np

def compute_similarities(image_tokens, W_E):
    # Normalize vectors for cosine similarity
    image_tokens_norm = image_tokens / image_tokens.norm(dim=-1, keepdim=True)
    W_E_norm = W_E / W_E.norm(dim=-1, keepdim=True)
    
    # Compute similarities [batch_size, seq_len, vocab_size]
    similarities = torch.matmul(image_tokens_norm, W_E_norm.t())

    return similarities
def top_token_similarities(similarities, processor, top_k=5, position=None):
    """
    Compute cosine similarity between image tokens and embedding matrix and return top matches.
    
    Args:
        image_tokens: Tensor of image token embeddings [batch_size, seq_len, d_model]
        W_E: Embedding matrix [vocab_size, d_model]
        processor: Processor containing the tokenizer
        top_k: Number of top matches to return
    
    Returns:
        top_tokens: List of top token matches with their scores
    """
    
    if position is not None:
        positions = [position]
    else:
        positions = torch.arange(similarities.shape[1])
    
    # Get top k most similar tokens and their scores
    top_scores, top_indices = torch.topk(similarities, k=top_k, dim=-1)
    
    # Get token strings for top matches
    tokenizer = processor.tokenizer
    top_tokens = []
    for batch in range(similarities.shape[0]):
        batch_tokens = []
        for seq_pos in positions:
            tokens = []
            for idx in top_indices[batch, seq_pos]:
                token = tokenizer.decode([idx])
                tokens.append((token, top_scores[batch, seq_pos, top_indices[batch, seq_pos] == idx].item()))
            batch_tokens.append(tokens)
        top_tokens.append(batch_tokens)
    
    return top_tokens


def read_top_k_images(ds, latent_id, top_k_images_dir, k=1, top_k_random_sample=False, partition=None, seed=42):
    latent_filename = os.path.join(top_k_images_dir, f"latent_{latent_id}.pt")
    data = torch.load(latent_filename, weights_only=False)
    top_ids = data.get("top_ids", [])
    if not isinstance(top_ids, list):
        top_ids = top_ids.tolist()
    if partition is not None:
        top_ids = top_ids[partition]
        heatmaps = data.get("heatmaps", [])[partition]
    else:
        heatmaps = data.get("heatmaps", [])
    if top_k_random_sample:
        # draw k **distinct** indices uniformly at random from the slice [start, end)
        sample_len = len(top_ids)
        rng = np.random.default_rng(seed=seed)  # reproducible? add seed=…
        random_indices = rng.choice(np.arange(0, sample_len), size=min(k, sample_len), replace=False)

        top_ids = [top_ids[i] for i in random_indices]
        if heatmaps:
            heatmaps = [heatmaps[i] for i in random_indices]

    else:
        top_ids = top_ids[0:k]
        if heatmaps:
            heatmaps = heatmaps[0:k]

    images = []
    for img_id in top_ids:
        image = ds[img_id]["image"]
        if isinstance(image, list):
            image = image[0]
        images.append(image.convert("RGB"))

    return {'images': images, 'heatmaps': heatmaps}
