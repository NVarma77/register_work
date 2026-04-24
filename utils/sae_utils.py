import sys
sys.path.append("..")

from einops import rearrange

def get_patch_sae_codes(sae, subset_layer_out, num_patches):
    """
    Process layer activations through the SAE to get sparse codes.
    
    This function encodes the layer activations using the SAE,
    then rearranges the resulting codes into a format suitable for further processing (batch_size, width, height, latent_dim).
    
    Parameters
    ----------
    subset_layer_out : torch.Tensor
        The layer activations to encode, typically of shape (batch_size * width * height, feature_dim).
    num_patches : int
        The number of patches in each dimension (width/height) of the image. We always work with square images.
        
    Returns
    -------
    torch.Tensor
        The encoded sparse codes, reshaped to (batch_size, width, height, latent_dim).
    """

    subset_codes = sae.encode(subset_layer_out)
    subset_codes = rearrange(subset_codes.detach(), '(n w h) d -> n w h d',  w=num_patches, h=num_patches)
    subset_codes = subset_codes.float()
    return subset_codes
