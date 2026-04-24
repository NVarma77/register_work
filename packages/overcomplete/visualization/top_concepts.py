"""
Module dedicated for visualizing top concepts in a batch of images.
"""

import numpy as np
import matplotlib.pyplot as plt
#from skimage import measure
import torch
from typing import List, Tuple, Callable, Union
from PIL import Image

from .plot_utils import show, interpolate_cv2, get_image_dimensions, np_channel_last, normalize
from .cmaps import VIRIDIS_ALPHA, TAB10_ALPHA


def _get_representative_ids(heatmaps, concept_id, top_k=10, aggregation_function="mean"):
    """
    Get the top 10 images based on the mean value of the heatmaps for a given concept.

    Parameters
    ----------
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.

    Returns
    -------
    torch.Tensor or np.ndarray
        Indices of the top 10 images based on the mean value of the heatmaps for a given concept.
    """
    if aggregation_function == "mean":
        if isinstance(heatmaps, torch.Tensor):
            return torch.mean(heatmaps[:, :, :, concept_id], dim=(1, 2)).argsort()[-top_k:]
        return np.mean(heatmaps[:, :, :, concept_id], axis=(1, 2)).argsort()[-top_k:]
    elif aggregation_function == "max":
        if isinstance(heatmaps, torch.Tensor):
            return torch.amax(heatmaps[:, :, :, concept_id], dim=(1, 2)).argsort()[-top_k:]
        return np.amax(heatmaps[:, :, :, concept_id], axis=(1, 2)).argsort()[-top_k:]
    else:
        raise ValueError(f"Invalid aggregation function: {aggregation_function}")

def _get_representative_ids_with_values(heatmaps, top_k=10, aggregation_function="mean"):
    
    """
    Get the top 10 images based on the mean value of the heatmaps for a given concept. 
    Also return the mean value of the heatmaps for each image.

    Parameters
    ----------
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.

    Returns
    -------
    torch.Tensor or np.ndarray
        Indices of the top 10 images based on the mean value of the heatmaps for a given concept.
    """
    mean_values = torch.mean(heatmaps, dim=(1, 2))  # shape: (batch_size, num_concepts)
        
    # Transpose to get (num_concepts, batch_size)
    mean_values_t = mean_values.transpose(0, 1)
    
    # Get top k indices and values for each concept (num_concepts, top_k)
    top_values, top_indices = torch.topk(mean_values_t, k=top_k, dim=1)
    
    return top_indices.detach(), top_values.detach()

    # if isinstance(heatmaps, torch.Tensor):
    #     mean_values = torch.mean(heatmaps[:, :, :, concept_id], dim=(1, 2))
    #     return mean_values.argsort()[-top_k:].detach(), mean_values[mean_values.argsort()[-top_k:]].detach()
    # mean_values = np.mean(heatmaps[:, :, :, concept_id], axis=(1, 2))
    # return mean_values.argsort()[-top_k:], mean_values[mean_values.argsort()[-top_k:]]

def overlay_top_heatmaps(images, heatmaps, concept_id, cmap=None, alpha=0.35, return_individual_heatmaps=False, aggregation_function="mean"):
    """
    Visualize the top activating image for a concepts and overlay the associated heatmap.

    This function sorts images based on the mean value of the heatmaps for a given concept and
    visualizes the top 10 images with their corresponding heatmaps.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    z_heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    cmap : str, optional
        Colormap for the heatmap, by default 'jet'.
    alpha : float, optional
        Transparency of the heatmap overlay, by default 0.35.
    return_individual_heatmaps : bool, optional
        If True, returns a list of image arrays with the heatmaps overlaid, by default False.
        
    Returns
    -------
    list or None
        If return_individual_heatmaps is True, returns a list of numpy arrays representing 
        the images with overlaid heatmaps. Otherwise, returns None.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    # if we handle the cmap, choose tab10 if number of concepts is less than 10
    # else choose a normal one
    if cmap is None:
        cmap = TAB10_ALPHA[concept_id] if heatmaps.shape[-1] < 10 else VIRIDIS_ALPHA
    if alpha is None:
        # and enforce the alpha value to one, as the alpha is already handled by the colormap
        alpha = 1.0

    best_ids = _get_representative_ids(heatmaps, concept_id, aggregation_function=aggregation_function)
    
    # Initialize a list to store the overlaid images if needed
    overlaid_images = [] if return_individual_heatmaps else None
    
    for i, idx in enumerate(best_ids):
        image = images[idx]
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))

        if return_individual_heatmaps:

            if not isinstance(image, np.ndarray):
                if hasattr(image, 'numpy'):  # torch tensor
                    image = image.numpy()
                else:  # PIL image
                    image = np.array(image)

            # Apply colormap to heatmap
            cmap_func = plt.get_cmap(cmap)
            heatmap = normalize(heatmap)
            colored_heatmap = cmap_func(heatmap)[:,:,:3]  # Get RGB from colormap
            
            # Blend image and heatmap directly using numpy
            blended_image = image.copy()
            if blended_image.dtype != np.uint8:
                blended_image = (blended_image * 255).astype(np.uint8)
            
            # Make sure we're working with RGB
            if blended_image.ndim == 2:  # Grayscale
                blended_image = np.stack([blended_image] * 3, axis=2)
            
            blended_image = np_channel_last(blended_image)
            blended_image = normalize(blended_image)


            # Blend using the alpha value
            for c in range(3):
                blended_image[:, :, c] = (1 - alpha) * blended_image[:, :, c]*255 + alpha * (colored_heatmap[:, :, c] * 255)
            
            # Ensure proper dtype
            blended_image = blended_image.astype(np.uint8)
            overlaid_images.append(blended_image)

        else:
            # Original visualization code
            plt.subplot(2, 5, i + 1)
            show(image)
            show(heatmap, cmap=cmap, alpha=alpha)
    
    return overlaid_images

def overlay_heatmaps_to_images(images: List[Image.Image], heatmaps: np.ndarray, cmap=None, alpha=0.35):
    """
    Overlay heatmaps onto images.
    
    Args:
        images (List[Image.Image]): List of images to overlay heatmaps on.
        heatmaps (np.ndarray): Array of heatmaps to overlay on the images.
        cmap (str, optional): Colormap to use for the heatmaps. Defaults to None, which uses VIRIDIS_ALPHA.
        alpha (float, optional): Transparency of the heatmap overlay. Defaults to 0.35.
        
    Returns:
        List[np.ndarray]: List of images with heatmaps overlaid.
    """

    # if we handle the cmap, choose tab10 if number of concepts is less than 10
    # else choose a normal one
    if cmap is None:
        cmap = VIRIDIS_ALPHA
    if alpha is None:
        alpha = 1.0
    assert len(images) == len(heatmaps)

    # Initialize a list to store the overlaid images if needed
    overlaid_images = []
    
    for image, heatmap in zip(images, heatmaps):
        width, height = get_image_dimensions(image)
        heatmap = interpolate_cv2(heatmap[:, :], (width, height))

        if not isinstance(image, np.ndarray):
            if hasattr(image, 'numpy'):  # torch tensor
                image = image.numpy()
            else:  # PIL image
                image = np.array(image)

        # Apply colormap to heatmap
        cmap_func = plt.get_cmap(cmap)
        heatmap = normalize(heatmap)
        colored_heatmap = cmap_func(heatmap)[:,:,:3]  # Get RGB from colormap
        
        # Blend image and heatmap directly using numpy
        blended_image = image.copy()
        if blended_image.dtype != np.uint8:
            blended_image = (blended_image * 255).astype(np.uint8)
        
        # Make sure we're working with RGB
        if blended_image.ndim == 2:  # Grayscale
            blended_image = np.stack([blended_image] * 3, axis=2)
        
        blended_image = np_channel_last(blended_image)
        blended_image = normalize(blended_image)


        # Blend using the alpha value
        for c in range(3):
            blended_image[:, :, c] = (1 - alpha) * blended_image[:, :, c]*255 + alpha * (colored_heatmap[:, :, c] * 255)
        
        # Ensure proper dtype
        blended_image = blended_image.astype(np.uint8)
        overlaid_images.append(blended_image)

    return overlaid_images

def numpy_to_pil(np_array: np.ndarray) -> Image.Image:
    """Convert NumPy array to PIL Image with normalization."""
    if np_array.dtype != np.uint8:
        np_array = ((np_array - np_array.min()) / 
                   (np_array.max() - np_array.min() + 1e-8) * 255).astype(np.uint8)
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3, 4]:
        np_array = np_array.transpose(1, 2, 0)
    if np_array.shape[-1] == 1:
        np_array = np_array.squeeze(-1)
    return Image.fromarray(np_array)

def overlay_heatmap_to_image(image, heatmap, cmap=None, alpha=0.35):
    """
    Overlay a heatmap onto an image.
    """
    return numpy_to_pil(overlay_heatmaps_to_images([image], [heatmap], cmap, alpha)[0])
    
def evidence_top_images(images, heatmaps, concept_id, percentiles=None):
    """
    Visualize the top activating image for a concept and highlight the top activating pixels.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then use the heatmap to highlights the top activating area depending on their percentile value.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    percentiles : list of int, optional
        List of percentiles to highlight, by default None.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    if percentiles is None:
        # gradation from 50% top activating pixels to 95% top activating pixels
        # with alpha 0 at start and 1 at the end
        percentiles = np.linspace(50, 95, 10)

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = images[idx]
        image = np_channel_last(image)
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))

        # use the heatmap to apply alpha depending on the percentile
        mask = np.zeros_like(heatmap)
        for percentile in percentiles:
            mask[heatmap > np.percentile(heatmap, percentile)] += 1.0
        mask = mask / len(percentiles)

        plt.subplot(2, 5, i + 1)
        show(image*mask[:, :, None])


def zoom_top_images(images, heatmaps, concept_id, zoom_size=100):
    """
    Zoom into the hottest point in the heatmaps for a specific concept.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then zooms into the hottest point of the heatmap for each of these images.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    zoom_size : int, optional
        Size of the zoomed area around the hottest point, by default 100.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = np_channel_last(images[idx])
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))
        hottest_point = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

        x_min = max(hottest_point[0] - zoom_size // 2, 0)
        x_max = min(hottest_point[0] + zoom_size // 2, image.shape[0])
        y_min = max(hottest_point[1] - zoom_size // 2, 0)
        y_max = min(hottest_point[1] + zoom_size // 2, image.shape[1])

        zoomed_image = image[x_min:x_max, y_min:y_max]

        plt.subplot(2, 5, i + 1)
        show(zoomed_image)


# def contour_top_image(images, heatmaps, concept_id, percentiles=None, cmap="viridis", linewidth=1.0):
#     """
#     Contour the best images for a specific concept using heatmap percentiles.

#     This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
#     then draws contours at specified percentiles on the heatmap overlaid on the original image.

#     Parameters
#     ----------
#     images : torch.Tensor or PIL.Image or np.ndarray
#         Batch of input images of shape (batch_size, channels, height, width).
#     heatmaps : torch.Tensor or np.ndarray
#         Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
#     concept_id : int
#         Index of the concept to visualize.
#     percentiles : list of int, optional
#         List of percentiles to contour, by default [70].
#     cmap : str, optional
#         Colormap for the contours, by default "viridis".
#     linewidth : float, optional
#         Width of the contour lines, by default 1.0.
#     """
#     assert len(images) == len(heatmaps)
#     assert heatmaps.shape[-1] > concept_id
#     assert heatmaps.ndim == 4

#     if percentiles is None:
#         percentiles = [70]

#     cmap = plt.get_cmap(cmap)
#     best_ids = _get_representative_ids(heatmaps, concept_id)

#     for i, idx in enumerate(best_ids):
#         image = images[idx]
#         width, height = get_image_dimensions(image)
#         plt.subplot(2, 5, i + 1)
#         show(image)

#         heatmap = heatmaps[idx, :, :, concept_id]
#         heatmap = interpolate_cv2(heatmap, (width, height))

#         for percentile in percentiles:
#             if len(percentiles) == 1:
#                 color_value = cmap(0.0)
#             else:
#                 # color value is a remap of percentile between [0, 1] depending on value of percentiles
#                 color_value = (percentile - percentiles[-1]) / (percentiles[0] - percentiles[-1])
#                 color_value = cmap(color_value)

#             cut_value = np.percentile(heatmap, percentile)
#             contours = measure.find_contours(heatmap, cut_value)
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color_value)
