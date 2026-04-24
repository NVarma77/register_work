import os
import sys
from dotenv import load_dotenv
load_dotenv()

REPO_DIR = os.environ.get('REPO_DIR')
# Check if REPO_DIR environment variable is set
if REPO_DIR is None:
    REPO_DIR = str(Path(__file__).resolve().parent)
    os.environ['REPO_DIR'] = REPO_DIR
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))


from pathlib import Path
import json
import torch
from transformers import (AutoModel,
                          AutoModelForCausalLM,
                          ViTForImageClassification,
                          Gemma3ForCausalLM,
                          Gemma3ForConditionalGeneration)
from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration
)
# CLIP stuff
from transformers import CLIPProcessor, CLIPModel
import gc
# ViT stuff
from transformers import ViTModel, ViTImageProcessor
import numpy as np
from PIL import Image
from functools import partial
from typing import Literal
from typing import List
import src.demo_config as demo_config
from tqdm import tqdm
import re
import yaml
from packages.overcomplete.visualization.top_concepts import overlay_heatmap_to_image

REPO_DIR = Path(__file__).parent.parent

import logging
logger = logging.getLogger(__name__)

def resolve_attr(obj, attr_path: str):
    """
    Resolve a nested attribute path like
    'vision_tower.vision_model.encoder.layers[2].self_attn'
    and return the referenced object.
    """
    parts = re.split(r'\.(?![^\[]*\])', attr_path)  # split by dots not inside []
    
    for part in parts:
        # handle list/tuple indexing, e.g. layers[2]
        match = re.match(r'([a-zA-Z0-9_]+)(\[(\d+)\])?', part)
        if not match:
            raise ValueError(f"Invalid path component: {part}")
        
        attr_name = match.group(1)
        index = match.group(3)
        
        obj = getattr(obj, attr_name)
        if index is not None:
            obj = obj[int(index)]
    
    return obj

def load_vision_model(model, cfg):
    """
    Load the vision model from a model.
    
    Args:
        model: The model to load the vision model from.
        cfg: The configuration object.
    
    Returns:
        The vision model.
    """
    assert demo_config.LLM_CONFIG[cfg.model_name].vision_model is not None, "vision model must be specified"
    vision_model = resolve_attr(model, demo_config.LLM_CONFIG[cfg.model_name].vision_model)
    if demo_config.LLM_CONFIG[cfg.model_name].vision_model_layer is not None:
        submodule = resolve_attr(model, demo_config.LLM_CONFIG[cfg.model_name].vision_model_layer)
        print("Submodule attached to vision model")
        vision_model.submodule = submodule[cfg.layer]
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()   

    return vision_model


def get_model_path(model_name: str) -> str:
    return demo_config.LLM_CONFIG[model_name].model_path

# vision models receive only images as input
# lm models receive text as input
# vlm models can receive both images and text as input
def get_model_type(model_name: str) -> str:
    return demo_config.LLM_CONFIG[model_name].model_type

# TODO: see if we can always read this from model's config
def get_model_patch_size(model_name: str) -> int | None:
    return demo_config.LLM_CONFIG[model_name].model_patch_size

def get_model_img_size(model_name: str) -> int | None:
    return demo_config.LLM_CONFIG[model_name].model_img_size

def load_encoder_tower(model_, encoder_tower):
    """
    Load the encoder tower from a CLIP-based model (CLIP, SigLip, etc.).
    
    Args:
        model_: The model to extract the encoder tower from.
        encoder_tower: Which encoder tower to load. Can be 'vision', 'text', 'both', or None.
                      If None, defaults to 'both'.
    
    Returns:
        The specified encoder tower(s) from the model.
        
    Raises:
        ValueError: If an invalid encoder tower is specified.
    """
    if encoder_tower == 'both':
        # No encoder tower specified, we load both as a dictionary
        return model_
    elif encoder_tower is None or encoder_tower == 'vision':
        print('Loading vision model')
        # default behavior
        model = model_.vision_model
    elif encoder_tower == 'text':
        print('Loading text model')
        model = model_.text_model
    else:
        raise ValueError(f"Invalid encoder tower: {encoder_tower}")
    return model
  
def load_hf_model(model_name, cfg, device="cuda:0", **kwargs):
    processor = None
    tokenizer = None
    model = None

    if 'paligemma' in model_name.lower():
        if "28b" in model_name.lower():
            model_ = PaliGemmaForConditionalGeneration.from_pretrained(
                        model_name, device_map="auto", torch_dtype=cfg.dtype)
        else:
            model_ = PaliGemmaForConditionalGeneration.from_pretrained(cfg.model_path, device_map=device)
        processor = PaliGemmaProcessor.from_pretrained(cfg.model_path, do_convert_rgb=True)
        tokenizer = processor.tokenizer
        if cfg.submodel == 'enc' and not cfg.get_full_model:
            model = model_.vision_tower.vision_model
            del model_
            model_ = None
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            model = model_
    elif 'vit' in model_name.lower() and 'clip' not in model_name.lower():
        model = ViTForImageClassification.from_pretrained(cfg.model_path, device_map=device)
        processor = ViTImageProcessor.from_pretrained(cfg.model_path, do_convert_rgb=True)
        tokenizer = None
    elif 'siglip' in model_name.lower():
        # We only load the vision tower from siglip
        model_ = AutoModel.from_pretrained(cfg.model_path, device_map=device)
        encoder_tower = kwargs.get('encoder_tower', None)
        model = load_encoder_tower(model_, encoder_tower)
        processor = AutoProcessor.from_pretrained(cfg.model_path, do_convert_rgb=True)
        tokenizer = processor.tokenizer
    elif "clip-vit" in model_name.lower():
        # 'encoder_tower' can be 'text' or 'vision' or 'both
        model_ = CLIPModel.from_pretrained(cfg.model_path) # doesn't work with device_map
        model_.to(device)
        encoder_tower = kwargs.get('encoder_tower', None)
        model = load_encoder_tower(model_, encoder_tower)
        
        processor = CLIPProcessor.from_pretrained(cfg.model_path, do_convert_rgb=True)
        tokenizer = None
    elif "dinov2" in model_name.lower():
        model = AutoModel.from_pretrained(cfg.model_path, device_map=device)
        processor = AutoImageProcessor.from_pretrained(cfg.model_path, do_convert_rgb=True)
        tokenizer = None
    elif "gemma-3" in model_name.lower() and "1b" not in model_name.lower() and "270m" not in model_name.lower():
        if "27b" in model_name.lower():
            model_ = Gemma3ForConditionalGeneration.from_pretrained(
                        model_name, device_map="auto", torch_dtype=cfg.dtype)
        else:
            model_ = Gemma3ForConditionalGeneration.from_pretrained(
                        model_name, device_map=device)
        processor = AutoProcessor.from_pretrained(cfg.model_path)
        tokenizer = processor.tokenizer
        if cfg.submodel == 'enc' and not cfg.get_full_model:
            model = load_vision_model(model_, cfg)    
        else:
            model = model_
    elif "gemma-3" in model_name.lower() and ("1b" in model_name.lower() or "270m" in model_name.lower()):
        # Gemma 3 1b is text-only (lm)
        model = Gemma3ForCausalLM.from_pretrained(cfg.model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    elif any(x in model_name.lower() for x in ('qwen2-vl', 'qwen2.5-vl', 'mimo-vl', 'aloe-vision-7b')):
        if "72b" in model_name.lower() or "32b" in model_name.lower():
            kwargs = {
                "device_map": "auto",
                "torch_dtype": cfg.dtype,
                "attn_implementation": "flash_attention_2"
            }
        else:
            kwargs = {
                "device_map": device,
                "torch_dtype": cfg.dtype,
                "attn_implementation": "flash_attention_2"
            }
        if "2-vl" in model_name.lower() or "aloe-vision-7b" in model_name.lower():
            model_ = Qwen2VLForConditionalGeneration.from_pretrained(
                cfg.model_path, **kwargs)
        elif "2.5-vl" in model_name.lower():
            model_ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_path, **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        # We resize images
        #orig_model_name = "Qwen/" + model_name.split("/")[-1] if "Qwen" in model_name else model_name
        # img_size = get_model_img_size(model_name)
        # pixels = img_size*img_size if img_size is not None else None
        # pixels = 256*28*28
        # print(f"Pixels: {pixels}")
        #processor = AutoProcessor.from_pretrained(cfg.model_path, max_pixels=pixels, min_pixels=pixels) if pixels is not None else AutoProcessor.from_pretrained(cfg.model_path)
        processor = AutoProcessor.from_pretrained(cfg.model_path)
        tokenizer = processor.tokenizer
        if cfg.context_length is not None:
            tokenizer.model_max_length = cfg.context_length
        if cfg.submodel == 'enc' and not cfg.get_full_model:
            model = load_vision_model(model_, cfg)    
        else:
            model = model_
    
    elif "internvl" in model_name.lower():
        model_ = AutoModel.from_pretrained(cfg.model_path, device_map=device,
                                                                torch_dtype=cfg.dtype,
                                                                attn_implementation="flash_attention_2",
                                                                trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, use_fast=False)
        if cfg.submodel == 'enc' and not cfg.get_full_model:
            model = model_.vision_model
            del model_
        else:
            model = model_
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, device_map=device, **kwargs)
        

    return model, tokenizer, processor

def load_model(model_name, cfg, dtype=torch.bfloat16, device="cuda:0", **kwargs):
    model, tokenizer, processor = load_hf_model(model_name, cfg, device, **kwargs)
    if 'llama' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(dtype=dtype)
    model.eval()
    return model, tokenizer, processor


def get_sae_run_id(sae_path: str) -> str:
    run_dir = Path(sae_path)
    if run_dir.name.startswith("trainer_"):
        run_dir = run_dir.parent

    run_name = run_dir.name
    match = re.search(r"_(\d+)(?:_(?:registers_only|outlier_patches.*))?$", run_name)
    if match:
        return match.group(1)
    return run_name.split("_")[-1]

def numpy_to_pil(numpy_img):
    """
    Convert a numpy array to a PIL Image.
    
    Parameters
    ----------
    numpy_img : numpy.ndarray
        Input image as numpy array. Should be in RGB format with values in range [0, 255].
        
    Returns
    -------
    PIL.Image.Image
        The converted PIL Image.
    """
    # Ensure the array is uint8 for PIL
    if numpy_img.dtype != np.uint8:
        if numpy_img.max() <= 1.0:
            numpy_img = (numpy_img * 255).astype(np.uint8)
        else:
            numpy_img = numpy_img.astype(np.uint8)
    
    # Create PIL image
    return Image.fromarray(numpy_img)

def save_json_data(data: dict, output_path: Path) -> None:
    """Save dictionary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved to {output_path}")


def create_output_path(base_dir: Path, subdirs: list[str]=None, filename: str=None) -> Path:
    """Create a full output path with subdirectories, ensuring they exist."""
    full_path = base_dir
    if subdirs:
        for subdir in subdirs:
            full_path /= subdir
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path / filename

def get_sae_path(config_str, target_model_name, target_layer):
    """
    Extract the sae_path for a specific model name and layer number.
    
    Args:
        config_str (str): The configuration string to parse
        target_model_name (str): The model name to search for
        target_layer (int): The layer number to search for
        
    Returns:
        str or None: The sae_path if found, None otherwise
    """
    try:
        with open(config_str, 'r') as file:
            config = yaml.safe_load(file)
        # Get the first key as model_id
        model_id = list(config.keys())[0]
        model_config = config[model_id]
        
        # Check if model name matches
        if model_config['model_name'] == target_model_name:
            # Search for the target layer in SAEs list
            for sae in model_config['saes']:
                if sae['layer'] == target_layer:
                    return sae['sae_path']
    except ArithmeticError:
        return None
    

def get_paths(subject_model_name, layer, path_to_get: Literal["sae_path", "top_k_images_id", "outputs_path"]="sae_path"):
    """
    Extract the sae_path for a specific model name and layer number.
    
    Args:
        config_str (str): The configuration string to parse
        subject_model_name (str): The model name to search for
        layer (int): The layer number to search for
        
    Returns:
        str or None: The path if found
    """
    config_str = f"{REPO_DIR}/config/saes.yaml"
    try:
        with open(config_str, 'r') as file:
            config = yaml.safe_load(file)

        for model_id in config.keys():
            model_config = config[model_id]
            # Check if model name matches
            if model_config['model_name'] == subject_model_name:
                # Search for the target layer in SAEs list
                for sae in model_config['saes']:
                    if sae['layer'] == layer:
                        print(f"sae: {sae}")
                        return sae[path_to_get]
    except Exception as e:
        logger.error(f"Error getting paths: {e}")

def load_active_latents(sae_path, dict_size=8192):
    """
    Load active latents from a SAE model.
    
    This function loads the latent activation frequency from a saved SAE model and returns
    a list of latent indices that have non-zero activation frequency. If the activation
    frequency file is not found, it returns all latent indices up to dict_size.
    
    Parameters
    ----------
    sae_path : str
        Path to the directory containing the SAE model and its metadata.
    dict_size : int
        The total number of latents in the dictionary.
        
    Returns
    -------
    list
        A list of latent indices that have non-zero activation frequency.
    """
    # Load latent activation frequency from the SAE path
    latent_activation_frequency_path = os.path.join(sae_path, "latent_activation_frequency.json")
    
    try:
        with open(latent_activation_frequency_path, 'r') as f:
            latent_activation_frequency = json.load(f)
        
        # Create a list of latent indices with activation frequency > 0
        latents_to_consider = [i for i, freq in enumerate(latent_activation_frequency) if freq > 0]
        print(f"Loaded {len(latents_to_consider)} active latents out of {len(latent_activation_frequency)}")
    except FileNotFoundError:
        print(f"Warning: latent_activation_frequency.json not found at {latent_activation_frequency_path}")
        print("Considering all latents")
        latents_to_consider = list(range(dict_size))
    return latents_to_consider

def get_outputs_path(subject_model_name, layer, run_type):
    outputs_path = get_paths(subject_model_name, layer, path_to_get="outputs_path")
    if run_type == "validation":
        # Workaround to add validation folder to the path
        outputs_path = '/'.join(outputs_path.split('/')[:-2]) + '/validation/' + '/'.join(outputs_path.split('/')[-2:])
    return outputs_path


def paper_plotly_plot(fig, tickangle=0):
    """
    Applies styling to the given plotly figure object targeting paper plot quality.
    """
    fig.update_layout({
        'template': 'plotly_white',
    })
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=tickangle,
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return fig


def list_features(top_k_dir) -> List[str]:
    """List all feature names from .pt files."""
    # Get all latent files
    latent_files = list(top_k_dir.glob("latent_*.pt"))
    
    # Extract latent numbers and sort numerically
    def extract_latent_number(file_path):
        try:
            # Extract the number from the filename (latent_X.pt)
            return int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            # If parsing fails, return a large number to sort it last
            return float('inf')
    
    # Sort files by the numeric value of the latent ID
    latent_files.sort(key=extract_latent_number)
    
    # Return the sorted stem names
    return [f.stem for f in latent_files]

html_colors = {
    'darkgreen' : '#138808',
    'green_drawio' : '#82B366',
    'dark_green_drawio' : '#557543',
    'dark_red_drawio' : '#990000',
    'blue_drawio' : '#6C8EBF',
    'orange_drawio' : '#D79B00',
    'red_drawio' : '#FF9999',# 990000
    'grey_drawio' : '#303030',
    'brown_D3' : '#8C564B',
    'orange_matplotlib': '#ff7f0e',
    'blue_matplotlib': '#1f77b4'}



def create_masked_image(image: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    """Apply heatmap mask to image and return PIL Image."""
    img_height, img_width = image.shape[:2]
    mask = (heatmap > 0).astype(np.uint8) * 255
    mask_resized = Image.fromarray(mask).resize((img_width, img_height), Image.NEAREST)
    mask_array = np.array(mask_resized).astype(bool)
    
    masked_img = image.copy()
    masked_img[~mask_array] = 0
    return numpy_to_pil(masked_img)

def get_or_create_attribution_images(
    image_arrays: List[np.ndarray], 
    heatmaps: List[np.ndarray], 
    heatmaps_flag: bool,
    masks_flag: bool
    ) -> tuple[List[Image.Image], int]:
    """Get existing masked images or create and save new ones for a feature."""
    attribution_images = []
    
    for idx, (image_array, heatmap) in enumerate(zip(image_arrays, heatmaps)):
        if heatmaps_flag:
            heatmap_image = overlay_heatmap_to_image(image_array, heatmap)
        if masks_flag:
            # IMPORTANT: This automatically resizes the heatmap to match the image size 
            masked_image = create_masked_image(image_array, heatmap)

        if heatmaps_flag and masks_flag:
            #attribution_images.append([heatmap_image, masked_image])
            attribution_images.extend((heatmap_image, masked_image))
        elif heatmaps_flag:
            attribution_images.append(heatmap_image)
        elif masks_flag:
            attribution_images.append(masked_image)
    
    return attribution_images


def read_segmentation_results(base_path):
    # Read segmentation results from all latent folders and extract metrics
    
    # Convert to Path object if it's a string
    base_path = Path(base_path)
    
    # Check if the directory exists
    if not base_path.exists():
        print(f"Directory not found: {base_path}")
        return None
    
    # List all latent directories
    latent_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("latent_")]
    print('LEN LATENT DIRS', len(latent_dirs))
    
    latent_explanations = {}
    for i, latent_dir in tqdm(enumerate(latent_dirs)):
        result_file = latent_dir / "explanations" / "explanation.json"
        # Extract the last folder name from latent_dir path
        # This will be something like "latent_123"
        latent_id = int(latent_dir.name.split("_")[-1])
        
        if result_file.exists():
            with open(result_file, "r") as f:
                data = json.load(f)
            
            latent_explanations[latent_id] = data['explanation'][0]
    
    return latent_explanations


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
