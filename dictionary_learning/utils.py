from datasets import load_dataset, Dataset, load_from_disk
from pathlib import Path
# import zstandard as zstd
import io
import json
import gzip
import os
import yaml

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)

import os
REPO_DIR = os.environ.get('REPO_DIR')

def load_dataset_from_yaml(dataset_name, split=None, streaming=False):

    # Resolve dataset configuration from YAML when column names are not provided
    dataset_config_path = os.path.join(REPO_DIR, "config", "dataset_config.yaml")
    if not os.path.isfile(dataset_config_path):
        raise FileNotFoundError(f"Dataset config YAML not found at {dataset_config_path}")

    with open(dataset_config_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    if dataset_name not in dataset_cfg:
        raise KeyError(f"Dataset '{dataset_name}' not found in {dataset_config_path}")

    ds_entry = dataset_cfg[dataset_name]

    text_column_name = ds_entry.get("text_column", None)
    image_column_name = ds_entry.get("image_column", None)

    # Determine where to load the dataset from
    yaml_path = ds_entry.get("path", None)

    if isinstance(yaml_path, str) and os.path.isdir(yaml_path):
        # Absolute or local on-disk dataset
        if streaming:
            raise ValueError(f"Streaming is not supported for on-disk dataset '{dataset_name}'")
        disk_path = str(Path(yaml_path) / split) if split else yaml_path
        dataset = load_from_disk(disk_path)
    elif yaml_path is None:
        # Load from the Hugging Face Hub using repo id in YAML
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    else:
        raise ValueError(f"Invalid dataset path configuration for '{dataset_name}' in YAML")

    return dataset, {'text': text_column_name, 'image': image_column_name}

def hf_dataset_to_generator(
    dataset_name,
    split="train",
    ratio_of_training_data=1,
    max_examples=None,
    streaming=False,
):

    dataset, column_names_dict = load_dataset_from_yaml(dataset_name, split=split, streaming=streaming)

    text_column_name = column_names_dict['text']
    image_column_name = column_names_dict['image']

    ds_split = dataset

    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")

    if not streaming:
        if ratio_of_training_data != 1:
            ds_split = ds_split.select(range(int(len(ds_split) * ratio_of_training_data)))
        if max_examples is not None:
            ds_split = ds_split.select(range(min(len(ds_split), max_examples)))
        dataset_len = len(ds_split)
    else:
        if ratio_of_training_data != 1 and max_examples is None:
            raise ValueError("streaming with ratio_of_training_data requires max_examples")
        dataset_len = max_examples

    if text_column_name is not None and image_column_name is not None:
        # Contains both text and images
        def gen_text_and_images():
            for idx, x in enumerate(iter(ds_split)):
                if max_examples is not None and idx >= max_examples:
                    break
                yield {"text": [x[text_column_name]], "image": [x[image_column_name]]}
        return gen_text_and_images(), dataset_len

    elif text_column_name is not None:
        # Only contains text
        def gen_text():
            for idx, x in enumerate(iter(ds_split)):
                if max_examples is not None and idx >= max_examples:
                    break
                yield {"text": [x[text_column_name]]}
        return gen_text(), dataset_len

    elif image_column_name is not None:
        # Only contains images
        def gen_images():
            for idx, x in enumerate(iter(ds_split)):
                if max_examples is not None and idx >= max_examples:
                    break
                yield {"image": [x[image_column_name]]}
        return gen_images(), dataset_len

    else:
        raise ValueError(f"Dataset {dataset_name} does not contain 'text' or 'image' columns")

def generator_from_files(files_list: list[str], field="text"):
    """
    Load text data from a list of files that can be either .jsonl or .jsonl.gz format.
    Each entry is assumed to have a 'text' field.
    
    Args:
        files_list (list[str]): List of file paths to .jsonl or .jsonl.gz files
        
    Returns:
        generator: Generator that yields text content from each entry
    """

    def generator():
        for file_path in files_list:
            if file_path.endswith(".jsonl.gz"):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            yield json.loads(line)[field]
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line")
                        except KeyError:
                            print(f"Skipping line without 'text' field")

            elif file_path.endswith(".jsonl"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            yield json.loads(line)[field]
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {file_path}")
                        except KeyError:
                            print(f"Skipping line without 'text' field in {file_path}")

    return generator()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def load_dictionary(base_path: str, device: str) -> tuple:
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config
