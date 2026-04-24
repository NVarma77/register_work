# Functions to process the inputs for the model
from PIL import Image
import torch as t

def reshape_images(input_ids, img_size=(336, 336)):
    """
    Reshape images to the specified size.
    """
    assert input_ids['image'] is not None, "Input ids must contain an image"
    assert isinstance(input_ids['image'][0][0], Image.Image), "Image must be a PIL Image"
    for i in range(len(input_ids['image'])):
        for j in range(len(input_ids['image'][i])):
            input_ids['image'][i][j] = input_ids['image'][i][j].resize(img_size, Image.LANCZOS)
    return input_ids

# Useful functions
def is_instruct_model(model_name):
    return '-it' in model_name.lower() or 'instruct' in model_name.lower() or 'llava-onevision' in model_name.lower()
def is_gemma_model(model_name):
    return 'gemma-3' in model_name and 'it' in model_name and not '1b' in model_name

def create_messages(inputs_dict):
    """
    Create a list of messages from a list of images, texts, or combinations of them.
    """
    messages_list = []
    image_inputs = inputs_dict.get('image', []) or []
    text_inputs = inputs_dict.get('text', []) or []

    max_items = max(len(image_inputs), len(text_inputs))
    for i in range(max_items):
        content_items = []
        
        # Add image if available
        if i < len(image_inputs):
            # image_inputs is a list of PIL images
            for image in image_inputs[i]:
                content_items.append({"type": "image", "image": image})
        
        # Add text if available
        if i < len(text_inputs):
            for text in text_inputs[i]:
                content_items.append({"type": "text", "text": text})
        
        if content_items:  # Only add message if there's content
            messages_list.append([{
                "role": "user",
                "content": content_items
            }])
    return messages_list

def internvl_processing(inputs_dict, cfg):
    """
    Process the messages list for InternVL models.
    """
    from utils.internvl_utils import load_image
    # For Top-K explanation
    # inputs_dict['text'] = [['describe the image', 'describe the image']]
    # inputs_dict['image'] = [[img_1,img_2...], [img_1,img_2...]]

    # For Raw Steering
    # inputs_dict['text'] = [['describe the image'], ['describe the image']]
    # inputs_dict['image'] = [[blank_img], [blank_img]]

    # print('inputs_dict', inputs_dict)
    # print()
    # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    list_pixels = []
    num_patches_list = []
    # question = '<image><image>\nDescribe the two images in detail.'
    questions = []
    for i, images_batch_elem in enumerate(inputs_dict['image']):
        question_batch_idx = ''
        for image in images_batch_elem:
            pixel_values = load_image(image, max_num=1).to(cfg.dtype).to(cfg.device)
            list_pixels.append(pixel_values)
            #question_batch_idx += '<image>'
        #     question_batch_idx = '<image>' + inputs_dict['text'][i][0]
        # questions.append(question_batch_idx)
        # This only works if max_num=1 in load_image
        num_patches_list.append(len(images_batch_elem))
    pixel_values = t.cat(list_pixels, dim=0)

    return {'pixel_values': pixel_values, 'questions': inputs_dict['text'], 'num_patches_list': num_patches_list}

def processing(inputs_dict, processor, cfg, tokenizer=None):
    """
    Process the messages list for Gemma models.
    """
    contains_images = inputs_dict['image'] is not None

    if is_gemma_model(cfg.lm_model_name.lower()):
        messages_list = create_messages(inputs_dict)
        if contains_images:
            processor_output = processor.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt", padding=True,
                                                        #do_pan_and_scan=False # TODO: handle this as processor_kwargs
                                                        ).to(cfg.dtype).to(cfg.device)
            if cfg.submodel == 'enc' and cfg.get_full_model == False:
                # If working with the encoder (siglip), we only need the pixel_values
                pixel_values = processor_output['pixel_values']
                processor_output = {'pixel_values': pixel_values}
        
        else:
            text = tokenizer.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=False)
            # We need to remove the BOS token so that the tokenizer doesn't add it again
            text = [text_[len(tokenizer.bos_token):] for text_ in text]
            if not contains_images:
                # If we have no images, we tokenize the text with the context length
                processor_output = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=cfg.context_length,
                    padding=True,
                    truncation=True,
                    return_overflowing_tokens=False
                ).to(cfg.device)
        
    elif 'paligemma2' in cfg.lm_model_name.lower():
        # PaliGemma models uses processor directly accepting a list of texts and a list of images
        # These lists length must be the same and are the batch_size
        assert contains_images, "PaliGemma models require images"
        processor_output = processor(text=inputs_dict['text'], images=inputs_dict['image'], return_tensors="pt").to(cfg.dtype).to(cfg.device)
        if cfg.submodel == 'enc' and cfg.get_full_model == False:
            # If working with the encoder (siglip), we only need the pixel_values
            pixel_values = processor_output['pixel_values']
            processor_output = {'pixel_values': pixel_values}
    elif any(x in cfg.lm_model_name.lower() for x in ('qwen2-vl', 'qwen2.5-vl', 'mimo-vl', 'aloe-vision-7b')):
        
        from qwen_vl_utils import process_vision_info
        if contains_images:
            # If we have images, we need reshape them
            inputs_dict = reshape_images(inputs_dict, img_size=(cfg.model_img_size, cfg.model_img_size))
        messages_list = create_messages(inputs_dict)
        text = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        if contains_images:
            image_inputs, video_inputs = process_vision_info(messages_list)
            processor_output = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                max_length=cfg.context_length,
                padding=True,
                return_tensors="pt",
            ).to(cfg.dtype).to(cfg.device)

        else:
            # If we have no images, we tokenize the text with the context length
            processor_output = tokenizer(
                text,
                return_tensors='pt',
                max_length=cfg.context_length,
                padding=True,
                truncation=True,
                return_overflowing_tokens=False
            ).to(cfg.device)

    elif 'internvl' in cfg.lm_model_name.lower():
        processor_output = internvl_processing(inputs_dict, cfg)
    else:
        # For other models, raise an error as processing is not implemented
        raise NotImplementedError(f"Processing for model {cfg.model_name} is not implemented. Only Gemma models are currently supported.")
    return processor_output

def tokenize_batch_vlm(inputs, processor, cfg, tokenizer=None):
    """
    Tokenize a batch of inputs for VLM models.
    """
    def load_inputs_dict(inputs, model_type='vllm'):
        # inputs is a list of dicts, e.g. [{'text': prompt, 'image': top_image}, ...]
        # Check if we have text and image inputs
        text_inputs = [input.get('text') for input in inputs]
        if all(x is None for x in text_inputs):
            if model_type == 'vllm':
                text_inputs = ['']*len(inputs)
            else:
                text_inputs = None
        image_inputs = [input.get('image') for input in inputs]
        if all(x is None for x in image_inputs):
            image_inputs = None

        inputs_dict = {
            'text': text_inputs,
            'image': image_inputs
        }
        # For Top-K explanation
        # inputs_dict['text'] = [['describe the image'], ['describe the image']]
        # inputs_dict['image'] = [[img_1,img_2...], [img_1,img_2...]]

        # For Raw Steering
        # inputs_dict['text'] = [['describe the image'], ['describe the image']]
        # inputs_dict['image'] = [[blank_img], [blank_img]]

        return inputs_dict

    inputs_dict = load_inputs_dict(inputs)

    if cfg.model_type == 'vision':
        # With vision-only models (e.g. ViT) we feed the model only with images
        flat_images = [img for batch in inputs_dict['image'] for img in batch]
        return processor(images=flat_images, return_tensors="pt").to(cfg.dtype).to(cfg.device)
    else:
        processor_output = processing(inputs_dict, processor, cfg, tokenizer)
        return processor_output

def tokenized_batch(inputs, tokenizer, cfg, processor=None):
    """
    Return a batch of tokenized inputs.
    Inputs is a list of dicts, e.g. [{'text': prompt, 'image': image}, ...]
    """
    if processor is not None:
        # If processor has 'tokenizer' attribute, use it as tokenizer
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = None
    return tokenize_batch_vlm(inputs, processor, cfg, tokenizer)