import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

"""
Dear chen, you just need to load raw_image, call segment_function, then get masked_image, 
"""
#raw_image = Image.open(<image_path>)
def segment_function(raw_image):
    w, h = raw_image.size
    print('raw image size')
    print(raw_image.size)
    input_points = [[[w // 2, h // 2]]]
    # print(raw_image)
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # masks = mask_generator.generate(image_name)
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    return masks, scores


def get_masked_image(image, masks):
    outputs = []
    for mask in masks:
        output  = image * np.expand_dims(mask, axis=-1)
        outputs.append(output)
    return outputs