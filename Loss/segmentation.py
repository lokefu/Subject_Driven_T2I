import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
#from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np

#use old:
#pip install transformers==4.22.2 accelerate==0.12.0

#pip install --upgrade transformers
#old: transformers 4.22.2
# fsspec-2023.10.0 huggingface-hub-0.17.3 safetensors-0.4.0 tokenizers-0.14.1 transformers-4.34.1

#import subprocess

# Define the pip command to upgrade the transformers package
#pip_command = ['pip', 'install', '--upgrade', 'transformers']

# Execute the pip command
#subprocess.check_call(pip_command)

#pip install --upgrade accelerate



#pip install --upgrade 'accelerate>=0.20.3'
#old: accelerate 0.12.0
# Define the pip command to upgrade the accelerate package
#pip_command1 = ['pip', 'install', '--upgrade', 'accelerate>=0.20.3']

# Execute the pip command
#subprocess.check_call(pip_command1)

def make_background_white(image, masks):
    outputs = []
    # print(image.size)
    # raise AttributeError
    for mask in masks:
        output = image.copy()
        output[~np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = 255
        outputs.append(output)
    return outputs

def sam(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


    #image_path = os.path.join(image_root, class_name, image_name)
    raw_image = Image.open(image_path)
    w, h = raw_image.size
    #print('raw image size')
    #print(raw_image.size)
    input_points = [[[w // 2, h // 2]]]
    # print(raw_image)
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # masks = mask_generator.generate(image_name)
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores

    #print(len(masks))
    #print(masks[0].shape)
    # print(type(masks))
    # print(masks.unique())
    #print(masks[0].unique())
    #print('scores:', scores)
    input_masks = masks[0].squeeze(0).numpy()
    outputs = make_background_white(np.array(raw_image), input_masks)
    output_image = Image.fromarray(outputs[0])
    return output_image
