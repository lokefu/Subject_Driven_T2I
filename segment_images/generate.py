import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# input_points = [[[450, 600]]]  # 2D location of a window in the image

# inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
# outputs = model(**inputs)

# masks = processor.image_processor.post_process_masks(
#     outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
# )
# scores = outputs.iou_scores
# print(len(masks))
# print(masks[0].unique)

# mask_generator = SamAutomaticMaskGenerator(model)
def get_masked_image(image, masks):
    outputs = []
    for mask in masks:
        output  = image * np.expand_dims(mask, axis=-1)
        outputs.append(output)
    return outputs

def make_background_white(image, masks):
    outputs = []
    # print(image.size)
    # raise AttributeError
    for mask in masks:
        output = image.copy()
        output[~np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = 255
        outputs.append(output)
    return outputs

import os
image_root = 'dreambooth/test'
save_root = 'dreambooth/one_data_seg'
if not os.path.exists(save_root):
    os.makedirs(save_root)
for class_name in os.listdir(image_root):
    if '.txt' in class_name:
        continue
    for image_name in os.listdir(os.path.join(image_root, class_name)):
        image_path = os.path.join(image_root, class_name, image_name)
        raw_image = Image.open(image_path)
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

        print(len(masks))
        print(masks[0].shape)
        # print(type(masks))
        # print(masks.unique())
        print(masks[0].unique())
        print('scores:', scores)
        input_masks = masks[0].squeeze(0).numpy()
        # outputs = get_masked_image(np. array(raw_image), input_masks)
        outputs = make_background_white(np. array(raw_image), input_masks)
        for i,output in enumerate(outputs):
            if not os.path.exists(os.path.join(save_root, class_name)):
                os.mkdir(os.path.join(save_root, class_name))
            Image.fromarray(output).save(os.path.join(save_root, class_name, image_name.split('.')[0] + '_' + str(i) + '.jpg'))
        # break
