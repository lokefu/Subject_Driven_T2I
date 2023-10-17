from huggingface_hub import hf_hub_download

chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth", local_dir = './')

# from segment_anything import build_sam, SamPredictor 
# predictor = SamPredictor(build_sam(checkpoint="ybelkada/segment-anything"))
import os 
print(os.path.exists("checkpoints/sam_vit_b_01ec64.pth"))
print(os.getcwd())
print(os.listdir('./'))
print(os.path.exists("checkpoints"))
print(os.listdir('./checkpoints'))
from segment_anything import build_sam, SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_b_01ec64.pth"))

import os
image_root = 'dreambooth/dataset'
for class_name in os.listdir(image_root):
    for image_name in os.list_dir(os.path.join(image_root, class_name, image_name)):
        
        masks = mask_generator.generate(image_name)
        print(masks.shape)
        print(type(masks))
        print(masks.unique())
