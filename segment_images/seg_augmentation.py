from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

class RandomColorShift:
    def __init__(self, shift_range=(-50, 50)):
        self.shift_range = shift_range

    def __call__(self, img):
        # 为每个通道随机选择一个值
        shift_values = np.random.randint(-100, 101, (3,))
        img_np = np.asarray(img).astype(np.int16)

        # 分别为R, G, B通道加上偏移量
        for i in range(3):
            img_np[:,:,i] += shift_values[i]

        shifted_img = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(shifted_img)

def apply_color_transforms_to_folder_for_n_epochs(input_folder, output_folder, epochs):
    # 定义 color jitter 和自定义的色彩偏移变换
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        RandomColorShift()
    ])

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for epoch in range(epochs):
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            img = Image.open(image_path)
            
            # 应用色彩变换
            transformed_img = transform(img)
            
            # 为输出图像名添加 epoch 信息，以便于区分
            output_image_name = f"epoch{epoch + 1}_transformed_{image_name}"
            output_path = os.path.join(output_folder, output_image_name)
            
            transformed_img.save(output_path)

        print(f"Completed epoch {epoch + 1}/{epochs}")

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

def change_target_color_and_save(input_image_path, mask, output_folder):
    """用于改变目标颜色并将其放回原始背景的函数"""

    # 读取原始图像
    original_image = Image.open(input_image_path)

    # 定义色彩变换
    transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        RandomColorShift()
    ])

    # 在应用颜色变换之前，找出纯黑和纯白的像素
    original_image_np = np.array(original_image)
    black_pixels = (original_image_np == [0, 0, 0]).all(axis=-1)
    white_pixels = (original_image_np == [255, 255, 255]).all(axis=-1)

    # 应用色彩变换到原始图像
    transformed_image = transform(original_image)
    transformed_image_np = np.array(transformed_image)

    # 恢复纯黑和纯白的像素
    transformed_image_np[black_pixels] = [0, 0, 0]
    transformed_image_np[white_pixels] = [255, 255, 255]

    # 根据遮罩形状，处理遮罩
    mask_np = mask.numpy()[0, 0, :, :].astype(bool)

    # 利用遮罩，将变换后的目标与原始背景合并
    combined_img_np = np.where(mask_np[..., None], transformed_image_np, original_image_np)

    combined_image = Image.fromarray(combined_img_np)

    # 保存合并后的图像
    output_path = os.path.join(output_folder, os.path.basename(input_image_path))
    combined_image.save(output_path)





# 调用这个函数
input_folder = 'dog6'
output_folder = 'dog_out'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    
    masks, _ = segment_function(Image.open(image_path))
    
    # 假设 segment_function 返回的是一系列的遮罩和分数，我们只取分数最高的那个
    mask = masks[0]  # 根据实际情况，您可能需要对这个索引进行调整
    print(f"Mask shape before squeeze: {mask.shape}")

    
    change_target_color_and_save(image_path, mask, output_folder)