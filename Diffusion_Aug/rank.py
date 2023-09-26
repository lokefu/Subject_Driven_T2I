"""
---
title: CLIP Image-to-Text Embedder
summary: >
 CLIP embedder to transform image embedding into prompt embeddings for stable diffusion
---

# CLIP Text Embedder

This is used to transform image embedding into prompt embeddings for [stable diffusion](../index.html).
It uses HuggingFace Transformers CLIP model.
"""
import shutil
from PIL import Image
from typing import List
from torch import nn
import torch
from transformers import CLIPProcessor, CLIPVisionModel
#from diffusers import StableDiffusionPipeline
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, default = '00.jpg')
parser.add_argument("--folder_name", type=str, default='SD_aug',help="Path to generated images.")
parser.add_argument("--top_k", type = int, default = 5, help = 'number of aug imaged to be selected')
parser.add_argument("--threshold", type = float, help = 'threshold for imaged to be selected, 0-1')

args = parser.parse_args()



class CLIPImage2TextEmbedder(nn.Module):
    """
    ## CLIP Image-to-Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-base-patch32", device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the processed image
        """
        super().__init__()
        self.device = device
        # Load the processor
        self.processor = CLIPProcessor.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = CLIPVisionModel.from_pretrained(version).eval()
        self.transformer = self.transformer.to(self.device)

        self.max_length = max_length

    def forward(self, image_path: List[str]):
        """
        :image_path: is the list of image_path to embed
        """
        # process image
        images = []
        for path in image_path:
            image = Image.open(path)
            images.append(image)
        batch_encoding = self.processor(text = None, images=images, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get pixel values
        tokens = batch_encoding["pixel_values"].to(self.device)
        tokens = tokens.cuda()
        # Get CLIP embeddings
        return self.transformer(pixel_values=tokens).last_hidden_state

class CLIPImageRanker:
    def __init__(self, version: str = "openai/clip-vit-base-patch32", device="cuda:0", max_length: int = 77):
        self.clip_embedder = CLIPImage2TextEmbedder(version, device, max_length)

    def rank_images(self, original_image_path: List[str], generated_images_path: List[str]) -> List[str]:
        
        original_embedding = self.clip_embedder.forward(original_image_path)
        generated_embeddings = self.clip_embedder.forward(generated_images_path)

        similarity_scores = nn.functional.cosine_similarity(original_embedding, generated_embeddings).mean(dim=-1)
        sorted_indices = torch.argsort(similarity_scores, descending=True)

        if args.threshold is not None:
            threshold_tensor = torch.tensor(args.threshold, device='cuda:0')
            selected_indices = []
            c = min(threshold_tensor, max(similarity_scores))
            print('threshold: ',threshold_tensor)
            print('max cos_similarity: ',max(similarity_scores))
            for idx in sorted_indices:
                if similarity_scores[idx] >= c:
                    selected_indices.append(idx)
                else:
                    break
                # Assuming the sorted_indices are in descending order, \
                # you can break the loop once the threshold is not met.
        else: #top K
            selected_indices = sorted_indices[:top_k]
        
        output_images = [generated_images_path[idx] for idx in selected_indices]

        return output_images

# Usage example:
original_image = [args.data_dir]
image_files = []
for file_name in os.listdir(args.folder_name):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        image_files.append(os.path.join(args.folder_name, file_name))

# image_files = ["00.jpg","01.jpg", "02.jpg"]
top_k = args.top_k


#output selected
# Specify the name of the new folder
if args.threshold is not None:
    folder_name = 'TH' + args.folder_name #threshold
else:
    folder_name = 'TO' + args.folder_name #top

# Get the current working directory
current_directory = os.getcwd()

# Combine the current directory path with the new folder name
new_folder_path = os.path.join(current_directory, folder_name)

# Check if the folder already exists
if not os.path.exists(new_folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(new_folder_path)
    print("Folder created successfully!")
else:
    print("Folder already exists!")
    shutil.rmtree(new_folder_path)
    print("Folder deleted successfully!")
    os.makedirs(new_folder_path)
    print("Folder created again!")

clip_ranker = CLIPImageRanker(device="cuda:0")
top_k_images_path = clip_ranker.rank_images(original_image, image_files)
#print(top_k_images_path)




for i in top_k_images_path:
    # Copy the selected image to the destination folder
    shutil.copy(i, new_folder_path)
