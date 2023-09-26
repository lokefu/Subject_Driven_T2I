from PIL import Image
import torch.nn as nn
import torch
from typing import List
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import numpy as np 

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, emb1, emb2):
        similarity = nn.functional.cosine_similarity(emb1, emb2)
        loss = 1 - similarity.mean()  # Minimize the difference by maximizing the similarity
        return loss


class CLIP_TextImages_Loss(nn.Module):
    def __init__(self):
        super(CLIP_TextImages_Loss, self).__init__()

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.cosine_similarity_loss = CosineSimilarityLoss()

    def forward(self, text, images_path: List[str]):
        # Check if images is a list
        if not isinstance(images_path, list):
            raise ValueError("Images must be a list.")
        # Check if the text is of type string
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")

        losses = []  # List to store individual losses

        for path in images_path:
            image = Image.open(path)
            # Get the embeddings of text and image
            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            text_emb = outputs.text_embeds
            image_emb = outputs.image_embeds

            # Compute and store the loss
            loss = self.cosine_similarity_loss(text_emb, image_emb)
            losses.append(loss)

        # Compute the sum of all losses
        total_loss = sum(losses)
        return total_loss

def load_image(p):
   '''     
   Function to load images from a defined path: string of path     
   '''
   return Image.open(p).convert('RGB').resize((512,512))
def pil_to_latents(image):
    '''     
    Function to convert image to latents     
    '''     
    init_image = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    #init_image = init_image
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  
def latents_to_pil(latents):     
    '''     
    Function to convert latents to images     
    '''     
    latents = (1 / 0.18215) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")     
    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images