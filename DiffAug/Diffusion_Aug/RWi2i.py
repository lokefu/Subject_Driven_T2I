import torch
from PIL import Image
import shutil
from diffusers import StableDiffusionImg2ImgPipeline
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_img", type = int, default = 5)
parser.add_argument("--data_dir", type = str, default = '00.jpg')
parser.add_argument("--steps", type = int, default = 30, help="denoising steps.")
parser.add_argument("--guidance", type = float, default = 3, help="A higher guidance scale \
                    value encourages the model to generate images closely linked to the \
                    text prompt at the expense of lower image quality. Guidance scale is \
                    enabled when guidance_scale > 1.")
#guidance: guidance from text, default at 7.5
parser.add_argument("--folder_name", type=str, default='SD_aug',help="Path to generated images.")
parser.add_argument("--label", type = str, default = '')
parser.add_argument("--strength", type = float, default = 0.2)
parser.add_argument("--total_num_img", type = int, default = 100)
#strength: noise to add: 1 is max - ignore input image with all noise

args = parser.parse_args()

# Specify the name of the new folder
folder_name = args.folder_name

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


device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)


init_image = Image.open(args.data_dir).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = args.label
count=0
for j in range(0, int(args.total_num_img / args.num_img)+1):
    images = pipe(prompt=prompt, image=init_image, strength=args.strength, guidance_scale=args.guidance, \
                  num_images_per_prompt=args.num_img, num_inference_steps=args.steps).images

    for i in range(0,len(images)): #args.num_img
        image_path = os.path.join(args.folder_name, f"aug_image_rwi2i_{j}_{i}.png")
        images[i].save(image_path)
        count+=1
print('number of images generated: ', count)
#images[0].save("i-1.png")