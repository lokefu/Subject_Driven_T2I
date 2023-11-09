from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--num_img", type = int, default = 5)
parser.add_argument("--data_dir", type = str, default = '00.jpg')
parser.add_argument("--steps", type = int, default = 30, help="denoising steps.")
parser.add_argument("--guidance", type = int, default = 3, help="A higher guidance scale \
                    value encourages the model to generate images closely linked to the \
                    text prompt at the expense of lower image quality. Guidance scale is \
                    enabled when guidance_scale > 1.")
parser.add_argument("--folder_name", type=str, default='SD_aug',help="Path to generated images.")

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



device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers", #'CompVis/stable-diffusion-v1-4'
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open(args.data_dir)
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)



out = sd_pipe(inp, num_images_per_prompt=args.num_img, guidance_scale=args.guidance, num_inference_steps=args.steps)

for i in range(0,len(out["images"])): #args.num_img
    image_path = os.path.join(args.folder_name, f"aug_image_{i}.png")
    out["images"][i].save(image_path)
    #out[i]["images"][0].save(image_path)
