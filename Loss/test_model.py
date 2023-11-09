from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler
import os
from PIL import Image
import shutil

model_path = "./model_base"  
#model_path = "./model1"
#prompt = "a sxc dog in the jungle"

unique_token = 'sxc'
class_token = 'dog'
num_per_prompt = 5 #num of image should be generated for each prompt
num_images_per_prompt = 3 #num of image each pipeline generate
# Directory where images will be saved

# Specify the name of the new folder
folder_name = 'model_base_dog6'
#folder_name = 'model1_dog6'

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



live_prompt_list = [
'a {0} {1} in the jungle'.format(unique_token, class_token),
'a {0} {1} in the snow'.format(unique_token, class_token),
'a {0} {1} on the beach'.format(unique_token, class_token),
'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
'a {0} {1} with a city in the background'.format(unique_token, class_token),
'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
'a {0} {1} wearing a red hat'.format(unique_token, class_token),
'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
'a {0} {1} in a chef outfit'.format(unique_token, class_token),
'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
'a {0} {1} in a police outfit'.format(unique_token, class_token),
'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
'a red {0} {1}'.format(unique_token, class_token),
'a purple {0} {1}'.format(unique_token, class_token),
'a shiny {0} {1}'.format(unique_token, class_token),
'a wet {0} {1}'.format(unique_token, class_token),
'a cube shaped {0} {1}'.format(unique_token, class_token)
]

object_prompt_list = [
'a {0} {1} in the jungle'.format(unique_token, class_token),
'a {0} {1} in the snow'.format(unique_token, class_token),
'a {0} {1} on the beach'.format(unique_token, class_token),
'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
'a {0} {1} with a city in the background'.format(unique_token, class_token),
'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
'a {0} {1} floating on top of water'.format(unique_token, class_token),
'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
'a {0} {1} on top of a mirror'.format(unique_token, class_token),
'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
'a {0} {1} on top of a white rug'.format(unique_token, class_token),
'a red {0} {1}'.format(unique_token, class_token),
'a purple {0} {1}'.format(unique_token, class_token),
'a shiny {0} {1}'.format(unique_token, class_token),
'a wet {0} {1}'.format(unique_token, class_token),
'a cube shaped {0} {1}'.format(unique_token, class_token)
]

pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=True,
        )
    )

def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy
pipe = pipe.to("cuda")

prompt_list = live_prompt_list


c = 0
for j in range(0,len(prompt_list)):
    print(f'the_{j}th prompt')
    prompt = prompt_list[j]
    prompt_folder_name = str(j)
    prompt_folder_path = os.path.join(new_folder_path, prompt_folder_name)
       # Check if the image file already exists
    if os.path.exists(prompt_folder_path):
        print('Skipping creating folder as it already exists')
        continue
    else:
        os.makedirs(prompt_folder_path)

    for m in range(0, int(num_per_prompt / num_images_per_prompt)+1):
        images = pipe(prompt=prompt, num_inference_steps=30, num_images_per_prompt=num_images_per_prompt).images
        
        for k in range(0, len(images)):
            c = m * num_images_per_prompt
            idx = c + k
            image_path = os.path.join(prompt_folder_path, f'image_{idx}.jpg')
            images[k].save(image_path)




#images[0].save("i-1.png")