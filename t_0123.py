import os
import torch
from p_0123 import p_0123
from diffusers.utils import load_image
from carvekit.api.high import HiInterface
import numpy as np
import PIL.Image as Image
from lovely_numpy import lo
import PIL
import cv2



model_id = "zhengdezhi-reprodece" 


def am(a, b, size=256):
    w, h = a.size
    r = Image.new(a.mode, (size, size), b)
    r.paste(a, ((size - w) // 2, (size - h) // 2))
    return r

def lp(it, im):
    image = im.convert('RGB')
    wob = it([image])[0]
    wob = np.array(wob)
    est_seg = wob > 127
    image = np.array(image)
    foreground = est_seg[:, :, -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y:y + h, x:x + w, :]
    image = PIL.Image.fromarray(np.array(image))
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = am(image, (255, 255, 255), size=256)
    image = np.array(image)

    return image

def pe(models, im, preprocess):
    if preprocess:
        im = lp(models['carvekit'], im)
        im = (im / 255.0).astype(np.float32)
    else:
        im = im.resize([256, 256], Image.Resampling.LANCZOS)
        im = np.asarray(im, dtype=np.float32) / 255.0
        aa = im[:, :, 3:4]
        white_im = np.ones_like(im)
        im = aa * im + (1.0 - aa) * white_im
        im = im[:, :, 0:3]

    return im

def ci():
    interface = HiInterface("object", 5,1,'cuda' if torch.cuda.is_available() else 'cpu',640,2048,231,30,5,False)
    return interface


p = p_0123.from_pretrained(model_id, torch_dtype=torch.float16)

p.enable_xformers_memory_efficient_attention()
p.enable_vae_tiling()
p.enable_attention_slicing()
p = p.to("cuda")

num_images_per_prompt = 4

# test inference pipeline
# x y z, Polar angle (vertical rotation in degrees) 	Azimuth angle (horizontal rotation in degrees) 	Zoom (relative distance from center)
query_pose1 = [0.0, 135.0, 0.0]
query_pose2 = [0.0, 90.0, 0.0]
query_pose3 = [0.0, 45.0, 0.0]

input_image1 = load_image("demo/00.jpg")

input_images = [input_image1, input_image1, input_image1]
query_poses = [query_pose1, query_pose2, query_pose3]


pre_images = []
models = dict()
models['carvekit'] = ci()
if not isinstance(input_images, list):
    input_images = [input_images]
for raw_im in input_images:
    input_im = pe(models, raw_im, True)
    H, W = input_im.shape[:2]
    pre_images.append(Image.fromarray((input_im * 255.0).astype(np.uint8)))
input_images = pre_images

images = p(input_imgs=input_images, prompt_imgs=input_images, poses=query_poses, height=H, width=W,
              guidance_scale=3.0, num_images_per_prompt=num_images_per_prompt, num_inference_steps=50).images


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
bs = len(input_images)
i = 0
for obj in range(bs):
    for idx in range(num_images_per_prompt):
        images[i].save(os.path.join(log_dir,f"obj{obj}_{idx}.jpg"))
        i += 1
