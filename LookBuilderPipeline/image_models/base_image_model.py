import numpy as np
import logging
import torch
from diffusers.utils import load_image
from LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.pose import detect_pose
from LookBuilderPipeline.utils.resize import resize_images
from PIL import Image, ImageOps
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import textwrap
import io
class BaseImageModel:
    def __init__(self, image, pose=None, mask=None, outfit=None, **kwargs):
        """
        Initialize the image model with common inputs.
        
        Args:
            pose (object): The detected pose generated earlier in the pipeline.
            clothes (object): The segmented clothes (outfit) generated earlier.
            mask (object): The mask generated earlier that defines the boundaries of the outfit.
            prompt (str): The text prompt to guide the image generation (e.g., style or additional details).
        """        
        if torch.backends.mps.is_available():
            device = torch.device('mps')  # Use MPS if available
        elif torch.cuda.is_available():
            device = torch.device('cuda')  # Use CUDA if available
        else:
            device = torch.device('cpu')
        logging.info(f"Device= {device}")

        if isinstance(image,str):
            self.image = load_image(image)
        elif isinstance(image,bytes):
            self.image = Image.open(io.BytesIO(image))
        else:
            self.image = image
        self.res = kwargs.get('res', 1024)
        image=resize_images([self.image],self.res,square=True)[0]
        self.original_image=image #resize_images([self.original_image],self.res,square=True)[0]
        self.image=image.copy()
        
        if pose is None:
            self.original_pose = detect_pose(image_path=image)
        else:
            self.original_pose = pose
        if mask is None:
            _, self.original_mask = segment_image(image_path=image,inverse=True)
        else:
            self.original_mask = mask
        if outfit is None:
            self.original_outfit, _ = segment_image(image_path=image,inverse=False)
        else:
            self.original_outfit = outfit
        
        self.image,self.mask,self.pose,self.outfit=resize_images([self.original_image,self.original_mask,self.original_pose,self.original_outfit],self.res,square=True)
        logging.info(f"Base parameters set")
        
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 6)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1.0)
        self.seed = kwargs.get('seed', 420042)
        self.strength = kwargs.get('strength', 1.0)
        self.LoRA = kwargs.get('LoRA', None)
        self.prompt = kwargs.get('prompt')
        
        # self.negative_prompt = kwargs.get('negative_prompt', "dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves")
        self.benchmark = kwargs.get('benchmark', False)
        self.control_guidance_start=kwargs.get('control_guidance_start', 0)
        self.control_guidance_end=kwargs.get('control_guidance_end', 1)
        self.lora_weight=kwargs.get('lora_weight', 1.0)
        self.device = device
        logging.info(f"BaseImageModel initialized")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BaseImageModel")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    args = parser.parse_args()
    image_model = BaseImageModel(
            args.image_path)
