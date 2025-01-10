import sys, os, gc, requests
import uuid
import logging
import time
import torch
import numpy as np
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from transformers import pipeline 
import torch.nn as nn
from compel import Compel, ReturnedEmbeddingsType
import glob
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from PIL import Image
import io
# Import required components from diffusers
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image, **kwargs):
        # Initialize the SDXL image model if not yet initialized
        super().__init__(image, **kwargs)

        # Set default values not used in prepared by base_image_model.py

        self.negative_prompt = kwargs.get('negative_prompt', "extra clothes, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, leggings, tights, sunglasses, stockings, pants, sleeves")


    def prepare_model(self):
        """
        Prepare model to generate a new image using the diffusion model.
        """
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # since we only have one GPU
        
        # Load the controlnet
        controlnet_model = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16,
        )

        # Load the pipeline + CN
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(#from_pretrained(
            "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",

            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )

        # Configure pipeline settings
        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.to(self.device)
        
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        

        # Activate compel for long prompts
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        if self.LoRA==None:
            ## compel for prompt embedding allowing >77 tokens
            self.conditioning, self.pooled = compel(self.prompt)
        else:
            self.conditioning, self.pooled = compel(self.prompt+self.LoRA)

            


    def generate_image(self):
        """
        Generate a new image using the diffusion model based on the pose, mask and prompt.
        """
        start_time = time.time()

        import cv2
        from PIL import ImageFilter
        mask_image = self.mask.filter(ImageFilter.GaussianBlur(radius=5))  # Adjust the radius
        mask_array = np.array(mask_image)
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size
        expanded_mask = cv2.dilate(mask_array, kernel, iterations=1)
        self.mask = Image.fromarray(expanded_mask)
        
        
        # Ensure self.image, self.mask, and self.pose are bytes-like objects
        try:        
            self.image = Image.open(io.BytesIO(self.image))
        except:
            pass
        try:        
            self.mask = Image.open(io.BytesIO(self.mask))
        except:
            pass
        try:        
            self.pose = Image.open(io.BytesIO(self.pose))
        except:
            pass

        # Generate the image using the pipeline
        image_res = self.pipe(
            # prompt=self.prompt,
            prompt_embeds=self.conditioning,
            pooled_prompt_embeds=self.pooled,
            image=self.image,
            padding_mask_crop=1,
            original_size=(self.res,self.res),
            mask_image=self.mask,
            control_image=self.pose,
            negative_prompt=self.negative_prompt,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            strength=self.strength,
        ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.clear_mem()
        
        try:        
            image_res = Image.open(io.BytesIO(image_res))
        except:
            pass
        return image_res #, save_path1
    

    def clear_mem(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageModelSDXL")
    parser.add_argument("--negative_prompt", default="dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves", help="Negative prompt")
    args = parser.parse_args()

    image_model = ImageModelSDXL(
        args.image_path, 
    )

    
    if args.benchmark==False:
        image_model.prepare_model()
        generated_image, generated_image_path = image_model.generate_image()
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clear_mem()
