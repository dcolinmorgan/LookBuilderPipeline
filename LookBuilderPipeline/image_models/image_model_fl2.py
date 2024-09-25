import sys, os
# sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
sys.path.insert(0,os.path.abspath('external_deps/flux-controlnet-inpaint/src'))
sys.path.insert(1,os.path.abspath('external_deps/ControlNetPlus'))
import torch
import numpy as np
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxMultiControlNetModel
import requests
import torch.nn as nn
from base_image_model import BaseImageModel
from LookBuilderPipeline.resize.resize import resize_images
import base64
from io import BytesIO
import os
import uuid

def closest_size_divisible_by_8(size):
    if size % 8 == 0:
        return size
    else:
        return size + (8 - size % 8) if size % 8 > 4 else size - (size % 8)

class ImageModelFlux(BaseImageModel):
    def __init__(self, image, pose, mask, prompt):
        super().__init__(image, pose, mask, prompt)

    def generate_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        ### init before loading model
        negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves"
        # prompt="photo realistic female fashion model with blonde hair on paris street corner"
        orig_image=load_image(self.image)
        pose_image=load_image(self.pose)
        mask_image=load_image(self.mask)
        width,height=orig_image.size
        if width // 8 != 0 or height // 8 != 0:
            print("resizing images")
            if width > height:
                newsize=closest_size_divisible_by_8(width)
            else:
                newsize=closest_size_divisible_by_8(height)

            orig_image=resize_images(orig_image,newsize,square=False)
            pose_image=resize_images(pose_image,newsize,square=False)
            mask_image=resize_images(mask_image,newsize,square=False)
        width,height=orig_image.size
        num_inference_steps=30
        guidance_scale=5
        controlnet_conditioning_scale=1
        seed=np.random.randint(0,100000000)
        generator = torch.Generator(device="cpu").manual_seed(seed)


        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model2 = 'InstantX/FLUX.1-dev-Controlnet-Union'

        controlnet_pose = FluxControlNetModel.from_pretrained(controlnet_model2, torch_dtype=torch.float16)
        # controlnet = FluxMultiControlNetModel([controlnet_pose,controlnet_pose])

        pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet_pose, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        pipe.enable_sequential_cpu_offload()

        

        image_res = pipe(
            self.prompt,
            negative_prompt,
            image=orig_image,
            control_image=pose_image,
            control_mode=4,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=0.9,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            joint_attention_kwargs={"scale": 1},    
        ).images[0]

        # Save image to a file
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "generated_images", filename)
        image_res.save(save_path)

        return jsonify(image_res)


    def generate_image_kids(self):
        """
        Generate a new image using the Flux model based on the canny image and prompt.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-dev'
        # controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'

        controlnet_kid = FluxControlNetModel.from_pretrained(controlnet_model2, torch_dtype=torch.float16)
        controlnet = FluxMultiControlNetModel([controlnet_kid])

        pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        pipe.enable_sequential_cpu_offload()

        
        # prompt = "beautiful female model on a brightly lit street"
        negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, sleeves, tights, stockings, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image_res = pipe(
            self.prompt,
            negative_prompt,
            image=self.image,
            control_image=self.pose_image,  # full canny image, not masked
            controlnet_conditioning_scale=0.8,
            mask_image=self.mask,
            strength=0.95,
            control_mode=4,
            num_inference_steps=20,
            guidance_scale=7,
            generator=generator,
            joint_attention_kwargs={"scale": 0.8},    
        ).images[0]

        return image_res
