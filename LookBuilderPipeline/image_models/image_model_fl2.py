import sys, os
sys.path.insert(0, os.path.abspath('/flux-controlnet-inpaint/src'))

import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxMultiControlNetModel
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
from controlnet_aux import CannyDetector


class ImageModelFlux(BaseImageModel):
    def __init__(self, pose, mask, prompt):
        super().__init__(pose, mask, prompt)

    def generate_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-dev'
        # controlnet_model2 = 'InstantX/FLUX.1-dev-Controlnet-Union'
        controlnet_model2 = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

        controlnet_pose = FluxControlNetModel.from_pretrained(controlnet_model2, torch_dtype=torch.float16)
        controlnet = FluxMultiControlNetModel([controlnet_pose,controlnet_pose])

        pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        pipe.enable_sequential_cpu_offload()

        
        prompt = "beautiful female model on a brightly lit street"
        negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, sleeves, tights, stockings, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image_res = pipe(
            prompt,
            negative_prompt,
            image=image,
            control_image=[canny_image,pose_image],  # try masked canny, fully pose
            control_mode=[0, 4],
            controlnet_conditioning_scale=[0.1,0.9],
            mask_image=mask,
            strength=0.95,
            num_inference_steps=20,
            guidance_scale=7,
            generator=generator,
            joint_attention_kwargs={"scale": 0.8},    
        ).images[0]

        return image_res


    def generate_image_kids(self):
        """
        Generate a new image using the Flux model based on the canny image and prompt.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'

        controlnet_kid = FluxControlNetModel.from_pretrained(controlnet_model2, torch_dtype=torch.float16)
        controlnet = FluxMultiControlNetModel([controlnet_kid])

        pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        pipe.enable_sequential_cpu_offload()

        
        prompt = "beautiful female model on a brightly lit street"
        negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, sleeves, tights, stockings, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image_res = pipe(
            prompt,
            negative_prompt,
            image=image,
            control_image=canny_image,  # full canny image, not masked
            controlnet_conditioning_scale=0.8,
            mask_image=mask,
            strength=0.95,
            num_inference_steps=20,
            guidance_scale=7,
            generator=generator,
            joint_attention_kwargs={"scale": 0.8},    
        ).images[0]

        return image_res
