
from base_image_model import BaseImageModel
from pose.pose import detect_pose
from segment.segment import segment_image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('..')
import cv2
import copy
import torch
import random
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
sys.path.insert(0, os.path.abspath('/ControlNetPlus'))
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline

from controlnet_aux import OpenposeDetector
from transformers import pipeline


class ImageModelSD3(BaseImageModel):
    def __init__(self, pose, mask, prompt):
        super().__init__(pose, mask, prompt)
       
        
        device=torch.device('cuda:0')

        # eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        # Note you should set the model and the config to the promax version manually, default is not the promax version. 
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
        # you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
        controlnet_model = ControlNetModel_Union.from_pretrained("controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16, use_safetensors=True,)


        pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
            "RunDiffusion/Juggernaut-XL-v8", controlnet=controlnet_model,
            # vae=vae,
            torch_dtype=torch.float16,
            # scheduler=ddim_scheduler,
            # scheduler=eulera_scheduler,
        )


        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        # # pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        super.pipe=pipe

    def generate_image(self):
        """
        Generate a new image using Stable Diffusion XL with ControlNet based on the pose, mask, and prompt.
        """
        image=load_image()
        pose_img, pose_image = detect_pose(image_path=self.pose)
        _,maskA,_ = segment_image(image_path=self.mask,inverse=True)
        _,maskB,_ = segment_image(image_path=self.mask)
        
        generator = torch.Generator(device="cuda").manual_seed(np.random.randint(0,10000000))

        # width, height = W, H
        negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves"
        # prompt="photo realistic female fashion model with blonde hair on paris street corner"

        gen_image = self.pipe(prompt=[prompt]*1,
                    image=original_img,
                    mask_image=maskA,
                    control_image_list=[pose_img, 0, 0, 0, 0, 0, 0, maskB], 
                    negative_prompt=[negative_prompt]*1,
                    generator=generator,
                    width=new_width, 
                    height=new_height,
                    num_inference_steps=50,
                    union_control=True,
                    guidance_scale=7.5,
                    union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0, 0, 1]),
                    # crops_coords_top_left=(0, 0),
                    # target_size=(new_width, new_height),
                    # original_size=(new_width * 2, new_height * 2),
                    ).images[0]

        
        return gen_image

