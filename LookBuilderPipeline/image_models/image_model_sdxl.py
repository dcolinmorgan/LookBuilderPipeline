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
from LookBuilderPipeline.segment.segment import segment_image
def closest_size_divisible_by_8(size):
    if size % 8 == 0:
        return size
    else:
        return size + (8 - size % 8) if size % 8 > 4 else size - (size % 8)

sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_inpaint_sd_xl import (
    StableDiffusionXLControlNetUnionInpaintPipeline
)
from huggingface_hub import snapshot_download

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image,pose, mask, prompt):
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

        ## model requires inverse mask too
        _, mask_image_inv_b,_=segment_image(self.image,inverse=True)  # clothes items not in mask
        _, mask_image_inv_a,_=segment_image(self.image,inverse=False)  # clothes items in mask
        
        # Set up the pipeline
        
        # device = torch.device('cuda:0')
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Download and set up the ControlNet model
        snapshot_download(
            repo_id="xinsir/controlnet-union-sdxl-1.0",
            local_dir='controlnet-union-sdxl-1.0'
        )
        controlnet_model = ControlNetModel_Union.from_pretrained(
            "controlnet-union-sdxl-1.0-promax",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # Set up the pipeline
        pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v8",
            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )

        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.enable_model_cpu_offload()


        image_res = self.pipe(
            prompt=self.prompt,
            image=mask_image_inv_b,
            mask_image=mask_image,
            control_image_list=[pose_image, 0, 0, 0, 0, 0, 0, mask_image_inv_a],
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            union_control=True,
            guidance_scale=guidance_scale,
            union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0, 0, 1]),
        ).images[0]
        
        # Save image to a file
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "generated_images", filename)
        image_res.save(save_path)

        return jsonify(image_res)

