from .base_image_model import BaseImageModel
from LookBuilderPipeline.pose.pose import detect_pose
from LookBuilderPipeline.segment.segment import segment_image

import os
import sys
import torch
import numpy as np
from diffusers.utils import load_image
import sys, os

sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_inpaint_sd_xl import (
    StableDiffusionXLControlNetUnionInpaintPipeline
)
from huggingface_hub import snapshot_download

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image,pose, mask, prompt):
        super().__init__(image, pose, mask, prompt)
        
        device = torch.device('cuda:0')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        self.pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v8",
            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )

        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.enable_model_cpu_offload()

    def generate_image(self):
        """
        Generate a new image using Stable Diffusion XL with ControlNet based on
        the pose, mask, and prompt.
        """
        image = load_image()
        pose_img, _ = detect_pose(image_path=self.pose)
        _, mask_a, _ = segment_image(image_path=self.mask, inverse=True)
        _, mask_b, _ = segment_image(image_path=self.mask)
        
        generator = torch.Generator(device="cuda").manual_seed(
            np.random.randint(0, 10000000)
        )

        negative_prompt = (
            "ugly, bad quality, bad anatomy, deformed body, deformed hands, "
            "deformed feet, deformed face, deformed clothing, deformed skin, "
            "bad skin, leggings, tights, sunglasses, stockings, pants, sleeves"
        )

        gen_image = self.pipe(
            prompt=[self.prompt],
            image=image,
            mask_image=mask_a,
            control_image_list=[pose_img, 0, 0, 0, 0, 0, 0, mask_b],
            negative_prompt=[negative_prompt],
            generator=generator,
            num_inference_steps=50,
            union_control=True,
            guidance_scale=7.5,
            union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0, 0, 1]),
        ).images[0]
        
        return gen_image

