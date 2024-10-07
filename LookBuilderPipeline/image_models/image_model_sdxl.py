import sys, os
import uuid
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 

import torch.nn as nn
from base_image_model import BaseImageModel
from LookBuilderPipeline.LookBuilderPipeline.resize import resize_images
from LookBuilderPipeline.LookBuilderPipeline.segment import segment_image

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
        
    def prepare_image(self):
        """
        Prepare the pose and mask images to generate a new image using the Flux model.
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
        self.width,self.height=orig_image.size
        

        ## model requires inverse mask too
        _, mask_image_inv_b,_=segment_image(self.image,inverse=True)  # clothes items not in mask
        _, mask_image_inv_a,_=segment_image(self.image,inverse=False)  # clothes items in mask
        
    def prepare_model(self):  # Set up the pipeline
        """
        Prepare model to generate a new image using the Flux model.
        """
        # device = torch.device('cuda:0')
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
        
        self.num_inference_steps=30
        self.guidance_scale=5
        self.controlnet_conditioning_scale=1
        self.seed=np.random.randint(0,100000000)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def generate_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        image_res = self.pipe(
            prompt=self.prompt,
            image=self.mask_image_inv_b,
            mask_image=self.mask_image,
            control_image_list=[self.pose_image, 0, 0, 0, 0, 0, 0, self.mask_image_inv_a],
            negative_prompt=self.negative_prompt,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            union_control=True,
            guidance_scale=guidance_scale,
            union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0, 0, 1]),
        ).images[0]
        
        # Save image to a file
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "generated_images", filename)
        image_res.save(save_path)

        return jsonify(image_res)

if __name__ == "__main__":    
    image_model = ImageModelSDXL(image_path, image_path, image_path, prompt_text)
    image_model.prepare_image()
    image_model.prepare_model()
    image_model.generate_image()
