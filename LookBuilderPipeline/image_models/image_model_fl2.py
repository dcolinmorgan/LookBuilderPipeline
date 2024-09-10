from image_models.base_image_model import BaseImageModel
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
# from PIL import Image

class ImageModelFlux(BaseImageModel):
    def __init__(self, pose, mask, prompt):
        super().__init__(pose, mask, prompt)

    def generate_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model_union = 'InstantX/FLUX.1-dev-Controlnet-Union'

        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.float16, token=token)
        controlnet = FluxMultiControlNetModel([controlnet_union])

        pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16, token=token)
        # pipe.to("cuda")
        # Enable CPU offloading
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        
        # prompt = 'A beautiful female model in paris.'
        control_mode_depth = 2 # defines the depth of the controlnet
        control_mode_pose = 4 # defines the pose of the controlnet

        width, height = self.pose.size

        out_img = pipe(
                self.prompt, 
                control_image=[self.mask, self.pose],
                control_mode=[control_mode_depth, control_mode_pose],
                width=width,
                height=height,
                controlnet_conditioning_scale=[0.2, 0.4],
                num_inference_steps=24, 
                guidance_scale=3.5,
                generator=torch.manual_seed(42),
            ).images[0]

        return out_img
