import sys, os
sys.path.insert(0, os.path.abspath('/flux-controlnet-inpaint/src'))
import torch
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxMultiControlNetModel
from PIL import Image
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from pathlib import Path
from controlnet_aux import CannyDetector
from transformers import pipeline 

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread


def showImagesHorizontally(list_of_files, output_path='output.png'):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = (list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to free up memory

class ImageModelFlux(BaseImageModel):
    def __init__(self, pose, mask, prompt):
        super().__init__(pose, mask, prompt)

    def generate_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        # Set up the pipeline
        
        base_model = 'black-forest-labs/FLUX.1-dev'
        # controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'
        controlnet_model2 = 'InstantX/FLUX.1-dev-Controlnet-Union'

        # controlnet_canny = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
        controlnet_pose = FluxControlNetModel.from_pretrained(controlnet_model2, torch_dtype=torch.float16)
        controlnet = FluxMultiControlNetModel([controlnet_pose,controlnet_pose])

        pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        pipe.text_encoder.to(torch.float16)
        pipe.controlnet.to(torch.float16)
        pipe.enable_sequential_cpu_offload()

        images = [image, canny_image, pose_image, mask]
        image, canny_image, pose_image, mask = [img.resize((512, 512)) for img in images]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_res = pipe(
            prompt,
            image=image,
            control_image=[canny_image,pose_image],
            control_mode=[0, 4],
            controlnet_conditioning_scale=[0.8,0.9],
            mask_image=mask,
            strength=0.95,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            joint_attention_kwargs={"scale": scale},    
        ).images[0]



        return out_img
