
import sys, os
sys.path.insert(2,os.path.abspath('LookBuilderPipeline/LookBuilderPipeline/'))

import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux import FluxControlNetInpaintPipeline
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image, check_min_version
from transformers import pipeline ,SegformerImageProcessor, AutoModelForSemanticSegmentation

import glob
from time import time
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
from controlnet_aux import OpenposeDetector
from pathlib import Path

from LookBuilderPipeline.LookBuilderPipeline.resize import resize_images
from LookBuilderPipeline.LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.LookBuilderPipeline.pose import detect_pose
from LookBuilderPipeline.plot_images import showImagesHorizontally

from diffusers import StableDiffusionXLInpaintPipeline,ControlNetModel,StableDiffusionXLControlNetInpaintPipeline
from transformers import pipeline 
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from quanto import quantize, freeze, qint4, qint8, qfloat8
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")#,device='cuda')


class ImageModelFlux(BaseImageModel):
    def __init__(self, image, pose, mask, prompt):
        super().__init__(image, pose, mask, prompt)

    def load_image(self):
        """
        Generate a new image using the Flux model based on the pose, mask and prompt.
        """
        ### init before loading model
        self.prompt+="no leggings, no tights, no sunglasses, no stockings, no pants, no sleeves, no bad anatomy, no deformation"
        
        self.orig_image=load_image(self.image)
        self.pose_image=load_image(self.pose)
        self.mask_image=load_image(self.mask)

    # def load_images(self):

    def load_model(self):
        self.width,self.height=self.orig_image.size
        self.num_inference_steps=30
        self.guidance_scale=7.5
        self.controlnet_conditioning_scale=0.5
        seed=np.random.randint(0,100000000)
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        dtype = torch.bfloat16
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('flux-fp8', subfolder="scheduler")  # flux1 schnell folder 
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        self.text_encoder_2 = T5EncoderModel.from_pretrained('flux-fp8', subfolder="text_encoder_2", torch_dtype=dtype)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained('flux-fp8', subfolder="tokenizer_2", torch_dtype=dtype)
        self.vae = AutoencoderKL.from_pretrained('flux-fp8', subfolder="vae", torch_dtype=dtype)
        transformer = FluxTransformer2DModel.from_pretrained('flux-fp8', subfolder="transformer", torch_dtype=dtype)
        
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model,use_safetensors=True, torch_dtype=torch.bfloat16, add_prefix_space=True,local_files_only=True,guidance_embeds=False)


        self.pipe = FluxControlNetInpaintPipeline.from_pretrained("ostris/OpenFLUX.1",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,)

        self.pipe.text_encoder_2 = text_encoder_2
        self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()


        
    def run_model(self):
        image_res = self.pipe(
            self.prompt,
            self.negative_prompt,
            image=self.orig_image,
            control_image=self.pose_image,
            control_mode=4,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            mask_image=self.mask_image,
            height=self.height,
            width=self.width,
            strength=0.9,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            joint_attention_kwargs={"scale": 1},    
        ).images[0]

        return image_res

