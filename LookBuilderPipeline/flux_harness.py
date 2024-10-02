# !pip install -U -q git+https://github.com/huggingface/diffusers
# !pip install transformers -U -q
# !pip install -U -q tokenizers==0.20
# !pip install -q- U controlnet_aux
# !pip install accelerate==0.32.1 -U
# !pip install mediapipe
# ! pip install protobuf==3.20
# !pip install sentencepiece -U

# !git clone https://github.com/Modegen/flux-controlnet-inpaint.git external_deps/
# !git clone https://github.com/Modegen/ControlNetPlus.git external_deps/
# !export GH_PAT=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
# !git clone https://${GH_PAT}@github.com/Modegen/LookBuilderPipeline.git


import sys, os
sys.path.insert(0,os.path.abspath('external_deps/flux-controlnet-inpaint/src/'))
sys.path.insert(1,os.path.abspath('external_deps/ControlNetPlus'))
sys.path.insert(2,os.path.abspath('LookBuilderPipeline/LookBuilderPipeline/'))

import torch
import torch.nn as nn
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxMultiControlNetModel
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


controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
controlnet_pose = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",controlnet=controlnet_pose, torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.text_encoder.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)
pipe.enable_sequential_cpu_offload()


dir=glob.glob('LookBuilderPipeline/LookBuilderPipeline/img/orig/*')

for input_image in dir:
    image = load_image(input_image)
    pose_image = detect_pose(input_image)
    mask,mask_image,_ = segment_image(input_image,inverse=True,additional_option='shoe')

    if image.size[0] > image.size[1]:
        sm_image=resize_images(image,image.size[1])
        sm_pose_image=resize_images(pose_image,image.size[1])
        sm_mask=resize_images(mask_image,image.size[1])
    else:
        sm_image=resize_images(image,image.size[0])
        sm_pose_image=resize_images(pose_image,image.size[0])
        sm_mask=resize_images(mask_image,image.size[0])
    
    width,height=sm_image.size
    
    # negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, \n deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves"
    # prompt="photo realistic fashion model with blonde hair with bare arms and legs"
    prompt= \
    "Hair: The model should have long, flowing blonde hair that cascades elegantly over her shoulders. \
    Skin: The model should have a flawless complexion, showcasing bare arms and legs, emphasizing a natural and healthy appearance.\
    Attire: The models arms and legs are bare. \
    Clothes: The outfit should stick 100% to the mask and not include any additonal piece."
    # Background: The background should be softly blurred flower garden to keep the focus on the model, with a neutral or pastel color palette that complements her features.\
    # Lighting: Use soft, natural lighting to highlight the model's features and create a warm, inviting atmosphere."

    # prompt2="no cloth, no shirt, no shoe, no cape, no leggings, no tights, no sunglasses, no stockings, no pants, no sleeves, no bad anatomy, no deformations"
    
    t=time()
    num_inference_steps=30
    guidance_scale=7.5
    controlnet_conditioning_scale=1
    seed=np.random.randint(0,100000000)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image_res = pipe(
            prompt,
            # prompt2,
            image=sm_image,
            control_image=sm_pose_image,
            control_mode=4,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            mask_image=sm_mask,
            height=height,
            width=width,
            strength=0.9,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    tt=t-time()
    filename='flux_test_output/flux_'+str(height)+'_'+str(width)+'_'+str(tt)+'_'+str(controlnet_conditioning_scale)+'_'+str(num_inference_steps)+'_'+str(guidance_scale)+'_'+str(seed)+'.png'
    showImagesHorizontally([sm_image,sm_mask,sm_pose_image,image_res],prompt,prompt2,'flux',height, width,controlnet_conditioning_scale,num_inference_steps,guidance_scale,seed,filename)
