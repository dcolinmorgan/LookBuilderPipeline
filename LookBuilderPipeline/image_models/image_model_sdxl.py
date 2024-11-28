import sys, os, gc, requests
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from compel import Compel, ReturnedEmbeddingsType
import glob
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.utils.resize import resize_images
from LookBuilderPipeline.annotate import annotate_images, image_blip, image_llava
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
from PIL import Image
import io
# Import required components from diffusers
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the SDXL image model
        super().__init__(image, pose, mask, prompt)
        if torch.backends.mps.is_available():
            device = torch.device('mps')  # Use MPS if available
        elif torch.cuda.is_available():
            device = torch.device('cuda')  # Use CUDA if available
        else:
            device = torch.device('cpu')
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 6)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1.0)
        self.seed = kwargs.get('seed', 420042)
        self.prompt = kwargs.get('prompt', prompt)
        self.image = kwargs.get('image', image)
        self.negative_prompt = kwargs.get('negative_prompt', "dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves")
        self.strength = kwargs.get('strength', 0.9)
        self.LoRA = kwargs.get('LoRA', None)
        self.model = 'sdxl_blur'
        self.benchmark = kwargs.get('benchmark', False)
        self.control_guidance_start=0
        self.control_guidance_end=1
        self.res = kwargs.get('res', 1024)
        self.control_guidance_start=0.0
        self.control_guidance_end=1.0
        self.lora_weight=kwargs.get('lora_weight', 1.0)
        self.device = device
    

    def prepare_model(self):
        """
        Prepare model to generate a new image using the diffusion model.
        """
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # since we only have one GPU
        
        # Load the controlnet
        controlnet_model = ControlNetModel.from_pretrained(
            # "xinsir/controlnet-union-sdxl-1.0",
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16,
            # use_safetensors=True,
        )

        # Load the pipeline + CN
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(#from_pretrained(
            "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            # "SG161222/RealVisXL_V5.0_Lightning",
            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )

        # Configure pipeline settings
        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.to("cuda")
        
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

        supermodel_face=231666
        female_face=273591
        better_face=301988
        diana=293406
        def download_lora(lora_id):
            url=f'https://civitai.com/api/download/models/{lora_id}'
            r = requests.get(url)
            fname=f'/LookBuilderPipeline/LookBuilderPipeline/image_models/{self.LoRA}.safetensors'
            open(fname , 'wb').write(r.content)
        # Load the LoRA
        if self.LoRA=='supermodel_face':
            download_lora(supermodel_face)
            self.pipe.load_lora_weights('LookBuilderPipeline/LookBuilderPipeline/image_models',weight_name='supermodel_face.safetensors',adapter_name=self.LoRA)
        elif self.LoRA=='female_face':
            download_lora(female_face)
            self.pipe.load_lora_weights('LookBuilderPipeline/LookBuilderPipeline/image_models',weight_name='female_face.safetensors',adapter_name=self.LoRA)
        elif self.LoRA=='better_face':
            download_lora(better_face)
            self.pipe.load_lora_weights('LookBuilderPipeline/LookBuilderPipeline/image_models',weight_name='better_face.safetensors',adapter_name=self.LoRA)
        elif self.LoRA=='diana':
            download_lora(diana)
            self.pipe.load_lora_weights('LookBuilderPipeline/LookBuilderPipeline/image_models',weight_name='diana.safetensors',adapter_name=self.LoRA)
        
        if self.LoRA!=None:
            # self.pipe.fuse_lora()
            # Activate the LoRA
            self.pipe.set_adapters(self.LoRA, adapter_weights=[self.lora_weight])
            # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

        # Activate compel for long prompts
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        if self.LoRA==None:
            ## compel for prompt embedding allowing >77 tokens
            self.conditioning, self.pooled = compel(self.prompt)
        else:
            self.conditioning, self.pooled = compel(self.prompt+self.LoRA)

            


    def generate_image(self):
        """
        Generate a new image using the diffusion model based on the pose, mask and prompt.
        """

        # image_model.prepare_model()
        start_time = time.time()
        ## try bluring mask for better outpaint 
        # self.sm_mask.save('testcv2.png')
        # self.sm_mask=cv2.imread('testcv2.png')
        # self.sm_mask = cv2.cvtColor(self.sm_mask, cv2.COLOR_BGR2GRAY)
        # self.sm_mask = cv2.GaussianBlur(self.sm_mask, (self.blur, self.blur), 0)
        # self.sm_mask = Image.fromarray(self.sm_mask)
        # Ensure self.image, self.mask, and self.pose are bytes-like objects
        try:        
            self.image = Image.open(io.BytesIO(self.image))
        except:
            pass
        try:        
            self.mask = Image.open(io.BytesIO(self.mask))
        except:
            pass
        try:        
            self.pose = Image.open(io.BytesIO(self.pose))
        except:
            pass
        from LookBuilderPipeline.utils.resize import resize_images
        self.image, self.mask, self.pose = resize_images(images=[self.image, self.mask, self.pose], target_size=self.res)
        # self.image=resize_images(self.image,self.image.size,aspect_ratio=None)
        # self.pose=resize_images(self.pose,self.image.size,aspect_ratio=None)#,aspect_ratio=self.image.size[0]/self.image.size[1])
        # self.mask=resize_images(self.mask,self.image.size,aspect_ratio=None)#,aspect_ratio=self.image.size[0]/self.image.size[1])
        
        from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
        self.image = BaseImageModel.resize_with_padding(self=None,img=self.image, expected_size=[self.res, self.res])
        self.pose = BaseImageModel.resize_with_padding(self=None,img=self.pose, expected_size=[self.res, self.res])
        self.mask = BaseImageModel.resize_with_padding(self=None,img=self.mask, expected_size=[self.res, self.res], color=255)
        
        # Generate the image using the pipeline
        image_res = self.pipe(
            # prompt=self.prompt,
            prompt_embeds=self.conditioning,
            pooled_prompt_embeds=self.pooled,
            image=self.image,
            padding_mask_crop=8,
            original_size=(self.res,self.res),
            mask_image=self.mask,
            control_image=self.pose,
            negative_prompt=self.negative_prompt,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            strength=self.strength,
        ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.clear_mem()
        
        try:        
            image_res = Image.open(io.BytesIO(image_res))
        except:
            pass
        return image_res #, save_path1
    

    def clear_mem(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageModelSDXL")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--pose_path", default=None, help="Path to the pose image")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=420042, help="Random seed")
    parser.add_argument("--negative_prompt", default="dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves", help="Negative prompt")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the transformation")
    parser.add_argument("--LoRA", type=str, default=None, help="use LoRA or not")
    parser.add_argument("--benchmark", type=bool, default=False, help="run benchmark with ranges pulled from user inputs +/-0.1")   
    parser.add_argument("--res", type=int, default=1024, help="Resolution of the image")
    parser.add_argument("--lora_weight", type=str, default=1.0, help="weight of the LoRA")


    args = parser.parse_args()

    # # Example usage of the ImageModelSDXL class with command-line arguments

    image_model = ImageModelSDXL(
        args.image_path, 
        args.pose_path, 
        args.mask_path, 
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        LoRA=args.LoRA
    )

    
    if args.benchmark==False:
        image_model.prepare_model()
        image_model.prepare_image(args.image_path, args.pose_path,args.mask_path)
        generated_image, generated_image_path = image_model.generate_image()
        # generated_image = image_model.upscale_image(generated_image)  ## will explode the VRAM at this point, need to unload pipe1 or run in series
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clear_mem()
