import sys, os
import uuid
import time
import torch
import numpy as np
import cv2
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from compel import Compel, ReturnedEmbeddingsType
import glob
from LookBuilderPipeline.image_generation.base_image_generation import BaseImageGeneration
from LookBuilderPipeline.resize import resize_images
from LookBuilderPipeline.annotate import annotate_images, image_blip, image_llava
from LookBuilderPipeline.image_generation.image_generation_fl2 import ImageGenerationFlux
from dotenv import load_dotenv
load_dotenv()

from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel

class ImageGenerationSD3(BaseImageGeneration):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the SD3 image model
        super().__init__(image, pose, mask, prompt)
        
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 7.5)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1.0)
        self.seed = kwargs.get('seed', 420042)
        self.prompt = kwargs.get('prompt', prompt)
        self.image = kwargs.get('image', image)
        self.negative_prompt = kwargs.get('negative_prompt', "dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves")
        self.strength = kwargs.get('strength', 1.0)
        self.model = 'sd3'
        self.benchmark = kwargs.get('benchmark', False)
        self.control_guidance_start=0
        self.control_guidance_end=1
        self.res = kwargs.get('res', 1280)
        self.control_guidance_start=0.0
        self.control_guidance_end=1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def prepare_model(self):
        """
        Prepare model to generate a new image using the diffusion model.
        """
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # since we only have one GPU

        controlnetA = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose")
        controlnetB = SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1)
        controlnet=SD3MultiControlNetModel([controlnetA,controlnetB])

        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # since we only have one GPU
        
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
        from diffusers import StableDiffusion3Pipeline
        import torch
        
        model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        from transformers import T5EncoderModel, BitsAndBytesConfig

        t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
        

        # Load the pipeline + CN
        self.pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo",
            token=os.getenv("HF_SD3_FLUX"),
            controlnet=controlnet,
            transformer=model_nf4,
            text_encoder_3=t5_nf4,
            torch_dtype=torch.bfloat16
        )
        
        self.pipe.enable_model_cpu_offload()
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)



    def generate_image(self):
        """
        Generate a new image using the diffusion model based on the pose, mask and prompt.
        """

        image_model.prepare_model()
        start_time = time.time()
        ## try bluring mask for better outpaint 
        # self.sm_mask.save('testcv2.png')
        # self.sm_mask=cv2.imread('testcv2.png')
        # self.sm_mask = cv2.cvtColor(self.sm_mask, cv2.COLOR_BGR2GRAY)
        # self.sm_mask = cv2.GaussianBlur(self.sm_mask, (self.blur, self.blur), 0)
        # self.sm_mask = Image.fromarray(self.sm_mask)
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

        self.conditioning, self.pooled = compel(self.prompt)
        # Generate the image using the pipeline
        image_res = self.pipe(
            # prompt=self.prompt,
            prompt_embeds=self.conditioning,
            pooled_prompt_embeds=self.pooled,
            # image=self.sm_image,
            control_mode=[4,None],
            padding_mask_crop=8,
            original_size=(self.res,self.res),
            mask_image=self.sm_mask,
            control_image=[self.sm_pose_image,self.sm_image],
            negative_prompt=self.negative_prompt,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=[self.controlnet_conditioning_scale,1.0],
            strength=self.strength,
        ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.i=os.path.basename(self.input_image).split('.')[0]
        
        # Save the generated image
        save_pathA=os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images",self.model)
        save_pathC=os.path.join("LookBuilderPipeline","LookBuilderPipeline","benchmark_images",self.model)

        os.makedirs(save_pathA, exist_ok=True)
        os.makedirs(save_pathC, exist_ok=True)
        bench_filename = 'img'+str(self.i)+'.png'

        save_path1 = os.path.join(save_pathA, bench_filename)
        save_path2 = os.path.join(save_pathC, bench_filename)
        image_res.save(save_path1)

        # self.annot='llava:'+image_llava(self,image_res)+'. blip2:'+image_blip(self,image_res)+'. desc:'+annotate_images(image_res)
        self.annot='llava: not yet. blip2:not yet either. desc:'+annotate_images(image_res)

        
        ImageGenerationSD3.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res,image_res2], output_path=save_path2)

        return image_res, save_path1
    
    def generate_bench(self,image_path,pose_path,mask_path):
        self.res=1280
        guidance_scale=self.guidance_scale
        strength=self.strength
        # blur=[51,101,151]
        image_model.prepare_model()
        for self.input_image in glob.glob(self.image):
            for self.guidance_scale in [guidance_scale]:
                for self.strength in [strength]:
                    for self.res in [768,1024,1280]:
                        self.prepare_image(self.input_image,pose_path,mask_path)
                        image_res, save_path = self.generate_image()
        

    def clearn_mem(self):
        # Clear CUDA memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageGenerationSD3")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--pose_path", default=None, help="Path to the pose image")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.7, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=420042, help="Random seed")
    parser.add_argument("--negative_prompt", default="dress, robe, clothing, flowing fabric, ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves", help="Negative prompt")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength of the transformation")
    parser.add_argument("--benchmark", type=bool, default=False, help="run benchmark with ranges pulled from user inputs +/-0.1")   
    parser.add_argument("--res", type=int, default=1280, help="Resolution of the image")


    args = parser.parse_args()

    # # Example usage of the ImageGenerationSD3 class with command-line arguments

    image_model = ImageGenerationSD3(
        args.image_path, 
        args.pose_path, 
        args.mask_path, 
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        strength=args.strength
    )

    
    if args.benchmark==False:
        image_model.prepare_model()
        image_model.prepare_image(args.image_path, args.pose_path,args.mask_path)
        generated_image, generated_image_path = image_model.generate_image()
        # generated_image = image_model.upscale_image(generated_image)  ## will explode the VRAM at this point, need to unload pipe1 or run in series
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clearn_mem()
