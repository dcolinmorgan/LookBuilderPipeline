import sys, os
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
from LookBuilderPipeline.resize import resize_images

# Import required components from diffusers
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the SDXL image model
        super().__init__(image, pose, mask, prompt)
        
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1.0)
        self.seed = kwargs.get('seed', 42)
        self.prompt = kwargs.get('prompt', prompt)
        self.image = kwargs.get('image', image)
        self.negative_prompt = kwargs.get('negative_prompt', "ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves")
        self.strength = kwargs.get('strength', 1.0)
        self.LoRA = kwargs.get('LoRA', False)
        self.model = 'sdxl'
        self.benchmark = kwargs.get('benchmark', False)
        self.control_guidance_start=0
        self.control_guidance_end=1
        self.res = kwargs.get('res', 1280)
        self.control_guidance_start=0.0
        self.control_guidance_end=1.0
        self.lora_weight=kwargs.get('lora_weight', 1.0)
    

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


        # if self.LoRA:
        from diffusers import EulerAncestralDiscreteScheduler

        # from scheduling_tcd import TCDScheduler 
        # pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        from diffusers import DDIMScheduler
    
        # self.pipe.load_lora_weights('ByteDance/Hyper-SD', weight_name='Hyper-SDXL-8steps-lora.safetensors', adapter_name="BD")

        # self.pipe.fuse_lora()
        

        # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # Load the LoRA
        # self.pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.extremely-detailed', weight_name='extremely detailed.safetensors', adapter_name="extremely detailed")
        if self.LoRA==0 or self.LoRA==False:
            self.loraout="noLora"
        # elif self.LoRA==1:
        #     self.loraout="photo in phst artstyle"
        #     self.pipe.load_lora_weights('lora-A', weight_name='EssenzStyleLoRav1.2Skynet.safetensors',adapter_name=self.loraout)
        elif self.LoRA==2:
            self.loraout="Cinematic"
            self.pipe.load_lora_weights('lora-A', weight_name='JuggernautCinematicXLLoRA.safetensors',adapter_name=self.loraout)
        elif self.LoRA==3:
            self.loraout="Analog"
            self.pipe.load_lora_weights('lora-A', weight_name='AnalogRedmondV2.safetensors',adapter_name=self.loraout)
        elif self.LoRA==4:
            self.loraout="highdetail"
            self.pipe.load_lora_weights('lora-A', weight_name='SDXLHighDetailv5.safetensors',adapter_name=self.loraout)
        elif self.LoRA==5:
            self.loraout="styleC"
            self.pipe.load_lora_weights('lora-styleC', weight_name='pytorch_lora_weights.safetensors',adapter_name=self.loraout)
        elif self.LoRA==6:
            self.loraout="h1t"
            self.pipe.load_lora_weights('h1t/TCD-SDXL-LoRA', weight_name='pytorch_lora_weights.safetensors',adapter_name=self.loraout)
        elif self.LoRA==7:
            self.loraout="12step"
            self.pipe.load_lora_weights('ByteDance/Hyper-SD', weight_name='Hyper-SDXL-12steps-CFG-lora.safetensors',adapter_name=self.loraout)
            # self.pipe.fuse_lora()
        elif self.LoRA==8:
            self.loraout="high fashion"
            self.pipe.load_lora_weights('studio', weight_name='pytorch_lora_weights.safetensors',adapter_name=self.loraout)


        if self.LoRA!=0:
            # self.pipe.fuse_lora()
            # Activate the LoRA
            self.pipe.set_adapters(self.loraout, adapter_weights=[self.lora_weight])
        
            

            # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

        # Activate the LoRA
        # self.pipe.set_adapters(["extremely detailed"], adapter_weights=[2.0])


        # Activate the LoRA
        # self.pipe.set_adapters(["winner"], adapter_weights=[2.0])
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        if self.LoRA==0:
            ## compel for prompt embedding allowing >77 tokens
            
            self.conditioning, self.pooled = compel(self.prompt)
        if self.LoRA!=0:
            self.conditioning, self.pooled = compel(self.prompt+self.loraout)

            


    def generate_image(self):
        """
        Generate a new image using the diffusion model based on the pose, mask and prompt.
        """

        
        start_time = time.time()
        # Generate the image using the pipeline
        image_res = self.pipe(
            # prompt=self.prompt,
            prompt_embeds=self.conditioning,
            pooled_prompt_embeds=self.pooled,
            image=self.sm_image,
            mask_image=self.sm_mask,
            control_image=self.sm_pose_image,
            negative_prompt=self.negative_prompt,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            strength=self.strength,
        ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.i=os.path.basename(self.input_image).split('.')[0]
        
        # Save the generated image
        save_pathA=os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images",self.model,self.loraout)
        save_pathC=os.path.join("LookBuilderPipeline","LookBuilderPipeline","benchmark_images",self.model,self.loraout)

        os.makedirs(save_pathA, exist_ok=True)
        os.makedirs(save_pathC, exist_ok=True)
        
        # bench_filename = 'img'+str(self.i)+'_g'+str(self.guidance_scale)+'_c'+str(self.controlnet_conditioning_scale)+'_s'+str(self.strength)+'_b'+str(self.control_guidance_start)+'_e'+str(self.control_guidance_end)+'.png'
        bench_filename = 'img'+str(self.i)+str(self.lora_weight)+'.png'

        save_path1 = os.path.join(save_pathA, bench_filename)
        save_path2 = os.path.join(save_pathC, bench_filename)
        image_res.save(save_path1)
        ImageModelSDXL.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res], output_path=save_path2)

        return image_res, save_path1
    
    def generate_bench(self,image_path,pose_path,mask_path):
        self.res=1024
        guidance_scale=self.guidance_scale
        strength=self.strength
        LoRA=[0,2,3,4,5,6,7,8]
        lora_weights=[0.5,1.0,1.5,2]

        for self.LoRA in LoRA:
            for self.lora_weight in lora_weights:
                image_model.prepare_model()
                for self.input_image in glob.glob(self.image):
                # for self.controlnet_conditioning_scale in [self.controlnet_conditioning_scale-0.2,self.controlnet_conditioning_scale,self.controlnet_conditioning_scale+0.2]:
                    for self.guidance_scale in [guidance_scale]:
                        for self.strength in [strength]:
                        # for self.res in [768,1024,1280]:
                                # for self.control_guidance_end in [self.control_guidance_end,self.control_guidance_end+0.1]:
                            self.prepare_image(self.input_image,pose_path,mask_path)
                            image_res, save_path = self.generate_image()
                        # image_res = self.upscale_image(image_res)
            self.pipe.unload_lora_weights()
        

    def clearn_mem(self):
        # Clear CUDA memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageModelSDXL")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--pose_path", default=None, help="Path to the pose image")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative_prompt", default="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves", help="Negative prompt")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength of the transformation")
    parser.add_argument("--LoRA", type=bool, default=False, help="use LoRA or not")
    parser.add_argument("--benchmark", type=bool, default=False, help="run benchmark with ranges pulled from user inputs +/-0.1")   
    parser.add_argument("--res", type=int, default=1280, help="Resolution of the image")
    parser.add_argument("--lora_weight", type=str, default=1.0, help="weight of the LoRA")


    args = parser.parse_args()

    # # Example usage of the ImageModelSDXL class with command-line arguments
    # if args.pose_path is None or args.mask_path is None:
    #     args.pose_path, args.mask_path = ImageModelSDXL.generate_image_extras(args.image_path,inv=True)

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

    
    if args.benchmark==None:
        image_model.prepare_model()
        image_model.prepare_image(args.image_path, args.pose_path,args.mask_path)
        generated_image, generated_image_path = image_model.generate_image()
        # generated_image = image_model.upscale_image(generated_image)  ## will explode the VRAM at this point, need to unload pipe1 or run in series
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clearn_mem()
