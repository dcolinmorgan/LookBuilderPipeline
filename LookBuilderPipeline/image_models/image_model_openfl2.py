import sys, os
import glob, inspect, shutil
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from compel import Compel, ReturnedEmbeddingsType

from LookBuilderPipeline.image_models.base_image_model import BaseImageModel, prepare_image
from LookBuilderPipeline.resize import resize_images
from quanto import qfloat8,qint4,qint8, quantize,freeze

# Import required components from diffusers
from diffusers import FluxPipeline, FluxInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

# ## change flux inpainting pipeline to allow negative-prompt in OpenFlux
PTH=(os.path.abspath(inspect.getfile(FluxPipeline)))
orig_pipe='/'.join(PTH.split('/')[:-1])+'/pipeline_flux_controlnet_inpainting.py'
mod_pipe='LookBuilderPipeline/LookBuilderPipeline/image_models/openflux/pipeline_flux_controlnet_inpainting.py'
shutil.copy(mod_pipe, orig_pipe)

class ImageModelOpenFLUX(BaseImageModel):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the FLUX image model
        super().__init__(image, pose, mask, prompt)
        
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 30)
        self.guidance_scale = kwargs.get('guidance_scale', 3.5)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 0.7)
        self.seed = kwargs.get('seed', np.random.randint(0, 100000000))
        self.prompt = kwargs.get('prompt', prompt)
        self.negative_prompt = kwargs.get('negative_prompt', 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves')
        self.image = kwargs.get('image', image)
        self.strength = kwargs.get('strength', 0.99)
        self.model = 'openflux'
        self.quantize = kwargs.get('quantize', None)
        self.LoRA = kwargs.get('LoRA', False)
        self.prompt_embeds=None
        self.pooled_prompt_embeds=None
        self.control_guidance_start = kwargs.get('control_guidance_start', 0.2)
        self.control_guidance_end = kwargs.get('control_guidance_end', 0.5)


    # def prepare_image(self):
    #     """
    #     Prepare the pose and mask images to generate a new image using the diffusion model.
    #     """
    #     # Load and resize images
    #     image = load_image(self.image)
    #     if isinstance(self.pose,str):
    #         pose_image = load_image(self.pose)
    #     else:
    #         pose_image = self.pose
    #     if isinstance(self.mask,str):
    #         mask_image = load_image(self.mask)
    #     else:
    #         mask_image = self.mask

    #     if pose_image.size[0] < image.size[0]:  ## resize to pose image size if it is smaller
    #         self.sm_image=resize_images(image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
    #         self.sm_pose_image=resize_images(pose_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
    #         self.sm_mask=resize_images(mask_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
            
    #     else:
    #         self.sm_image=resize_images(image,image.size,aspect_ratio=image.size[0]/image.size[1])
    #         self.sm_pose_image=resize_images(pose_image,image.size,aspect_ratio=image.size[0]/image.size[1])
    #         self.sm_mask=resize_images(mask_image,image.size,aspect_ratio=image.size[0]/image.size[1])
            
    #     self.width, self.height = self.sm_image.size

    def prepare_model(self):
        """
        Load the FLUX model and controlnet.
        """

        # Set up the pipeline
        base_model = 'ostris/OpenFLUX.1'
        controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union' ## may need to change this to FLUX.1-schnell-Controlnet-Union or train our own https://huggingface.co/xinsir/controlnet-union-FLUX-1.0/discussions/28

        controlnet_pose = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16,guidance_embeds=False)#,add_prefix_space=True)
        # controlnet = FluxMultiControlNetModel([controlnet_pose,controlnet_pose])

        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet_pose, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.enable_sequential_cpu_offload()
        if self.LoRA:
            # self.pipe.load_lora_weights('hugovntr/flux-schnell-realism', weight_name='schnell-realism_v2.3.safetensors', adapter_name="winner")
            self.pipe.load_lora_weights('XLabs-AI/flux-RealismLora', weight_name='lora.safetensors', adapter_name="winner") 

            # Activate the LoRA
            self.pipe.set_adapters(["winner"], adapter_weights=[2.0])
            # from diffusers import EulerAncestralDiscreteScheduler
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        
    def prepare_quant_model(self):
        """
        Load the FLUX model and controlnet as individual components.
        """
        if not glob.glob('flux-fp8'):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="ostris/OpenFLUX.1",local_dir='flux-fp8') 
        
        from diffusers import FluxControlNetInpaintPipeline, FluxTransformer2DModel
        from torchao.quantization import quantize_, int8_weight_only
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

        controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16,add_prefix_space=True,guidance_embeds=False)
        
        # model_path = "black-forest-labs/FLUX.1-schnell"

        transformer = FluxTransformer2DModel.from_pretrained(
            'flux-fp8',
            subfolder = "transformer",
            torch_dtype = torch.bfloat16
        )
        quantize_(transformer, int8_weight_only())

        # from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
        # from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        # from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        # from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
        # dtype = torch.bfloat16

        # # Import required components from diffusers
        # from diffusers import FluxControlNetInpaintPipeline
        
        # scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('flux-fp8', subfolder="scheduler")
        # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        # # text_encoder_2 = T5EncoderModel.from_pretrained('flux-fp8', subfolder="text_encoder_2", torch_dtype=dtype)
        # tokenizer_2 = T5TokenizerFast.from_pretrained('flux-fp8', subfolder="tokenizer_2", torch_dtype=dtype)
        # vae = AutoencoderKL.from_pretrained('flux-fp8', subfolder="vae", torch_dtype=dtype)
        # # transformer = FluxTransformer2DModel.from_pretrained('flux-fp8', subfolder="transformer", torch_dtype=dtype)

        # if glob.glob('qopenflux/text_encoder_2'+str(self.quantize)+'/*.safetensors'):
        #     text_encoder_2 = T5EncoderModel.from_pretrained('qopenflux', subfolder="text_encoder_2"+str(self.quantize), torch_dtype=dtype)
        # else:
        #     text_encoder_2 = T5EncoderModel.from_pretrained('flux-fp8', subfolder="text_encoder_2", torch_dtype=dtype)
        #     quantize(text_encoder_2, weights=eval(self.quantize))
        #     freeze(text_encoder_2)
        #     text_encoder_2.save_pretrained('qopenflux/text_encoder_2'+self.quantize)

        # # if glob.glob('qopenflux/transformer/*.safetensors'):
        # try:
        #     transformer = FluxTransformer2DModel.from_pretrained('qopenflux', subfolder="transformer"+str(self.quantize), torch_dtype=dtype)
        # # else:
        # except:
        #     transformer = FluxTransformer2DModel.from_pretrained('flux-fp8', subfolder="transformer", torch_dtype=dtype)
        #     quantize(transformer, weights=eval(self.quantize))
        #     freeze(transformer)
        #     transformer.save_pretrained('qopenflux/transformer'+self.quantize)
            
        # self.pipe = FluxControlNetInpaintPipeline(
        #     controlnet=controlnet,
        #     scheduler=scheduler,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     text_encoder_2=None,
        #     tokenizer_2=tokenizer_2,
        #     vae=vae,
        #     transformer=None,
        # )
        # self.pipe.text_encoder_2 = text_encoder_2
        # self.pipe.transformer = transformer
        # self.pipe.enable_model_cpu_offload()
        
        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(
            'flux-fp8',
            controlnet=controlnet,
            transformer = transformer,
            torch_dtype = torch.bfloat16,
        )

        self.pipe.enable_model_cpu_offload()
        
        self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
            self.pipe= self.pipe,
            self.prompt= self.prompt
        )

        if self.LoRA:
            self.pipe.load_lora_weights('hugovntr/flux-schnell-realism', weight_name='schnell-realism_v2.3.safetensors', adapter_name="winner")

            # Activate the LoRA
            self.pipe.set_adapters(["winner"], adapter_weights=[2.0])
            from diffusers import EulerAncestralDiscreteScheduler
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

        
    def generate_image(self):
        ## compel for prompt embedding allowing >77 tokens
        # compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        # conditioning, pooled = compel(self.prompt)
        
        start_time = time.time()
        # Generate the image using the pipeline
        if self.negative_prompt!=None and self.negative_prompt!=None:
            image_res = self.pipe(
                prompt=self.prompt,
                # prompt_embeds=conditioning,
                # pooled_prompt_embeds=pooled,
                negative_prompt=self.negative_prompt,
                image=self.sm_image,
                control_image=self.sm_pose_image,
                control_mode=4,
                control_guidance_start=self.control_guidance_start,
                control_guidance_end=self.control_guidance_end,
                # padding_mask_crop=32,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                mask_image=self.sm_mask,
                height=self.height,
                width=self.width,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
            ).images[0]
        else:
            image_res = self.pipe(
                # prompt=self.prompt,
                prompt_embeds=self.prompt_embeds,
                pooled_prompt_embeds=self.pooled_prompt_embeds,
                negative_prompt=self.negative_prompt,
                image=self.sm_image,
                control_image=self.sm_pose_image,
                control_mode=4,
                # padding_mask_crop=32,
                control_guidance_start=self.control_guidance_start,
                control_guidance_end=self.control_guidance_end,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                mask_image=self.sm_mask,
                height=self.height,
                width=self.width,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
            ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
    
        # Save the generated image
        os.makedirs(os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model), exist_ok=True)

        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model, filename)
        image_res.save(save_path)
        bench_filename = f"{uuid.uuid4()}.png"
        bench_save_path = os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", "openflux", 'bench'+bench_filename)
        ImageModelOpenFLUX.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res],output_path=bench_save_path)
     

        return image_res, save_path

    def clearn_mem(self):
        # Clear CUDA memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageModelFLUX")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--pose_path", default=None, help="Path to the pose image")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default='ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves', help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strength", type=float, default=0.99, help="Strength of the transformation")  # Add strength argument
    parser.add_argument("--quantize", default=None, help="None,qfloat8,qint8,qint4")
    parser.add_argument("--LoRA", default=False, help="to add LoRA or not")
    parser.add_argument("--control_guidance_start", type=float, default=0.2, help="ControlNet guidance start")
    parser.add_argument("--control_guidance_end", type=float, default=0.5, help="ControlNet guidance end")
    args = parser.parse_args()

    # Example usage of the ImageModelFLUX class with command-line arguments
    if args.pose_path is None or args.mask_path is None:
        args.pose_path, args.mask_path = ImageModelOpenFLUX.generate_image_extras(args.image_path,inv=True)

    image_model = ImageModelOpenFLUX(
        args.image_path, 
        args.pose_path, 
        args.mask_path, 
        args.prompt,
        args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        quantize=args.quantize,
        LoRA=args.LoRA,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
    )
    image_model.prepare_image()
    if args.quantize==None:
        image_model.prepare_model()
    else:
        image_model.prepare_quant_model()
    generated_image, generated_image_path = image_model.generate_image()
    print(f"Generated image saved at: {generated_image_path}")
    image_model.clearn_mem()
