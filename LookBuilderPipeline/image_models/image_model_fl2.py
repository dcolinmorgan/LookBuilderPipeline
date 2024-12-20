import sys, os, shutil, glob, gc
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.utils.resize import resize_images
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

# Import required components from diffusers
from diffusers import FluxControlNetInpaintPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel, FluxControlNetPipeline, FluxMultiControlNetModel
from diffusers.models.controlnet_flux import FluxControlNetModel

# ## change flux inpainting pipeline to allow negative-prompt in OpenFlux
# PTH=(os.path.abspath(inspect.getfile(Fluxpipeline)))
# orig_pipe='/'.join(PTH.split('/')[:-1])+'/pipeline_flux_controlnet_inpainting.py'
# mod_pipe='LookBuilderpipeline/LookBuilderpipeline/image_models/openflux/pipeline_flux_controlnet_inpainting.py'
# shutil.copy(mod_pipe, orig_pipe)

class ImageModelFlux(BaseImageModel):
    def __init__(self, image, **kwargs):
        # Initialize the FLUX image model
        super().__init__(image, **kwargs)

        # Set default values
        # self.num_inference_steps = kwargs.get('num_inference_steps', 10)
        # self.guidance_scale = kwargs.get('guidance_scale', 7.0)
        # self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1.0)
        # self.seed = kwargs.get('seed', np.random.randint(0, 100000000))
        # self.prompt = kwargs.get('prompt', prompt)
        # self.image = kwargs.get('image', image)
        # self.strength = kwargs.get('strength', 0.99)
        self.model = kwargs.get('model', 'schnell')
        self.quantize = kwargs.get('quantize', True)
        # self.LoRA = kwargs.get('LoRA', False)
        # self.prompt_embeds=kwargs.get('prompt_embeds', None)
        # self.pooled_prompt_embeds=kwargs.get('pooled_prompt_embeds', None)
        # self.control_guidance_start = kwargs.get('control_guidance_start', 0)
        # self.control_guidance_end = kwargs.get('control_guidance_end', 1)
        # self.benchmark = kwargs.get('benchmark', False)
        self.res = kwargs.get('res', 1024)
        self.openflux = kwargs.get('openflux', False)
        # self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe=self.flux_pipe, prompt=self.prompt)
    

    def prepare_inpaint_model(self):
        """
        Load the FLUX model and controlnet.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-schnell'

        controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Union", torch_dtype=torch.bfloat16
)


        transformer=ImageModelFlux.prepare_quant_model(self)  # quant or not
        self.flux_pipe = FluxControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        # self.flux_pipe.text_encoder.to(torch.bfloat16)
        # self.flux_pipe.controlnet.to(torch.bfloat16)
        # self.flux_pipe.enable_model_cpu_offload()
        # self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe=self.flux_pipe, prompt=self.prompt)

    def prepare_img2img_model(self):
        """
        Load the FLUX model and controlnet.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-schnell'

        transformer= ImageModelFlux.prepare_quant_model(self)  # quant or not

        self.flux_pipe = FluxImg2ImgPipeline.from_pretrained(
            base_model,
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.to("cuda")

        # self.flux_pipe.text_encoder.to(torch.bfloat16)
        # self.flux_pipe.controlnet.to(torch.bfloat16)
        # self.flux_pipe.enable_sequential_cpu_offload()
        # self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe=self.flux_pipe, prompt=self.prompt)
        
    def prepare_upscale_image(self):
        base_model = 'black-forest-labs/FLUX.1-schnell'
        # Load pipeline
        controlnet2 = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=torch.bfloat16
        )
        transformer=ImageModelFlux.prepare_quant_model(self)  # quant or not
        self.flux_pipe = FluxControlNetPipeline.from_pretrained(base_model,
            controlnet=controlnet2, 
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.flux_pipe.to("cpu")

        # self.flux_pipe.text_encoder.to(torch.bfloat16)
        # self.flux_pipe.controlnet.to(torch.bfloat16)
        # self.flux_pipe.enable_model_cpu_offload()
        # self.flux_pipe.enable_sequential_cpu_offload()


    def prepare_quant_model(self):
        """
        Load the FLUX model and controlnet as individual components.
        """
        from torchao.quantization import quantize_, int8_weight_only

        if not glob.glob('flux-schnell-fp8'):
            from huggingface_hub import snapshot_download
            if self.openflux==True:
                snapshot_download(repo_id="ostris/OpenFLUX.1",local_dir='flux-schnell-fp8')
            else:
                snapshot_download(repo_id="black-forest-labs/FLUX.1-schnell",local_dir='flux-schnell-fp8')
                
        if self.quantize and not os.path.exists('flux-schnell-fp8/quant_transformer/diffusion_pytorch_model-00001-of-00002.bin'):
            self.transformer = FluxTransformer2DModel.from_pretrained(
                'flux-schnell-fp8',
                subfolder = "transformer",
                torch_dtype = torch.bfloat16
            ) 
            quantize_(self.transformer, int8_weight_only())
            self.transformer.save_pretrained('flux-schnell-fp8/quant_transformer/',safe_serialization=False)
        if self.quantize and os.path.exists('flux-schnell-fp8/quant_transformer/diffusion_pytorch_model-00001-of-00002.bin'):
            self.transformer = FluxTransformer2DModel.from_pretrained(
                'flux-schnell-fp8',
                subfolder = "quant_transformer",
                torch_dtype = torch.bfloat16,
                use_safetensors=False
                )
        elif not self.quantize:
            self.transformer = FluxTransformer2DModel.from_pretrained(
                'flux-schnell-fp8',
                subfolder = "transformer",
                torch_dtype = torch.bfloat16
            )
        return self.transformer
        

    def generate_image(self):
        self.generator = torch.Generator(device="cuda").manual_seed(self.seed)
        # self.control_guidance_start=np.round(self.control_guidance_start,2)
        # self.control_guidance_end=np.round(self.control_guidance_end,2)
        
        # if self.LoRA:
        #     self.flux_pipe.load_lora_weights('prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0', weight_name='Canopus-LoRA-Flux-UltraRealism.safetensors', adapter_name="ultra")
            # self.flux_pipe.load_lora_weights('prithivMLmods/Canopus-LoRA-Flux-FaceRealism', weight_name='Canopus-LoRA-Flux-FaceRealism.safetensors', adapter_name="face")
            # self.flux_pipe.load_lora_weights('prithivMLmods/Canopus-Clothing-Flux-LoRA', weight_name='Canopus-Clothing-Flux-Dev-Florence2-LoRA.safetensors', adapter_name="clothes")
            # self.flux_pipe.load_lora_weights('hugovntr/flux-schnell-realism', weight_name='schnell-realism_v2.3.safetensors', adapter_name="winner3")
            # self.flux_pipe.load_lora_weights('XLabs-AI/flux-RealismLora', weight_name='lora.safetensors', adapter_name="winner2") 
            # self.flux_pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-add-details', weight_name='FLUX-dev-lora-add_details.safetensors', adapter_name="winner") 
            # self.flux_pipe.load_lora_weights('ByteDance/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors', adapter_name="HYSD") 
            # Activate the LoRA
            # self.flux_pipe.set_adapters(["ultra"], adapter_weights=[2.0])
            # self.flux_pipe.set_adapters(["face"], adapter_weights=[2.0])        
        
        start_time = time.time()
        
        image_res = self.flux_pipe(  ## only inpainting now
                prompt=self.prompt,
                image=self.image,
                control_image=self.pose,
                control_mode=4,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                mask_image=self.mask,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
                # prompt_embeds=self.prompt_embeds,
                # pooled_prompt_embeds=self.pooled_prompt_embeds,

            ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.clear_mem()
        # self.negative_prompt=None
        # self.i=os.path.basename(self.input_image).split('.')[0]

        # Save the generated image
        # save_pathA=os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images",self.model,self.loraout)
        # save_pathC=os.path.join("LookBuilderPipeline","LookBuilderPipeline","benchmark_images",self.model,self.loraout)

        # os.makedirs(save_pathA, exist_ok=True)
        # os.makedirs(save_pathC, exist_ok=True)
        
        # # bench_filename = 'img'+str(self.i)+'_g'+str(self.guidance_scale)+'_c'+str(self.controlnet_conditioning_scale)+'_s'+str(self.strength)+'_b'+str(self.control_guidance_start)+'_e'+str(self.control_guidance_end)+'.png'
        # bench_filename = 'img'+str(self.i)+str(self.lora_weight)+'.png'

        # save_path1 = os.path.join(save_pathA, bench_filename)
        # save_path2 = os.path.join(save_pathC, bench_filename)
        # image_res.save(save_path1)

        # self.annot='llava:'+image_llava(self,image_res)+'. blip2:'+image_blip(self,image_res)+'. desc:'+annotate_images(image_res)
        
        # ImageModelFlux.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res], output_path=save_path2)

        return image_res #, save_path1

    def generate_bench(self,image_path,pose_path,mask_path):
        for self.input_image in glob.glob(self.image):
            # for self.controlnet_conditioning_scale in [self.controlnet_conditioning_scale-0.2,self.controlnet_conditioning_scale,self.controlnet_conditioning_scale+0.2]:
            for self.guidance_scale in [self.guidance_scale-0.5,self.guidance_scale,self.guidance_scale+0.5]:
                for self.strength in [self.strength]:#,self.strength+0.009]:
                    for self.res in [768,1024,1280]:
                            # for self.control_guidance_end in [self.control_guidance_end,self.control_guidance_end+0.1]:
                        self.prepare_image(self.input_image,pose_path,mask_path)
                        image_res, save_path = self.generate_image()
                    # image_res = self.upscale_image(image_res)
                    
    def clear_mem(self):
        del self.flux_pipe
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageModelFLUX")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--pose_path", default=None, help="Path to the pose image")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strength", type=float, default=0.99, help="Strength of the transformation")
    parser.add_argument("--model", type=str, default='schnell', help="Model to use")
    parser.add_argument("--quantize", type=bool, default=True, help="Quantization to use")
    parser.add_argument("--LoRA", type=bool, default=False, help="LoRA to use")
    parser.add_argument("--prompt_embeds", type=str, default=None, help="Prompt embeds to use")
    parser.add_argument("--pooled_prompt_embeds", type=str, default=None, help="Pooled prompt embeds to use")
    parser.add_argument("--control_guidance_start", type=float, default=0, help="Control guidance start")
    parser.add_argument("--control_guidance_end", type=float, default=1, help="Control guidance end")
    parser.add_argument("--benchmark", type=bool, default=False, help="run benchmark with ranges pulled from user inputs +/-0.1")
    parser.add_argument("--res", type=int, default=1280, help="Resolution of the image")
    parser.add_argument("--openflux", type=bool, default=False, help="Use OpenFLUX")


    args = parser.parse_args()

    # Example usage of the ImageModelFLUX class with command-line arguments

    image_model = ImageModelFlux(
        args.image_path, 
        args.pose_path, 
        args.mask_path, 
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        # neg_prompt=args.neg_prompt,
        strength=args.strength,
        model=args.model,
        quantize=args.quantize,
        LoRA=args.LoRA,
        prompt_embeds=args.prompt_embeds,
        pooled_prompt_embeds=args.pooled_prompt_embeds,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end
    )
    
    if args.quantize==None:
        image_model.prepare_model()
    else:
        image_model.prepare_quant_model()
    if args.benchmark==None:
        image_model.prepare_image(args.image_path, args.pose_path,args.mask_path)
        generated_image, generated_image_path = image_model.generate_image()
        # generated_image = image_model.upscale_image(generated_image)  ## will explode the VRAM at this point, need to unload pipe1 or run in series
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clear_mem()
