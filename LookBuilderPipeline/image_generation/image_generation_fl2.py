import sys, os, shutil, glob, gc
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from LookBuilderPipeline.image_generation.base_image_generation import BaseImageGeneration
from LookBuilderPipeline.utils.resize import resize_images
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

# Import required components from diffusers
from diffusers import FluxControlNetInpaintPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel, FluxControlNetPipeline, FluxMultiControlNetModel
from diffusers.models.controlnet_flux import FluxControlNetModel


class ImageGenerationFlux(BaseImageGeneration):
    def __init__(self, image, **kwargs):
        # Initialize the FLUX image model
        super().__init__(image, **kwargs)

        # Set default values not used in prepared by base_image_generation.py
        self.model = kwargs.get('model', 'schnell')
        self.quantize = kwargs.get('quantize', True)
        self.res = kwargs.get('res', 1024)
        self.openflux = kwargs.get('openflux', False)
        ## may need to add this back in later to handle larger prompts
        # self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe=self.flux_pipe, prompt=self.prompt)
    

    # def prepare_inpaint_model(self):
        """
        Load the FLUX model and controlnet.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-schnell'

        controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Union", torch_dtype=torch.bfloat16
)

        transformer=ImageGenerationFlux.prepare_quant_model(self)  # quant or not
        self.flux_pipe = FluxControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    def prepare_img2img_model(self):
        """
        Load the FLUX model and controlnet.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-schnell'

        transformer= ImageGenerationFlux.prepare_quant_model(self)  # quant or not

        self.flux_pipe = FluxImg2ImgPipeline.from_pretrained(
            base_model,
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    def prepare_upscale_image(self):
        base_model = 'black-forest-labs/FLUX.1-schnell'
        # Load pipeline
        controlnet2 = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=torch.bfloat16
        )
        transformer=ImageGenerationFlux.prepare_quant_model(self)  # quant or not
        self.flux_pipe = FluxControlNetPipeline.from_pretrained(base_model,
            controlnet=controlnet2, 
            transformer = transformer,
            torch_dtype=torch.bfloat16)
        self.flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.flux_pipe.to("cpu")


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

            ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.clear_mem()
        
        return image_res

    def generate_bench(self,image_path,pose_path,mask_path):
        for self.input_image in glob.glob(self.image):
            for self.guidance_scale in [self.guidance_scale-0.5,self.guidance_scale,self.guidance_scale+0.5]:
                for self.strength in [self.strength]:#,self.strength+0.009]:
                    for self.res in [768,1024,1280]:
                        self.prepare_image(self.input_image,pose_path,mask_path)
                        image_res, save_path = self.generate_image()

                    
    def clear_mem(self):
        del self.flux_pipe
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageGenerationFLUX")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default='schnell', help="Model to use")
    parser.add_argument("--quantize", type=bool, default=True, help="Quantization to use")
    parser.add_argument("--res", type=int, default=1280, help="Resolution of the image")
    parser.add_argument("--openflux", type=bool, default=False, help="Use OpenFLUX")


    args = parser.parse_args()

    # Example usage of the ImageGenerationFLUX class with command-line arguments

    image_model = ImageGenerationFlux(
        args.image_path, 
        
    )
    
    if args.quantize:
        image_model.prepare_quant_model()
    if args.benchmark==None:
        generated_image, generated_image_path = image_model.generate_image()
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clear_mem()
