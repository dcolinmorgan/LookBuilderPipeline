import sys, os, shutil, glob
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel#, prepare_image
from LookBuilderPipeline.resize import resize_images

# Import required components from diffusers
from diffusers import FluxControlNetInpaintPipeline, FluxPipeline, FluxInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

# ## change flux inpainting pipeline to allow negative-prompt in OpenFlux
# PTH=(os.path.abspath(inspect.getfile(FluxPipeline)))
# orig_pipe='/'.join(PTH.split('/')[:-1])+'/pipeline_flux_controlnet_inpainting.py'
# mod_pipe='LookBuilderPipeline/LookBuilderPipeline/image_models/openflux/pipeline_flux_controlnet_inpainting.py'
# shutil.copy(mod_pipe, orig_pipe)

class ImageModelFlux(BaseImageModel):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the FLUX image model
        super().__init__(image, pose, mask, prompt)
        
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 30)
        self.guidance_scale = kwargs.get('guidance_scale', 3.5)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 0.7)
        self.seed = kwargs.get('seed', np.random.randint(0, 100000000))
        self.prompt = kwargs.get('prompt', prompt)
        self.image = kwargs.get('image', image)
        self.strength = kwargs.get('strength', 0.99)
        self.model = kwargs.get('model', 'schnell')
        self.quantize = kwargs.get('quantize', None)
        self.LoRA = kwargs.get('LoRA', False)
        self.prompt_embeds=kwargs.get('prompt_embeds', None)
        self.pooled_prompt_embeds=kwargs.get('pooled_prompt_embeds', None)
        self.control_guidance_start = kwargs.get('control_guidance_start', 0.2)
        self.control_guidance_end = kwargs.get('control_guidance_end', 0.5)
        self.benchmark = kwargs.get('benchmark', False)
        

    def prepare_model(self):
        """
        Load the FLUX model and controlnet.
        """
        # Set up the pipeline
        base_model = 'black-forest-labs/FLUX.1-schnell'
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
            # snapshot_download(repo_id="ostris/OpenFLUX.1",local_dir='flux-fp8')
            snapshot_download(repo_id="black-forest-labs/FLUX.1-schnell",local_dir='flux-schnell-fp8')
        
        from diffusers import FluxControlNetInpaintPipeline, FluxTransformer2DModel
        from torchao.quantization import quantize_, int8_weight_only
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

        controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16,add_prefix_space=True,guidance_embeds=False)
        
        # model_path = "black-forest-labs/FLUX.1-schnell"

        transformer = FluxTransformer2DModel.from_pretrained(
            'flux-schnell-fp8',
            subfolder = "transformer",
            torch_dtype = torch.bfloat16
        )
        quantize_(transformer, int8_weight_only())
        
        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(
            'flux-schnell-fp8',
            controlnet=controlnet,
            transformer = transformer,
            torch_dtype = torch.bfloat16,
        )

        self.pipe.enable_model_cpu_offload()
        
        self.prompt_embeds, self.pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
            pipe=self.pipe,
            prompt=self.prompt
        )
        
        if self.LoRA:
            self.pipe.load_lora_weights('hugovntr/flux-schnell-realism', weight_name='schnell-realism_v2.3.safetensors', adapter_name="winner")

            # Activate the LoRA
            self.pipe.set_adapters(["winner"], adapter_weights=[2.0])
            from diffusers import EulerAncestralDiscreteScheduler
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        
    def generate_image(self):
        start_time = time.time()
        image_res = self.pipe(
                prompt=self.prompt,
                image=self.sm_image,
                control_image=self.sm_pose_image,
                control_mode=4,
                # padding_mask_crop=32,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                mask_image=self.sm_mask,
                height=self.height,
                width=self.width,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
                control_guidance_start=self.control_guidance_start,
                control_guidance_end=self.control_guidance_end,
            ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        self.negative_prompt=None
        # Save the generated image
        os.makedirs(os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model), exist_ok=True)

        # filename = f"{uuid.uuid4()}.png"
        # save_path = os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model, filename)
        # 
        i=os.path.basename(self.input_image).split('.')[0]
        bench_filename = 'img'+str(i)+'_g'+str(self.guidance_scale)+'_c'+str(self.controlnet_conditioning_scale)+'_s'+str(self.strength)+'_b'+str(self.control_guidance_start)+'_e'+str(self.control_guidance_end)+'.png'
        save_path1 = os.path.join("generated_images", self.model, bench_filename)
        image_res.save(save_path1)
        save_path2 = os.path.join("benchmark_images", self.model, bench_filename)
        ImageModelFlux.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res],output_path=save_path2)
     

        return image_res, save_path
    
    def generate_bench(self,image_path,pose_path,mask_path):
        for self.input_image in glob.glob(self.image):
            for self.controlnet_conditioning_scale in [self.controlnet_conditioning_scale-0.1,self.controlnet_conditioning_scale,self.controlnet_conditioning_scale+0.1]:
                for self.guidance_scale in [self.guidance_scale-1,self.guidance_scale,self.guidance_scale+1]:
                        for self.strength in [self.strength-0.1,self.strength,self.strength+0.1]:
                            for self.control_guidance_start in [self.control_guidance_start-0.1,self.control_guidance_start,self.control_guidance_start+0.1]:
                                for self.control_guidance_end in [self.control_guidance_end-0.1,self.control_guidance_end,self.control_guidance_end+0.1]:
                                    self.prepare_image(self.input_image,pose_path,mask_path)
                                    self.generate_image()

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
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.4, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the transformation")
    parser.add_argument("--model", type=str, default='schnell', help="Model to use")
    parser.add_argument("--quantize", type=str, default=None, help="Quantization to use")
    parser.add_argument("--LoRA", type=bool, default=False, help="LoRA to use")
    parser.add_argument("--prompt_embeds", type=str, default=None, help="Prompt embeds to use")
    parser.add_argument("--pooled_prompt_embeds", type=str, default=None, help="Pooled prompt embeds to use")
    parser.add_argument("--control_guidance_start", type=float, default=0.2, help="Control guidance start")
    parser.add_argument("--control_guidance_end", type=float, default=0.5, help="Control guidance end")
    parser.add_argument("--benchmark", type=bool, default=False, help="run benchmark with ranges pulled from user inputs +/-0.1")


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
        print(f"Generated image saved at: {generated_image_path}")
    else:
        image_model.generate_bench(args.image_path, args.pose_path,args.mask_path)
    
    image_model.clearn_mem()
