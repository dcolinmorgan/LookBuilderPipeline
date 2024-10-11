import sys, os, shutil
import uuid
import time
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.resize import resize_images

# Import required components from diffusers
from diffusers import FluxControlNetInpaintPipeline, FluxPipeline, FluxInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

# ## change flux inpainting pipeline to allow negative-prompt in OpenFlux
PTH=(os.path.abspath(inspect.getfile(FluxPipeline)))
orig_pipe='/'.join(PTH.split('/')[:-1])+'/pipeline_flux_controlnet_inpainting.py'
mod_pipe='LookBuilderPipeline/LookBuilderPipeline/image_models/openflux/pipeline_flux_controlnet_inpainting.py'
shutil.copy(mod_pipe, orig_pipe)

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
        self.model = 'flux'


    def prepare_image(self):
        """
        Prepare the pose and mask images to generate a new image using the diffusion model.
        """
        # Load and resize images
        image = load_image(self.image)
        if isinstance(self.pose,str):
            pose_image = load_image(self.pose)
        else:
            pose_image = self.pose
        if isinstance(self.mask,str):
            mask_image = load_image(self.mask)
        else:
            mask_image = self.mask

        if pose_image.size[0] < image.size[0]:  ## resize to pose image size if it is smaller
            self.sm_image=resize_images(image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
            self.sm_pose_image=resize_images(pose_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
            self.sm_mask=resize_images(mask_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
            
        else:
            self.sm_image=resize_images(image,image.size,aspect_ratio=image.size[0]/image.size[1])
            self.sm_pose_image=resize_images(pose_image,image.size,aspect_ratio=image.size[0]/image.size[1])
            self.sm_mask=resize_images(mask_image,image.size,aspect_ratio=image.size[0]/image.size[1])
            
        self.width, self.height = self.sm_image.size

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
            ).images[0]
        end_time = time.time()
        self.time = end_time - start_time
        
        # Save the generated image
        os.makedirs(os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model), exist_ok=True)

        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("LookBuilderPipeline","LookBuilderPipeline","generated_images", self.model, filename)
        image_res.save(save_path)
        bench_filename = f"{uuid.uuid4()}.png"
        bench_save_path = os.path.join("generated_images", self.model, 'bench'+bench_filename)
        ImageModelFlux.showImagesHorizontally(self,list_of_files=[self.sm_image,self.sm_pose_image,self.sm_mask,image_res],output_path=bench_save_path)
     

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
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strength", type=float, default=0.99, help="Strength of the transformation")  # Add strength argument

    args = parser.parse_args()

    # Example usage of the ImageModelFLUX class with command-line arguments
    if args.pose_path is None or args.mask_path is None:
        args.pose_path, args.mask_path = ImageModelFlux.generate_image_extras(args.image_path,inv=True)

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
        strength=args.strength
    )
    image_model.prepare_image()
    image_model.prepare_model()
    generated_image, generated_image_path = image_model.generate_image()
    print(f"Generated image saved at: {generated_image_path}")
    image_model.clearn_mem()
