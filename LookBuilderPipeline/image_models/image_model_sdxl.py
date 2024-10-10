import sys, os
import uuid
import torch
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline 
import torch.nn as nn
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.resize import resize_images

# Import required components from diffusers
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class ImageModelSDXL(BaseImageModel):
    def __init__(self, image, pose, mask, prompt, *args, **kwargs):
        # Initialize the SDXL image model
        super().__init__(image, pose, mask, prompt)
        
        # Set default values
        self.num_inference_steps = kwargs.get('num_inference_steps', 30)
        self.guidance_scale = kwargs.get('guidance_scale', 5)
        self.controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 1)
        self.seed = kwargs.get('seed', np.random.randint(0, 100000000))
        self.prompt = kwargs.get('prompt', prompt)
        self.image = kwargs.get('image', image)
        self.negative_prompt = kwargs.get('neg_prompt', "ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves")
        self.strength = kwargs.get('strength', 0.8)
        self.model = 'sdxl'

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
        Prepare model to generate a new image using the diffusion model.
        """
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # since we only have one GPU
        
        # Load the controlnet
        controlnet_model = ControlNetModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # Load the pipeline + CN
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v8",
            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )

        # Configure pipeline settings
        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)
        self.pipe.to("cuda")
        
        # Set generator
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def generate_image(self):
        """
        Generate a new image using the diffusion model based on the pose, mask and prompt.
        """
        # Generate the image using the pipeline
        image_res = self.pipe(
            prompt=self.prompt,
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
        
        # Save the generated image
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("generated_images", "sdxl", filename)
        image_res.save(save_path)
        bench_filename = f"{uuid.uuid4()}.png"
        bench_save_path = os.path.join("generated_images", "sdxl", 'bench'+bench_filename)
        ImageModelSDXL.showImagesHorizontally([self.sm_image,self.sm_pose_image,self.sm_mask,image_res],bench_save_path)
        return image_res, save_path

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
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5, help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--neg_prompt", default=None, help="Negative prompt")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength of the transformation")  # Add strength argument

    args = parser.parse_args()

    # Example usage of the ImageModelSDXL class with command-line arguments
    if args.pose_path is None or args.mask_path is None:
        args.pose_path, args.mask_path = ImageModelSDXL.generate_image_extras(args.image_path,inv=True)

    image_model = ImageModelSDXL(
        args.image_path, 
        args.pose_path, 
        args.mask_path, 
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        neg_prompt=args.neg_prompt,
        strength=args.strength
    )
    image_model.prepare_image()
    image_model.prepare_model()
    generated_image, generated_image_path = image_model.generate_image()
    print(f"Generated image saved at: {generated_image_path}")
    image_model.clearn_mem()
