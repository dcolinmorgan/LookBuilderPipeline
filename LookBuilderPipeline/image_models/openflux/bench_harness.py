import torch
import torch.nn as nn
from diffusers.utils import load_image
# from diffusers import FluxControlNetInpaintPipeline #, FluxImg2ImgPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

from diffusers.utils import load_image, check_min_version
from transformers import pipeline ,SegformerImageProcessor, AutoModelForSemanticSegmentation
import os
import time
import cProfile
from pstats import Stats
from optimum.quanto import qfloat8,qint4,qint8, quantize,freeze

from LookBuilderPipeline.resize import resize_images
from LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.pose import detect_pose
# from LookBuilderPipeline.plot_images import showImagesHorizontally

def image_setup(input_image):
    image = load_image(input_image)
    pose_image = detect_pose(input_image)
    mask,mask_image,mask_array = segment_image(input_image,inverse=True,additional_option='shoe')
    mask2,mask_image2,mask_array2 = segment_image(input_image,inverse=False,additional_option='shoe')
            
    # if image.size[0] < image.size[0]:
    #             sm_image=resize_images(image,sm_pose_image.size,aspect_ratio=sm_pose_image.size[0]/sm_pose_image.size[1])
    #             sm_pose_image=resize_images(pose_image,sm_pose_image.size,aspect_ratio=None)
    #             sm_mask=resize_images(mask_image,sm_pose_image.size,aspect_ratio=sm_pose_image.size[0]/sm_pose_image.size[1])
        
    # else:
    sm_image=resize_images(image,target_size=image.size,aspect_ratio=None)
    sm_pose_image=resize_images(pose_image,image.size,aspect_ratio=image.size[0]/image.size[1])
    sm_mask=resize_images(mask_image,image.size,aspect_ratio=image.size[0]/image.size[1])
    return sm_image, sm_pose_image, sm_mask

import numpy as np
def showImagesHorizontally(list_of_files, prompt,negative_prompt,model,time,height,width,controlnet_conditioning_scale,num_inference_steps, guidance_scale,seed,strength,output_path):
        from matplotlib.pyplot import figure, imshow, axis
        from matplotlib.image import imread
        import matplotlib.pyplot as plt

        fig = figure(figsize=(10,5))
        number_of_files = len(list_of_files)
        for i in range(number_of_files):
            a=fig.add_subplot(1,number_of_files,i+1)
            image = (list_of_files[i])
            imshow(image,cmap='Greys_r')
            axis('off')

        # Add text to the image
        fig.text(0.5, 0.01, f"Prompt: {prompt}       Neg_Prompt: {negative_prompt} \n Model: {model}  Time(s): {np.round(time,2)}  Time(m): {np.round(time/60,2)}  height: {height}  width: {width}  cond_scale: {controlnet_conditioning_scale}  steps: {num_inference_steps}  guidance: {guidance_scale} strength: {strength} seed: {seed}", ha='center', fontsize=10, color='black', wrap=True)
        text_to_save = f"""
        Prompt: {prompt} 
        Neg_Prompt: {negative_prompt}
        Model: {model}
        Time: {time}
        height: {height}
        width: {width}
        cond_scale: {controlnet_conditioning_scale}
        steps: {num_inference_steps}
        guidance: {guidance_scale}
        seed: {seed}
        strength:{strength}
        time: {time}"""
        
         # Save the text to a .txt file
        with open(output_path+'.txt', 'w') as file:  # Specify the desired file name
            file.write(text_to_save)  # Write the text to the file
        
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory





from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
dtype = torch.bfloat16
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('flux-fp8', subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained('flux-fp8', subfolder="tokenizer_2", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained('flux-fp8', subfolder="vae", torch_dtype=dtype)

controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16,add_prefix_space=True,guidance_embeds=False)


import glob
from random import shuffle
dir=glob.glob('test-ai/paulo_orig/*')
dir+=glob.glob('test-ai/orig/*')
# shuffle(dir)


if glob.glob('qopenflux/text_encoder_2_qfloat8/*.safetensors'):
    text_encoder_2 = T5EncoderModel.from_pretrained('qopenflux', subfolder="text_encoder_2_qfloat8", torch_dtype=dtype)
else:
    text_encoder_2 = T5EncoderModel.from_pretrained('flux-fp8', subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)
    text_encoder_2.save_pretrained('qopenflux/text_encoder_2_qfloat8')

# if glob.glob('qopenflux/transformer_qfloat8/*.safetensors'):
try:
    transformer = FluxTransformer2DModel.from_pretrained('qopenflux', subfolder="transformer_qfloat8", torch_dtype=dtype)
# else:
except:
    transformer = FluxTransformer2DModel.from_pretrained('flux-fp8', subfolder="transformer", torch_dtype=dtype)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)
    transformer.save_pretrained('qopenflux/transformer_qfloat8')


from diffusers import FluxControlNetInpaintPipeline #, 


def parse_args():
    parser = argparse.ArgumentParser(description="OpenFlux Benchmark")
    parser.add_argument("--guidance_scales", nargs="+", type=float, default=[3, 5, 7],
                        help="List of guidance scale values")
    parser.add_argument("--conditioning_scales", nargs="+", type=float, default=[0.5, 0.7, 0.9],
                        help="List of controlnet conditioning scale values")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.7, 0.8, 0.9, 0.99],
                        help="List of strength values")
    parser.add_argument("--inference_steps", nargs="+", type=int, default=[15, 20],
                        help="List of number of inference steps")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings, pants, sleeves",
                        help="Negative prompt for image generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    return parser.parse_args()

def main():
    args = parse_args()

    pipe = FluxControlNetInpaintPipeline(
        controlnet=controlnet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )

    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    pipe.enable_model_cpu_offload()

    sm_image, sm_pose_image, sm_mask = image_setup(args.input_image)
    width, height = sm_image.size

    for guidance_scale in args.guidance_scales:
        for controlnet_conditioning_scale in args.conditioning_scales:
            for strength in args.strengths:
                for num_inference_steps in args.inference_steps:
                    generator = torch.Generator(device="cuda").manual_seed(args.seed)
                    
                    pr = cProfile.Profile()
                    pr.enable()
                    start1 = time.time()

                    image_res = pipe(
                        args.prompt,
                        negative_prompt=args.negative_prompt,
                        image=sm_image,
                        control_image=sm_pose_image,
                        control_mode=4,
                        padding_mask_crop=32,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        mask_image=sm_mask,
                        height=height,
                        width=width,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]

                    end1 = time.time()
                    tt = end1 - start1
                    
                    pr.disable()

                    filename = f'qopenflux_benchmark/qopenflux_bench_guid_{guidance_scale}_cond_{controlnet_conditioning_scale}_strength_{strength}_ints_{num_inference_steps}.png'
                    showImagesHorizontally([sm_image, sm_mask, sm_pose_image, image_res], 
                                           args.prompt, args.negative_prompt, 'openflux', tt, 
                                           height, width, controlnet_conditioning_scale, 
                                           num_inference_steps, guidance_scale, args.seed, 
                                           strength, output_path=filename)

if __name__ == "__main__":
    main()
