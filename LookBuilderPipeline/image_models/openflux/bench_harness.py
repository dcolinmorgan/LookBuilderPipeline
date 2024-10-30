import torch
import torch.nn as nn
from diffusers.utils import load_image
# from diffusers import FluxControlNetInpaintPipeline #, FluxImg2ImgPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

from diffusers import FluxControlNetInpaintPipeline, FluxControlNetPipeline

from diffusers.utils import load_image, check_min_version
from transformers import pipeline ,SegformerImageProcessor, AutoModelForSemanticSegmentation
import os,sys
import time
import cProfile
from pstats import Stats
# from optimum.quanto import qfloat8,qint4,qint8, quantize,freeze
from quanto import qfloat8,qint4,qint8, quantize,freeze


# controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'
controlnet_model = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
# controlnet_model = 'alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha'
from diffusers.models import FluxMultiControlNetModel,FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16,add_prefix_space=True,guidance_embeds=False)
# controlnet = FluxMultiControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)#,add_prefix_space=True,guidance_embeds=False)

import glob
from random import shuffle
# dir=glob.glob('test-ai/orig/*')
# # dir+=glob.glob('test-ai/jcrew_orig/*')
# # dir=glob.glob('test-ai/paulo_orig/*')
# dir+=glob.glob('test-ai/upscaled/*')
dir=glob.glob('test-ai/orig/p05.jpeg')

# dir+=glob.glob('test-ai/more_paulo/*')

shuffle(dir)

from LookBuilderPipeline.resize import resize_images
from LookBuilderPipeline.segment import segment_image, full_mask
from LookBuilderPipeline.pose import detect_pose


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

from diffusers.utils import load_image  # For loading images
from transformers import pipeline  # For using pre-trained models
import numpy as np  # For numerical operations on arrays
from PIL import Image, ImageOps  # For image manipulation


# Initialize the segmentation model using a pre-trained model from Hugging Face
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")


def no_back(original_image):
    seg_img = load_image(sm_image)
    segments = segmenter(seg_img)
    segment_include = ["Background"]
        
    # seg_img = load_image(flux_image)
    # segments = segmenter(seg_img)
    # segment_include += ["Hat","Hair","Sunglasses","Face","Left-leg","Right-leg","Left-arm","Right-arm"]
        
        
    mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
        
    final_array = final_mask.copy()
    final_mask = Image.fromarray(final_mask)
    seg_img.putalpha(final_mask)
        
    return final_mask, seg_img
    
import numpy as np
def showImagesHorizontally(list_of_files, prompt,negative_prompt,model,time,height,width,controlnet_conditioning_scale,num_inference_steps, guidance_scale,seed,strength,control_guidance_start,control_guidance_end,output_path):
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
        fig.text(0.5, 0.01, f"Prompt: {prompt}       Neg_Prompt: {negative_prompt} \n Model: {model}  Time(s): {np.round(time,2)}  Time(m): {np.round(time/60,2)}  height: {height}  width: {width}    steps: {num_inference_steps}   seed: {seed}" "\n"
        f"cond_scale: {controlnet_conditioning_scale} guidance: {guidance_scale} strength: {strength}  Begin Cond Ratio: {control_guidance_start} End Cond Ratio:{control_guidance_end}", ha='center', fontsize=10, color='black', wrap=True)
        # fig.text(0.5, 0.0, f"cond_scale: {controlnet_conditioning_scale} guidance: {guidance_scale} strength: {strength}  Begin Cond Ratio: {control_guidance_start} End Cond Ratio:{control_guidance_end}", ha='center', fontsize=10, color='black', wrap=True)
        fig.text(1, 0.7, f"C: {controlnet_conditioning_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.65, f"G: {guidance_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.6, f"S: {strength}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.55, f"B: {control_guidance_start}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.5, f"E: {control_guidance_end}", ha='center', fontsize=8, color='black', wrap=True)



    
        # text_to_save = f"""
        # Prompt: {prompt} 
        # Neg_Prompt: {negative_prompt}
        # Model: {model}
        # Time: {time}
        # height: {height}
        # width: {width}
        # cond_scale: {controlnet_conditioning_scale}
        # steps: {num_inference_steps}
        # guidance: {guidance_scale}
        # seed: {seed}
        # strength:{strength}
        # time: {time}"""
        
        #  # Save the text to a .txt file
        # with open(output_path+'.txt', 'w') as file:  # Specify the desired file name
            # file.write(text_to_save)  # Write the text to the file
        
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory



from diffusers import FluxControlNetInpaintPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
import torch
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

# model_path = "black-forest-labs/FLUX.1-schnell"

transformer = FluxTransformer2DModel.from_pretrained(
    'flux-schnell-fp8',
    subfolder = "transformer",
    torch_dtype = torch.bfloat16
)
quantize_(transformer, int8_weight_only())

# text_encoder_2 = FluxTransformer2DModel.from_pretrained(
#     'flux-schnell-fp8',
#     subfolder = "text_encoder_2",
#     torch_dtype = torch.bfloat16
# )
# quantize_(text_encoder_2, int8_weight_only())


prompts=[ #'A photo realitic black Congolese model wearing a bikini with smokey eye makeup in red tones enjoying a tropical day at the beach in midday sun under palm trees and surfers on waves in the sea.'
# 'A very tall and slender and longer-than-average arms and neck poses looking straight into the camera. 1. Model characteristics: - Height: Very tall and slender -- Arms: Longer than average - Neck: Longer than average - Facial features: Very sharp elegant - Expression: Very intriguing - Makeup: Completely nude look - Hair: Wet and slicked back with a wide-toothed comb 2. Lighting and effects: - Eyes: Shining intensely with the flash light - Skin: Completely glossy, highlighting the lines from the flash - Shadows: Strongly defined by the flash, giving the impression the model is floating - Optical distortion: Continued - Camera settings: Aperture at f/2.8',
# 'A supermodel sits elegantly on a black volcanic beach in Iceland, surrounded by dark, rugged terrain and towering volcanic peaks. She wears high-fashion clothing, contrasting with the dramatic landscape. Her hair flows gently in the wind as she gazes into the distance, under a moody sky with soft light breaking through the clouds. The scene blends natural beauty with modern glamour, highlighting the model against the striking volcanic background.',
# prompt='A Congolese model with short, intense curly hair poses with smokey eye makeup in red tones. The lighting is soft and low, creating a warm atmosphere at a color temperature of 5400K. The background is a gradient dark gray that lightens towards the top, with her legs positioned high in the frame as she gazes toward her right ankle.'
'A supermodel sits elegantly on a black volcanic beach in Iceland, surrounded by dark, rugged terrain and towering volcanic peaks. She wears high-fashion clothing, contrasting with the dramatic landscape. Her hair flows gently in the wind as she gazes into the distance, under a moody sky with soft light breaking through the clouds. The scene blends natural beauty with modern glamour, highlighting the model against the striking volcanic background'
# 'An Italian model with long, dark blonde hair gazes flirtatiously into the camera, biting her lips. Her glossy skin and smokey blue eye makeup stand out, with bold red lips. The background is bright white with a red-magenta film burn effect. Soft, diffuse lighting highlights her flawless skin and nude-painted toenails, creating a cool, modern look.'
]
# prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
#     pipe        = pipe
#     , prompt    = prompt
# )

prompt='A supermodel sits elegantly on a black volcanic beach in Iceland, surrounded by dark, rugged terrain and towering volcanic peaks. She wears high-fashion clothing, contrasting with the dramatic landscape. She has tan skin and her blonde hair flows gently in the wind as she gazes into the distance, under a moody sky with soft light breaking through the clouds. The scene blends natural beauty with modern glamour, highlighting the model against the striking volcanic background'

# prompt_embeds2, pooled_prompt_embeds2 = get_weighted_text_embeddings_flux1(
#     pipe        = pipe2
#     , prompt    = prompt
# )


# prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
#     pipe        = pipe
#     , prompt    = prompt
# )

# promptA= 'very skinny naked person on black volcanic beach in Iceland. rugged terrain and towering volcanic peaks. striking volcanic background.'

num_inference_steps=10
model='qschnell'
shuffle(dir)
seed=42
generator = torch.Generator(device="cuda").manual_seed(seed)
negative_prompt=None
# guidance_scale=3.5
strength=0.9
controlnet_conditioning_scale=0.6
# prompt= 'a photo realistic image of a fashion model whos face is very sharp and elegant not smiling or showing teeth staring into the f/2.6 camera, at an ancient roman temple with many stone large columns and stone-paved streets at sunset'
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
blockPrint()

for input_image in dir:
    sm_image, sm_pose_image, sm_mask= image_setup(input_image)
    _,sm_image_A=no_back(sm_image)
    width,height=sm_image_A.size
    sm_image_A.save('sm_image_A.png')
    from diffusers import FluxControlNetPipeline
    pipe2 = FluxControlNetPipeline.from_pretrained(
        'flux-schnell-fp8',
        controlnet=controlnet,
        transformer = transformer,
        # text_encoder_2 = text_encoder_2,
        torch_dtype = torch.bfloat16,
    )
    
    pipe2.enable_model_cpu_offload()
    start1 = time.time()
    with torch.no_grad():
        image_res = pipe2(
                prompt=prompt, #'A photo realitic bald black Congolese model wearing a bikini with smokey eye makeup in red tones enjoying a tropical day at the beach in midday sun under palm trees and surfers on waves in the sea.',
                # prompt_embeds=prompt_embeds2,
                # pooled_prompt_embeds=pooled_prompt_embeds2,
                # image=sm_image,
                control_image=sm_pose_image,
                control_mode=4,
                # padding_mask_crop=32,
                controlnet_conditioning_scale=0.9,
                # mask_image=sm_mask,
                # control_guidance_start=control_guidance_start,
                # control_guidance_end=control_guidance_end,
                height=height,
                width=width,
                # strength=0.99,
                # num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
    image_res.save('image_res.png')

    from PIL import Image

    background = Image.open("image_res.png")
    foreground = Image.open("sm_image_A.png")
    
    background.paste(foreground, (0, 0), foreground)
    background.show()
    del pipe2
    torch.cuda.empty_cache()

    pipe = FluxControlNetInpaintPipeline.from_pretrained(
        'flux-schnell-fp8',
        controlnet=controlnet,
        transformer = transformer,
        # transformer = text_encoder_2,
        torch_dtype = torch.bfloat16,
    )
    
    pipe.enable_model_cpu_offload()


    for control_guidance_start in [0]:
        for control_guidance_end in [1]:
            for guidance_scale in [7]:
                for strength in [0.99]:
                    for controlnet_condixtioning_scale in [0.9]:
                        for jjj,prompt in enumerate(prompts):
                            i=os.path.basename(input_image).split('.')[0]
                            
                            # prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                            #     pipe        = pipe
                            #     , prompt    = prompt
                            # )
    


                            
                            # image_res = pipe(
                            #         prompt='A bald black Congolese model wearing a bikini with smokey eye makeup in red tones. The background is a tropical beach at midday with a sky that lightens towards the top.',

                            #         # prompt_embeds=prompt_embeds,
                            #         # pooled_prompt_embeds=pooled_prompt_embeds,
    
                            #         image=background,
                            #         control_image=sm_pose_image,
                            #         control_mode=4,
                            #         # padding_mask_crop=32,
                            #         # controlnet_conditioning_scale=controlnet_conditioning_scale,
                            #         mask_image=sm_mask,
                            #         # control_guidance_start=control_guidance_start,
                            #         # control_guidance_end=control_guidance_end,
                            #         height=height,
                            #         width=width,
                            #         # strength=strength,
                            #         # num_inference_steps=num_inference_steps,
                            #         # guidance_scale=guidance_scale,
                            #         generator=generator,
                            #     ).images[0]

                            # final_mask=full_mask(sm_image_A,image_res)
                            # final_mask=resize_images(final_mask,sm_image_A.size,aspect_ratio=sm_image_A.size[0]/sm_image_A.size[1])
                            with torch.no_grad():
                                image_res_A = pipe(
                                    prompt=prompt, #'A photo realitic black Congolese model wearing a bikini with smokey eye makeup in red tones enjoying a tropical day at the beach in midday sun under palm trees and surfers on waves in the sea.',
                                    # prompt_embeds=prompt_embeds,
                                    # pooled_prompt_embeds=pooled_prompt_embeds,
    
                                    image=background,
                                    control_image=sm_pose_image,
                                    control_mode=4,
                                    # padding_mask_crop=32,
                                    # controlnet_conditioning_scale=controlnet_conditioning_scale,
                                    mask_image=sm_mask,
                                    # control_guidance_start=control_guidance_start,
                                    # control_guidance_end=control_guidance_end,
                                    height=height,
                                    width=width,
                                    # strength=strength,
                                    # num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    generator=generator,
                                ).images[0]
            
                    
                            
                            image_res_A.save('image_res_A.png')
                            # image_res_A.size
                            from gradio_client import Client, file
                            access_token = "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"

                            client = Client("lllyasviel/IC-Light",hf_token=access_token)
                            result = client.predict(
                            		input_fg=file('image_res_A.png'),
                            		prompt=prompt,
                            		image_width=704,
                            		image_height=1024,
                            		num_samples=1,
                            		seed=42,
                            		steps=25,
                            		a_prompt="best quality",
                            		n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
                            		cfg=2,
                            		highres_scale=1.5,
                            		highres_denoise=0.5,
                            		lowres_denoise=0.9,
                            		bg_source="None",
                            		api_name="/process_relight"
                            )
                            AA=load_image(result[1][0]['image'])

                            end1 = time.time()
                            tt=end1-start1
                            # image_res.save('qschnell_benchmark/pitch/img'+str(i)+'_p'+str(jjj)+'_g'+str(guidance_scale)+'_c'+str(controlnet_conditioning_scale)+'_s'+str(strength)+'_b'+str(control_guidance_start)+'_e'+str(control_guidance_end)+'.png')
                            filename='qschnell_benchmark/bench/img'+str(i)+'_p'+str(jjj)+'_g'+str(guidance_scale)+'_c'+str(controlnet_conditioning_scale)+'_s'+str(strength)+'_b'+str(control_guidance_start)+'_e'+str(control_guidance_end)+'.png'
                            showImagesHorizontally([sm_image,sm_pose_image,image_res,background,sm_mask,image_res_A,AA], prompt,negative_prompt,model,tt,height, width, controlnet_conditioning_scale,num_inference_steps,guidance_scale,seed,strength,control_guidance_start,control_guidance_end,output_path=filename)
                            # filename='qschnell_benchmark/deck/img'+str(i)+'_p'+str(jjj)+'_g'+str(guidance_scale)+'_c'+str(controlnet_conditioning_scale)+'_s'+str(strength)+'_b'+str(control_guidance_start)+'_e'+str(control_guidance_end)+'.png'
                            # showImagesHorizontally([sm_image,sm_image_A,image_res], prompt,negative_prompt,model,tt,height, width, controlnet_conditioning_scale,num_inference_steps,guidance_scale,seed,strength,control_guidance_start,control_guidance_end,output_path=filename)

