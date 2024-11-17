import numpy as np
from diffusers.utils import load_image
from LookBuilderPipeline.segment import segment_image, no_back, full_mask
from LookBuilderPipeline.pose import detect_pose
from LookBuilderPipeline.utils.resize import resize_images
from PIL import Image, ImageOps
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import textwrap
class BaseImageModel:
    def __init__(self, img, pose, mask, prompt):
        """
        Initialize the image model with common inputs.
        
        Args:
            pose (object): The detected pose generated earlier in the pipeline.
            clothes (object): The segmented clothes (outfit) generated earlier.
            mask (object): The mask generated earlier that defines the boundaries of the outfit.
            prompt (str): The text prompt to guide the image generation (e.g., style or additional details).
        """
        self.image = img
        self.pose = pose
        self.mask = mask
        self.prompt = prompt


    def generate_image_extras(self,imageA,inv=False):
        """
        used to generate extra images for the various controlnets
        """
        print("running on:",imageA)
        # label = str(np.random.randint(100000000))
        image=load_image(imageA)
        pose_image = detect_pose(self,imageA)
        _,final_mask,_ = segment_image(self,imageA,inverse=inv)
        
        # _,clothes = no_back(image)

        # clothes = clothes.convert("RGBA")

        # pixdata = clothes.load()
        
        # width, height = clothes.size
        # for y in range(height):
        #     for x in range(width):
        #         # Change transparent pixels to white
        #         if pixdata[x, y][3] == 0:  # Check if the alpha value is 0 (transparent)
        #             pixdata[x, y] = (255, 255, 255, 255)  # Set to white
            

        return pose_image, final_mask#, clothes

    def showImagesHorizontally(self,list_of_files, output_path):

        fig = figure(figsize=(10,5))
        number_of_files = len(list_of_files)
        for i in range(number_of_files):
            a=fig.add_subplot(1,number_of_files,i+1)
            image = (list_of_files[i])
            imshow(image,cmap='Greys_r')
            axis('off')

        # Add text to the image
        fig.text(0.5, 0.1, f"Prompt: {textwrap.fill(self.prompt,width=200)}", ha='center', fontsize=10, color='black',wrap=True)
        fig.text(0.5, 0.05, f"Neg_Prompt: {textwrap.fill(self.negative_prompt,width=200)}", ha='center', fontsize=10, color='black',wrap=True)

        fig.text(0.5, 0.01, f" Model: {self.model}  Time(s): {np.round(self.time,2)}  Time(m): {np.round(self.time/60,2)}  height: {self.height}  width: {self.width}    steps: {self.num_inference_steps}   seed: {self.seed} \n cond_scale: {self.controlnet_conditioning_scale} guidance: {self.guidance_scale} strength: {self.strength}", ha='center', fontsize=10, color='black', wrap=True)
        fig.text(1, 0.8, f"l: {self.LoRA}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.75, f"i: {self.i}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.7, f"C: {self.controlnet_conditioning_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.65, f"G: {self.guidance_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.6, f"S: {self.strength}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.55, f"B: {self.blur}", ha='center', fontsize=8, color='black', wrap=True)
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.savefig(output_path, dpi=300,bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory

    def resize_with_padding(self, img, expected_size, color=0):
        """Resize image with padding to maintain aspect ratio"""
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding, color)


    def prepare_image(self,input_image,pose_path,mask_path):
        """
        Prepare the pose and mask images to generate a new image using the diffusion model.
        """

        if pose_path is None or mask_path is None:
            self.pose, self.mask = self.generate_image_extras(input_image,inv=True)
            
        # Load and resize images
        image = load_image(input_image)
        if isinstance(self.pose,str):
            pose_image = load_image(self.pose)
        else:
            pose_image = self.pose
        if isinstance(self.mask,str):
            mask_image = load_image(self.mask)
        else:
            mask_image = self.mask

        image=resize_images(image,image.size,aspect_ratio=None)
        pose_image=resize_images(pose_image,image.size,aspect_ratio=image.size[0]/image.size[1])
        mask_image=resize_images(mask_image,image.size,aspect_ratio=image.size[0]/image.size[1])

        self.sm_image = self.resize_with_padding(image, [self.res, self.res])
        self.sm_pose_image = self.resize_with_padding(pose_image, [self.res, self.res])
        self.sm_mask = self.resize_with_padding(mask_image, [self.res, self.res], 255)

        self.width, self.height = self.sm_image.size

