import numpy as np
from diffusers.utils import load_image
from LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.pose import detect_pose
from LookBuilderPipeline.resize import resize_images

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


    def generate_image_extras(self,image,inv=False):
        """
        used to generate extra images for the various controlnets
        """
        print("running on:",image)
        # label = str(np.random.randint(100000000))
        image=load_image(image)
        pose_image = detect_pose(image)
        _,final_mask,_ = segment_image(image,inverse=inv)

        return pose_image, final_mask



    def showImagesHorizontally(self,list_of_files, output_path):
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
        fig.text(0.5, 0.01, f"Prompt: {self.prompt}       Neg_Prompt: {self.negative_prompt} \n Model: {self.model}  Time(s): {np.round(self.time,2)}  Time(m): {np.round(self.time/60,2)}  height: {self.height}  width: {self.width}    steps: {self.num_inference_steps}   seed: {self.seed}" "\n"
        f"cond_scale: {self.controlnet_conditioning_scale} guidance: {self.guidance_scale} strength: {self.strength}  Begin Cond Ratio: {self.control_guidance_start} End Cond Ratio:{self.control_guidance_end}", ha='center', fontsize=10, color='black', wrap=True)
        # fig.text(0.5, 0.0, f"cond_scale: {controlnet_conditioning_scale} guidance: {guidance_scale} strength: {strength}  Begin Cond Ratio: {control_guidance_start} End Cond Ratio:{control_guidance_end}", ha='center', fontsize=10, color='black', wrap=True)
        fig.text(1, 0.7, f"C: {self.controlnet_conditioning_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.65, f"G: {self.guidance_scale}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.6, f"S: {self.strength}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.55, f"B: {self.control_guidance_start}", ha='center', fontsize=8, color='black', wrap=True)
        fig.text(1, 0.5, f"E: {self.control_guidance_end}", ha='center', fontsize=8, color='black', wrap=True)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory



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

        # if pose_image.size[0] < image.size[0]:  ## resize to pose image size if it is smaller
        #     self.sm_image=resize_images(image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
        #     self.sm_pose_image=resize_images(pose_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
        #     self.sm_mask=resize_images(mask_image,pose_image.size,aspect_ratio=pose_image.size[0]/pose_image.size[1])
            
        # else:
        self.sm_image=resize_images(image,image.size,aspect_ratio=None)
        self.sm_pose_image=resize_images(pose_image,image.size,aspect_ratio=image.size[0]/image.size[1])
        self.sm_mask=resize_images(mask_image,image.size,aspect_ratio=image.size[0]/image.size[1])
            
        self.width, self.height = self.sm_image.size

