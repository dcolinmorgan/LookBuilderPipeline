import numpy as np
from diffusers.utils import load_image
from LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.pose import detect_pose

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


    @staticmethod
    def generate_image_extras(image,inv=False):
        """
        used to generate extra images for the various controlnets
        """
        
        # label = str(np.random.randint(100000000))
        image=load_image(image)
        pose_image = detect_pose(image)
        _,final_mask,_ = segment_image(image,inverse=inv)

        return pose_image, final_mask


    def showImagesHorizontally(self,list_of_files,output_path):
        from matplotlib.pyplot import figure, imshow, axis
        from matplotlib.image import imread
        import matplotlib.pyplot as plt

        fig = figure()
        number_of_files = len(list_of_files)
        for i in range(number_of_files):
            a=fig.add_subplot(1,number_of_files,i+1)
            image = (list_of_files[i])
            imshow(image,cmap='Greys_r')
            axis('off')

        # Add text to the image
        fig.text(0.5, 0.01, f"Prompt: {self.prompt} \n Neg_Prompt: {self.negative_prompt} \n Model: {self.model} \n height: {self.height} width: {self.width} \n cond_scale: {self.controlnet_conditioning_scale} steps: {self.num_inference_steps} \n guidance: {self.guidance_scale} seed: {self.generator}", ha='center', fontsize=10, color='black')

        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory
