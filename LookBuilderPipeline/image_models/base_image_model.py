import numpy as np
from diffusers.utils import load_image
from LookBuilderPipeline.LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.LookBuilderPipeline.pose import detect_pose

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


    def showImagesHorizontally(list_of_files, output_path='output.png'):
        fig = figure()
        number_of_files = len(list_of_files)
        for i in range(number_of_files):
            a=fig.add_subplot(1,number_of_files,i+1)
            image = (list_of_files[i])
            imshow(image,cmap='Greys_r')
            axis('off')
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory
        
    @staticmethod
    def generate_image_extras(image,inv=False):
        """
        used to generate extra images for the various controlnets
        """
        
        label = str(np.random.randint(100000000))
        pose_image = detect_pose(image)
        final_mask = segment(image,inv)

        showImagesHorizontally([image,pose_image,final_mask],'input'+label+'.png')

        return pose_image, final_mask
