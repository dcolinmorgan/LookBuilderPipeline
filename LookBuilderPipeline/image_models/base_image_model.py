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
