# Base class for image generation models (SD3, Flux, etc.)
# This contains the shared logic for handling the inputs: pose, clothes, mask, and prompt.

class BaseImageModel:
    def __init__(self, pose, clothes, mask, prompt):
        """
        Initialize the image model with common inputs.
        
        Args:
            pose (object): The detected pose generated earlier in the pipeline.
            clothes (object): The segmented clothes (outfit) generated earlier.
            mask (object): The mask generated earlier that defines the boundaries of the outfit.
            prompt (str): The text prompt to guide the image generation (e.g., style or additional details).
        """
        self.pose = pose
        self.clothes = clothes
        self.mask = mask
        self.prompt = prompt

    def generate_image(self):
        """
        Method to be implemented by subclasses. 
        Each model (SD3, Flux, etc.) will implement this method differently.
        """
        raise NotImplementedError("Subclasses must implement this method")
