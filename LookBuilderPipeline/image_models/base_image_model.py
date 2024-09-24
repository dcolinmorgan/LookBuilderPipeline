# Base class for image generation models (SD3, Flux, etc.)
# This contains the shared logic for handling the inputs: pose, mask, and prompt.

class BaseImageModel:
    def __init__(self, image, pose, mask, prompt):
        """
        Initialize the image model with common inputs.
        
        Args:
            pose (object): The detected pose generated earlier in the pipeline.
            mask (object): The mask generated earlier that defines the
                           boundaries of the outfit.
            prompt (str): The text prompt to guide the image generation
                          (e.g., style or additional details).
        """
        self.image = image
        self.pose = pose
        self.mask = mask
        self.prompt = prompt

    def generate_image(self):
        """
        Generate a new image using the specific model implementation.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_image()")
