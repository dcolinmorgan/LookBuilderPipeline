from diffusers.utils import load_image  # For loading images
from .resize import resize_images
from PIL import Image  # Importing PIL for image manipulation

def pad_image(image_path, resize=False, size=(512, 512)):
    """
    
    Args:
        image_path (str): Path to the input image.
        resize (bool): Whether to resize the output image. Default is False.
        size (tuple): The target size for resizing the output image. Default is (512, 512).
        
    Returns:
        PIL.Image: The padded image.
    """
    # Load the image from the specified path and convert it to RGB format
    image = load_image(image_path).convert("RGB")
    
    # Calculate the size of the padding needed
    max_size = max(image.size)
    new_image = Image.new("RGB", (max_size, max_size), (255, 255, 255))  # Create a new white square image
    new_image.paste(image, ((max_size - image.size[0]) // 2, (max_size - image.size[1]) // 2))  # Center the original image
    
    if resize:
        new_image = new_image.resize(size)  # Resize if specified
    
    return new_image  # Return the padded image
