from PIL import Image
from typing import Union, List

def resize_images(images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                  target_size: int = 512,
                  resample: int = Image.LANCZOS) -> Union[Image.Image, List[Image.Image]]:
    """
    Resize a single image or a list of images to the specified target size while maintaining aspect ratio.
    Ensures each dimension is divisible by 8.

    Args:
        images (str, Image.Image, or List[Union[str, Image.Image]]): 
            A single image (as file path or PIL Image object) or a list of images.
        target_size (int): The target size for the longer dimension of the resized image(s). Default is 512.
        resample (int): The resampling filter. Default is PIL.Image.LANCZOS for high-quality downsampling.

    Returns:
        Union[Image.Image, List[Image.Image]]: The resized image(s) as PIL Image object(s).

    Raises:
        ValueError: If the input type is not recognized.
    """
    def resize_single_image(img):
        if isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Calculate the new size while maintaining aspect ratio
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        return img.resize((new_width, new_height), resample=resample)

    if isinstance(images, (str, Image.Image)):
        return resize_single_image(images)
    elif isinstance(images, list):
        return [resize_single_image(img) for img in images]
    else:
        raise ValueError(f"Unsupported input type: {type(images)}")
