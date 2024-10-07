from PIL import Image, ImageOps
from typing import Union, List

def resize_images(images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                  target_size: int = 512,
                  aspect_ratio: float = 1.0,
                  resample: int = Image.LANCZOS,
                  square: bool = False) -> Union[Image.Image, List[Image.Image]]:
    """
    Resize a single image or a list of images to the specified target size while maintaining aspect ratio.
    Ensures each dimension is divisible by 8. Optionally pads the image to make it square.

    Args:
        images (str, Image.Image, or List[Union[str, Image.Image]]): 
            A single image (as file path or PIL Image object) or a list of images.
        target_size (int): The target size for the longer dimension of the resized image(s). Default is 512.
        aspect_ratio (float): The aspect ratio to maintain. Default is 1.0 (square).
        resample (int): The resampling filter. Default is PIL.Image.LANCZOS for high-quality downsampling.
        square (bool): If True, pad the image to make it square. Default is False.

    Returns:
        Union[Image.Image, List[Image.Image]]: The resized image(s) as PIL Image object(s).

    Raises:
        ValueError: If the input type is not recognized.
    """
    def resize_single_image(img,target_size,aspect_ratio,square):
        if isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        original_width, original_height = img.size
        if aspect_ratio is None:            
            aspect_ratio = original_width / original_height
        if isinstance(target_size, int):
            try:
                new_width = target_size
                new_height = int(new_width / aspect_ratio)
            except:
                new_height = target_size
                new_width = int(new_height / aspect_ratio)
        elif original_width >= original_height:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        img = img.resize((new_width, new_height), resample=resample)

        if square:
            # Calculate padding to make the image square
            max_dim = max(new_width, new_height)
            padding = (
                (max_dim - new_width) // 2,
                (max_dim - new_height) // 2,
                (max_dim - new_width + 1) // 2,
                (max_dim - new_height + 1) // 2
            )
            img = ImageOps.expand(img, padding, fill=(0, 0, 0))

        return img

    if isinstance(images, (str, Image.Image)):
        return resize_single_image(images,target_size,aspect_ratio,square=False)
    elif isinstance(images, list):
        return [resize_single_image(img) for img in images]
    else:
        raise ValueError(f"Unsupported input type: {type(images)}")
