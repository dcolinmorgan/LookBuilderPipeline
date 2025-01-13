from PIL import Image, ImageOps
from typing import Union, List
from io import BytesIO

def resize_images(images: Union[str, bytes, Image.Image, List[Union[str, bytes, Image.Image]]],
                  target_size: int = 512,
                  aspect_ratio: float = 1.0,
                  resample: int = Image.LANCZOS,
                  square: bool = False) -> Union[Image.Image, List[Image.Image]]:
    """
    Resize a single image or a list of images to the specified target size.

    Args:
        images: Image as file path, bytes, PIL Image object, or list of these
        target_size: Target size for the longer dimension
        aspect_ratio: Aspect ratio to maintain. The aspect ratio to maintain. Default is 1.0 (square).
        resample: Resampling filter. Default is PIL.Image.LANCZOS for high-quality downsampling.
        square: If True, pad to square
    """
    
    def resize_single_image(img, target_size=512, aspect_ratio=1.0, square=False):
        try:
            colors = img.convert('RGB').getcolors(maxcolors=256)  # Get colors and their counts
            most_prevalent_color = max(colors, key=lambda item: item[0])[1][1]  # Get the color with the highest count
        except:
            most_prevalent_color=0
        
        # Convert input to PIL Image
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        elif isinstance(img, Image.Image):
            img = img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Get original dimensions
        original_width, original_height = img.size
        
        # Convert target_size to int if it's not already
        if isinstance(target_size, (int, float)):
            target_size = int(target_size)
        
        # Calculate resize dimensions maintaining aspect ratio
        if original_width >= original_height:
            resize_height = int((original_height / original_width) * target_size)
            resize_width = target_size
        else:
            resize_width = int((original_width / original_height) * target_size)
            resize_height = target_size
        
        # Resize first
        img = img.resize((resize_width, resize_height), resample=resample)
        
        # Add padding to make square if requested
        if square:
            # Calculate padding to reach target size
            padding = (
                (target_size - resize_width) // 2,
                (target_size - resize_height) // 2,
                (target_size - resize_width + 1) // 2,
                (target_size - resize_height + 1) // 2
            )
            img = ImageOps.expand(img, padding, fill=most_prevalent_color)
            print(f"Debug - Final size with padding: {img.size}")

        return img
    if isinstance(images, (str, bytes, Image.Image)):
        return resize_single_image(images, target_size, aspect_ratio, square)
    elif isinstance(images, list):
        return [resize_single_image(img, target_size, aspect_ratio, square) for img in images]
    else:
        raise ValueError(f"Unsupported input type: {type(images)}")

