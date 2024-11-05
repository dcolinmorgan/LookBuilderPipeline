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
        # Convert input to PIL Image
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        elif isinstance(img, Image.Image):
            img = img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Rest of the resize logic remains the same
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

    if isinstance(images, (str, bytes, Image.Image)):
        return resize_single_image(images, target_size, aspect_ratio, square)
    elif isinstance(images, list):
        return [resize_single_image(img) for img in images]
    else:
        raise ValueError(f"Unsupported input type: {type(images)}")
