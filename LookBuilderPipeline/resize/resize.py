from PIL import Image
from typing import Union, List, Tuple

def resize_images(images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                  size: Tuple[int, int] = (512, 512),
                  resample: int = Image.LANCZOS) -> Union[Image.Image, List[Image.Image]]:
    """
    Resize a single image or a list of images to the specified size.

    Args:
        images (str, Image.Image, or List[Union[str, Image.Image]]): 
            A single image (as file path or PIL Image object) or a list of images.
        size (Tuple[int, int]): The target size for the resized image(s). Default is (512, 512).
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
        return img.resize(size, resample=resample)

    if isinstance(images, (str, Image.Image)):
        return resize_single_image(images)
    elif isinstance(images, list):
        return [resize_single_image(img) for img in images]
    else:
        raise ValueError(f"Unsupported input type: {type(images)}")
