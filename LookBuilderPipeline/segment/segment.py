from diffusers.utils import load_image
from transformers import pipeline
import numpy as np
from PIL import Image

segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")


def segment_image(image_path, additional_option=None, resize=(512,512)):
    """
    Function for segmenting an image and returning the outfit with optional additions.
    Args:
        image (PIL.Image or similar): The actual image object to be processed.
        additional_option (str): The additional item to segment (e.g., "shoes", "handbag"). Optional.
    Returns:
        tuple: (segmented_outfit, mask)
        segmented_outfit (object): The image of the segmented outfit.
        mask (object): A binary mask highlighting the segmented area (outfit and any additional items).
    """
    image = load_image(image_path)
    # Convert image to a format suitable for processing (e.g., a numpy array)
    segments = segmenter(image)

    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf"]
    if additional_option in ["shoe"]:
        segment_include.extend(["Left-shoe", "Right-shoe"])
    if additional_option in ["bag"]:
        segment_include.extend(["Bag"])

        
    mask_list = [np.array(s['mask']) for s in segments if s['label'] in segment_include]
    final_mask = np.sum(mask_list, axis=0)
    seg_img = Image.fromarray(final_mask.astype(np.uint8) * 255)
    
    if resize!=None:
        seg_img = seg_img.resize(resize)
    
    # Return both the segmented outfit and the mask
    return seg_img, final_mask.astype(np.uint8)
